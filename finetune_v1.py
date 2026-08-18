"""
finetune_v1.py - Fine-tune the trained v1 model to boost SSIM.

Strategy (low-risk, targeted):
  - START from the existing v1 best.pth weights (epoch 83). We are NOT
    training from scratch - just refining a working model.
  - ADD a gradient/edge loss term to the existing L1+SSIM loss. This
    penalises differences in image gradients (Sobel-style), which
    directly rewards recovering sharp edges and fine structure - exactly
    what SSIM measures. Restoration models often under-sharpen; this
    pushes back on that.
  - LOW learning rate (2e-5) so we refine, not destroy, the learned
    weights. Short schedule (15 epochs).
  - Saves to checkpoints_ft/ - your original v1 stays untouched, so this
    can only help, never hurt (worst case, we keep the original).

Run:
    python finetune_v1.py --epochs 15 --batch_size 4 --num_workers 0
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders
from losses import CombinedLoss, ssim as ssim_metric
from model import RestorationNet, count_parameters

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Sobel edge/gradient loss ------------------------------------------------
class GradientLoss(nn.Module):
    """L1 distance between the image gradients (Sobel) of pred and target.
    Rewards matching edges / high-frequency structure -> helps SSIM."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, pred, target):
        # Match kernel dtype to input (handles AMP float16 vs float32).
        kx = self.kx.to(device=pred.device, dtype=pred.dtype)
        ky = self.ky.to(device=pred.device, dtype=pred.dtype)
        gx_p = F.conv2d(pred, kx, padding=1)
        gy_p = F.conv2d(pred, ky, padding=1)
        gx_t = F.conv2d(target, kx, padding=1)
        gy_t = F.conv2d(target, ky, padding=1)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


class FineTuneLoss(nn.Module):
    """Combined L1+SSIM (as before) plus a weighted gradient term."""

    def __init__(self, grad_weight=0.1):
        super().__init__()
        self.base = CombinedLoss()
        self.grad = GradientLoss()
        self.grad_weight = grad_weight

    def forward(self, pred, target):
        base_loss, parts = self.base(pred, target)
        # Compute gradient loss in float32 to avoid AMP dtype conflicts.
        g = self.grad(pred.float(), target.float())
        total = base_loss + self.grad_weight * g
        parts["grad"] = g.item()
        return total, parts


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/train")
    p.add_argument("--init", default="models/best.pth",
                   help="Starting v1 weights to fine-tune from")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--grad_weight", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--checkpoint_dir", default="checkpoints_ft")
    p.add_argument("--log_dir", default="runs/exp_ft")
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def train():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    train_loader, val_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        val_fraction=0.1, num_workers=args.num_workers,
    )

    model = RestorationNet().to(device)
    ckpt = torch.load(args.init, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Fine-tuning from {args.init} (epoch {ckpt.get('epoch','?')}, "
          f"val PSNR {ckpt.get('best_val_psnr','?')})")
    print(f"Params: {count_parameters(model):,}")

    criterion = FineTuneLoss(grad_weight=args.grad_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    writer = SummaryWriter(args.log_dir)

    # Establish the starting-point val score so we only save improvements.
    def evaluate():
        model.eval(); vp, vs, n = 0.0, 0.0, 0
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                lr_img, gt_img = lr_img.to(device), gt_img.to(device)
                pred = model(lr_img)
                vp += psnr(pred, gt_img).item(); vs += ssim_metric(pred, gt_img); n += 1
        return vp / n, vs / n

    base_psnr, base_ssim = evaluate()
    base_combined = base_psnr / 40.0 + base_ssim
    print(f"Starting val: PSNR {base_psnr:.3f} dB  SSIM {base_ssim:.4f}")
    best_combined = base_combined

    for epoch in range(args.epochs):
        model.train(); ep_loss = 0.0; t0 = time.time()
        for i, (lr_img, gt_img) in enumerate(train_loader):
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(lr_img)
            # Loss in float32 (standard AMP practice, avoids dtype errors).
            loss, parts = criterion(pred.float(), gt_img)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            ep_loss += loss.item()
            if i % 40 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} grad={parts['grad']:.4f}")
        scheduler.step()

        vp, vs = evaluate()
        combined = vp / 40.0 + vs
        print(f"[Epoch {epoch}] val_psnr={vp:.3f}dB val_ssim={vs:.4f} "
              f"time={time.time()-t0:.1f}s")
        writer.add_scalar("PSNR/val", vp, epoch)
        writer.add_scalar("SSIM/val", vs, epoch)

        if combined > best_combined:
            best_combined = combined
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_psnr": vp, "val_ssim": vs, "model_type": "v1",
                "best_val_psnr": vp,
            }, os.path.join(args.checkpoint_dir, "best.pth"))
            print(f"  -> Improved & saved (PSNR={vp:.3f} SSIM={vs:.4f})")

    writer.close()
    print("\nFine-tuning complete.")
    print(f"Original v1:  PSNR {base_psnr:.3f}  SSIM {base_ssim:.4f}")
    print("Fine-tuned best saved to checkpoints_ft/best.pth (only if it improved).")
    print("Compare with: python evaluate_pro.py data/train --paired --limit 200 "
          "--checkpoint checkpoints_ft/best.pth --tta")


if __name__ == "__main__":
    train()

