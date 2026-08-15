"""
train_v3.py - AMP training for RestorationNet v3 (RRDB + PixelShuffle)

Tuned for stable convergence of the deeper RRDB network:
  - Lower LR (1e-4) than v2's unstable 2e-4.
  - Short linear warmup (5 epochs) then cosine annealing. Warmup is
    important for deep residual-dense nets - it prevents the early
    large-gradient divergence that made v2 wobble.
  - AMP mixed precision (fits the 6M-param model on 8GB).
  - Gradient clipping (max_norm=1.0).
  - Selects best checkpoint on combined PSNR+SSIM (same as v2).

Run (batch 4 is safest for 8GB with this bigger model):
    python train_v3.py --epochs 80 --batch_size 4 --num_workers 0
"""

import os
import time
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders
from losses import CombinedLoss, ssim as ssim_metric
from model_v3 import RestorationNetV3, count_parameters

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/train")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_v3")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log_dir", type=str, default="runs/exp_v3")
    p.add_argument("--no_amp", action="store_true")
    return p.parse_args()


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def train():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Using device: {device} | AMP: {use_amp} | model: v3 (RRDB)")

    train_loader, val_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        val_fraction=args.val_fraction, num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model = RestorationNetV3().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # warmup + cosine schedule
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        # cosine from 1.0 down to ~0 over the remaining epochs
        import math
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_score = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(args.log_dir)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, (lr_img, gt_img) in enumerate(train_loader):
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(lr_img)
                loss, parts = criterion(pred, gt_img)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} l1={parts['l1']:.4f} "
                      f"ssim_loss={parts['ssim_loss']:.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        model.eval()
        vp, vs, nv = 0.0, 0.0, 0
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                lr_img, gt_img = lr_img.to(device), gt_img.to(device)
                pred = model(lr_img)
                vp += psnr(pred, gt_img).item()
                vs += ssim_metric(pred, gt_img)
                nv += 1

        avg_val_psnr = vp / nv
        avg_val_ssim = vs / nv
        elapsed = time.time() - t0
        combined = (avg_val_psnr / 40.0) + avg_val_ssim

        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} "
              f"val_psnr={avg_val_psnr:.2f}dB val_ssim={avg_val_ssim:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} time={elapsed:.1f}s")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("PSNR/val", avg_val_psnr, epoch)
        writer.add_scalar("SSIM/val", avg_val_ssim, epoch)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
            "val_psnr": avg_val_psnr,
            "val_ssim": avg_val_ssim,
            "model_type": "v3",
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pth"))

        if combined > best_score:
            best_score = combined
            ckpt["best_score"] = best_score
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pth"))
            print(f"  -> New best (PSNR={avg_val_psnr:.2f}dB SSIM={avg_val_ssim:.4f})")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    train()
