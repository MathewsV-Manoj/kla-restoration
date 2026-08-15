"""
train_v2.py - AMP-enabled training for RestorationNet v2

Improvements over train.py:
  1. Automatic Mixed Precision (AMP) via torch.cuda.amp - trains ~2x
     faster and uses less VRAM on your RTX 5050, letting you fit the
     larger v2 model and/or a bigger batch.
  2. Uses RestorationNetV2 (channel attention) by default. A --model
     flag lets you switch back to v1 for comparison.
  3. Tracks val SSIM alongside PSNR (SSIM was the weaker metric, so we
     want to watch it directly), and selects the best checkpoint on a
     combined score so we don't optimize PSNR at SSIM's expense.
  4. Gradient clipping for training stability with AMP.

Run:
    python train_v2.py --epochs 80 --batch_size 8 --num_workers 0
"""

import os
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders
from losses import CombinedLoss, ssim as ssim_metric

# Blackwell cuDNN stability (same as v1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/train")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints_v2")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log_dir", type=str, default="runs/exp_v2")
    p.add_argument("--model", type=str, default="v2", choices=["v1", "v2"])
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    return p.parse_args()


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def build_model(name):
    if name == "v2":
        from model_v2 import RestorationNetV2, count_parameters
        return RestorationNetV2(), count_parameters
    else:
        from model import RestorationNet, count_parameters
        return RestorationNet(), count_parameters


def train():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    print(f"Using device: {device} | AMP: {use_amp} | model: {args.model}")

    train_loader, val_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        val_fraction=args.val_fraction, num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model, count_parameters = build_model(args.model)
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    best_score = 0.0  # combined PSNR + SSIM selection score

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
            with torch.cuda.amp.autocast(enabled=use_amp):
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

        # validation - track both PSNR and SSIM
        model.eval()
        val_psnr_total = 0.0
        val_ssim_total = 0.0
        n_val = 0
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                lr_img, gt_img = lr_img.to(device), gt_img.to(device)
                pred = model(lr_img)
                val_psnr_total += psnr(pred, gt_img).item()
                val_ssim_total += ssim_metric(pred, gt_img)
                n_val += 1

        avg_val_psnr = val_psnr_total / n_val
        avg_val_ssim = val_ssim_total / n_val
        elapsed = time.time() - t0

        # combined selection score: normalize PSNR to ~[0,1] range and
        # add SSIM, so both metrics influence which checkpoint we keep.
        combined = (avg_val_psnr / 40.0) + avg_val_ssim

        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} "
              f"val_psnr={avg_val_psnr:.2f}dB val_ssim={avg_val_ssim:.4f} "
              f"time={elapsed:.1f}s")

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
            "model_type": args.model,
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pth"))

        if combined > best_score:
            best_score = combined
            ckpt["best_score"] = best_score
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pth"))
            print(f"  -> New best model saved "
                  f"(PSNR={avg_val_psnr:.2f}dB SSIM={avg_val_ssim:.4f})")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    train()
