"""
train.py - Training loop for RestorationNet
"""

import os
import time
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloaders
from model import RestorationNet, count_parameters
from losses import CombinedLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/train")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log_dir", type=str, default="runs/exp1")
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
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size,
        val_fraction=args.val_fraction, num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model = RestorationNet().to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_psnr = 0.0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_psnr = ckpt.get("best_val_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    writer = SummaryWriter(args.log_dir)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, (lr_img, gt_img) in enumerate(train_loader):
            lr_img, gt_img = lr_img.to(device), gt_img.to(device)

            optimizer.zero_grad()
            pred = model(lr_img)
            loss, parts = criterion(pred, gt_img)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} l1={parts['l1']:.4f} "
                      f"ssim_loss={parts['ssim_loss']:.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        # validation
        model.eval()
        val_psnr_total = 0.0
        val_loss_total = 0.0
        with torch.no_grad():
            for lr_img, gt_img in val_loader:
                lr_img, gt_img = lr_img.to(device), gt_img.to(device)
                pred = model(lr_img)
                loss, _ = criterion(pred, gt_img)
                val_loss_total += loss.item()
                val_psnr_total += psnr(pred, gt_img).item()

        avg_val_loss = val_loss_total / len(val_loader)
        avg_val_psnr = val_psnr_total / len(val_loader)
        elapsed = time.time() - t0

        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} "
              f"val_loss={avg_val_loss:.4f} val_psnr={avg_val_psnr:.2f}dB "
              f"time={elapsed:.1f}s")

        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("PSNR/val", avg_val_psnr, epoch)

        # save checkpoints
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_psnr": best_val_psnr,
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pth"))

        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            ckpt["best_val_psnr"] = best_val_psnr
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pth"))
            print(f"  -> New best model saved (PSNR={best_val_psnr:.2f}dB)")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    train()
