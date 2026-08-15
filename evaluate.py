"""
evaluate.py - Standalone inference + evaluation for RestorationNet

Two modes:

1) TEST INFERENCE (default): runs the trained model on every .npy in the
   test folder, saves restored 256x256 outputs, and reports inference
   speed (total + per-image latency + throughput).

       python evaluate.py --checkpoint checkpoints/best.pth \
           --input data/Test_NoisyLR/NoisyLR --output outputs/restored

2) PAIRED METRICS (--paired): runs on a folder that has both GT/ and
   NoisyLR/ subfolders (i.e. your training data) and reports mean
   PSNR and SSIM against ground truth. Use this to get the numbers
   for your innovation slide.

       python evaluate.py --checkpoint checkpoints/best.pth \
           --input data/train --paired

The script is fully self-contained: given a checkpoint and an input
folder it reproduces submission outputs from scratch. This is the
standalone evaluation deliverable required by the hackathon.
"""

import os
import glob
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model import RestorationNet

# Same Blackwell cuDNN stability flags used during training.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return (20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)).item()


def gaussian_window(window_size, sigma, channels, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = g.unsqueeze(0) * g.unsqueeze(1)
    w2d = w2d.unsqueeze(0).unsqueeze(0)
    return w2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    channels = img1.size(1)
    window = gaussian_window(window_size, sigma, channels, img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def load_model(checkpoint_path, device):
    model = RestorationNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    best = ckpt.get("best_val_psnr", None)
    msg = f"Loaded checkpoint from epoch {epoch}"
    if best is not None:
        msg += f" (val PSNR {best:.3f} dB)"
    print(msg)
    return model


# --------------------------------------------------------------------------
# Mode 1: test inference
# --------------------------------------------------------------------------
def run_test_inference(model, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not files:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    print(f"Found {len(files)} test images. Running inference...")

    # GPU warm-up (first CUDA call includes one-time init cost that would
    # otherwise pollute the timing of the first real image).
    if device.type == "cuda":
        dummy = torch.randn(1, 1, 128, 128, device=device)
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()

    per_image_times = []
    with torch.no_grad():
        for f in files:
            lr = np.load(f).astype(np.float32)          # (128,128), unclamped
            lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            pred = model(lr_t)                          # (1,1,256,256), clamped [0,1]

            if device.type == "cuda":
                torch.cuda.synchronize()
            per_image_times.append(time.perf_counter() - t0)

            out = pred.squeeze().cpu().numpy().astype(np.float32)
            out_name = os.path.basename(f)
            np.save(os.path.join(output_dir, out_name), out)

    times = np.array(per_image_times)
    total = times.sum()
    print("\n----- Inference speed -----")
    print(f"Images processed : {len(files)}")
    print(f"Total time       : {total:.3f} s")
    print(f"Mean per image   : {times.mean()*1000:.2f} ms")
    print(f"Median per image : {np.median(times)*1000:.2f} ms")
    print(f"Throughput       : {len(files)/total:.1f} images/sec")
    print(f"Restored outputs saved to: {output_dir}")


# --------------------------------------------------------------------------
# Mode 2: paired metrics (needs GT)
# --------------------------------------------------------------------------
def run_paired_eval(model, root_dir, device, limit=None):
    gt_dir = os.path.join(root_dir, "GT")
    lr_dir = os.path.join(root_dir, "NoisyLR")
    files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    if limit:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No GT files found in {gt_dir}")

    print(f"Evaluating on {len(files)} paired samples...")

    psnr_vals, ssim_vals = [], []
    with torch.no_grad():
        for f in files:
            name = os.path.basename(f)
            gt = np.clip(np.load(f).astype(np.float32), 0.0, 1.0)
            lr = np.load(os.path.join(lr_dir, name)).astype(np.float32)

            lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
            gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)

            pred = model(lr_t)
            psnr_vals.append(psnr(pred, gt_t))
            ssim_vals.append(ssim(pred, gt_t))

    print("\n----- Paired metrics -----")
    print(f"Samples   : {len(files)}")
    print(f"Mean PSNR : {np.mean(psnr_vals):.3f} dB")
    print(f"Mean SSIM : {np.mean(ssim_vals):.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    p.add_argument("--input", type=str, default="data/Test_NoisyLR/NoisyLR")
    p.add_argument("--output", type=str, default="outputs/restored")
    p.add_argument("--paired", action="store_true",
                   help="Run paired PSNR/SSIM eval (input must have GT/ and NoisyLR/)")
    p.add_argument("--limit", type=int, default=None,
                   help="In paired mode, evaluate only the first N samples")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, device)

    if args.paired:
        run_paired_eval(model, args.input, device, limit=args.limit)
    else:
        run_test_inference(model, args.input, args.output, device)


if __name__ == "__main__":
    main()
