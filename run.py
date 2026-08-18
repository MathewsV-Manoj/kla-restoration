"""
run.py - Official submission entry point (KLA Problem Statement,
         AI-Based Restoration of Degraded Images).

USAGE
    python run.py <input-dir> <output-dir>

    <input-dir>  : directory containing degraded .npy images (NoisyLR)
    <output-dir> : directory where restored .npy images are written
                   (created automatically if it does not exist)

BEHAVIOUR / GUARANTEES
    * Every *.npy file in <input-dir> is read and restored.
    * Exactly one output file is written per input file, with the SAME
      filename as its input.
    * Each output is a grayscale float32 array of shape (H, W) at the
      target resolution (2x the input, e.g. 128x128 -> 256x256).
    * Output values are finite and inside [0, 1] - NaN/Inf are scrubbed
      and the result is clipped before saving.
    * Weights ship with the repository (models/best.pth). No internet
      access, API keys, extra downloads, or manual configuration are
      needed. Runs on an NVIDIA GPU when available, CPU otherwise.
    * If a single image fails for any reason, a bicubic upsample of that
      input is written instead, so the output set is never incomplete.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Repository root, so the script works from any working directory.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Deterministic, stable cuDNN settings (also safe on Blackwell / sm_120).
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEFAULT_WEIGHTS = [
    os.path.join(ROOT, "models", "best.pth"),
    os.path.join(ROOT, "checkpoints", "best.pth"),
]
SCALE = 2  # 2x super-resolution: (H, W) -> (2H, 2W)


def resolve_checkpoint(explicit=None):
    """Return the path to the model weights bundled with the repo."""
    candidates = [explicit] if explicit else DEFAULT_WEIGHTS
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Model weights not found. Expected one of: "
        + ", ".join(p for p in candidates if p)
    )


def build_model(checkpoint_path, device):
    """Load the trained restoration network onto `device`, in eval mode."""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:  # older torch, or a checkpoint holding non-tensor objects
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model_type = ckpt.get("model_type") if isinstance(ckpt, dict) else None
    if model_type is None:
        # Infer the architecture from the weight keys.
        keys = list(state.keys())
        if any(k.startswith("body.0.rdb1") for k in keys):
            model_type = "v3"
        elif any("fc" in k for k in keys):
            model_type = "v2"
        else:
            model_type = "v1"

    if model_type == "v3":
        from model_v3 import RestorationNetV3 as Net
    elif model_type == "v2":
        from model_v2 import RestorationNetV2 as Net
    else:
        from model import RestorationNet as Net

    model = Net().to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {model_type} weights from {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'})")
    return model


def to_2d_float(arr):
    """Normalise an arbitrary input array to a 2-D float32 (H, W) image."""
    arr = np.asarray(arr)

    # Integer images are assumed to use the full dynamic range of their dtype.
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float32, copy=False)

    arr = np.squeeze(arr)  # (H,W,1) / (1,H,W) / (1,1,H,W) -> (H,W)
    if arr.ndim == 3:      # multi-channel: collapse to grayscale
        arr = arr.mean(axis=-1 if arr.shape[-1] <= 4 else 0)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {np.asarray(arr).shape}")

    # NoisyLR values legitimately fall outside [0,1] (speckle noise) and are
    # deliberately NOT clipped here - only non-finite values are repaired.
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)


def sanitize_output(arr, target_hw):
    """Guarantee a finite, [0,1], float32 (H, W) array at the target size."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.shape != target_hw:
        raise ValueError(f"output shape {arr.shape} != expected {target_hw}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def bicubic_fallback(lr):
    """Plain bicubic 2x upsample - used only if the network fails on an image."""
    t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0)
    up = F.interpolate(t, scale_factor=SCALE, mode="bicubic", align_corners=False)
    return up.squeeze().numpy()


def restore_directory(input_dir, output_dir, checkpoint, device):
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not files:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    model = build_model(checkpoint, device)
    print(f"Found {len(files)} .npy file(s). Restoring -> {output_dir}")

    # Warm-up so the first timed image is not charged for CUDA/cuDNN init.
    if device.type == "cuda":
        with torch.no_grad():
            model(torch.zeros(1, 1, 128, 128, device=device))
        torch.cuda.synchronize()

    times, failures = [], 0
    with torch.no_grad():
        for path in files:
            name = os.path.basename(path)
            t0 = time.perf_counter()

            lr = to_2d_float(np.load(path))
            target_hw = (lr.shape[0] * SCALE, lr.shape[1] * SCALE)

            try:
                inp = torch.from_numpy(lr)[None, None].to(device)
                out = model(inp).squeeze().detach().cpu().numpy()
                out = sanitize_output(out, target_hw)
            except Exception as exc:  # never leave an input without an output
                failures += 1
                print(f"  [warn] {name}: {exc} - falling back to bicubic")
                out = sanitize_output(bicubic_fallback(lr), target_hw)

            np.save(os.path.join(output_dir, name), out)

            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = np.asarray(times)
    total = float(times.sum())
    print("\n----- Summary -----")
    print(f"Images restored  : {len(files)}")
    if failures:
        print(f"Bicubic fallback : {failures}")
    print(f"Total time       : {total:.3f} s")
    print(f"Mean per image   : {times.mean() * 1000:.2f} ms")
    print(f"Median per image : {float(np.median(times)) * 1000:.2f} ms")
    print(f"Throughput       : {len(files) / total:.1f} images/sec")
    print(f"Output directory : {os.path.abspath(output_dir)}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Restore degraded .npy images (denoise + 2x super-resolution)."
    )
    p.add_argument("input_dir", help="Directory containing degraded .npy images")
    p.add_argument("output_dir", help="Directory for restored .npy images")
    p.add_argument("--checkpoint", default=None,
                   help="Optional path to model weights "
                        "(default: models/best.pth)")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"],
                   help="Force a device (default: cuda when available)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = resolve_checkpoint(args.checkpoint)
    restore_directory(args.input_dir, args.output_dir, checkpoint, device)


if __name__ == "__main__":
    main()
