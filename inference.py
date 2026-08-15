"""
inference.py - Standalone inference script (KLA Hackathon 2026, Track 1)

This is the file KLA's evaluators run AS-IS to restore degraded images and
benchmark inference time. It runs with zero manual edits.

USAGE (as specified in the submission requirements):

    python inference.py <input_dir> <output_dir>

    - <input_dir>  : directory containing degraded NoisyLR .npy images
    - <output_dir> : directory where restored .npy images are written

Example:
    python inference.py data/Test_NoisyLR/NoisyLR outputs/restored

The trained weights load automatically from ./checkpoints/best.pth. Each
restored image is saved as a 256x256 float32 .npy under the same filename
as its input. The script prints end-to-end inference-speed statistics
(mean / median per-image latency and throughput).

End-to-end timing here includes disk read, CPU->GPU transfer, model
execution, GPU->CPU transfer, and saving - matching the runtime
definition in the KLA help document.
"""

import os
import glob
import time
import argparse

import numpy as np
import torch

# Blackwell (sm_120) cuDNN stability - safe defaults on any GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state"]
    model_type = ckpt.get("model_type")
    if model_type is None:
        # infer architecture from weight keys
        is_v2 = any("body" in k and "fc" in k for k in state.keys())
        model_type = "v2" if is_v2 else "v1"

    if model_type == "v3":
        from model_v3 import RestorationNetV3
        model = RestorationNetV3().to(device)
    elif model_type == "v2":
        from model_v2 import RestorationNetV2
        model = RestorationNetV2().to(device)
    else:
        from model import RestorationNet
        model = RestorationNet().to(device)

    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {model_type} model from {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', '?')})")
    return model


def run_inference(input_dir, output_dir, checkpoint, device):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not files:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    model = load_model(checkpoint, device)
    print(f"Found {len(files)} images. Running inference...")

    # GPU warm-up so the first real image isn't penalised by CUDA init.
    if device.type == "cuda":
        warm = torch.randn(1, 1, 128, 128, device=device)
        with torch.no_grad():
            _ = model(warm)
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for f in files:
            t0 = time.perf_counter()

            lr = np.load(f).astype(np.float32)          # (128,128), unclamped
            lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(lr_t)                           # (1,1,256,256), [0,1]
            out = pred.squeeze().cpu().numpy().astype(np.float32)
            np.save(os.path.join(output_dir, os.path.basename(f)), out)

            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times = np.array(times)
    total = times.sum()
    print("\n----- Inference speed (end-to-end) -----")
    print(f"Images processed : {len(files)}")
    print(f"Total time       : {total:.3f} s")
    print(f"Mean per image   : {times.mean()*1000:.2f} ms")
    print(f"Median per image : {np.median(times)*1000:.2f} ms")
    print(f"Throughput       : {len(files)/total:.1f} images/sec")
    print(f"Restored outputs saved to: {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Standalone restoration inference")
    p.add_argument("input", help="Input directory of degraded NoisyLR .npy images")
    p.add_argument("output", nargs="?", default="outputs/restored",
                   help="Output directory for restored .npy images")
    p.add_argument("--checkpoint", default="checkpoints/best.pth",
                   help="Path to model weights (default: checkpoints/best.pth)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run_inference(args.input, args.output, args.checkpoint, device)


if __name__ == "__main__":
    main()
