#!/usr/bin/env python3
"""KLA submission entry point. Usage: python run.py <input-dir> <output-dir>"""
import os, sys, glob, inspect
import numpy as np
import torch


def _find_checkpoint():
    for p in ("models/best.pth", "checkpoints/best.pth"):
        if os.path.exists(p):
            return p
    return "checkpoints/best.pth"


def _sanitize_outputs(output_dir):
    # Guarantee the graded contract on every saved file: grayscale, [0,1], no NaN/Inf.
    for f in glob.glob(os.path.join(output_dir, "*.npy")):
        a = np.load(f).astype(np.float32)
        a = np.squeeze(a)
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        a = np.clip(a, 0.0, 1.0).astype(np.float32)
        np.save(f, a)


def main():
    if len(sys.argv) < 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)
    input_dir, output_dir = sys.argv[1], sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _find_checkpoint()

    import inference as inf
    fn = inf.run_inference
    names = list(inspect.signature(fn).parameters)
    provided = {
        "input_dir": input_dir, "input": input_dir, "in_dir": input_dir,
        "output_dir": output_dir, "output": output_dir, "out_dir": output_dir,
        "checkpoint": ckpt, "checkpoint_path": ckpt, "ckpt": ckpt, "weights": ckpt,
        "device": device,
    }
    kwargs = {n: provided[n] for n in names if n in provided}
    has_in = any(k in kwargs for k in ("input_dir", "input", "in_dir"))
    has_out = any(k in kwargs for k in ("output_dir", "output", "out_dir"))
    if has_in and has_out:
        fn(**kwargs)
    else:
        args = [input_dir, output_dir]
        if len(names) >= 3: args.append(ckpt)
        if len(names) >= 4: args.append(device)
        fn(*args)

    _sanitize_outputs(output_dir)


if __name__ == "__main__":
    main()
