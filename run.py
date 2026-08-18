#!/usr/bin/env python3
"""KLA submission entry point. Usage: python run.py <input-dir> <output-dir>"""
import os, sys, glob, inspect
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_checkpoint():
    for rel in (("models", "best.pth"), ("checkpoints", "best.pth")):
        p = os.path.join(_HERE, *rel)
        if os.path.exists(p):
            return p
    return os.path.join(_HERE, "checkpoints", "best.pth")


def _sanitize_outputs(output_dir):
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

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
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
