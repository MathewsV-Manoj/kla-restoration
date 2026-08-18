"""
evaluate_pro.py - Advanced inference: Test-Time Augmentation + Ensemble

Extracts maximum quality from already-trained models WITHOUT retraining.

Three techniques, each measurable independently:

1. TTA (Test-Time Augmentation / self-ensemble):
   Run each image through the model in all 8 dihedral orientations
   (4 rotations x 2 flips), invert each transform on the output, and
   average. Because the model makes slightly different errors on each
   orientation, averaging cancels noise -> reliably +0.3-0.7 dB. This
   is standard practice in every competitive super-resolution pipeline
   (EDSR, RCAN all report "self-ensemble" / "+" results).

2. Model ensemble:
   Average the outputs of two independently trained models (v1 and v3).
   Their errors are partially uncorrelated, so the average is often
   better than either alone - even when one model is individually weaker.

3. Both combined: TTA applied to the ensemble.

USAGE

  Metrics on validation data (needs GT/ and NoisyLR/):
    # single model, no TTA (baseline)
    python evaluate_pro.py data/train --paired --limit 200

    # single model + TTA
    python evaluate_pro.py data/train --paired --limit 200 --tta

    # ensemble v1+v3 + TTA (maximum quality)
    python evaluate_pro.py data/train --paired --limit 200 --tta \
        --checkpoint models/best.pth --checkpoint2 checkpoints_v3/best.pth

  Generate final restored test outputs (maximum quality):
    python evaluate_pro.py data/Test_NoisyLR/NoisyLR outputs/restored --tta \
        --checkpoint models/best.pth --checkpoint2 checkpoints_v3/best.pth
"""

import os
import glob
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ------------------------- metrics -------------------------
def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse))


def _win(ws, sigma, ch, device):
    c = torch.arange(ws, dtype=torch.float32, device=device) - ws // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    w = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return w.expand(ch, 1, ws, ws).contiguous()


def ssim(a, b, ws=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    ch = a.size(1); w = _win(ws, sigma, ch, a.device)
    mu1 = F.conv2d(a, w, padding=ws//2, groups=ch)
    mu2 = F.conv2d(b, w, padding=ws//2, groups=ch)
    m1s, m2s, m12 = mu1**2, mu2**2, mu1*mu2
    s1 = F.conv2d(a*a, w, padding=ws//2, groups=ch) - m1s
    s2 = F.conv2d(b*b, w, padding=ws//2, groups=ch) - m2s
    s12 = F.conv2d(a*b, w, padding=ws//2, groups=ch) - m12
    smap = ((2*m12+C1)*(2*s12+C2)) / ((m1s+m2s+C1)*(s1+s2+C2))
    return float(smap.mean())


_LPIPS = None
def try_lpips(pred, target, device):
    global _LPIPS
    try:
        import lpips
    except ImportError:
        return None
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net="alex").to(device); _LPIPS.eval()
    p = (pred*2-1).repeat(1,3,1,1); t = (target*2-1).repeat(1,3,1,1)
    with torch.no_grad():
        return float(_LPIPS(p, t).mean())


# ------------------------- model loading -------------------------
def load_one(path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state"]
    mt = ckpt.get("model_type")
    if mt is None:
        mt = "v2" if any("fc" in k and "body" in k for k in state) else "v1"
    if mt == "v3":
        from model_v3 import RestorationNetV3 as M; m = M()
    elif mt == "v2":
        from model_v2 import RestorationNetV2 as M; m = M()
    else:
        from model import RestorationNet as M; m = M()
    m.load_state_dict(state); m.to(device).eval()
    print(f"Loaded {mt} from {path} (epoch {ckpt.get('epoch','?')})")
    return m


# ------------------------- TTA -------------------------
# 8 dihedral transforms and their inverses, applied on (1,1,H,W) tensors.
def _forward_transform(x, i):
    # i in 0..7 : rotations k=i%4, with optional horizontal flip
    k = i % 4
    xf = torch.flip(x, dims=[3]) if i >= 4 else x
    return torch.rot90(xf, k, dims=[2, 3])

def _inverse_transform(y, i):
    k = i % 4
    y = torch.rot90(y, -k, dims=[2, 3])
    if i >= 4:
        y = torch.flip(y, dims=[3])
    return y


def infer(models, lr_t, use_tta):
    """Run (ensemble of) models on one image, optionally with 8x TTA.
    Averages over all models and all TTA orientations."""
    outs = []
    orientations = range(8) if use_tta else [0]
    with torch.no_grad():
        for i in orientations:
            xin = _forward_transform(lr_t, i) if use_tta else lr_t
            preds = [m(xin) for m in models]
            pred = torch.stack(preds, 0).mean(0)
            if use_tta:
                pred = _inverse_transform(pred, i)
            outs.append(pred)
    return torch.stack(outs, 0).mean(0)


# ------------------------- paired eval -------------------------
def run_paired(models, root, device, use_tta, limit):
    gt_dir = os.path.join(root, "GT"); lr_dir = os.path.join(root, "NoisyLR")
    files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    if limit: files = files[:limit]

    ps, ss, lp = [], [], []
    lpips_ok = True
    for f in files:
        name = os.path.basename(f)
        gt = np.clip(np.load(f).astype(np.float32), 0, 1)
        lr = np.load(os.path.join(lr_dir, name)).astype(np.float32)
        lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)
        pred = infer(models, lr_t, use_tta)
        ps.append(psnr(pred, gt_t)); ss.append(ssim(pred, gt_t))
        if lpips_ok:
            v = try_lpips(pred, gt_t, device)
            if v is None: lpips_ok = False
            else: lp.append(v)

    print("\n----- Paired metrics -----")
    print(f"Models   : {len(models)} | TTA: {use_tta} | Samples: {len(files)}")
    print(f"Mean PSNR : {np.mean(ps):.3f} dB")
    print(f"Mean SSIM : {np.mean(ss):.4f}")
    print(f"Mean LPIPS: {np.mean(lp):.4f}" if lp else "Mean LPIPS: (install lpips)")


# ------------------------- test inference -------------------------
def run_test(models, in_dir, out_dir, device, use_tta):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "*.npy")))
    print(f"Found {len(files)} images | Models: {len(models)} | TTA: {use_tta}")

    if device.type == "cuda":
        d = torch.randn(1,1,128,128,device=device)
        infer(models, d, use_tta); torch.cuda.synchronize()

    times = []
    for f in files:
        lr = np.load(f).astype(np.float32)
        lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = infer(models, lr_t, use_tta)
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.perf_counter()-t0)
        out = pred.squeeze().cpu().numpy().astype(np.float32)
        np.save(os.path.join(out_dir, os.path.basename(f)), out)

    t = np.array(times)
    print("\n----- Inference speed -----")
    print(f"Images     : {len(files)}")
    print(f"Total time : {t.sum():.3f} s")
    print(f"Mean/image : {t.mean()*1000:.2f} ms")
    print(f"Throughput : {len(files)/t.sum():.1f} img/s")
    print(f"Saved to   : {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output", nargs="?", default="outputs/restored")
    p.add_argument("--checkpoint", default="models/best.pth")
    p.add_argument("--checkpoint2", default=None,
                   help="Second checkpoint to ensemble (e.g. checkpoints_v3/best.pth)")
    p.add_argument("--tta", action="store_true", help="Enable 8x test-time augmentation")
    p.add_argument("--paired", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    models = [load_one(args.checkpoint, device)]
    if args.checkpoint2:
        models.append(load_one(args.checkpoint2, device))
    if args.paired:
        run_paired(models, args.input, device, args.tta, args.limit)
    else:
        run_test(models, args.input, args.output, device, args.tta)


if __name__ == "__main__":
    main()
