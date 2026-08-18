"""
make_figures.py - Generate before/after comparison figures for the slide deck.

Creates side-by-side panels: NoisyLR input (upscaled for display) | Model
output | Ground truth, for a few validation samples. Saves PNGs to
outputs/figures/ for dropping into the presentation.
"""
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import RestorationNet

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RestorationNet().to(device)
ckpt = torch.load("models/best.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

os.makedirs("outputs/figures", exist_ok=True)

# pick a few varied samples
gt_files = sorted(glob.glob("data/train/GT/*.npy"))
picks = [gt_files[i] for i in [5, 100, 500, 1500, 2500]]

for idx, gtf in enumerate(picks):
    name = os.path.basename(gtf)
    gt = np.clip(np.load(gtf).astype(np.float32), 0, 1)
    lr = np.load(os.path.join("data/train/NoisyLR", name)).astype(np.float32)
    with torch.no_grad():
        lr_t = torch.from_numpy(lr).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(lr_t).squeeze().cpu().numpy()

    # metrics for this sample
    mse = np.mean((pred - gt) ** 2)
    psnr = 20*np.log10(1.0) - 10*np.log10(mse) if mse > 0 else 99

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(np.clip(lr, 0, 1), cmap="gray"); axes[0].set_title("Degraded input (NoisyLR, 128x128)", fontsize=11)
    axes[1].imshow(pred, cmap="gray"); axes[1].set_title(f"Restored (ours, 256x256)  PSNR={psnr:.2f} dB", fontsize=11)
    axes[2].imshow(gt, cmap="gray"); axes[2].set_title("Ground truth (256x256)", fontsize=11)
    for a in axes: a.axis("off")
    plt.tight_layout()
    out = f"outputs/figures/compare_{idx+1}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}  (PSNR {psnr:.2f} dB)")

print("\nDone. Figures in outputs/figures/ - use these on the results slide.")
