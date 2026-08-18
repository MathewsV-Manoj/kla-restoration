# AI-Based Restoration of Degraded Semiconductor Inspection Images

**SEMICON India Hackathon 2026 — Track 1 (KLA Challenge)**
**Team PixelForge** — Mathews V Manoj, Shone Reji (Muthoot Institute of Technology and Science)

A deep-learning pipeline for **joint denoising + 2× super-resolution** of degraded
grayscale semiconductor inspection images. Given a noisy, low-resolution input
(speckle noise, additive Gaussian noise, and 2× downsampling, applied in arbitrary
order), the model reconstructs a clean, full-resolution image.

| | Input (NoisyLR) | Output (Restored) | Target (GT) |
|---|---|---|---|
| Size | 128×128 | 256×256 | 256×256 |
| Range | may exceed [0,1] | [0,1] | [0,1] |

---

## Results (validation, 200 samples)

| Metric | Value |
|---|---|
| PSNR | **27.99 dB** |
| SSIM | **0.769** |
| LPIPS | **0.303** (lower is better) |
| Inference (fast mode) | ~44 ms/image on RTX 5050 |
| Inference (with TTA) | ~145 ms/image on RTX 5050 |

Trained on 3,200 paired samples (seeded 90/10 train/val split, no leakage).
Timing method: end-to-end wall-clock, CUDA-synchronized, per image.

---

## Repository structure

```
kla-restoration/
├── README.md               # this file
├── requirements.txt        # pinned dependencies
├── dataset.py              # dataset loader, train/val split, augmentation
├── model.py                # RestorationNet v1 (final model)
├── model_v2.py             # v2 (channel attention) - ablation
├── model_v3.py             # v3 (RRDB + PixelShuffle) - ablation
├── losses.py               # CombinedLoss (L1 + SSIM)
├── train.py                # training script (v1, final)
├── train_v2.py             # training script (v2, ablation)
├── train_v3.py             # training script (v3, ablation)
├── finetune_v1.py          # optional edge-loss fine-tuning experiment
├── inference.py            # STANDALONE inference (input dir -> output dir)
├── evaluate.py             # inference + optional paired metrics
├── evaluate_pro.py         # TTA + ensemble evaluation
├── make_figures.py         # before/after comparison figures
├── checkpoints/
│   └── best.pth            # final trained weights (epoch 83)
└── outputs/
    ├── figures/            # before/after comparison images
    └── restored_final/     # 400 restored test outputs
```

---

## Setup

Requires Python 3.10+ and an NVIDIA GPU with CUDA. Tested on Windows 11 with an
RTX 5050 (Blackwell, sm_120).

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# 2. Install PyTorch with CUDA support
#    (cu128 wheel required for Blackwell / RTX 50-series GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt
```

---

## Running inference (the standalone evaluation script)

The inference script takes an **input directory** of degraded `.npy` images and an
**output directory** for the restored results. It runs with no manual edits:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python inference.py data/Test_NoisyLR/NoisyLR outputs/restored
```

It loads weights automatically from `checkpoints/best.pth`, restores every `.npy`
in the input directory, saves each 256×256 float32 result under the same filename,
and reports inference-speed statistics.

### Computing quality metrics (optional, needs ground truth)

```bash
python evaluate.py data/train --paired --limit 200
```

Reports PSNR, SSIM, and LPIPS. For test-time augmentation / ensembling experiments,
see `evaluate_pro.py`.

---

## Reproducing training

```bash
python train.py --epochs 100 --batch_size 4 --num_workers 0
```

Checkpoints are written to `checkpoints/` (best + last); TensorBoard logs to `runs/`.

---

## Method summary

- **Architecture**: pre-upsample residual CNN (12 residual blocks, 64 channels,
  ~0.96M params). The input is bicubically upsampled to 256×256, then the network
  predicts an image-space *correction* on top of that baseline (residual learning).
- **Loss**: combined L1 + SSIM (α = 0.84), following Zhao et al. (2017).
- **Key data insight**: degraded (NoisyLR) inputs contain pixel values **outside
  [0,1]** due to speckle noise. These are **never clipped** on input — the
  out-of-range values carry real signal. Only the ground truth is guaranteed [0,1].
- **Augmentation**: geometric only (flips + 90° rotations), applied identically to
  input and target; no intensity jitter (preserves noise statistics).
- **Model selection**: an ablation (v1 residual CNN vs. v2 channel attention vs.
  v3 RRDB) selected v1 as the best accuracy–speed tradeoff. Test-time augmentation
  gives a small additional quality gain at inference.

---

## Hardware & environment

- GPU: NVIDIA RTX 5050 Laptop (8 GB, Blackwell sm_120)
- OS: Windows 11
- Python 3.x, PyTorch (CUDA 12.8 / cu128 wheel), mixed-precision (AMP) for v2/v3

---

## External resources disclosure

- **LPIPS** (Zhang et al. 2018), AlexNet backbone — used **only for evaluation
  reporting**, not part of the restoration model. Package: `lpips` (BSD-3-Clause).
- No external training datasets or pretrained weights are used in the model itself;
  the network is trained from scratch on the provided KLA dataset.

---

## References

1. H. Zhao, O. Gallo, I. Frosio, J. Kautz. "Loss Functions for Image Restoration
   with Neural Networks." *IEEE Transactions on Computational Imaging*, 2017.
2. B. Lim, S. Son, H. Kim, S. Nah, K. M. Lee. "Enhanced Deep Residual Networks for
   Single Image Super-Resolution (EDSR)." *CVPRW*, 2017.
3. Y. Zhang et al. "Image Super-Resolution Using Very Deep Residual Channel
   Attention Networks (RCAN)." *ECCV*, 2018.
4. R. Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual
   Metric (LPIPS)." *CVPR*, 2018.

## Run (official)

python run.py <input-dir> <output-dir>

Weights: models/best.pth (also in checkpoints/best.pth). Outputs are .npy, grayscale, in [0,1], NaN/Inf-free.
