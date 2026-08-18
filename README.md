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

## Quick start

```bash
pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

`run.py` reads every `.npy` in `<input-dir>`, creates `<output-dir>` if needed, and
writes one restored `.npy` per input under the same filename. Model weights ship in
`models/best.pth` — no internet access, API keys, extra downloads, or manual
configuration are required at inference time.

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
├── run.py                  # SUBMISSION ENTRY POINT (input dir -> output dir)
├── requirements.txt        # pinned dependencies
├── README.md               # this file
├── models/
│   └── best.pth            # final trained weights (epoch 83), ships with repo
├── dataset.py              # dataset loader, train/val split, augmentation
├── model.py                # RestorationNet v1 (final model)
├── model_v2.py             # v2 (channel attention) - ablation
├── model_v3.py             # v3 (RRDB + PixelShuffle) - ablation
├── losses.py               # CombinedLoss (L1 + SSIM)
├── train.py                # training script (v1, final)
├── train_v2.py             # training script (v2, ablation)
├── train_v3.py             # training script (v3, ablation)
├── finetune_v1.py          # optional edge-loss fine-tuning experiment
├── inference.py            # inference + speed benchmark (same model as run.py)
├── evaluate.py             # inference + optional paired metrics
├── evaluate_pro.py         # TTA + ensemble evaluation
├── make_figures.py         # before/after comparison figures
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

# 2. Install all dependencies
#    requirements.txt already declares the PyTorch cu128 index, which is
#    required for Blackwell / RTX 50-series GPUs. On other NVIDIA GPUs any
#    CUDA-enabled build of torch >= 2.1 works.
pip install -r requirements.txt
```

No further configuration is needed. The trained weights are committed in
`models/best.pth`, so inference requires **no internet access, no API keys and
no additional model downloads**.

---

## Running the solution

`run.py` is the entry point. It takes an **input directory** of degraded `.npy`
images and an **output directory** for the restored results, and runs with no
manual edits:

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py data/Test_NoisyLR/NoisyLR outputs/restored
```

Weights load automatically from `models/best.pth`. The output directory is
created if it does not already exist.

### Input / output contract

| | Input | Output |
|---|---|---|
| Format | `.npy` | `.npy`, same filename as its input |
| Shape | `(H, W)` grayscale, e.g. 128×128 | `(2H, 2W)` grayscale, e.g. 256×256 |
| dtype | any numeric | `float32` |
| Range | may fall outside `[0,1]` (speckle noise) | strictly `[0,1]`, no `NaN` / `Inf` |

Every `.npy` file in the input directory produces exactly one output file.
Outputs are clipped to `[0,1]` and scrubbed of non-finite values before saving.
Inputs shaped `(H, W, 1)` or `(1, H, W)` are also accepted and squeezed to
`(H, W)`. If the network were to fail on any single image, a bicubic upsample of
that input is written instead, so the output set is never incomplete.

Optional flags: `--checkpoint <path>` to use different weights, and
`--device cpu|cuda` to force a device (default: CUDA when available).

### Inference speed benchmark (optional)

`inference.py` runs the same model and additionally reports end-to-end
per-image latency and throughput:

```bash
python inference.py <input_dir> <output_dir>
```

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
Those are local training artifacts - the weights shipped for evaluation live in
`models/best.pth`.

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

## Submission checklist

| Requirement | Status |
|---|---|
| `run.py` present, invoked as `python run.py <input-dir> <output-dir>` | ✅ |
| Reads all `.npy` files from the input directory | ✅ |
| Creates the output directory if it does not exist | ✅ |
| One restored `.npy` per input file | ✅ |
| Output filename matches its input filename | ✅ |
| Outputs are grayscale arrays of shape `(H, W)` | ✅ |
| Output values within `[0,1]`, no `NaN` / `Inf` | ✅ clipped + scrubbed before saving |
| Correct target resolution (2× the input, 128×128 → 256×256) | ✅ |
| Model weights and supporting files included | ✅ `models/best.pth` (committed) |
| `requirements.txt` with pinned versions | ✅ |
| `README.md` with setup and execution instructions | ✅ this file |
| Runs on an NVIDIA GPU with no internet, API keys, downloads, or manual config | ✅ CPU fallback included |

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
