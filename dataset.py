"""
dataset.py

Dataset loader for the KLA i4C Hackathon 2026 - Track 1
(AI-Based Restoration of Degraded Semiconductor Inspection Images)

Data format (confirmed empirically from provided .npy files):
    GT/*.npy       -> float32, shape (256, 256), range strictly [0.0, 1.0]
    NoisyLR/*.npy  -> float32, shape (128, 128), range EXCEEDS [0,1]
                      (e.g. observed min -0.0033, max 1.54 across samples)

CRITICAL (confirmed by KLA webinar slides, verbatim):
    "Image GL ranges: NoisyLR will have value beyond [0,1] whereas GT only
    has [0,1] - it is a feature not a bug!"
    "Images are normalized a-priori. Welcome to re-normalize. Note that
    GT is always [0,1]."

    => Do NOT clip/clamp NoisyLR values into [0,1]. The out-of-range
       values ARE the speckle noise signal the model must learn to
       remove. Clipping them before training would destroy information
       the network needs to see.
    => GT is guaranteed to already be [0,1] - no rescaling needed there.

Degradations (per KLA slides): Gaussian noise, speckle noise, and
downsampling/blur, applied in an unknown/arbitrary order. The model
must learn the joint inverse function f^-1: NoisyLR -> GT directly;
it does not need to know or infer degradation order.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KLARestorationDataset(Dataset):
    """
    Paired dataset for joint denoising + 2x super-resolution.

    Each sample returns:
        lr : torch.FloatTensor, shape (1, 128, 128), UNCLAMPED range
        gt : torch.FloatTensor, shape (1, 256, 256), range [0, 1]

    Args:
        root_dir:   path containing GT/ and NoisyLR/ subfolders
        file_ids:   list of filenames (e.g. "000000.npy") to include in
                    this split. If None, uses every file found in GT/.
        augment:    if True, apply random flips/rotations (same transform
                    applied identically to both lr and gt to keep them
                    spatially aligned)
    """

    def __init__(self, root_dir, file_ids=None, augment=False):
        self.gt_dir = os.path.join(root_dir, "GT")
        self.lr_dir = os.path.join(root_dir, "NoisyLR")
        self.augment = augment

        if file_ids is None:
            gt_files = sorted(glob.glob(os.path.join(self.gt_dir, "*.npy")))
            self.file_ids = [os.path.basename(f) for f in gt_files]
        else:
            self.file_ids = file_ids

        if len(self.file_ids) == 0:
            raise RuntimeError(f"No .npy files found under {self.gt_dir}")

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fname = self.file_ids[idx]

        gt = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
        lr = np.load(os.path.join(self.lr_dir, fname)).astype(np.float32)

        # Sanity guard: GT should already be [0,1]. Clamp defensively
        # in case of rare float rounding just past the boundary
        # (e.g. 1.0000001), WITHOUT touching lr.
        gt = np.clip(gt, 0.0, 1.0)

        # DELIBERATELY NOT clamping lr - see module docstring.

        if self.augment:
            lr, gt = self._joint_augment(lr, gt)

        # add channel dim: (H, W) -> (1, H, W)
        lr = torch.from_numpy(lr.copy()).unsqueeze(0)
        gt = torch.from_numpy(gt.copy()).unsqueeze(0)

        return lr, gt

    @staticmethod
    def _joint_augment(lr, gt):
        """Apply identical random flip/rotation to both lr and gt so
        they remain pixel-aligned. Only geometric transforms are used -
        no brightness/contrast jitter, since altering pixel intensity
        would corrupt the exact noise statistics the model must learn."""

        # random horizontal flip
        if random.random() < 0.5:
            lr = np.fliplr(lr)
            gt = np.fliplr(gt)

        # random vertical flip
        if random.random() < 0.5:
            lr = np.flipud(lr)
            gt = np.flipud(gt)

        # random 90-degree rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            lr = np.rot90(lr, k)
            gt = np.rot90(gt, k)

        return lr, gt


def get_train_val_split(root_dir, val_fraction=0.1, seed=42):
    """
    Splits the training set into train/val file-id lists.
    Returns (train_ids, val_ids).
    """
    gt_files = sorted(glob.glob(os.path.join(root_dir, "GT", "*.npy")))
    file_ids = [os.path.basename(f) for f in gt_files]

    rng = random.Random(seed)
    rng.shuffle(file_ids)

    n_val = max(1, int(len(file_ids) * val_fraction))
    val_ids = file_ids[:n_val]
    train_ids = file_ids[n_val:]

    return train_ids, val_ids


def get_dataloaders(root_dir, batch_size=16, val_fraction=0.1,
                     num_workers=4, seed=42):
    """
    Convenience function: builds train and val DataLoaders in one call.

    Usage:
        train_loader, val_loader = get_dataloaders("data/train")
    """
    train_ids, val_ids = get_train_val_split(root_dir, val_fraction, seed)

    train_ds = KLARestorationDataset(root_dir, file_ids=train_ids, augment=True)
    val_ds = KLARestorationDataset(root_dir, file_ids=val_ids, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Quick smoke test - run `python dataset.py` from the project root
    # after activating your venv to sanity-check everything loads.
    train_loader, val_loader = get_dataloaders("data/train", batch_size=4)

    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    lr, gt = next(iter(train_loader))
    print(f"LR batch: shape={tuple(lr.shape)} dtype={lr.dtype} "
          f"min={lr.min():.4f} max={lr.max():.4f}")
    print(f"GT batch: shape={tuple(gt.shape)} dtype={gt.dtype} "
          f"min={gt.min():.4f} max={gt.max():.4f}")
