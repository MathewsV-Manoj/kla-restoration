"""
model_v2.py - RestorationNet v2 (with channel attention)

Upgrade over v1: each residual block now includes a Channel Attention
module (RCAB-style, from Zhang et al. "Image Super-Resolution Using Very
Deep Residual Channel Attention Networks", ECCV 2018).

Why this helps our task specifically:
  - Channel attention lets the network learn *which* feature channels
    matter most for reconstructing fine texture, rather than weighting
    all channels equally. This directly targets structural fidelity
    (SSIM), which was the weaker metric for v1 (~0.77).
  - We keep every proven element of v1: bicubic pre-upsample, global
    residual over the body, and the final image-space residual
    correction (predict the correction to the upsampled input, not the
    image from scratch). Only the attention is added, so this is a
    low-risk incremental upgrade, not a rewrite.

Also: BatchNorm is REMOVED from the residual blocks. For image
restoration/SR, BN is known to hurt (it normalizes away absolute
intensity information the model needs, and couples training/inference
statistics). Modern SR nets (EDSR, RCAN) drop it. This both improves
quality and slightly speeds inference.

Input:  (B, 1, 128, 128) unclamped NoisyLR
Output: (B, 1, 256, 256) restored image, clamped to [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class RCAB(nn.Module):
    """Residual Channel Attention Block (no BatchNorm)."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            ChannelAttention(channels, reduction),
        )

    def forward(self, x):
        return x + self.body(x)


class RestorationNetV2(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_blocks=16,
                 reduction=16):
        super().__init__()

        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.body = nn.Sequential(
            *[RCAB(base_channels, reduction) for _ in range(num_blocks)]
        )
        self.body_out = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)

        feat = self.head(x_up)
        res = self.body(feat)
        res = self.body_out(res)
        feat = feat + res                     # global residual

        out = self.tail(feat)
        out = x_up + out                      # image-space residual correction

        return torch.clamp(out, 0.0, 1.0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = RestorationNetV2()
    n_params = count_parameters(model)
    print(f"Total trainable parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}  min={y.min():.4f} max={y.max():.4f}")
