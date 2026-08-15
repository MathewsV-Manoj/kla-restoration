"""
model.py - RestorationNet
Pre-upsample residual CNN for joint denoising + 2x super-resolution.
Input:  (B, 1, 128, 128) unclamped NoisyLR
Output: (B, 1, 256, 256) restored image, clamped to [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class RestorationNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_blocks=12):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_blocks)]
        )

        self.body_out = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        # bicubic pre-upsample 128x128 -> 256x256
        x_up = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)

        feat = self.head(x_up)
        res = self.body(feat)
        res = self.body_out(res)
        feat = feat + res  # global residual connection

        out = self.tail(feat)
        out = x_up + out  # image-space residual correction

        return torch.clamp(out, 0.0, 1.0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = RestorationNet()
    n_params = count_parameters(model)
    print(f"Total trainable parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}  min={y.min():.4f} max={y.max():.4f}")
