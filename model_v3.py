"""
model_v3.py - RestorationNet v3 (RRDB + sub-pixel upsampling)

This is a targeted redesign after v2 (channel attention) failed to beat
the v1 baseline. The diagnosis of v2's shortcomings drove every choice
here:

WHY v3 SHOULD BEAT v1 AND v2
----------------------------
1. POST-UPSAMPLING via PixelShuffle (sub-pixel convolution), instead of
   bicubic pre-upsampling.
   - v1/v2 first blew the image up to 256x256 with bicubic interpolation,
     then spent network capacity *cleaning up* interpolation blur.
   - v3 processes features at the native 128x128 resolution (cheaper and
     sharper), and learns the 2x upsampling itself in feature space at
     the very end. This is the ESPCN/EDSR design and is consistently
     better at recovering genuine high-frequency detail -> higher PSNR.

2. RRDB (Residual-in-Residual Dense Blocks), the ESRGAN backbone.
   - Dense connections reuse features across layers with strong gradient
     flow. Empirically the strongest general-purpose restoration/SR block.
   - Replaces v1's plain residual blocks and v2's attention blocks.

3. STABILITY FIXES that v2 lacked:
   - No BatchNorm (correct for SR: BN removes intensity info and hurts).
   - Residual scaling (beta=0.2) inside each dense block, the standard
     ESRGAN trick that keeps deep residual nets from diverging.
   - Designed to train with a lower LR (1e-4) than v2's unstable 2e-4.

4. Global skip: a bicubic upsample of the input is still ADDED at the end
   (image-space residual), so the network only predicts the *correction*
   on top of a reasonable baseline -- easier optimisation, faster
   convergence, keeps the proven idea from v1.

Input:  (B, 1, 128, 128) unclamped NoisyLR
Output: (B, 1, 256, 256) restored, clamped [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Dense block with 5 convs and local residual scaling (ESRGAN-style)."""

    def __init__(self, channels, growth=32, beta=0.2):
        super().__init__()
        self.beta = beta
        self.conv1 = nn.Conv2d(channels + 0 * growth, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + 1 * growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * growth, channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + self.beta * x5


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 dense blocks + outer residual."""

    def __init__(self, channels, growth=32, beta=0.2):
        super().__init__()
        self.beta = beta
        self.rdb1 = ResidualDenseBlock(channels, growth, beta)
        self.rdb2 = ResidualDenseBlock(channels, growth, beta)
        self.rdb3 = ResidualDenseBlock(channels, growth, beta)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + self.beta * out


class RestorationNetV3(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_rrdb=8, growth=32):
        super().__init__()

        # Feature extraction at native LR resolution (128x128)
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.body = nn.Sequential(
            *[RRDB(base_channels, growth) for _ in range(num_rrdb)]
        )
        self.body_out = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        # 2x sub-pixel upsampling: conv to 4*C channels then PixelShuffle(2)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),                       # 128->256, C stays
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        feat = self.head(x)
        body = self.body(feat)
        body = self.body_out(body)
        feat = feat + body                            # global residual (LR space)

        up = self.upsample(feat)                       # -> (B, C, 256, 256)
        out = self.tail(up)

        # image-space residual: add bicubic upsample of the input
        x_up = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        out = x_up + out

        return torch.clamp(out, 0.0, 1.0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = RestorationNetV3()
    n = count_parameters(model)
    print(f"Total trainable parameters: {n:,} ({n/1e6:.2f}M)")
    x = torch.randn(2, 1, 128, 128)
    y = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}  min={y.min():.4f} max={y.max():.4f}")
