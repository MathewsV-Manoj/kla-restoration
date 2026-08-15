"""
losses.py - CombinedLoss (L1 + SSIM)

Combines pixel-wise L1 loss with structural similarity (SSIM) loss.
Rationale: L1 optimizes raw pixel accuracy; SSIM optimizes perceived
structural/textural fidelity, which is what visual restoration quality
is ultimately judged on.

Reference:
Zhao, H., Gallo, O., Frosio, I., & Kautz, J. (2017). "Loss functions
for image restoration with neural networks." IEEE Transactions on
Computational Imaging, 3(1), 47-57.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_window(window_size, sigma, channels, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(0) * g.unsqueeze(1)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, sigma=1.5, C1=0.01 ** 2, C2=0.03 ** 2):
    channels = img1.size(1)
    window = gaussian_window(window_size, sigma, channels, img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, img1, img2):
        return 1.0 - ssim(img1, img2, self.window_size, self.sigma)


class CombinedLoss(nn.Module):
    """
    L_total = alpha * L1 + (1 - alpha) * SSIM_loss

    Default alpha=0.84 follows Zhao et al. (2017), who found this
    weighting balances pixel accuracy and structural fidelity best
    for restoration tasks.
    """

    def __init__(self, alpha=0.84, window_size=11, sigma=1.5):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size, sigma)

    def forward(self, pred, target):
        l1_val = self.l1(pred, target)
        ssim_val = self.ssim_loss(pred, target)
        total = self.alpha * l1_val + (1 - self.alpha) * ssim_val
        return total, {"l1": l1_val.item(), "ssim_loss": ssim_val.item(), "total": total.item()}


if __name__ == "__main__":
    criterion = CombinedLoss()
    pred = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    loss, parts = criterion(pred, target)
    print(f"Loss: {loss.item():.4f}  Parts: {parts}")
