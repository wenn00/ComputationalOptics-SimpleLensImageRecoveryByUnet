"""Loss functions and metrics for image deblurring.

- L1Loss: pixel-level reconstruction loss
- SSIM Loss: structural similarity (via pytorch-msssim)
- Combined Loss: L1 + weight * (1 - SSIM)
- PSNR: peak signal-to-noise ratio metric
"""

import torch
import torch.nn as nn
from pytorch_msssim import ssim


class L1Loss(nn.Module):
    """Standard L1 (mean absolute error) loss."""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class SSIMLoss(nn.Module):
    """SSIM-based loss: 1 - SSIM.

    Uses pytorch-msssim with data_range=1.0 (images in [0, 1]).
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim(pred, target, data_range=1.0, size_average=True)


class CombinedLoss(nn.Module):
    """Combined L1 + SSIM loss.

    total = L1 + ssim_weight * (1 - SSIM)

    Args:
        use_ssim: Whether to include SSIM component.
        ssim_weight: Weight for the SSIM loss term.
    """

    def __init__(self, use_ssim: bool = False, ssim_weight: float = 0.1):
        super().__init__()
        self.l1 = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.l1(pred, target)
        if self.use_ssim:
            loss = loss + self.ssim_weight * self.ssim_loss(pred, target)
        return loss


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between prediction and target.

    Both tensors should be in [0, 1] range (data_range = 1.0).

    Args:
        pred: Predicted image tensor.
        target: Ground truth image tensor.

    Returns:
        PSNR value in dB.
    """
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM between prediction and target.

    Both tensors should be in [0, 1] range.

    Args:
        pred: Predicted image tensor (B, C, H, W).
        target: Ground truth image tensor (B, C, H, W).

    Returns:
        SSIM value (0 to 1, higher is better).
    """
    return ssim(pred, target, data_range=1.0, size_average=True).item()
