"""Residual UNet for image deblurring.

Architecture:
- Encoder: 4 downsampling levels (64 -> 128 -> 256 -> 512)
- Bottleneck: 512 -> 512
- Decoder: 4 upsampling levels with skip connections
- Global residual: output = input + network(input)
- GroupNorm for batch-size independent normalization
- Bilinear upsample + Conv3x3 (no checkerboard artifacts)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv3x3 -> GroupNorm -> ReLU -> Conv3x3 -> GroupNorm -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Use 8 groups for GroupNorm (channels must be divisible by num_groups)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling: MaxPool2x2 -> ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling: Bilinear upsample + Conv1x1 (channel align) -> Concat skip -> ConvBlock."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = ConvBlock(out_channels * 2, out_channels)  # *2 for skip concat

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up_conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResidualUNet(nn.Module):
    """Residual UNet for image deblurring.

    The network learns the residual (difference) between blurry and sharp images.
    Final output: input + network(input), clamped to [0, 1] at inference time.

    Args:
        in_channels: Number of input channels (3 for RGB).
        out_channels: Number of output channels (3 for RGB).
        channels: List of channel counts for each encoder level.
        bottleneck: Channel count for the bottleneck layer.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: list[int] | None = None,
        bottleneck: int = 512,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512]

        # Encoder
        self.enc1 = ConvBlock(in_channels, channels[0])       # 64
        self.enc2 = DownBlock(channels[0], channels[1])        # 128
        self.enc3 = DownBlock(channels[1], channels[2])        # 256
        self.enc4 = DownBlock(channels[2], channels[3])        # 512

        # Bottleneck
        self.bottleneck = DownBlock(channels[3], bottleneck)   # 512

        # Decoder
        self.dec4 = UpBlock(bottleneck, channels[3])           # 512
        self.dec3 = UpBlock(channels[3], channels[2])          # 256
        self.dec2 = UpBlock(channels[2], channels[1])          # 128
        self.dec1 = UpBlock(channels[1], channels[0])          # 64

        # Final 1x1 conv to map to output channels
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with encoder-decoder skip connections and global residual.

        Args:
            x: Input tensor of shape (B, C, H, W), values in [0, 1].

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        # Encoder — each level extracts features and halves spatial resolution
        e1 = self.enc1(x)          # (B, 64,  H,    W)
        e2 = self.enc2(e1)         # (B, 128, H/2,  W/2)
        e3 = self.enc3(e2)         # (B, 256, H/4,  W/4)
        e4 = self.enc4(e3)         # (B, 512, H/8,  W/8)

        # Bottleneck — deepest, most abstract representation
        b = self.bottleneck(e4)    # (B, 512, H/16, W/16)

        # Decoder — upsample and merge with encoder features (skip connections)
        d4 = self.dec4(b, e4)      # (B, 512, H/8,  W/8)
        d3 = self.dec3(d4, e3)     # (B, 256, H/4,  W/4)
        d2 = self.dec2(d3, e2)     # (B, 128, H/2,  W/2)
        d1 = self.dec1(d2, e1)     # (B, 64,  H,    W)

        # Output — 1x1 conv to get 3 channels, then add original input (residual)
        residual = self.final_conv(d1)
        return x + residual
