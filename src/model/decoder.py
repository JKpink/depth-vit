"""渐进式上采样解码器"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv2d + BN + ReLU ×2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DepthDecoder(nn.Module):
    """渐进上采样解码器: 48×48 → 384×384"""

    def __init__(self, in_channels: int = 1152, out_channels: int = 1):
        """
        in_channels: 3 × embed_dim (L4+L8+L12 concat)
        """
        super().__init__()
        self.up1 = nn.Sequential(
            ConvBlock(in_channels, 256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )  # 48 → 96
        self.up2 = nn.Sequential(
            ConvBlock(256, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )  # 96 → 192
        self.up3 = nn.Sequential(
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )  # 192 → 384
        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1152, 48, 48]
        returns: [B, 1, 384, 384]
        """
        x = self.up1(x)  # 96
        x = self.up2(x)  # 192
        x = self.up3(x)  # 384
        return self.head(x)
