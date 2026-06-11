"""渐进式上采样解码器"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ResidualConvUnit(nn.Module):
    """后激活残差双卷积 (Conv→BN→ReLU → Conv→BN → +skip → ReLU)"""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)  # 后激活残差


class FeatureFusionBlock(nn.Module):
    """后激活融合: skip→refine→add, 再次 refine→upsample→proj"""

    def __init__(self, features: int):
        super().__init__()
        self.res_unit1 = ResidualConvUnit(features)
        self.res_unit2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        if skip is not None:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=True)
            x = x + self.res_unit1(skip)
        x = self.res_unit2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.out_conv(x)
        return x


class ProgressiveDecoder(nn.Module):
    """DA-V2 式底部向上渐进解码器 + sigmoid 输出"""

    def __init__(
        self,
        features: int = 256,
        num_stages: int = 4,
        max_depth: float = 10.0,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.num_stages = num_stages

        self.stage0 = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
        )

        self.fusion = nn.ModuleList([
            FeatureFusionBlock(features) for _ in range(num_stages - 1)
        ])

        # Head → raw → sigmoid × max_depth
        self.head = nn.Sequential(
            nn.Conv2d(features, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = self.stage0(features[-1])

        for i in range(self.num_stages - 2, -1, -1):
            x = self.fusion[self.num_stages - 2 - i](x, features[i])

        x = self.head(x)
        return torch.sigmoid(x) * self.max_depth


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
