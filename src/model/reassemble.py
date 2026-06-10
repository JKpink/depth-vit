"""Token → 空间特征图重建"""

from typing import List
import torch
import torch.nn as nn


class Reassemble(nn.Module):
    """将 ViT token 序列还原为空间特征图"""

    def __init__(self, image_size: int = 384, patch_size: int = 16, embed_dim: int = 384, out_dim: int = 384):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.conv = nn.Conv2d(embed_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, D] where N = (H/P)^2
        returns: [B, out_dim, H/P, H/P]
        """
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)
        x = tokens.transpose(1, 2).reshape(B, D, H, W)
        return self.conv(x)


class MultiScaleReassemble(nn.Module):
    """多尺度 Token → 统一尺寸特征图"""

    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 16,
        embed_dim: int = 384,
        target_size: int = 48,  # 384/8 = 48
    ):
        super().__init__()
        self.target_size = target_size
        self.reassemble = Reassemble(image_size, patch_size, embed_dim, embed_dim)
        # 上采样到统一尺寸
        self.upsample = nn.Upsample(size=(target_size, target_size), mode="bilinear", align_corners=False)

    def forward(self, multi_scale_tokens: List[torch.Tensor]) -> torch.Tensor:
        """三尺度特征 → concat → 统一尺寸"""
        reassembled: List[torch.Tensor] = []
        for tokens in multi_scale_tokens:
            feat = self.reassemble(tokens)  # [B, D, H, W]
            feat = self.upsample(feat)      # [B, D, T, T]
            reassembled.append(feat)
        return torch.cat(reassembled, dim=1)  # [B, 3D, T, T]
