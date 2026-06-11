"""DepthFormer v2 —— DINOv2 + HAHI Neck + ProgressiveDecoder"""

from typing import List
import torch
import torch.nn as nn

from .dino_encoder import FrozenDINOv2Encoder
from .hahi_neck import HAHINeckLight
from .decoder import ProgressiveDecoder


class DepthFormer(nn.Module):
    """深度估计模型 v2 — DINOv2 编码器 + HAHI 特征融合 + 渐进解码器

    Args:
        image_size: 输入分辨率
        neck_dim: HAHI 输出通道数
        num_heads: 注意力头数
    """

    def __init__(
        self,
        image_size: int = 384,
        neck_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.image_size = image_size

        # ── Encoder: frozen DINOv2-S/14 ──
        self.encoder = FrozenDINOv2Encoder(
            model_name="dinov2_vits14", image_size=image_size,
        )
        embed_dim = self.encoder.embed_dim   # 384
        patch_size = self.encoder.patch_size  # 14

        # ── Neck: HAHI token 跨尺度融合 ──
        self.neck = HAHINeckLight(
            embed_dim=embed_dim,
            out_dim=neck_dim,
            num_heads=num_heads,
            num_layers=4,
            patch_size=patch_size,
            image_size=image_size,
        )

        # ── Decoder: FeatureFusionBlock + sigmoid ──
        self.decoder = ProgressiveDecoder(features=neck_dim, num_stages=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(x)             # [L0, L4, L8, L11], each [B, N, 384]
        feats = self.neck(tokens)            # [feat0..feat3], each [B, 256, H, W]
        depth = self.decoder(feats)          # [B, 1, ~H, ~W]
        return depth
