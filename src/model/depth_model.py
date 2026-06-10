"""DepthFormer 完整模型 —— Plan A 或 Plan B 统一接口"""

from typing import Literal
import torch
import torch.nn as nn

from .vit_encoder import CustomViTEncoder
from .dino_encoder import FrozenDINOv2Encoder
from .reassemble import MultiScaleReassemble
from .decoder import DepthDecoder


class DepthFormer(nn.Module):
    """深度估计模型

    Args:
        plan: "A" (自研ViT) 或 "B" (冻结DINOv2)
        image_size: 输入分辨率
        embed_dim: Token 嵌入维度（仅 Plan A 有效）
        num_layers: ViT 层数（仅 Plan A 有效）
        num_heads: 注意力头数（仅 Plan A 有效）
    """

    def __init__(
        self,
        plan: Literal["A", "B"] = "A",
        image_size: int = 384,
        embed_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
    ):
        super().__init__()
        self.plan = plan
        self.image_size = image_size

        # ── Encoder ──
        if plan == "A":
            self.encoder = CustomViTEncoder(
                img_size=image_size, embed_dim=embed_dim,
                num_layers=num_layers, num_heads=num_heads,
            )
            patch_size = 16
            encoder_dim = embed_dim
        else:  # plan == "B"
            self.encoder = FrozenDINOv2Encoder(
                model_name="dinov2_vits14", image_size=image_size,
            )
            patch_size = 14
            encoder_dim = self.encoder.embed_dim  # 384

        # ── Token → 特征图 ──
        target_size = image_size // (patch_size // 2) if plan == "A" else image_size // 14 * 2
        if plan == "A":
            target_size = 48  # 384/8
        else:
            target_size = 54  # 384/14*2

        self.reassemble = MultiScaleReassemble(
            image_size=image_size, patch_size=patch_size,
            embed_dim=encoder_dim, target_size=target_size,
        )

        # ── 解码器 ──
        decoder_in = encoder_dim * 3  # L4+L8+L12 concat
        self.decoder = DepthDecoder(in_channels=decoder_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)           # [L4, L8, L12]
        fused = self.reassemble(features)    # [B, 3D, T, T]
        depth = self.decoder(fused)          # [B, 1, H, W]
        return depth
