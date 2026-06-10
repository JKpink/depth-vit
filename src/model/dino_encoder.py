"""冻结 DINOv2 Encoder（Plan B: 保底方案）"""

from typing import List
import torch
import torch.nn as nn


class FrozenDINOv2Encoder(nn.Module):
    """封装 torch Hub DINOv2-Small，输出多尺度特征"""

    def __init__(self, model_name: str = "dinov2_vits14", image_size: int = 384):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.image_size = image_size

        # 冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # DINOv2-S: 12 blocks, embed_dim=384, patch_size=14
        self.patch_size = 14
        self.embed_dim = 384
        self.num_patches = (image_size // self.patch_size) ** 2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回多尺度特征 L4, L8, L12"""
        with torch.no_grad():
            B, _, H, W = x.shape
            # DINOv2 forward 取中间层
            x = self.backbone.prepare_tokens_with_masks(x)

            features: List[torch.Tensor] = []
            extract_at: set[int] = {4, 8, 12}
            for i, blk in enumerate(self.backbone.blocks, start=1):
                x = blk(x)
                if i in extract_at:
                    # 去掉 CLS token
                    features.append(x[:, 1:, :])  # [B, N_patches, D]

        return features  # [L4, L8, L12]
