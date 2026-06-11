"""冻结 DINOv2 Encoder（Plan B）—— 通过 HuggingFace Hub 加载，支持国内镜像"""

from typing import List
import os
import torch
import torch.nn as nn
from transformers import AutoModel


class FrozenDINOv2Encoder(nn.Module):
    """从 HuggingFace 加载冻结 DINOv2-Small，输出多尺度特征

    设置环境变量 HF_ENDPOINT=https://hf-mirror.com 使用国内镜像加速下载。
    """

    def __init__(self, model_name: str = "dinov2_vits14", image_size: int = 384):
        super().__init__()

        # 映射旧 torch.hub 名称到 HuggingFace model id
        hf_model_id = {
            "dinov2_vits14": "facebook/dinov2-small",
            "dinov2_vitb14": "facebook/dinov2-base",
            "dinov2_vitl14": "facebook/dinov2-large",
            "dinov2_vitg14": "facebook/dinov2-giant",
        }.get(model_name, model_name)

        self.backbone = AutoModel.from_pretrained(hf_model_id)
        self.image_size = image_size

        # 默认启用梯度 (微调 encoder)
        for param in self.backbone.parameters():
            param.requires_grad = True

        # DINOv2-S: 12 blocks, embed_dim=384, patch_size=14
        self.patch_size = self.backbone.config.patch_size  # 14
        self.embed_dim = self.backbone.config.hidden_size  # 384

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回多尺度特征 L2, L5, L8, L11（对齐 DA-V2, 去掉 CLS token）"""
        with torch.no_grad():
            outputs = self.backbone(x, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            features = [hidden_states[i][:, 1:, :] for i in (2, 5, 8, 11)]
        return features  # [L2_tokens, L5_tokens, L8_tokens, L11_tokens]
