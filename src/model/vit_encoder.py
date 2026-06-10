"""自研 ViT Encoder（Plan A: 从头训练）"""

import math
from typing import Tuple, List
import torch
import torch.nn as nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    """RGB → Patch 序列"""

    def __init__(self, img_size: int = 384, patch_size: int = 16, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] → [B, N, D]
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, embed_dim: int = 384, num_heads: int = 6):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, N, d]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """ViT Block: MHSA + MLP"""

    def __init__(self, embed_dim: int = 384, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CustomViTEncoder(nn.Module):
    """自研 ViT Encoder"""

    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        embed_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回多尺度特征 L4, L8, L12"""
        x = self.patch_embed(x)  # [B, N, D]
        x = x + self.pos_embed

        features: List[torch.Tensor] = []
        # 按比例提取: 1/3, 2/3, 3/3 位置
        L = len(self.blocks)
        extract_at = [L // 3, L * 2 // 3, L]
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            if i in extract_at:
                features.append(x)
        return features  # [L1/3, L2/3, L_all]
