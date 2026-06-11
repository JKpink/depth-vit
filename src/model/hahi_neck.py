"""HAHI Neck — 跨尺度 Transformer 特征融合
参考: DepthFormer (https://arxiv.org/abs/2203.14211)
简化版: 去掉 deformable attention / mmcv, 用标准 PyTorch MultiheadAttention
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenSelfAttention(nn.Module):
    """单尺度 token 自注意力 + FFN"""

    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x


class TokenCrossAttention(nn.Module):
    """跨尺度交叉注意力: 高层 token 查询低层 token"""

    def __init__(self, dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # query, key_value: [B, N, D]
        y, _ = self.attn(self.norm_q(query), self.norm_kv(key_value), self.norm_kv(key_value))
        x = query + y
        x = x + self.ffn(self.norm_out(x))
        return x


class HAHINeckLight(nn.Module):
    """轻量 HAHI Neck — 4 层 DINOv2 token → 融合 → 4 个尺度输出

    Args:
        embed_dim: DINOv2 token 维度 (384 for vits14)
        out_dim:   输出特征图通道数
        num_heads: 注意力头数
        num_layers: 特征层数
        patch_size: ViT patch 尺寸 (14 for DINOv2)
        image_size: 输入分辨率
    """

    def __init__(
        self,
        embed_dim: int = 384,
        out_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        patch_size: int = 14,
        image_size: int = 384,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.grid_size = image_size // patch_size  # eg. 384 // 14 = 27 (truncated)

        # Token → feature map: 1×1 conv 投影
        self.token_proj = nn.ModuleList([
            nn.Linear(embed_dim, out_dim) for _ in range(num_layers)
        ])

        # 自注意力: 每层独立
        self.self_attn = nn.ModuleList([
            TokenSelfAttention(out_dim, num_heads) for _ in range(num_layers)
        ])

        # 交叉注意力: 低层查询高层 (L_i queries L_{i-1})
        self.cross_attn = nn.ModuleList([
            TokenCrossAttention(out_dim, num_heads) for _ in range(num_layers - 1)
        ])

        # 位置编码 (全局, 所有 scale 共用相同 token 数)
        max_tokens = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, out_dim) * 0.02)

        # 可学习 resize 层 (Stage 5, 对齐 DA-V2)
        self.resize_l4x = nn.ConvTranspose2d(out_dim, out_dim, kernel_size=4, stride=4)
        self.resize_l2x = nn.ConvTranspose2d(out_dim, out_dim, kernel_size=2, stride=2)
        self.resize_down = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)

        # Conv 分支: 保持局部空间结构
        self.conv_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True),
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(True),
            ) for _ in range(num_layers)
        ])

    def _tokens_to_2d(self, tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
        """tokens [B, N, D] → feature map [B, D, H, W]"""
        B, N, D = tokens.shape
        H = W = grid_size
        if N != H * W:
            # 处理非整除情况: pad 或 truncate
            H = W = int(N ** 0.5)
            N_use = H * W
            tokens = tokens[:, :N_use, :]  # truncate to square
        return tokens.transpose(1, 2).reshape(B, D, H, W)

    def _feature_to_tokens(self, feature: torch.Tensor) -> torch.Tensor:
        """feature map [B, D, H, W] → tokens [B, H*W, D]"""
        B, D, H, W = feature.shape
        return feature.flatten(2).transpose(1, 2)

    def forward(self, token_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            token_list: [L0, L4, L8, L11] DINOv2 token features, each [B, N, 384]
        Returns:
            feats: 4 个特征图 [B, 256, H, W], 从粗到细 (L11→L0)
        """
        B = token_list[0].shape[0]
        num_scales = len(token_list)

        # ── Stage 1: Token 投影 + position ──
        tokens_proj = []
        for i in range(num_scales):
            t = self.token_proj[i](token_list[i])  # [B, N, 384] → [B, N, 256]
            N = t.shape[1]
            t = t + self.pos_embed[:, :N, :]       # add position
            tokens_proj.append(t)

        # ── Stage 2: 自注意力 (每层独立) ──
        tokens_sa = [self.self_attn[i](t) for i, t in enumerate(tokens_proj)]

        # ── Stage 3: 交叉注意力 (低层查高层) ──
        # L0 ← L4, L4 ← L8, L8 ← L11
        tokens_merged = tokens_sa.copy()
        for i in range(num_scales - 1):
            # i=0: L0 queries L4, i=1: L4 queries L8, i=2: L8 queries L11
            tokens_merged[i] = self.cross_attn[i](
                tokens_merged[i], tokens_sa[i + 1]
            )

        # ── Stage 4: Token → Conv 双通路融合 ──
        raw_feats = []
        for i in range(num_scales):
            feat_t = self._tokens_to_2d(tokens_merged[i], self.grid_size)
            feat_s = self._tokens_to_2d(tokens_sa[i], self.grid_size)
            raw_feats.append(feat_t + self.conv_refine[i](feat_s))

        # ── Stage 5: 可学习多尺度金字塔 (对齐 DA-V2) ──
        # L2→4× up, L5→2× up, L8→keep, L11→0.5× down
        outputs = [
            self.resize_l4x(raw_feats[0]),     # [B, 256, 108, 108]
            self.resize_l2x(raw_feats[1]),     # [B, 256,  54,  54]
            raw_feats[2],                       # [B, 256,  27,  27]
            self.resize_down(raw_feats[3]),     # [B, 256,  14,  14]
        ]
        return outputs
