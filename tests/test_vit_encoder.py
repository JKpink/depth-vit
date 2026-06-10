"""测试 Plan A ViT Encoder"""

import sys, os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model.vit_encoder import PatchEmbed, MultiHeadSelfAttention, TransformerBlock, CustomViTEncoder


class TestPatchEmbed:
    def test_output_shape(self, batch_input):
        pe = PatchEmbed(img_size=384, patch_size=16, embed_dim=384)
        out = pe(batch_input)
        assert out.shape == (2, 576, 384)  # (384/16)^2 = 576 patches

    def test_num_patches(self):
        pe = PatchEmbed(img_size=384, patch_size=16, embed_dim=384)
        assert pe.num_patches == 576


class TestMHSA:
    def test_output_shape(self):
        attn = MultiHeadSelfAttention(embed_dim=384, num_heads=6)
        x = torch.randn(2, 576, 384)
        out = attn(x)
        assert out.shape == (2, 576, 384)

    def test_attention_not_changing_shape(self):
        attn = MultiHeadSelfAttention(embed_dim=384, num_heads=6)
        x = torch.randn(2, 100, 384)
        out = attn(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    def test_output_shape(self):
        blk = TransformerBlock(embed_dim=384, num_heads=6)
        x = torch.randn(2, 576, 384)
        out = blk(x)
        assert out.shape == (2, 576, 384)

    def test_not_identity(self):
        """训练过的 block 不应该输出与输入完全一致"""
        blk = TransformerBlock(embed_dim=384, num_heads=6)
        x = torch.randn(2, 10, 384)
        out = blk(x)
        assert not torch.allclose(out, x)


class TestCustomViTEncoder:
    def test_output_length(self, batch_input):
        vit = CustomViTEncoder(img_size=384, embed_dim=384, num_layers=12, num_heads=6)
        features = vit(batch_input)
        assert len(features) == 3  # L4, L8, L12

    def test_feature_shapes(self, batch_input):
        vit = CustomViTEncoder(img_size=384, embed_dim=384, num_layers=12, num_heads=6)
        features = vit(batch_input)
        for feat in features:
            assert feat.shape == (2, 576, 384)  # [B, N_patches, embed_dim]

    def test_small_model(self, batch_input):
        """6 层小模型也正常输出"""
        vit = CustomViTEncoder(img_size=384, embed_dim=192, num_layers=6, num_heads=4)
        features = vit(batch_input)
        assert len(features) == 3
        assert features[-1].shape == (2, 576, 192)
