"""完整管道维度测试 —— 验证每层 Tensor Shape"""

import sys, os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model.vit_encoder import PatchEmbed, MultiHeadSelfAttention, TransformerBlock, CustomViTEncoder
from model.reassemble import Reassemble, MultiScaleReassemble
from model.decoder import ConvBlock, DepthDecoder
from model.depth_model import DepthFormer


class TestFullPipelineShapes:
    """验证从输入到输出的每一层维度"""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_patch_embed(self, batch_size):
        x = torch.randn(batch_size, 3, 384, 384)
        pe = PatchEmbed(384, 16, 384)
        out = pe(x)
        assert out.shape == (batch_size, 576, 384)  # (384/16)^2 = 576

    def test_mhsa_batch_invariant(self):
        """MHSA 对不同 batch size 维度正确"""
        for bs in [1, 2, 4, 8]:
            attn = MultiHeadSelfAttention(384, 6)
            x = torch.randn(bs, 576, 384)
            out = attn(x)
            assert out.shape == (bs, 576, 384)

    @pytest.mark.parametrize("embed_dim,num_heads", [(192, 4), (384, 6), (768, 12)])
    def test_mhsa_configs(self, embed_dim, num_heads):
        attn = MultiHeadSelfAttention(embed_dim, num_heads)
        x = torch.randn(2, 100, embed_dim)
        out = attn(x)
        assert out.shape == x.shape

    def test_transformer_preserves_shape(self):
        for i in range(3):
            blk = TransformerBlock(384, 6)
            x = torch.randn(2, 576, 384)
            out = blk(x)
            assert out.shape == x.shape, f"pass {i}: {x.shape} → {out.shape}"

    @pytest.mark.parametrize("plan", ["A", "B"])
    def test_depthformer_pipeline(self, plan, batch_input):
        """端到端: RGB → 深度图"""
        if plan == "B":
            pytest.skip("需要 torch hub 下载 DINOv2")
        model = DepthFormer(plan="A")
        out = model(batch_input)
        assert out.shape == (2, 1, 384, 384)
        assert torch.isfinite(out).all()

    def test_encoder_feature_count(self, batch_input):
        """12 层 ViT 应输出 3 个中间特征"""
        vit = CustomViTEncoder(num_layers=12)
        features = vit(batch_input)
        assert len(features) == 3

    def test_encoder_feature_shapes_12(self, batch_input):
        vit = CustomViTEncoder(num_layers=12, embed_dim=384)
        features = vit(batch_input)
        for i, f in enumerate(features):
            assert f.shape == (2, 576, 384), f"L{i}: {f.shape} != (2, 576, 384)"

    def test_reassemble_concat_dim(self):
        """三尺度 concat 后通道数 = 3 × embed_dim"""
        mr = MultiScaleReassemble(384, 16, 384, 48)
        tokens = [torch.randn(2, 576, 384) for _ in range(3)]
        out = mr(tokens)
        assert out.shape == (2, 1152, 48, 48)  # 3×384=1152

    def test_decoder_progression(self):
        """解码器逐层放大"""
        decoder = DepthDecoder(1152)
        x = torch.randn(2, 1152, 48, 48)
        out = decoder(x)
        assert out.shape == (2, 1, 384, 384)  # 8x up

    @pytest.mark.parametrize("img_size,patch_size,expected", [
        (384, 16, 576),
        (256, 16, 256),
        (512, 32, 256),
    ])
    def test_patch_counts(self, img_size, patch_size, expected):
        pe = PatchEmbed(img_size, patch_size, 128)
        x = torch.randn(2, 3, img_size, img_size)
        out = pe(x)
        assert out.shape[1] == expected
