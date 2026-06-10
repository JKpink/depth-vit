"""测试 Reassemble + Decoder"""

import sys, os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model.reassemble import Reassemble, MultiScaleReassemble
from model.decoder import ConvBlock, DepthDecoder


class TestReassemble:
    def test_output_shape(self):
        r = Reassemble(image_size=384, patch_size=16, embed_dim=384, out_dim=384)
        tokens = torch.randn(2, 576, 384)
        out = r(tokens)
        assert out.shape == (2, 384, 24, 24)  # 384/16=24

    def test_non_square(self):
        """非方形也可以"""
        r = Reassemble(image_size=384, patch_size=16, embed_dim=192, out_dim=192)
        tokens = torch.randn(2, 576, 192)
        out = r(tokens)
        assert out.shape == (2, 192, 24, 24)


class TestMultiScaleReassemble:
    def test_output_shape(self, multi_scale_tokens):
        mr = MultiScaleReassemble(image_size=384, patch_size=16, embed_dim=384, target_size=48)
        out = mr(multi_scale_tokens)
        assert out.shape == (2, 1152, 48, 48)  # 3D channels


class TestConvBlock:
    def test_output_shape(self):
        c = ConvBlock(256, 128)
        x = torch.randn(2, 256, 48, 48)
        out = c(x)
        assert out.shape == (2, 128, 48, 48)


class TestDepthDecoder:
    def test_output_shape(self):
        decoder = DepthDecoder(in_channels=1152)
        x = torch.randn(2, 1152, 48, 48)
        out = decoder(x)
        assert out.shape == (2, 1, 384, 384)

    def test_output_finite(self):
        """输出不含 NaN 或 inf"""
        decoder = DepthDecoder(in_channels=1152)
        x = torch.randn(2, 1152, 48, 48)
        out = decoder(x)
        assert torch.isfinite(out).all()
