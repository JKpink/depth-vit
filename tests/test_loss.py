"""测试损失函数"""

import sys, os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loss import gradient_loss, SSIM, DepthLoss


class TestGradientLoss:
    def test_perfect_match(self):
        pred = torch.rand(2, 1, 100, 100)
        gt = pred.clone()
        loss = gradient_loss(pred, gt)
        assert loss.item() < 1e-6

    def test_different_returns_positive(self):
        pred = torch.rand(2, 1, 100, 100)
        gt = torch.rand(2, 1, 100, 100)
        loss = gradient_loss(pred, gt)
        assert loss.item() > 0


class TestSSIM:
    def test_perfect_match(self):
        ssim = SSIM()
        x = torch.rand(2, 1, 100, 100)
        loss = ssim(x, x)
        assert loss.item() < 1e-3

    def test_range_zero_to_one(self):
        ssim = SSIM()
        pred = torch.rand(2, 1, 100, 100)
        gt = torch.rand(2, 1, 100, 100)
        loss = ssim(pred, gt)
        assert 0.0 <= loss.item() <= 1.0


class TestDepthLoss:
    def test_returns_scalar_and_metrics(self):
        dl = DepthLoss()
        pred = torch.rand(2, 1, 384, 384)
        gt = torch.rand(2, 1, 384, 384)
        loss, metrics = dl(pred, gt)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert "l1" in metrics
        assert "ssim" in metrics
        assert "grad" in metrics
        assert "total" in metrics

    def test_edge_mask_exists(self):
        dl = DepthLoss(grad_threshold=0.05)
        pred = torch.rand(2, 1, 128, 128)
        gt = torch.rand(2, 1, 128, 128)
        _, metrics = dl(pred, gt)
        assert "edge_ratio" in metrics
        assert 0.0 <= metrics["edge_ratio"] <= 1.0

    def test_ablation_no_grad(self):
        """去掉梯度损失后 metric 里 grad 仍存在"""
        dl = DepthLoss(lambda_grad=0.0)
        pred = torch.rand(2, 1, 64, 64)
        gt = torch.rand(2, 1, 64, 64)
        _, metrics = dl(pred, gt)
        assert metrics["grad"] >= 0
