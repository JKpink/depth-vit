"""共享 fixtures"""

import sys, os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch_input():
    """[B=2, C=3, H=384, W=384]"""
    return torch.randn(2, 3, 384, 384)


@pytest.fixture
def batch_depth_gt():
    """[B=2, 1, 384, 384]"""
    return torch.rand(2, 1, 384, 384)


@pytest.fixture
def batch_tokens():
    """[B=2, N=576, D=384]"""
    return torch.randn(2, 576, 384)


@pytest.fixture
def multi_scale_tokens():
    """L4, L8, L12 tokens, each [B=2, N=576, D=384]"""
    return [torch.randn(2, 576, 384) for _ in range(3)]
