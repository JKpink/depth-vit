"""梯度感知损失: L1 + SSIM + Gradient"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """计算预测与真值在梯度域的 L1 差"""
    dx_pred = pred[..., :-1] - pred[..., 1:]    # [H-1, W]
    dy_pred = pred[..., :-1, :] - pred[..., 1:, :]  # [H, W-1]
    dx_gt = gt[..., :-1] - gt[..., 1:]
    dy_gt = gt[..., :-1, :] - gt[..., 1:, :]
    # 统一为最小尺寸
    h = min(dx_pred.shape[-2], dy_pred.shape[-2])
    w = min(dx_pred.shape[-1], dy_pred.shape[-1])
    dx_pred, dy_pred = dx_pred[..., :h, :w], dy_pred[..., :h, :w]
    dx_gt, dy_gt = dx_gt[..., :h, :w], dy_gt[..., :h, :w]
    return F.l1_loss(dx_pred, dx_gt) + F.l1_loss(dy_pred, dy_gt)


class SSIM(nn.Module):
    """结构相似度（简化版）"""

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # 强制 float32 防止 autocast 下精度不够产生 NaN
        pred = pred.float()
        gt = gt.float()
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        mu_p = F.avg_pool2d(pred, self.window_size, 1)
        mu_g = F.avg_pool2d(gt, self.window_size, 1)
        sigma_p2 = F.avg_pool2d(pred * pred, self.window_size, 1) - mu_p * mu_p
        sigma_g2 = F.avg_pool2d(gt * gt, self.window_size, 1) - mu_g * mu_g
        sigma_pg = F.avg_pool2d(pred * gt, self.window_size, 1) - mu_p * mu_g
        ssim_map = ((2 * mu_p * mu_g + C1) * (2 * sigma_pg + C2)) / (
            (mu_p ** 2 + mu_g ** 2 + C1) * (sigma_p2 + sigma_g2 + C2)
        )
        return torch.clamp(1 - ssim_map.mean(), 0, 1)


class DepthLoss(nn.Module):
    """L1 + SSIM + Gradient 组合损失"""

    def __init__(
        self,
        lambda_l1: float = 0.6,
        lambda_ssim: float = 0.3,
        lambda_grad: float = 0.1,
        grad_threshold: float = 0.05,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_grad = lambda_grad
        self.grad_threshold = grad_threshold
        self.ssim = SSIM()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # 统一尺寸
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)

        l1 = F.l1_loss(pred, gt)
        ssim_loss = self.ssim(pred, gt)
        grad_l = gradient_loss(pred, gt)

        # 边界增强权重: 梯度大的区域给更多权重
        dx = gt[..., :-1] - gt[..., 1:]
        dy = gt[..., :-1, :] - gt[..., 1:, :]
        h, w = min(dx.shape[-2], dy.shape[-2]), min(dx.shape[-1], dy.shape[-1])
        gt_grad = torch.abs(dx[..., :h, :w]) + torch.abs(dy[..., :h, :w])
        edge_mask = (F.interpolate(gt_grad if gt_grad.dim() == 4 else gt_grad,
                                   size=pred.shape[-2:], mode="bilinear", align_corners=False)
                     > self.grad_threshold).float().mean()

        # δ1: 每张图独立算，避免 batch 像素混在一起
        delta1 = 0.0
        valid = gt > 0.01
        if valid.any():
            with torch.no_grad():
                ratio = torch.max(
                    pred.float() / gt.float().clamp(min=0.01),
                    gt.float() / pred.float().clamp(min=0.01),
                )
                delta1 = (ratio < 1.25).float().mean().item()

        total = self.lambda_l1 * l1 + self.lambda_ssim * ssim_loss + self.lambda_grad * grad_l

        metrics = {
            "l1": l1.item(),
            "ssim": ssim_loss.item(),
            "grad": grad_l.item(),
            "edge_ratio": edge_mask.item(),
            "delta1": delta1,
            "total": total.item(),
        }
        return total, metrics


class SiLogLoss(nn.Module):
    """Scale-Invariant Logarithmic Loss (对齐 DA-V2)
    L = sqrt(α * mean(d²) - λ * mean(d)²)  where d = log(pred) - log(gt)
    """

    def __init__(self, var_lambda: float = 0.85):
        super().__init__()
        self.var_lambda = var_lambda

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)

        valid = gt > 0.1  # 排除无效像素
        if valid.any():
            pred = pred[valid]
            gt = gt[valid]
            diff = torch.log(pred.clamp(min=1e-4)) - torch.log(gt.clamp(min=1e-4))
            loss = torch.sqrt((diff ** 2).mean() - self.var_lambda * diff.mean() ** 2)
        else:
            loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

        # δ1 for monitoring
        valid_mon = (gt > 0.01) if pred.numel() == 0 else (gt > 0.01)
        with torch.no_grad():
            ratio = torch.max(pred / gt.clamp(min=0.01), gt / pred.clamp(min=0.01))
            delta1 = (ratio < 1.25).float().mean().item()

        metrics = {"l1": loss.item(), "ssim": 0.0, "grad": 0.0,
                    "edge_ratio": 0.0, "delta1": delta1, "total": loss.item()}
        return loss, metrics
