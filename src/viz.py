"""可视化: 深度图对比 + 边界放大 + Baseline 对比图"""

from pathlib import Path
from typing import List, Dict
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def colorize_depth(depth: torch.Tensor, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """深度图伪彩色（红=近，蓝=远）"""
    depth = (depth - vmin) / (vmax - vmin + 1e-8)
    depth = torch.clamp(depth, 0, 1).squeeze().cpu().numpy()
    cmap = plt.cm.jet
    colored = cmap(depth)[..., :3]  # RGBA → RGB
    return (colored * 255).astype(np.uint8)


def save_depth_comparison(
    rgb: torch.Tensor,
    pred: torch.Tensor,
    gt: torch.Tensor,
    output_path: str,
):
    """保存深度对比图：RGB + Pred + GT + Error"""
    rgb_np = (rgb.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pred_colored = colorize_depth(pred)
    gt_colored = colorize_depth(gt)

    # 误差热力图
    error = torch.abs(pred - gt).squeeze().cpu().numpy()
    error_img = (plt.cm.hot(error / (error.max() + 1e-8))[..., :3] * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["RGB", "Prediction", "Ground Truth", "Error"]
    imgs = [rgb_np, pred_colored, gt_colored, error_img]
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_boundary_zoom(
    pred: torch.Tensor,
    gt: torch.Tensor,
    roi_xy: tuple = (100, 100),
    roi_size: int = 64,
    output_path: str = None,
):
    """边界区域局部放大对比"""
    x, y = roi_xy
    pred_roi = pred[..., x:x+roi_size, y:y+roi_size]
    gt_roi = gt[..., x:x+roi_size, y:y+roi_size]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, d, title in zip(axes, [pred_roi, gt_roi], ["Prediction", "GT"]):
        ax.imshow(colorize_depth(d))
        ax.set_title(f"{title} ({roi_size}×{roi_size})", fontsize=12)
        ax.axis("off")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_baseline_comparison(
    metrics: Dict[str, Dict[str, float]],
    output_path: str,
):
    """Baseline 对比柱状图: RMSE, δ1 两个指标"""
    names = list(metrics.keys())
    rmse_vals = [metrics[n]["RMSE"] for n in names]
    delta1_vals = [metrics[n]["δ1"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    ax1.bar(names, rmse_vals, color=colors[:len(names)])
    ax1.set_title("RMSE ↓", fontsize=14)
    for i, v in enumerate(rmse_vals):
        ax1.text(i, v + max(rmse_vals) * 0.02, f"{v:.3f}", ha="center")

    ax2.bar(names, delta1_vals, color=colors[:len(names)])
    ax2.set_title("δ₁ ↑", fontsize=14)
    ax2.set_ylim(0, 1)
    for i, v in enumerate(delta1_vals):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha="center")

    plt.suptitle("Depth Estimation Baseline Comparison", fontsize=16)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_path: str,
):
    """训练曲线"""
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train", linewidth=2)
    plt.plot(epochs, val_losses, label="Val", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
