"""评测: RMSE, AbsRel, δ1, 边界 RMSE
参数默认值从 configs/default.yaml 读取。
"""

import argparse, json, yaml
from pathlib import Path
from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.depth_model import DepthFormer
from data.nyu_dataset import NYUDepthDataset
from viz import save_depth_comparison, plot_boundary_zoom


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent.parent / path
    with open(cfg_path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, grad_threshold: float = 0.5) -> Dict[str, float]:
    """计算标准深度评测指标"""
    # 保存原始形状用于边界计算
    pred_2d = pred.squeeze(0)
    gt_2d = gt.squeeze(0)

    mask = gt > 0.1
    pred_masked = pred[mask]
    gt_masked = gt[mask]

    if pred_masked.numel() == 0:
        return {"RMSE": float("nan"), "AbsRel": float("nan"), "δ1": float("nan"), "BoundaryRMSE": float("nan")}

    rmse = torch.sqrt(F.mse_loss(pred_masked, gt_masked)).item()
    absrel = (torch.abs(pred_masked - gt_masked) / gt_masked).mean().item()
    ratio = torch.max(pred_masked / gt_masked, gt_masked / pred_masked)
    delta1 = (ratio < 1.25).float().mean().item()

    # 边界 RMSE — 使用原始 2D 张量
    dx = torch.abs(gt_2d[:, :-1] - gt_2d[:, 1:])
    dy = torch.abs(gt_2d[:-1, :] - gt_2d[1:, :])
    h, w = min(dx.shape[0], dy.shape[0]), min(dx.shape[1], dy.shape[1])
    gt_grad = dx[:h, :w] + dy[:h, :w]
    edge_mask = gt_grad > grad_threshold

    pred_edge = pred_2d[:h, :w][edge_mask]
    gt_edge = gt_2d[:h, :w][edge_mask]
    bd_rmse = torch.sqrt(F.mse_loss(pred_edge, gt_edge)).item() if pred_edge.numel() > 0 else float("nan")

    return {"RMSE": rmse, "AbsRel": absrel, "δ1": delta1, "BoundaryRMSE": bd_rmse}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    metrics_sum: Dict[str, float] = {"RMSE": 0.0, "AbsRel": 0.0, "δ1": 0.0, "BoundaryRMSE": 0.0}
    count = 0

    for rgb, depth in loader:
        rgb, depth = rgb.to(device), depth.to(device)
        pred = model(rgb)
        if pred.shape[-2:] != depth.shape[-2:]:
            pred = F.interpolate(pred, size=depth.shape[-2:], mode="bilinear", align_corners=False)
        for i in range(rgb.shape[0]):
            m = compute_metrics(pred[i], depth[i])
            for k in metrics_sum:
                v = m.get(k, 0.0)
                if not (v != v):  # skip NaN
                    metrics_sum[k] += v
            count += 1

    return {k: v / count for k, v in metrics_sum.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--plan", choices=["A", "B"])
    parser.add_argument("--data", default="data/nyu")
    parser.add_argument("--eval-dataset", choices=["nyu", "sunrgbd"], default="nyu")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", default="outputs/eval.json")
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_dir   = args.data       or "data/nyu"
    image_size = cfg["training"].get("image_size", 504)
    batch_size = args.batch_size or 8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_dataset == "sunrgbd":
        from data.sun_dataset import SUNRGBDDataset
        val_ds = SUNRGBDDataset(data_dir, "val", image_size, augment=False)
        print(f"SUN RGB-D val: {len(val_ds)}")
    else:
        val_ds = NYUDepthDataset(data_dir, "val", image_size, augment=False)

    # SUN RGB-D 太大，随机取 1/10 评测
    if args.eval_dataset == "sunrgbd":
        indices = torch.randperm(len(val_ds))[:max(1, len(val_ds) // 10)].tolist()
        val_ds = torch.utils.data.Subset(val_ds, indices)
        print(f"Subset: {len(val_ds)} images (1/10 of test)")

    loader = DataLoader(val_ds, batch_size, num_workers=2, pin_memory=True)

    model = DepthFormer(image_size=image_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = evaluate(model, loader, device)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    # ── 出图: 深度对比 + 边界放大 ──
    viz_dir = Path(args.checkpoint).parent / "viz"
    viz_dir.mkdir(exist_ok=True)
    snip = Path(args.checkpoint).parent.name
    for i, (rgb, depth) in enumerate(loader):
        if i >= 4:
            break
        rgb, depth = rgb.to(device), depth.to(device)
        pred = model(rgb)
        if pred.shape[-2:] != depth.shape[-2:]:
            pred = F.interpolate(pred, size=depth.shape[-2:], mode="bilinear", align_corners=False)
        save_depth_comparison(rgb[0].cpu(), pred[0].cpu(), depth[0].cpu(),
                              str(viz_dir / f"{snip}_depth_{i}.png"))
        plot_boundary_zoom(pred[0], depth[0],
                           roi_xy=(80, 100), roi_size=64,
                           output_path=str(viz_dir / f"{snip}_boundary_{i}.png"))
        print(f"  saved {snip}_depth_{i}.png, boundary_{i}.png")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(args.output, "w"), indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
