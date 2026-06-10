"""评测: RMSE, AbsRel, δ1, 边界 RMSE"""

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.depth_model import DepthFormer
from data.nyu_dataset import NYUDepthDataset


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, grad_threshold: float = 0.05) -> Dict[str, float]:
    """计算标准深度评测指标"""
    # 屏蔽无效深度值
    mask = (gt > 0.01) & (gt < 1.0)
    pred = pred[mask]
    gt = gt[mask]

    if pred.numel() == 0:
        return {"RMSE": float("nan"), "AbsRel": float("nan"), "δ1": float("nan"), "BoundaryRMSE": float("nan")}

    rmse = torch.sqrt(F.mse_loss(pred, gt)).item()
    absrel = (torch.abs(pred - gt) / gt).mean().item()
    ratio = torch.max(pred / gt, gt / pred)
    delta1 = (ratio < 1.25).float().mean().item()

    # 边界 RMSE
    dx = torch.abs(gt[..., :-1] - gt[..., 1:])
    dy = torch.abs(gt[..., :-1, :] - gt[..., 1:, :])
    h, w = min(dx.shape[-2], dy.shape[-2]), min(dx.shape[-1], dy.shape[-1])
    gt_grad = dx[..., :h, :w] + dy[..., :h, :w]
    edge_mask = gt_grad > grad_threshold

    pred_h = pred.reshape(-1, *gt.shape[-2:])
    gt_h = gt.reshape(-1, *gt.shape[-2:])
    pred_edge = pred_h[..., :h, :w][edge_mask]
    gt_edge = gt_h[..., :h, :w][edge_mask]
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
    parser.add_argument("--plan", choices=["A", "B"], default="A")
    parser.add_argument("--data", default="data/nyu")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="outputs/eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        NYUDepthDataset(args.data, "val", 384, augment=False),
        args.batch_size, num_workers=2, pin_memory=True,
    )

    model = DepthFormer(plan=args.plan).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    metrics = evaluate(model, loader, device)
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(args.output, "w"), indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
