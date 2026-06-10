"""训练脚本 —— Plan A 或 Plan B, 含 checkpoint 断点续训"""

import argparse, os, sys
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))
from model.depth_model import DepthFormer
from data.nyu_dataset import NYUDepthDataset
from loss import DepthLoss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: DepthLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> Dict[str, float]:
    model.train()
    total_metrics = {"l1": 0.0, "ssim": 0.0, "grad": 0.0, "edge_ratio": 0.0, "total": 0.0}
    n = 0

    for rgb, depth in loader:
        rgb, depth = rgb.to(device), depth.to(device)
        optimizer.zero_grad()

        with autocast():
            pred = model(rgb)
            loss, metrics = loss_fn(pred, depth)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = rgb.shape[0]
        for k in total_metrics:
            total_metrics[k] += metrics.get(k, 0.0) * bs
        n += bs

    avg = {k: v / n for k, v in total_metrics.items()}
    writer.add_scalar("train/l1", avg["l1"], epoch)
    writer.add_scalar("train/total", avg["total"], epoch)
    return avg


@torch.no_grad()
def val_epoch(model: nn.Module, loader: DataLoader, loss_fn: DepthLoss,
              device: torch.device, epoch: int, writer: SummaryWriter) -> Dict[str, float]:
    model.eval()
    total_metrics = {"l1": 0.0, "ssim": 0.0, "grad": 0.0, "edge_ratio": 0.0, "total": 0.0}
    n = 0

    for rgb, depth in loader:
        rgb, depth = rgb.to(device), depth.to(device)
        pred = model(rgb)
        _, metrics = loss_fn(pred, depth)
        bs = rgb.shape[0]
        for k in total_metrics:
            total_metrics[k] += metrics.get(k, 0.0) * bs
        n += bs

    avg = {k: v / n for k, v in total_metrics.items()}
    writer.add_scalar("val/l1", avg["l1"], epoch)
    writer.add_scalar("val/total", avg["total"], epoch)
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", choices=["A", "B"], default="A")
    parser.add_argument("--data", default="data/nyu")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output) / f"plan_{args.plan}"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir))

    # 数据
    train_ds = NYUDepthDataset(args.data, "train", args.image_size, augment=True)
    val_ds = NYUDepthDataset(args.data, "val", args.image_size, augment=False)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, num_workers=args.workers, pin_memory=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 模型
    model = DepthFormer(plan=args.plan).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    loss_fn = DepthLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()
    start_epoch = 0
    best_val = float("inf")

    # 断点续训
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best val: {best_val:.4f}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, scaler, device, epoch, writer)
        val_metrics = val_epoch(model, val_loader, loss_fn, device, epoch, writer)

        print(f"E {epoch:3d} | train:{train_metrics['total']:.4f} val:{val_metrics['total']:.4f} "
              f"l1:{val_metrics['l1']:.4f} grad:{val_metrics['grad']:.4f}")

        # 保存最佳
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "best_val": best_val},
                       out_dir / "best.pt")

        # 断点
        if (epoch + 1) % 20 == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "best_val": best_val},
                       out_dir / f"ckpt_{epoch:03d}.pt")

    print(f"Done. Best val: {best_val:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
