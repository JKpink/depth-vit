"""训练脚本 —— Plan A 或 Plan B, 含 checkpoint 断点续训
所有默认参数从 configs/default.yaml 读取，CLI 可覆盖。
"""

import argparse, json, os, sys, yaml, time
from datetime import datetime
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model.depth_model import DepthFormer
from data.nyu_dataset import NYUDepthDataset
from data.sun_dataset import SUNRGBDDataset
from loss import DepthLoss, SiLogLoss  # noqa: F401


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        cfg_path = Path(__file__).resolve().parent.parent / path  # project root
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: DepthLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    grad_clip: float,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    lr_scheduler = None,
) -> Dict[str, float]:
    model.train()
    total_metrics = {"l1": 0.0, "ssim": 0.0, "grad": 0.0, "edge_ratio": 0.0, "total": 0.0, "delta1": 0.0}
    n = 0

    pbar = tqdm(loader, desc=f"Train E{epoch:3d}", leave=False, ncols=130)
    for rgb, depth in pbar:
        rgb, depth = rgb.to(device), depth.to(device)
        optimizer.zero_grad()

        with autocast():
            pred = model(rgb)
            loss, metrics = loss_fn(pred, depth)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # 检测 NaN 梯度并跳过
        grad_ok = True
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_ok = False
                break
        if not grad_ok:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()  # 必须调 update 重置 unscale_ 状态
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if lr_scheduler is not None:
            lr_scheduler.step()

        bs = rgb.shape[0]
        for k in total_metrics:
            total_metrics[k] += metrics.get(k, 0.0) * bs
        n += bs

        pbar.set_postfix(loss=metrics["total"], l1=metrics["l1"],
                         grad=metrics["grad"], delta1=metrics["delta1"])

    avg = {k: v / n for k, v in total_metrics.items()}
    writer.add_scalar("train/loss", avg["total"], epoch)
    writer.add_scalar("train/delta1", avg["delta1"], epoch)
    return avg


@torch.no_grad()
def val_epoch(model: nn.Module, loader: DataLoader, loss_fn: DepthLoss,
              device: torch.device, epoch: int, writer: SummaryWriter) -> Dict[str, float]:
    model.eval()
    total_metrics = {"l1": 0.0, "ssim": 0.0, "grad": 0.0, "edge_ratio": 0.0, "total": 0.0, "delta1": 0.0}
    n = 0

    pbar = tqdm(loader, desc=f"Val  E{epoch:3d}", leave=False, ncols=130)
    for rgb, depth in pbar:
        rgb, depth = rgb.to(device), depth.to(device)
        pred = model(rgb)
        _, metrics = loss_fn(pred, depth)
        bs = rgb.shape[0]
        for k in total_metrics:
            total_metrics[k] += metrics.get(k, 0.0) * bs
        n += bs
        pbar.set_postfix(loss=metrics["total"], l1=metrics["l1"],
                         delta1=metrics["delta1"])

    avg = {k: v / n for k, v in total_metrics.items()}
    writer.add_scalar("val/loss", avg["total"], epoch)
    writer.add_scalar("val/delta1", avg["delta1"], epoch)
    return avg


def main():
    # ── 加载默认 config ──
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["nyu", "sunrgbd"], default="nyu")
    parser.add_argument("--data")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--ablation", default=None, help="ablation tag (full, nograd, nossim, pure_l1)")
    parser.add_argument("--resume", default=None, help="checkpoint path")
    parser.add_argument("--output", default="")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── 参数解析：CLI > config ──
    data_dir   = args.data       or "data/nyu"
    image_size = args.image_size or cfg["training"].get("image_size", 504)
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    epochs     = args.epochs     or cfg["training"]["epochs"]
    lr         = args.lr         or cfg["training"]["lr"]
    num_workers = args.workers    or cfg["data"].get("num_workers", 4)

    grad_clip     = cfg["training"]["grad_clip"]
    weight_decay  = cfg["training"]["weight_decay"]

    dataset_name = args.dataset or "nyu"

    # ── 设备 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.output:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        ablation_tag = args.ablation or "full"
        out_dir = Path("outputs") / f"{ablation_tag}_{dataset_name}_bs{batch_size}_ep{epochs}_{timestamp}"
    else:
        out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 日志同时写到文件和终端
    class Tee:
        def __init__(self, f1, f2):
            self.f1 = f1; self.f2 = f2
        def write(self, s):
            self.f1.write(s); self.f2.write(s); self.f1.flush(); self.f2.flush()
        def flush(self):
            self.f1.flush(); self.f2.flush()

    log_f = open(out_dir / "train.log", "a")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)

    writer = SummaryWriter(str(out_dir))

    print(f"Config: {args.config}")
    print(f"  dataset={dataset_name}, image_size={image_size}, batch_size={batch_size}, epochs={epochs}, lr={lr}")
    print(f"  loss: SiLog")

    # ── 数据 ──
    if dataset_name == "sunrgbd":
        train_ds = SUNRGBDDataset(data_dir, "train", image_size, augment=True)
    else:
        train_ds = NYUDepthDataset(data_dir, "train", image_size, augment=True)
    # 评估始终用 NYU
    val_ds = NYUDepthDataset("data/nyu", "val", image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size, num_workers=num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── 模型 ──
    model = DepthFormer(image_size=image_size).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── 损失 ──
    ablation = args.ablation or "full"
    if ablation == "pure_l1":
        loss_fn = DepthLoss(lambda_l1=1.0, lambda_ssim=0.0, lambda_grad=0.0)
    elif ablation == "nossim":
        loss_fn = DepthLoss(lambda_l1=0.8, lambda_ssim=0.0, lambda_grad=0.2)
    elif ablation == "nograd":
        loss_fn = DepthLoss(lambda_l1=0.8, lambda_ssim=0.2, lambda_grad=0.0)
    else:  # full
        loss_fn = SiLogLoss()

    # ── 优化器 (对齐 DA-V2: split lr, encoder 10× lower) ──
    encoder_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder"):
            encoder_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": lr},
        {"params": encoder_params, "lr": lr * 0.1},
    ], lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    scaler = GradScaler()

    # ── 多项式 LR 衰减 (对齐 DA-V2) ──
    from torch.optim.lr_scheduler import LambdaLR
    decay_power = cfg["training"].get("lr_decay_power", 0.9)
    total_iters = epochs * len(train_loader)
    poly_scheduler = LambdaLR(optimizer, lambda it: (1 - it / total_iters) ** decay_power)

    start_epoch = 0
    best_epoch = 0
    best_val = float("inf")
    early_stop = cfg["training"].get("early_stop_patience", 20)

    # ── 断点续训 ──
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        best_epoch = ckpt.get("best_epoch", start_epoch - 1)
        print(f"Resumed from epoch {start_epoch}, best val: {best_val:.4f} (E{best_epoch})")

    # ── 训练 ──
    history = []
    for epoch in range(start_epoch, epochs):
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, scaler,
                                     grad_clip, device, epoch, writer, poly_scheduler)
        val_metrics = val_epoch(model, val_loader, loss_fn, device, epoch, writer)

        # Polynomial LR (per-batch, already done in train_epoch; keep for consistency)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"E {epoch:3d} | loss:{train_metrics['total']:.4f} val:{val_metrics['total']:.4f} "
              f"l1:{val_metrics['l1']:.4f} grad:{val_metrics['grad']:.4f} "
              f"δ1:{val_metrics['delta1']:.3f}  lr:{lr_now:.2e}")

        history.append({"epoch": epoch, "train_loss": train_metrics["total"],
                        "val_loss": val_metrics["total"], "delta1": val_metrics["delta1"]})

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch},
                       out_dir / "best.pt")

        if (epoch + 1) % 20 == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch},
                       out_dir / f"ckpt_{epoch:03d}.pt")

        # 早停已禁用

    print(f"Done. Best: E{best_epoch} val={best_val:.4f}")
    # 保存训练历史
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    writer.close()


if __name__ == "__main__":
    main()
