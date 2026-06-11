#!/bin/bash
# 连续跑实验: SiLog vs L1+SSIM+Grad (消融)
set -e

DATA="data/sunrgbd"
EPOCHS=40
LR="1e-5"

echo "=== 实验 1: SiLog ==="
python src/train.py --dataset sunrgbd --data "$DATA" --lr "$LR" --epochs "$EPOCHS"

echo "=== 实验 2: L1+SSIM+Grad ==="
python src/train.py --dataset sunrgbd --data "$DATA" --ablation full --lr "$LR" --epochs "$EPOCHS"

echo "=== 完成 ==="
echo "评测:"
echo "  python src/eval.py --checkpoint outputs/<run1>/best.pt"
echo "  python src/eval.py --checkpoint outputs/<run2>/best.pt"
