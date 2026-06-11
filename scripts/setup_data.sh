#!/bin/bash
# 一键下载并解压 NYU Depth v2 数据集
# 用法:
#   bash setup_data.sh                                 # 直连下载
#   bash setup_data.sh --proxy                         # 代理 127.0.0.1:7890
#   bash setup_data.sh --proxy 7890                    # 代理 127.0.0.1:指定端口
#   bash setup_data.sh --proxy 192.168.1.100:7890      # 代理 指定IP:端口
#   bash setup_data.sh --no-proxy                      # 强制直连
#   bash setup_data.sh /path/to/data --proxy 192.168.1.100:7890

set -e

DATA_DIR="data/nyu"
USE_PROXY=""
PROXY_ADDR=""

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --proxy)
            USE_PROXY="yes"
            shift
            if [[ $# -gt 0 && ! "$1" =~ ^- ]]; then
                PROXY_ADDR="$1"
                shift
            fi
            ;;
        --no-proxy)
            USE_PROXY="no"
            shift
            ;;
        -*)
            echo "未知参数: $1"
            echo "用法: bash setup_data.sh [数据目录] [--proxy [ip:port]] [--no-proxy]"
            exit 1
            ;;
        *)
            DATA_DIR="$1"
            shift
            ;;
    esac
done

# ── 解析代理地址 ──
if [[ "$USE_PROXY" == "yes" ]]; then
    if [[ -z "$PROXY_ADDR" ]]; then
        PROXY_ADDR="127.0.0.1:7890"
    elif [[ "$PROXY_ADDR" =~ ^[0-9]+$ ]]; then
        # 纯数字 → 端口，IP 默认 127.0.0.1
        PROXY_ADDR="127.0.0.1:${PROXY_ADDR}"
    elif [[ ! "$PROXY_ADDR" =~ : ]]; then
        # 只有 IP 没有端口 → 补默认端口
        PROXY_ADDR="${PROXY_ADDR}:7890"
    fi
    echo "使用代理: http://${PROXY_ADDR}"
    export http_proxy="http://${PROXY_ADDR}"
    export https_proxy="http://${PROXY_ADDR}"
else
    echo "直连下载（不使用代理）"
fi

echo "=== 1. 下载 NYU Depth v2 ==="
if [ -f "$MAT_FILE" ]; then
    echo "文件已存在: $MAT_FILE"
else
    echo "下载中... (约 2.8GB)"
    wget -q --show-progress "$URL" -O "$MAT_FILE" || {
        echo "wget 失败，试试 Kaggle 方式:"
        echo "  pip install kagglehub && python -c \"import kagglehub; print(kagglehub.dataset_download('soumikrakshit/nyu-depth-v2'))\""
        exit 1
    }
fi

echo ""
echo "=== 2. 解压为图片 ==="
pip install -q h5py scipy 2>/dev/null

python3 << 'PYEOF'
import h5py
import numpy as np
from PIL import Image
from pathlib import Path
import os

DATA_DIR = os.environ.get("DATA_DIR", "data/nyu")
MAT_FILE = "/tmp/nyu_depth_v2_labeled.mat"

rgb_dir = Path(DATA_DIR) / "rgb"
depth_dir = Path(DATA_DIR) / "depth"
rgb_dir.mkdir(parents=True, exist_ok=True)
depth_dir.mkdir(parents=True, exist_ok=True)

print(f"读取 {MAT_FILE} ...")
f = h5py.File(MAT_FILE, "r")

images = f["images"]   # (1449, 3, 480, 640)
depths = f["depths"]   # (1449, 480, 640)

n = images.shape[0]
print(f"共 {n} 张，解压中...")

for i in range(n):
    rgb = images[i]                # (3, 480, 640)
    rgb = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)
    Image.fromarray(rgb).save(rgb_dir / f"{i+1:05d}.jpg")

    depth = depths[i]              # (480, 640)
    depth = depth.astype(np.float32)
    np.save(str(depth_dir / f"{i+1:05d}.npy"), depth)

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{n}")

f.close()
print(f"✅ 完成。RGB: {rgb_dir}, Depth: {depth_dir}")
PYEOF

echo ""
echo "=== 3. 验证 ==="
echo "RGB 数量: $(ls "$DATA_DIR/rgb/" | wc -l)"
echo "Depth 数量: $(ls "$DATA_DIR/depth/" | wc -l)"
echo ""
echo "✅ 数据集准备完成: $DATA_DIR"
echo "   python src/train.py --plan A --data $DATA_DIR"
