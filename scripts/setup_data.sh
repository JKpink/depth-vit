#!/bin/bash
# 一键下载并解压 NYU Depth v2 数据集
# 用法: bash setup_data.sh [数据目录，默认 ./data/nyu]

set -e

DATA_DIR="${1:-data/nyu}"
MAT_FILE="/tmp/nyu_depth_v2_labeled.mat"
URL="http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"

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
