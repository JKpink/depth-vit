#!/bin/bash
# 下载 SUN RGB-D (已预处理的配对版本)
# 来源: chrischoy/SUN_RGBD (斯坦福)
set -e

DATA_DIR="data/sunrgbd"
URL="http://cvgl.stanford.edu/data2/sun_rgbd.tgz"
TGZ_FILE="/tmp/sun_rgbd.tgz"

mkdir -p "$DATA_DIR"

echo "=== 下载 (~15GB) ==="
if [ -f "$TGZ_FILE" ]; then
    echo "已有 $TGZ_FILE, 跳过下载"
else
    wget -O "$TGZ_FILE" "$URL" || curl -L -o "$TGZ_FILE" "$URL"
fi

echo "=== 解压 ==="
tar -xzf "$TGZ_FILE" -C "$DATA_DIR" --strip-components=1

echo "=== 完成 ==="
echo "文件在 $DATA_DIR/"
ls "$DATA_DIR/" 2>/dev/null || echo "(目录为空, 检查解压)"
