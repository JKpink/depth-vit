"""NYU Depth v2 数据加载"""

from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class NYUDepthDataset(Dataset):
    """NYU Depth v2"""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 384,
        augment: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and split == "train"

        # 扫描图片文件
        rgb_dir = self.data_dir / "rgb"
        depth_dir = self.data_dir / "depth"
        if not rgb_dir.exists():
            raise FileNotFoundError(
                f"{rgb_dir} 不存在。NYU v2 目录结构: {data_dir}/rgb/ 和 {data_dir}/depth/"
            )

        self.rgb_files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
        self.depth_files = sorted(depth_dir.glob("*.png")) + sorted(depth_dir.glob("*.npy"))

        if not self.rgb_files:
            raise FileNotFoundError(f"{rgb_dir} 中没有找到图片文件")
        if not self.depth_files:
            raise FileNotFoundError(f"{depth_dir} 中没有找到深度文件")

        # 训练/验证划分: 前 1449 训练, 后 654 验证
        total = len(self.rgb_files)
        train_count = min(1449, total - 654) if total > 654 else total
        if split == "train":
            self.rgb_files = self.rgb_files[:train_count]
            self.depth_files = self.depth_files[:train_count]
        else:
            self.rgb_files = self.rgb_files[train_count:]
            self.depth_files = self.depth_files[train_count:]

        # 预处理
        self.rgb_transform = self._build_rgb_transform()
        self.depth_transform = self._build_depth_transform()

    def _build_rgb_transform(self):
        t = [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()]
        return transforms.Compose(t)

    def _build_depth_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
        ])

    def _load_depth(self, path: Path) -> torch.Tensor:
        if path.suffix == ".npy":
            depth = np.load(str(path))
        else:
            depth = np.array(Image.open(str(path)), dtype=np.float32)

        # 保留米单位 [0, 10m] (对齐 DA-V2)
        if depth.max() > 20:  # 可能是毫米单位
            depth = depth / 1000.0
        depth = np.clip(depth, 0.0, 10.0)
        return torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]

    def __len__(self) -> int:
        return len(self.rgb_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb = self.rgb_transform(Image.open(str(self.rgb_files[idx])).convert("RGB"))

        depth_path = self.depth_files[idx]
        depth = self._load_depth(depth_path)
        depth = self.depth_transform(depth)

        # 仅水平翻转（对齐 DA-V2）
        if self.augment and torch.rand(1).item() > 0.5:
            rgb = transforms.functional.hflip(rgb)
            depth = transforms.functional.hflip(depth)

        return rgb, depth
