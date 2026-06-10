# DepthFormer — 基于 ViT 的轻量级单目深度估计

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-38/38_passed-brightgreen.svg)](.)

单目深度估计课程项目。使用 Vision Transformer 编码器 + 多尺度特征重建 + 渐进上采样解码器，从单张 RGB 图像预测深度图。支持两种 Encoder：自研轻量 ViT（Plan A）和冻结 DINOv2（Plan B）。

![overview]()

## 架构

```
RGB (384×384×3)
    │
    ▼
┌─────────────────────┐
│  ViT / DINOv2       │  Plan A: 12层自研ViT (patch=16, embed=384, head=6)
│  Encoder            │  Plan B: 冻结 DINOv2-S/14 (22M特征提取)
└──────┬──────────────┘
       │ [L4 tokens, L8 tokens, L12 tokens]
       ▼
┌─────────────────────┐
│  MultiScale         │  Token → 空间特征图重建
│  Reassemble         │  3层特征 → concat → 单一大特征图
└──────┬──────────────┘
       │ [B, 3×384, H/patch, W/patch]
       ▼
┌─────────────────────┐
│  Depth Decoder      │  渐进上采样: 48ch → 24ch → 12ch → 1ch
│                     │  ×2 ×2 ×2 = ×8 恢复原图分辨率
└──────┬──────────────┘
       │
       ▼
  Depth Map (384×384×1)
```

### 损失函数

```
L_total = 0.6·L1 + 0.3·SSIM + 0.1·GradientLoss
```

GradientLoss 计算预测深度图和真值深度图在 x/y 方向梯度的 L1 差，鼓励边缘对齐。

## 项目结构

```
depth-vit/
├── src/
│   ├── model/
│   │   ├── vit_encoder.py      # 自研 ViT (PatchEmbed + MHSA + TransformerBlock)
│   │   ├── dino_encoder.py     # 冻结 DINOv2 封装
│   │   ├── reassemble.py       # Token → 空间特征图
│   │   ├── decoder.py          # 渐进上采样解码器
│   │   └── depth_model.py      # DepthFormer 完整模型 (Plan A/B 统一接口)
│   ├── data/
│   │   └── nyu_dataset.py      # NYU Depth v2 数据加载器
│   ├── loss.py                 # L1 + SSIM + Gradient 组合损失
│   ├── train.py                # 训练脚本 (AMP + checkpoint + TensorBoard)
│   ├── eval.py                 # 评测 (RMSE, AbsRel, δ1, Boundary RMSE)
│   └── viz.py                  # 可视化 (深度对比、边界放大、Baseline柱状图)
├── configs/
│   └── default.yaml            # 模型/训练/损失参数
├── tests/
│   ├── conftest.py             # 共享 fixtures
│   ├── test_vit_encoder.py     # 8 tests: PatchEmbed, MHSA, Transformer, ViT
│   ├── test_decoder.py         # 6 tests: Reassemble, MultiScale, DepthDecoder
│   ├── test_loss.py            # 7 tests: Gradient, SSIM, DepthLoss, 消融
│   └── test_tensor_shapes.py   # 17 tests: 全管线 + 参数化维度检查
├── configs/default.yaml
├── requirements.txt
├── setup.sh
└── README.md
```

## 快速开始

### 1. 环境搭建

```bash
bash setup.sh
source venv/bin/activate
```

### 2. 下载数据集

下载 [NYU Depth v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) 数据，解压到 `data/nyu/`，结构如下：

```
data/nyu/
├── rgb/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
└── depth/
    ├── 00001.png
    ├── 00002.png
    └── ...
```

### 3. 运行测试（验证环境）

```bash
python -m pytest tests/ -v
```

应显示 **38 passed, 1 skipped**。

### 4. 训练

```bash
# Plan A — 自研 ViT
python src/train.py --plan A --data data/nyu --batch-size 16 --epochs 100

# Plan B — 冻结 DINOv2
python src/train.py --plan B --data data/nyu --batch-size 16 --epochs 100

# 断点续训
python src/train.py --plan A --resume outputs/plan_A/ckpt_019.pt
```

TensorBoard 日志保存在 `outputs/plan_A/`（或 `plan_B/`），每20个epoch保存checkpoint，自动保留最佳模型 `best.pt`。

### 5. 评测

```bash
python src/eval.py --plan A --checkpoint outputs/plan_A/best.pt --data data/nyu
```

输出 RMSE / AbsRel / δ1 / BoundaryRMSE，结果保存到 `outputs/eval.json`。

## 指标说明

| 指标 | 含义 | 方向 |
|------|------|------|
| RMSE | 均方根误差 | ↓ 越低越好 |
| AbsRel | 绝对相对误差 \|pred-gt\|/gt | ↓ 越低越好 |
| δ1 | 误差<1.25的像素比例 | ↑ 越高越好 |
| BoundaryRMSE | 仅在深度梯度>0.05的边界像素上算RMSE | ↓ 越低越好 |

## 实验设计

| 实验 | 说明 |
|------|------|
| Plan A | 自研12层ViT，完整梯度损失 |
| Plan B | 冻结DINOv2-S，完整梯度损失 |
| 消融: 无梯度 | λ_grad=0，仅 L1+SSIM |

### Baseline 对比

| Baseline | 来源 | 规模 |
|----------|------|------|
| Lite-Mono | CVPR 2023 | 3.5M |
| DenseDepth | 2019 | 33M |
| Depth Anything V2-Small | NeurIPS 2024 | 24M |

## 硬件要求

| GPU | 显存 | Batch 32 | Batch 16 |
|-----|------|----------|----------|
| RTX 3060 12GB | 12GB | ❌ OOM | ✅ |
| RTX 3090 24GB | 24GB | ✅ (~18GB) | ✅ |

推荐：RTX 3090 24GB，日租约 ¥26.4，100 epochs 约 2 小时。

## 技术要点

- **多尺度特征提取**: L4/L8/L12 三层 Token 分别重建特征图后融合，保留不同感受野信息
- **梯度感知损失**: 在深度梯度域计算 L1，对物体边界区域施加更强监督
- **SSIM 损失**: 保持预测深度图的结构一致性
- **AMP 混合精度**: GradScaler + autocast，节省显存的同时加速训练
- **Plan A/B 双轨**: 同一训练/评测管线，`--plan` 一键切换

## 相关工作综述

项目设计阶段调研了以下方向：

### 单目深度估计（MDE）演进

| 阶段 | 代表工作 | 关键思路 |
|------|---------|---------|
| CNN 时代 (2014–2020) | Eigen et al. (NeurIPS 2014), DenseDepth (ICCV 2019) | 编解码器 + 多尺度特征 |
| Transformer 引入 (2021–2023) | DPT (ICCV 2021), AdaBins (CVPR 2021), MiDaS v3.1 (ICCV 2021) | ViT 做 backbone，token → 空间重建 |
| 大模型泛化 (2023–2024) | ZoeDepth (2023), Depth Anything V1/V2 (NeurIPS 2024) | 大规模预训练 + 零样本泛化 |
| 轻量高效 (2023–2025) | Lite-Mono (CVPR 2023), MobileViT, EfficientFormer | 小模型实时推理 |
| 前沿 (2025–2026) | PatchRefiner V2 (ICLR 2026), DepthLM (ICLR 2026) | 扩散模型、语言引导深度 |

### ViT 与 Dense Prediction

- **ViT** (Dosovitskiy et al., ICLR 2021) — Patch Embed + 自注意力取代卷积，全局感受野
- **DPT** (Ranftl et al., ICCV 2021) — 首个系统性地用 ViT 做密集预测，提出 token 重组策略
- **DINOv2** (Oquab et al., 2023) — 自监督预训练 ViT，冻结后可做通用特征提取器

### 边缘/边界问题

轻量模型在物体边界处深度模糊是已知瓶颈，主流应对策略：

| 策略 | 代表工作 | 代价 |
|------|---------|------|
| 显式边缘分支 | Hu et al., EdgeNet | +1~2M 参数 |
| 梯度域监督 | Ummenhofer et al., 2017 | 零参数增量 |
| 多尺度边界聚合 | PatchRefiner V2 (ICLR 2026) | 额外细化网络 |
| 高频增强解码器 | LeReS, Adabins | 解码器设计 |

本项目选择 **梯度域监督**（Gradient Loss），对物体边界处深度跳变施加 L1 约束，零额外参数，适合轻量模型。

### 关键参考论文

| 论文 | 会议 | 与本项目关联 |
|------|------|-------------|
| Eigen et al. — Depth Map Prediction from a Single Image using a Multi-Scale Deep Network | NeurIPS 2014 | MDE 奠基工作 |
| DenseDepth — High Quality Monocular Depth Estimation via Transfer Learning | ICCV 2019 | 编解码器范式 |
| DPT — Vision Transformers for Dense Prediction | ICCV 2021 | ViT → 密集预测 |
| Lite-Mono — A Lightweight CNN and Transformer Architecture for Self-Supervised MDE | CVPR 2023 | 轻量 baseline |
| DINOv2 — Learning Robust Visual Features without Supervision | 2023 | Plan B 依赖 |
| Depth Anything V2 — Boosting MDE via Synthetic Captions | NeurIPS 2024 | SOTA upper bound |
| PatchRefiner V2 — Fast and Accurate Monocular Depth Estimation | ICLR 2026 | 边界细化参考 |
| SSIM Loss — Wang et al., Image Quality Assessment | TIP 2004 | 结构相似度损失 |

## License

MIT
