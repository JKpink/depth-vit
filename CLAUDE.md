# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment setup
bash setup.sh && source venv/bin/activate

# Run all tests (expect 38 passed, 1 skipped — Plan B test skips without DINOv2)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_vit_encoder.py -v

# Run a single test
python -m pytest tests/test_tensor_shapes.py::TestFullPipelineShapes::test_depthformer_pipeline -v

# Train Plan A (self-built ViT, from scratch)
python src/train.py --plan A --data data/nyu --batch-size 16 --epochs 100

# Train Plan B (frozen DINOv2-S)
python src/train.py --plan B --data data/nyu --batch-size 16 --epochs 100

# Resume from checkpoint
python src/train.py --plan A --resume outputs/plan_A/ckpt_019.pt

# Evaluate
python src/eval.py --plan A --checkpoint outputs/plan_A/best.pt --data data/nyu
```

## Architecture

**DepthFormer** is a monocular depth estimation model: ViT encoder → multi-scale token reassembly → progressive upsampling decoder. Single RGB image → single-channel depth map.

### Data flow (inference)

```
[B, 3, 384, 384]                         # RGB input
    │
    ▼
Encoder (Plan A or B)
    │  Plan A: CustomViTEncoder — 12 Transformer blocks, patch=16, embed=384, 6 heads
    │  Plan B: FrozenDINOv2Encoder — torch.hub DINOv2-S/14, all params frozen
    │  Both return: [L4_tokens, L8_tokens, L12_tokens]  each [B, N, D]
    ▼
MultiScaleReassemble
    │  Reassemble each token set: reshape [B,N,D] → [B,D,H,W] + Conv2d
    │  Upsample all to uniform target_size (48×48 for A, 54×54 for B)
    │  Concat → [B, 3*D, T, T]  (typically 1152 channels)
    ▼
DepthDecoder
    │  ConvBlock + Upsample ×3:  48→96→192→384
    │  Head: Conv2d(64, 1)
    │  Progressive channel reduction: 1152→256→128→64→1
    ▼
[B, 1, 384, 384]                         # Depth map output (normalized 0–1)
```

### Loss function

`L_total = 0.6·L1 + 0.3·SSIM + 0.1·GradientLoss`

GradientLoss computes L1 difference of x/y directional gradients between pred and GT — zero extra parameters, encourages edge alignment. The loss returns both the scalar and a dict of per-component metrics (`l1`, `ssim`, `grad`, `edge_ratio`, `total`).

### Key design decisions

- **Multi-scale features at 3 depths**: Extracts tokens from blocks at 1/3, 2/3, and final layers (L4, L8, L12 out of 12) — captures features at different receptive field sizes.
- **Plan A vs Plan B share the same pipeline**: `DepthFormer(plan="A"/"B")` is the only entry point; the encoder is swapped internally. Plan B uses `patch_size=14` (DINOv2-S) vs Plan A's 16, which changes target sizes in reassembly.
- **AMP mixed precision**: Training uses `torch.cuda.amp` (GradScaler + autocast) for VRAM efficiency.
- **Checkpoint format**: Dict `{"model": state_dict, "optimizer": state_dict, "epoch": int, "best_val": float}`. Saved every 20 epochs + best model separately.
- **Test design**: Tests are pure shape/dimension validations on CPU — no GPU required, no real data needed. `conftest.py` provides shared fixtures: `batch_input`, `batch_depth_gt`, `batch_tokens`, `multi_scale_tokens`.
