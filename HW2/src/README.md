# NYCU Visual Recognition HW2 — Digit Detection with Deformable DETR

## Introduction

**Deformable DETR** based digit detection on the SVHN-style dataset.
Uses **ResNet-50** backbone (pretrained on ImageNet) with **multi-scale deformable
attention** — a pure PyTorch implementation (no custom CUDA ops required).

### Key differences from standard DETR

| Feature | DETR  | Deformable DETR (src) |
|---|---|---|
| Backbone features | Single-scale (stride 32) | Multi-scale (strides 8, 16, 32, 64) |
| Attention | Dense global attention | Sparse deformable attention |
| Encoder input | Flattened single feature map | Concatenated multi-scale features |
| Reference points | N/A (learned queries) | Predicted per query, optionally refined |
| Box refinement | No | Optional (`--with_box_refine`) |
| FFN dim (default) | 2048 | 1024 |

## Environment Setup

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
cd src

# Basic training
python train.py \
    --train_img_dir ../dataset/train \
    --train_ann ../dataset/train.json \
    --val_img_dir ../dataset/valid \
    --val_ann ../dataset/valid.json \
    --epochs 200 \
    --batch_size 4 \
    --accumulate_steps 4 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --num_queries 50 \
    --num_feature_levels 4 \
    --aux_loss \
    --output_dir ./output

# With iterative box refinement
python train.py \
    --with_box_refine \
    --epochs 200 \
    --output_dir ./output_refine

# With focal loss
python train.py \
    --focal_loss \
    --output_dir ./output_focal
```

### Inference

```bash
python inference.py \
    --checkpoint ./output/best.pth \
    --test_img_dir ../dataset/test \
    --output_file pred.json \
    --score_threshold 0.3
```

This generates `pred.json` and `pred.zip` ready for CodaBench submission.

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--num_feature_levels` | 4 | Number of multi-scale feature levels |
| `--enc_n_points` | 4 | Sampling points per head per level (encoder) |
| `--dec_n_points` | 4 | Sampling points per head per level (decoder) |
| `--with_box_refine` | False | Enable iterative bounding box refinement |
| `--dim_feedforward` | 1024 | FFN hidden dimension |
| `--lr` | 2e-4 | Learning rate (2x higher than DETR) |
| `--lr_backbone` | 2e-5 | Backbone learning rate |
