# HW4 — All-in-One Image Restoration (PromptIR + NAFBlock)

NYCU Visual Recognition using Deep Learning, Homework 4. The task is to
restore degraded images (rain + snow) with a **single** model. We follow the
PromptIR architecture (Potlapalli et al., NeurIPS 2023) but swap its
TransformerBlock (MDTA + GDFN) for the **NAFBlock** from NAFNet (Chen et al.,
ECCV 2022) at every internal site. PromptGenBlock and the three prompt
injection points are kept verbatim, since the prompt mechanism is the
PromptIR contribution we are required to elaborate on.

## Introduction

- **Backbone**: 4-level U-Net with PixelUnshuffle / PixelShuffle resizing.
- **Encoder / decoder block**: NAFBlock — LayerNorm2d → 1×1 conv → 3×3 DWConv
  → SimpleGate → Simplified Channel Attention → 1×1 conv → β-scaled residual,
  followed by LayerNorm2d → 1×1 conv → SimpleGate → 1×1 conv → γ-scaled
  residual. Attention-free, activation-free.
- **Prompt mechanism**: at the latent and the two coarsest decoder levels,
  a `PromptGenBlock` produces a soft mixture of learnable prompt parameters
  (weighted by GAP-projected input features), interpolates it to the feature
  spatial size, and concatenates it with the feature map before another
  NAFBlock and a 1×1 reducer.
- **Output**: residual added to the degraded input.

## Environment setup

```
bash env_setup.sh        # creates conda env VRDL_4 and pip-installs deps
conda activate VRDL_4
```

Dependencies: PyTorch 2.1.2 (CUDA 12.1), torchvision 0.16.2, einops, pillow,
tqdm, wandb, scikit-image, numpy<2.

## Dataset layout

```
HW4/dataset/
├── train/
│   ├── degraded/  rain-1.png … rain-1600.png, snow-1.png … snow-1600.png
│   └── clean/     rain_clean-1.png … rain_clean-1600.png, snow_clean-1.png …
└── test/
    └── degraded/  0.png … 99.png   (all 256×256 RGB)
```

## Usage

### Train

```
python train.py --data_root HW4/dataset --ckpt_dir HW4/ckpt/run1
```

Every hyperparameter is an argparse switch. Highlights:

| Flag | Default | Notes |
|------|---------|-------|
| `--dim`, `--num_blocks`, `--num_refinement_blocks` | 48, [4 6 6 8], 4 | Model size |
| `--dw_expand`, `--ffn_expand`, `--drop_path` | 2, 2, 0.0 | NAFBlock |
| `--decoder_prompt` | true | Enable PromptIR prompts |
| `--prompt_dims`, `--prompt_len`, `--prompt_sizes` | [64 128 320], 5, [64 32 16] | PromptGenBlock |
| `--patch_size`, `--batch_size`, `--epochs` | 128, 8, 200 | Training schedule |
| `--lr`, `--wd`, `--warmup_epochs` | 2e-4, 1e-4, 15 | AdamW + warmup-cosine |
| `--loss` | charbonnier | `l1 / l2 / charbonnier / psnr` |
| `--ssim_weight`, `--fft_weight` | 0, 0 | Auxiliary loss weights |
| `--use_ema`, `--ema_decay` | true, 0.999 | Weight EMA |
| `--amp`, `--grad_clip` | true, 1.0 | Mixed precision + clipping |
| `--aug_flip`, `--aug_rot90`, `--aug_rgb_shuffle`, `--aug_mixup_p` | true, true, false, 0 | Augmentations |

Checkpoints (`last.pt`, `best.pt`, `ema_best.pt`) and `args.json` are written
to `--ckpt_dir`.

### Weights & Biases logging (optional)

Set `--wandb_project` to enable. Once-off `wandb login` first, or run with
`WANDB_MODE=offline` for a local-only run that can be `wandb sync`'d later.

| Flag | Default | Notes |
|------|---------|-------|
| `--wandb_project` | "" | Empty = disabled |
| `--wandb_entity`, `--run_name`, `--wandb_tags`, `--wandb_notes` | "" | Standard run metadata |
| `--wandb_log_images` | true | Log `(deg \| pred \| clean)` triplets from a fixed set of val samples |
| `--wandb_image_count` | 4 | How many val samples to visualize |
| `--wandb_image_every` | 5 | Log images every N epochs (also whenever val PSNR improves) |
| `--wandb_watch_freq` | 0 | `wandb.watch` log_freq for grad/param histograms; 0 = off |

Logged per epoch: `train/loss`, `train/lr`, `train/grad_norm`,
`train/time_s`, `train/samples_per_s`, `train/gpu_mem_gb`,
`val/psnr_all`, `val/psnr_rain`, `val/psnr_snow`, `val/best_psnr`,
plus `val/samples` image grid. Run summary keeps `params_M`, `best_psnr`,
`final_best_psnr`, dataset sizes.

### Inference (build `pred.npz`)

```
python inference.py \
    --ckpt HW4/ckpt/run1/ema_best.pt \
    --data_root HW4/dataset \
    --output pred.npz \
    --tta true
```

Useful switches:

| Flag | Default | Notes |
|------|---------|-------|
| `--tta` | true | 8-way self-ensemble (identity / flips / rot90 combos) |
| `--tile`, `--overlap` | 0, 32 | Set `--tile 256` etc. for sliding-window inference |
| `--ckpt_kind` | auto | `auto` prefers EMA weights; `raw` forces non-EMA |

The output `pred.npz` is a dict keyed by the test filenames
(`"0.png"` … `"99.png"`) whose values are uint8 arrays of shape `(3, H, W)` —
exactly the format expected by CodaBench (matches
`example/example_img2npz.py`).

### One-shot recipe

```
bash best_run.sh           # original baseline (dim=48, 300 epochs)
bash best_run_new.sh       # scale-up + multi-stage fine-tune (dim=64, ~650 epochs)
```

### Multi-stage fine-tune (`best_run_new.sh`)

Two-stage NAFNet-style recipe that pushes another +0.5–1.0 dB on top of
`best_run.sh`:

- **Stage A** — `dim=64`, `num_blocks=[4,6,8,10]`, `num_refinement_blocks=6`,
  `prompt_dims=[64,128,384]` (~47 M params). 500 epochs of Charbonnier + SSIM(0.1) +
  FFT(0.05) at `patch=128`, `bs=8`.
- **Stage B** — Same model architecture, **weights initialised from Stage A's EMA**
  via `--init_from ckpt/big_stageA/ema_best.pt --init_from_kind ema`, then 150
  epochs of pure `PSNRLoss` at `lr=5e-5`. This directly optimises the
  evaluation metric.
- **Inference** — `ema_best.pt` from Stage B with 8-way TTA.

VRAM: ~13 GB peak, fits on a single 4090. Wall-clock: Stage A ≈ 14 h, Stage B
≈ 4 h.

New checkpoint-loading flags (only used by Stage B):

| Flag | Default | Notes |
|------|---------|-------|
| `--init_from` | "" | Path to a `.pt`; loads model (+EMA) weights only |
| `--init_from_kind` | ema | `ema` (recommended) or `model` |
| `--resume` | "" | **Full** resume (opt + sched + scaler + epoch); use only when continuing the same run |

`--init_from` and `--resume` are mutually exclusive — if both are passed,
`--resume` wins.

## Performance snapshot

Held-out 5 % validation PSNR (training in progress, table to update):

| Run | Loss | EMA | TTA | val PSNR (all) | val PSNR (rain / snow) |
|-----|------|-----|-----|----------------|------------------------|
| baseline | charbonnier | ✓ | (eval only) | TBD | TBD |

## References

- Potlapalli, Zamir, Khan, Khan. *PromptIR: Prompting for All-in-One Blind
  Image Restoration.* NeurIPS 2023. arXiv:2306.13090.
- Chen, Chu, Zhang, Sun. *Simple Baselines for Image Restoration.* ECCV 2022.
  arXiv:2204.04676.
- Zamir, Arora, Khan, Hayat, Khan, Yang. *Restormer: Efficient Transformer
  for High-Resolution Image Restoration.* CVPR 2022.
