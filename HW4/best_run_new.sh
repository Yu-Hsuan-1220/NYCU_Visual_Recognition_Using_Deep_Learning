#!/usr/bin/env bash
# Scale-up + multi-stage fine-tune recipe (target +0.5~1.0 dB over best_run.sh).
#
#   Stage A: bigger backbone (dim=64, blocks [4,6,8,10], refine 6)
#            with the same loss bundle as best_run.sh
#            (Charbonnier + SSIM + FFT, EMA, AMP).
#   Stage B: PSNRLoss fine-tune from Stage A's EMA weights
#            (NAFNet recipe — directly optimises -log MSE).
#   Inference: Stage B's EMA with 8-way TTA.
set -e

DATA=dataset
A=ckpt/big_stageA
B=ckpt/big_stageB

COMMON_MODEL="--dim 64 --num_blocks 4 6 8 10 --num_refinement_blocks 6 \
              --prompt_dims 64 128 384"

# ============ Stage A — long training on the bigger backbone ============
python train.py \
    --data_root "$DATA" --ckpt_dir "$A" \
    $COMMON_MODEL \
    --epochs 500 --batch_size 8 --patch_size 128 \
    --lr 2e-4 --warmup_epochs 20 \
    --loss charbonnier --ssim_weight 0.1 --fft_weight 0.05 \
    --use_ema true --ema_decay 0.999 --amp true \
    --aug_flip true --aug_rot90 true \
    --wandb_project VRDL_4 --run_name big_stageA

# ============ Stage B — PSNRLoss fine-tune from Stage A's EMA ============
python train.py \
    --data_root "$DATA" --ckpt_dir "$B" \
    $COMMON_MODEL \
    --init_from "$A/ema_best.pt" --init_from_kind ema \
    --epochs 150 --batch_size 8 --patch_size 128 \
    --lr 5e-5 --warmup_epochs 5 \
    --loss psnr \
    --use_ema true --ema_decay 0.9995 --amp true \
    --aug_flip true --aug_rot90 true \
    --wandb_project VRDL_4 --run_name big_stageB

# ============ Inference — Stage B EMA + 8-way TTA ============
python inference.py \
    --ckpt "$B/ema_best.pt" --data_root "$DATA" \
    --output pred.npz --tta true
