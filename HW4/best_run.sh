#!/usr/bin/env bash
# Full training + inference recipe with the recommended score-boost bundle:
#   Charbonnier loss, EMA, AMP, gradient clipping, warmup-cosine LR,
#   flip+rot90 augmentation; inference with 8-way TTA.


python train.py \
    --data_root "dataset"   \
    --ckpt_dir "ckpt/run8"  \ 
    --epochs 300   \
    --batch_size 8     \
    --patch_size 128     \
    --lr 2e-4     \
    --warmup_epochs 15     \
    --loss charbonnier     \
    --use_ema true     \
    --amp true     \
    --aug_flip true     \
    --aug_rot90 true \
    --ssim_weight 0.1 \
    --fft_weight 0.05

python inference.py \
    --ckpt "ckpt/run8/ema_best.pt" \
    --data_root "dataset" \
    --output pred.npz \
    --tta true
