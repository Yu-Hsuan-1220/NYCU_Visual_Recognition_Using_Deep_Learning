#!/usr/bin/env bash
# improved_run.sh — trains with full augmentation suite and runs ensemble + rotation TTA.
#
# After training completes, replace the _epoch_XX placeholders below with the
# actual best checkpoint filenames printed by each fold's training run.
#
# Changes vs best_run.sh:
#   Training: --color_jitter  --diagonal_flip  --resize_ratio_min 0.1 --resize_ratio_max 3.0
#             --bbox_loss giou  --mask_roi_size 28
#   Inference: --tta_rotation  --mask_roi_size 28

cd "$(dirname "$0")"

python prepare_coco_dataset.py

python train.py --fold 0 --epochs 100 --amp \
  --color_jitter --diagonal_flip \
  --resize_ratio_min 0.1 --resize_ratio_max 3.0 \
  --bbox_loss giou --mask_roi_size 28

python train.py --fold 1 --epochs 50 --amp \
  --color_jitter --diagonal_flip \
  --resize_ratio_min 0.1 --resize_ratio_max 3.0 \
  --bbox_loss giou --mask_roi_size 28

python train.py --fold 2 --epochs 50 --amp \
  --color_jitter --diagonal_flip \
  --resize_ratio_min 0.1 --resize_ratio_max 3.0 \
  --bbox_loss giou --mask_roi_size 28

python train.py --fold 3 --epochs 50 --amp \
  --color_jitter --diagonal_flip \
  --resize_ratio_min 0.1 --resize_ratio_max 3.0 \
  --bbox_loss giou --mask_roi_size 28

python train.py --fold 4 --epochs 50 --amp \
  --color_jitter --diagonal_flip \
  --resize_ratio_min 0.1 --resize_ratio_max 3.0 \
  --bbox_loss giou --mask_roi_size 28

# Replace _epoch_XX with actual best checkpoint filenames after training
python inference.py \
  --checkpoints \
    ../work_dirs/fold0/best_coco_segm_mAP_50_epoch_XX.pth \
    ../work_dirs/fold1/best_coco_segm_mAP_50_epoch_XX.pth \
    ../work_dirs/fold2/best_coco_segm_mAP_50_epoch_XX.pth \
    ../work_dirs/fold3/best_coco_segm_mAP_50_epoch_XX.pth \
    ../work_dirs/fold4/best_coco_segm_mAP_50_epoch_XX.pth \
  --output ../test-results-improved.json \
  --tta --tta_rotation \
  --load_all_models \
  --bbox_loss giou \
  --mask_roi_size 28
