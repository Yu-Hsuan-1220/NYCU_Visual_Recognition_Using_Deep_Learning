python prepare_coco_dataset.py

python train.py --fold 0 --epochs 50 --amp
python train.py --fold 1 --epochs 50 --amp
python train.py --fold 2 --epochs 50 --amp
python train.py --fold 3 --epochs 50 --amp
python train.py --fold 4 --epochs 50 --amp

python inference.py \
  --checkpoints ../work_dirs/fold0/best_coco_segm_mAP_50_epoch_20.pth \
                ../work_dirs/fold1/best_coco_segm_mAP_50_epoch_45.pth \
                ../work_dirs/fold2/best_coco_segm_mAP_50_epoch_45.pth \
                ../work_dirs/fold3/best_coco_segm_mAP_50_epoch_30.pth \
                ../work_dirs/fold4/best_coco_segm_mAP_50_epoch_40.pth \
  --output ../test-results-ensemble.json --tta --load_all_models
