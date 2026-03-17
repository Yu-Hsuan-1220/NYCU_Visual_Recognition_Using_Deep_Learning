cd /project2/YuHsuanLee/VRDL/HW1/src
# training
python /project2/YuHsuanLee/VRDL/HW1/src/train.py --backbone resnet50 --img_size 224 --batch_size 64 --epochs 100 \
--lr 1e-3 --lr_backbone 1e-4 --weight_decay 5e-4 --optimizer adamw --scheduler cosine --warmup_epochs 5 \
--use_RandAugment --use_mixup_and_cutmix --use_label_smoothing --label_smoothing 0.1 --dropout 0.5 \
--wandb_run_name resnet50-v2-strong-recipe
# inference
python /project2/YuHsuanLee/VRDL/HW1/src/inference.py --backbone resnet50 --dropout 0.5 \
--checkpoint ./checkpoints/best_resnet50.pth --output_csv prediction.csv --use_tta