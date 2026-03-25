cd /project2/YuHsuanLee/VRDL/HW1/src
# training
python /project2/YuHsuanLee/VRDL/HW1/src/train.py --backbone resnet50 --img_size 224 --batch_size 64 --epochs 100 \
--lr 1e-3 --lr_backbone 1e-4 --weight_decay 5e-4 --optimizer adamw --scheduler cosine --warmup_epochs 5 \
--use_RandAugment --use_mixup_and_cutmix --use_label_smoothing --label_smoothing 0.1 --dropout 0.5 \
--wandb_run_name resnet50-v2-strong-recipe

python /project2/YuHsuanLee/VRDL/HW1/src/train.py --backbone resnet101 --img_size 224 --batch_size 64 --epochs 100 \
--lr 1e-3 --lr_backbone 1e-4 --weight_decay 5e-4 --optimizer adamw --scheduler cosine --warmup_epochs 5 \
--use_RandAugment --use_mixup_and_cutmix --use_label_smoothing --label_smoothing 0.1 --dropout 0.5 --save_dir checkpoints_1\
--wandb_run_name resnet101-v2-strong-recipe

python /project2/YuHsuanLee/VRDL/HW1/src/train.py --backbone resnet152 --img_size 224 --batch_size 64 --epochs 200 \
--lr 1e-3 --lr_backbone 1e-4 --weight_decay 5e-4 --optimizer adamw --scheduler cosine --warmup_epochs 10 --save_dir checkpoints_2\
--use_RandAugment --use_mixup_and_cutmix --use_label_smoothing --label_smoothing 0.1 --dropout 0.5 \
--wandb_run_name resnet152-v2-strong-recipe

# inference
python /project2/YuHsuanLee/VRDL/HW1/src/inference.py --backbone resnet50 --dropout 0.5 \
--checkpoint ./checkpoints/best_resnet50.pth --output_csv prediction1.csv --use_tta

python /project2/YuHsuanLee/VRDL/HW1/src/inference.py --backbone resnet101 --dropout 0.5 \
--checkpoint ./checkpoints_1/best_resnet101.pth --output_csv prediction2.csv --use_tta

python /project2/YuHsuanLee/VRDL/HW1/src/inference.py --backbone resnet152 --dropout 0.5 \
--checkpoint ./checkpoints_2/best_resnet152.pth --output_csv prediction3.csv --use_tta

# voting

python /project2/YuHsuanLee/VRDL/HW1/src/voting.py --input_csvs prediction3.csv prediction2.csv prediction1.csv \