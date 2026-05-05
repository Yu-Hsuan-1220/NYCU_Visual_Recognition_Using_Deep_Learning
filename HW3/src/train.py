"""
Training script for Cascade Mask R-CNN with ConvNeXt-V2-Base.

Usage:
    python train.py --fold 0 --epochs 100 --batch_size 2 --lr 1e-4
    python train.py --fold all --epochs 100  # Train all 5 folds sequentially
"""
import argparse
import os
import sys

# Ensure the src directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Train Cascade Mask R-CNN")

    # Data
    parser.add_argument("--data_root", type=str, default="../dataset",
                        help="Root directory of the dataset")
    parser.add_argument("--ann_dir", type=str, default="../dataset/annotations",
                        help="Directory containing COCO annotation JSONs")
    parser.add_argument("--fold", type=str, default="0",
                        help="Fold index (0-4) or 'all' to train all folds")

    # Model
    parser.add_argument("--backbone", type=str, default="convnextv2_base",
                        help="Backbone model name (timm)")
    parser.add_argument("--fpn_channels", type=int, default=256,
                        help="FPN output channels")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of foreground classes")
    parser.add_argument("--drop_path_rate", type=float, default=0.4,
                        help="Stochastic depth rate")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use ImageNet pretrained backbone")
    parser.add_argument("--no_pretrained", action="store_true", default=False,
                        help="Do not use pretrained backbone")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate")
    parser.add_argument("--backbone_lr_mult", type=float, default=0.1,
                        help="Learning rate multiplier for backbone")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                        help="Weight decay for AdamW")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Number of warmup epochs")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01,
                        help="Minimum LR ratio for cosine schedule")

    # Augmentation
    parser.add_argument("--img_scale", type=int, nargs=2, default=[1024, 1024],
                        help="Training image scale (H, W)")
    parser.add_argument("--multiscale", action="store_true", default=True,
                        help="Use multi-scale training")
    parser.add_argument("--no_multiscale", action="store_true", default=False,
                        help="Disable multi-scale training")
    parser.add_argument("--color_jitter", action="store_true", default=False,
                        help="Add PhotoMetricDistortion (brightness/contrast/saturation/hue)")
    parser.add_argument("--diagonal_flip", action="store_true", default=False,
                        help="Add diagonal flip for full D4 symmetry coverage")
    parser.add_argument("--resize_ratio_min", type=float, default=0.5,
                        help="Min scale ratio for RandomResize (default 0.5)")
    parser.add_argument("--resize_ratio_max", type=float, default=2.0,
                        help="Max scale ratio for RandomResize (default 2.0)")
    parser.add_argument("--random_rotate", action="store_true", default=False,
                        help="RandomRotate: arbitrary-angle rotation (±rotate_max_angle°), "
                             "handles bboxes and masks. Requires mmdet RandomRotate.")
    parser.add_argument("--rotate_max_angle", type=float, default=30.0,
                        help="Max rotation angle in degrees for --random_rotate (default 30)")
    parser.add_argument("--gridmask", action="store_true", default=False,
                        help="GridMask: randomly mask rectangular grid regions for regularization")
    parser.add_argument("--gaussian_noise", action="store_true", default=False,
                        help="GaussianNoise: add pixel-level Gaussian noise (custom transform)")
    parser.add_argument("--noise_std", type=float, default=15.0,
                        help="Std of Gaussian noise in uint8 pixel range (default 15)")
    parser.add_argument("--albu", action="store_true", default=False,
                        help="Pixel-level transforms: GaussianBlur + CLAHE (cv2-based, no extra deps)")

    # Architecture
    parser.add_argument("--bbox_loss", type=str, default="smoothl1",
                        choices=["smoothl1", "giou"],
                        help="Bbox regression loss for Cascade bbox heads (default: smoothl1)")
    parser.add_argument("--mask_head_convs", type=int, default=4,
                        help="FCNMaskHead conv layers (default 4; try 6 or 8)")
    parser.add_argument("--mask_roi_size", type=int, default=14,
                        help="Mask RoIAlign output size (default 14; try 28)")

    # Misc
    parser.add_argument("--work_dir", type=str, default="../work_dirs",
                        help="Working directory for logs and checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Validation interval (epochs)")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Checkpoint save interval (epochs)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from checkpoint")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Use automatic mixed precision (off by default, can cause NaN with ConvNeXt)")
    parser.add_argument("--no_amp", action="store_true", default=False,
                        help="Disable AMP")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--wandb_project", type=str, default="VRDL_HW3",
                        help="Wandb project name (set to 'none' to disable)")

    return parser.parse_args()


def build_config(args, fold_idx):
    """Build the full MMDetection config from args."""
    from dataset import get_train_pipeline, get_val_pipeline
    from model_config import get_model_config

    pretrained = args.pretrained and not args.no_pretrained
    multiscale = args.multiscale and not args.no_multiscale
    use_amp = args.amp and not args.no_amp
    img_scale = tuple(args.img_scale)

    # Model config
    model = get_model_config(
        backbone_name=args.backbone,
        pretrained=pretrained,
        fpn_out_channels=args.fpn_channels,
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path_rate,
        bbox_loss=args.bbox_loss,
        mask_head_convs=args.mask_head_convs,
        mask_roi_size=args.mask_roi_size,
    )

    # Dataset configs
    train_pipeline = get_train_pipeline(
        img_scale=img_scale,
        multiscale_mode=multiscale,
        color_jitter=args.color_jitter,
        diagonal_flip=args.diagonal_flip,
        resize_ratio_min=args.resize_ratio_min,
        resize_ratio_max=args.resize_ratio_max,
        random_rotate=args.random_rotate,
        rotate_max_angle=args.rotate_max_angle,
        gridmask=args.gridmask,
        gaussian_noise=args.gaussian_noise,
        noise_std=args.noise_std,
        albu=args.albu,
    )
    val_pipeline = get_val_pipeline(img_scale=img_scale)

    train_ann_file = os.path.join(args.ann_dir, f"fold{fold_idx}_train.json")
    val_ann_file = os.path.join(args.ann_dir, f"fold{fold_idx}_val.json")

    train_dataloader = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        sampler=dict(type="DefaultSampler", shuffle=True),
        batch_sampler=dict(type="AspectRatioBatchSampler"),
        dataset=dict(
            type="CocoDataset",
            data_root=args.data_root,
            ann_file=train_ann_file,
            data_prefix=dict(img=""),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline,
            metainfo=dict(
                classes=("class1", "class2", "class3", "class4"),
            ),
        ),
    )

    val_dataloader = dict(
        batch_size=1,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        drop_last=False,
        sampler=dict(type="DefaultSampler", shuffle=False),
        dataset=dict(
            type="CocoDataset",
            data_root=args.data_root,
            ann_file=val_ann_file,
            data_prefix=dict(img=""),
            test_mode=True,
            pipeline=val_pipeline,
            metainfo=dict(
                classes=("class1", "class2", "class3", "class4"),
            ),
        ),
    )

    val_evaluator = dict(
        type="CocoMetric",
        ann_file=val_ann_file,
        metric=["bbox", "segm"],
        format_only=False,
    )

    # Optimizer
    total_iters = args.epochs  # We use epoch-based runner
    warmup_iters = args.warmup_epochs

    # Gradient clipping config
    clip_grad_cfg = None
    if args.grad_clip > 0:
        clip_grad_cfg = dict(max_norm=args.grad_clip, norm_type=2)

    optim_wrapper = dict(
        type="AmpOptimWrapper" if use_amp else "OptimWrapper",
        optimizer=dict(
            type="AdamW",
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        ),
        clip_grad=clip_grad_cfg,
        # Apply lower LR to backbone
        paramwise_cfg=dict(
            custom_keys={
                "backbone": dict(lr_mult=args.backbone_lr_mult),
                # LayerNorm and bias should not be decayed
                "norm": dict(decay_mult=0.0),
                "bias": dict(decay_mult=0.0),
            }
        ),
    )

    # Learning rate scheduler (guard against warmup >= epochs)
    warmup_end = min(warmup_iters, max(args.epochs - 1, 1))
    param_scheduler = [
        # Linear warmup
        dict(
            type="LinearLR",
            start_factor=0.001,
            by_epoch=True,
            begin=0,
            end=warmup_end,
        ),
        # Cosine annealing after warmup
        dict(
            type="CosineAnnealingLR",
            eta_min=args.lr * args.min_lr_ratio,
            by_epoch=True,
            begin=warmup_end,
            end=args.epochs,
        ),
    ]

    # Runner config
    train_cfg_runner = dict(type="EpochBasedTrainLoop", max_epochs=args.epochs, val_interval=args.val_interval)
    val_cfg = dict(type="ValLoop")

    # Hooks
    default_hooks = dict(
        timer=dict(type="IterTimerHook"),
        logger=dict(type="LoggerHook", interval=50),
        param_scheduler=dict(type="ParamSchedulerHook"),
        checkpoint=dict(
            type="CheckpointHook",
            interval=args.save_interval,
            save_best="coco/segm_mAP_50",
            rule="greater",
            max_keep_ckpts=3,
        ),
        sampler_seed=dict(type="DistSamplerSeedHook"),
    )

    # Visualizer and logger
    vis_backends = [dict(type="LocalVisBackend")]
    wandb_enabled = args.wandb_project and args.wandb_project.lower() != "none"
    if wandb_enabled:
        vis_backends.append(
            dict(
                type="WandbVisBackend",
                init_kwargs=dict(
                    project=args.wandb_project,
                    name=f"fold{fold_idx}_{args.backbone}_ep{args.epochs}_lr{args.lr}",
                ),
            )
        )

    visualizer = dict(
        type="DetLocalVisualizer",
        vis_backends=vis_backends,
        name="visualizer",
    )

    log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)

    # Work directory
    work_dir = os.path.join(args.work_dir, f"fold{fold_idx}")

    # Environment config
    env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
        dist_cfg=dict(backend="nccl"),
    )

    # Full config
    cfg = dict(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        val_evaluator=val_evaluator,
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        train_cfg=train_cfg_runner,
        val_cfg=val_cfg,
        default_hooks=default_hooks,
        visualizer=visualizer,
        log_processor=log_processor,
        work_dir=work_dir,
        env_cfg=env_cfg,
        default_scope="mmdet",
        randomness=dict(seed=args.seed),
        load_from=None,
        resume=args.resume_from is not None,
    )

    return cfg


def train_fold(args, fold_idx):
    """Train a single fold."""
    # Import here so dataset.py registers the backbone
    import dataset  # noqa: F401  -- registers ConvNeXtV2Backbone

    from mmengine.config import Config
    from mmengine.runner import Runner

    print(f"\n{'='*60}")
    print(f"  Training Fold {fold_idx}")
    print(f"{'='*60}\n")

    cfg_dict = build_config(args, fold_idx)
    cfg = Config(cfg_dict)

    if args.resume_from:
        cfg.load_from = args.resume_from

    runner = Runner.from_cfg(cfg)

    # Print model parameter count
    model = runner.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    assert trainable_params < 200e6, (
        f"Trainable params ({trainable_params / 1e6:.2f}M) exceed 200M limit!"
    )

    runner.train()
    print(f"\nFold {fold_idx} training complete!")
    print(f"Best checkpoint saved at: {cfg_dict['work_dir']}")


def main():
    args = parse_args()

    if args.fold == "all":
        for fold_idx in range(5):
            train_fold(args, fold_idx)
    else:
        fold_idx = int(args.fold)
        train_fold(args, fold_idx)


if __name__ == "__main__":
    main()
