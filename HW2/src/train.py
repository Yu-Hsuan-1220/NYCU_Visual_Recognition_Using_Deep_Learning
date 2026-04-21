"""Main training script for Deformable DETR digit detection."""

import argparse
import os
import random
import time

import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from criterion import build_criterion
from dataset import CocoDetection, collate_fn, make_transforms
from engine import ModelEMA, adjust_learning_rate, evaluate, train_one_epoch
from model import build_model


def get_args():
    p = argparse.ArgumentParser("Deformable DETR Digit Detection Training")

    # --- Data ---
    p.add_argument("--train_img_dir", default="../dataset/train")
    p.add_argument("--train_ann", default="../dataset/train.json")
    p.add_argument("--val_img_dir", default="../dataset/valid")
    p.add_argument("--val_ann", default="../dataset/valid.json")
    p.add_argument("--num_workers", type=int, default=8)

    # --- Model ---
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--num_encoder_layers", type=int, default=6)
    p.add_argument("--num_decoder_layers", type=int, default=6)
    p.add_argument("--dim_feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_queries", type=int, default=100)
    p.add_argument("--num_feature_levels", type=int, default=4)
    p.add_argument("--enc_n_points", type=int, default=4)
    p.add_argument("--dec_n_points", type=int, default=4)
    p.add_argument("--aux_loss", action="store_true", default=True)
    p.add_argument("--no_aux_loss", dest="aux_loss", action="store_false")
    p.add_argument("--with_box_refine", action="store_true", default=False)
    p.add_argument("--pretrained_backbone", action="store_true", default=True)
    p.add_argument(
        "--no_pretrained_backbone", dest="pretrained_backbone",
        action="store_false"
    )
    p.add_argument(
        "--freeze_at", type=int, default=1,
        help="Freeze backbone up to stage (0=none, 1=layer0+1, 2=+layer2)"
    )

    # --- Loss ---
    p.add_argument("--cost_class", type=float, default=2.0)
    p.add_argument("--cost_bbox", type=float, default=5.0)
    p.add_argument("--cost_giou", type=float, default=2.0)
    p.add_argument("--loss_ce_coef", type=float, default=2.0)
    p.add_argument("--loss_bbox_coef", type=float, default=5.0)
    p.add_argument("--loss_giou_coef", type=float, default=2.0)
    p.add_argument(
        "--eos_coef", type=float, default=0.1,
        help="No-object class weight for CE loss"
    )
    p.add_argument("--focal_loss", action="store_true", default=False)
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # --- Training ---
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_backbone", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_max_norm", type=float, default=0.1)
    p.add_argument("--accumulate_steps", type=int, default=2)
    p.add_argument(
        "--lr_scheduler", choices=["step", "cosine"], default="cosine"
    )
    p.add_argument("--lr_drop_epochs", type=int, nargs="+", default=[150])
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument(
        "--lr_min_ratio", type=float, default=1e-2,
        help="Min LR ratio for cosine scheduler"
    )
    p.add_argument(
        "--amp", action="store_true", default=False,
        help="Use automatic mixed precision"
    )

    # --- EMA ---
    p.add_argument(
        "--ema_decay", type=float, default=0.0,
        help="EMA decay (0 = disabled)"
    )

    # --- Augmentation ---
    p.add_argument("--fixed_h", type=int, default=320)
    p.add_argument("--fixed_w", type=int, default=640)
    p.add_argument("--min_size", type=int, default=256)
    p.add_argument("--max_size_train", type=int, default=384)
    p.add_argument("--val_size", type=int, default=384)
    p.add_argument("--max_size", type=int, default=512)
    p.add_argument("--color_jitter", type=float, default=0.4)
    p.add_argument(
        "--rotate_max_angle", type=float, default=10,
        help="Max rotation angle for training augmentation"
    )
    p.add_argument(
        "--gaussian_blur_p", type=float, default=0.2,
        help="Probability of applying random Gaussian blur"
    )

    # --- ISO Noise Augmentation ---
    p.add_argument("--aug_iso_noise", action="store_true", default=False)
    p.add_argument("--aug_iso_noise_p", type=float, default=0.2)
    p.add_argument("--aug_iso_noise_intensity", type=float, default=0.05)

    # --- Translation Augmentation ---
    p.add_argument("--aug_translation", action="store_true", default=False)
    p.add_argument("--aug_translation_p", type=float, default=0.3)
    p.add_argument("--aug_translation_max_shift", type=float, default=0.1)
    p.add_argument(
        "--aug_translation_min_area_ratio", type=float, default=0.25,
        help="Drop bbox if remaining area < ratio * original"
    )

    # --- Expand (Zoom-out) Augmentation ---
    p.add_argument("--aug_expand", action="store_true", default=False)
    p.add_argument("--aug_expand_p", type=float, default=0.3)
    p.add_argument(
        "--aug_expand_max_ratio", type=float, default=0.8,
        help="Max expand ratio"
    )

    # --- Misc ---
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="", help="Path to checkpoint to resume")
    p.add_argument(
        "--load_weights", default="",
        help="Path to trained model weights only for fine-tuning"
    )
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--eval_freq", type=int, default=1)
    p.add_argument("--print_freq", type=int, default=50)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", default="DeformDETR-Digit-Detection")
    p.add_argument("--wandb_run_name", default=None)

    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ---------- W&B ----------
    if args.wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )
    elif args.wandb:
        print("Warning: wandb not installed, skipping logging.")

    # ---------- Datasets ----------
    train_ds = CocoDetection(
        args.train_img_dir, args.train_ann, make_transforms("train", args),
    )
    val_ds = CocoDetection(
        args.val_img_dir, args.val_ann, make_transforms("val", args),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True,
    )
    coco_val = COCO(args.val_ann)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ---------- Model & Criterion ----------
    model = build_model(args).to(device)
    criterion = build_criterion(args).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ---------- Optimizer ----------
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters()
                       if "backbone" not in n and p.requires_grad],
            "lr": args.lr, "initial_lr": args.lr
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone, "initial_lr": args.lr_backbone
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay,
    )

    # ---------- AMP & EMA ----------
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # ---------- Load pretrained weights (model only) ----------
    if args.load_weights:
        ckpt = torch.load(
            args.load_weights, map_location=device, weights_only=False
        )
        state = ckpt["model"] if "model" in ckpt else ckpt
        # Only load matching keys
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"Warning: missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"Warning: unexpected keys: {unexpected[:5]}...")
        print(f"Loaded model weights from {args.load_weights}")

    # ---------- Resume ----------
    start_epoch = 0
    best_map = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_map" in ckpt:
            best_map = ckpt["best_map"]
        if ema is not None and "ema" in ckpt:
            ema.module.load_state_dict(ckpt["ema"])
        print(f"Resumed from epoch {start_epoch}, best_map={best_map:.4f}")

    # ---------- Eval only ----------
    if args.eval_only:
        eval_model = ema.module if ema is not None else model
        stats = evaluate(
            eval_model, val_loader, device, coco_val,
            args.num_classes, args.focal_loss,
        )
        print(f"Eval: mAP={stats['mAP']:.4f} mAP50={stats['mAP_50']:.4f}")
        return

    # ---------- Training loop ----------
    print(f"\nStarting training for {args.epochs} epochs ...")
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        t0 = time.time()

        avg_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, args,
            scaler=scaler, ema=ema,
        )

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        if len(optimizer.param_groups) > 1:
            lr_bb = optimizer.param_groups[1]["lr"]
        else:
            lr_bb = lr_now
        print(
            f"Epoch [{epoch+1}/{args.epochs}]  loss={avg_loss:.4f}  "
            f"lr={lr_now:.2e}  time={elapsed:.0f}s"
        )

        # Log train metrics
        if args.wandb and wandb is not None and wandb.run is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_loss,
                "train/lr": lr_now,
                "train/lr_backbone": lr_bb,
                "train/epoch_time_s": elapsed,
            }, step=epoch + 1)

        # Evaluate
        do_eval = (
            (epoch + 1) % args.eval_freq == 0
            or epoch + 1 == args.epochs
        )
        if do_eval:
            eval_model = ema.module if ema is not None else model
            stats = evaluate(
                eval_model, val_loader, device, coco_val,
                args.num_classes, args.focal_loss,
            )
            current_map = stats["mAP"]
            print(
                f"  >> mAP={stats['mAP']:.4f}  "
                f"mAP50={stats['mAP_50']:.4f}  "
                f"mAP75={stats['mAP_75']:.4f}"
            )

            # Log val metrics
            if args.wandb and wandb is not None and wandb.run is not None:
                wandb.log({
                    "val/mAP": stats["mAP"],
                    "val/mAP_50": stats["mAP_50"],
                    "val/mAP_75": stats["mAP_75"],
                }, step=epoch + 1)

            is_best = current_map > best_map
            if is_best:
                best_map = current_map

            # Save checkpoint
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_map": best_map,
                "args": vars(args),
            }
            if ema is not None:
                ckpt["ema"] = ema.module.state_dict()

            torch.save(ckpt, os.path.join(args.output_dir, "last.pth"))
            if is_best:
                torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))
                print(f"  >> New best mAP: {best_map:.4f}")

    print(f"\nTraining complete. Best mAP: {best_map:.4f}")
    if args.wandb and wandb is not None and wandb.run is not None:
        wandb.log({"best_mAP": best_map})
        wandb.finish()


if __name__ == "__main__":
    main()
