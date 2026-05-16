"""Train PromptIR-NAF on HW4 rain+snow restoration.

All hyperparameters and tricks are exposed via argparse so any switch can be
flipped from the command line without touching code.

Usage (defaults reproduce the recommended bundle):
    python train.py --data_root HW4/dataset --ckpt_dir HW4/ckpt/run1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader

# allow running from project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import HW4TrainDataset
from src.ema import ModelEMA
from src.losses import CompositeLoss, build_primary
from src.promptir_naf import PromptIRNAF
from src.schedulers import LinearWarmupCosineAnnealingLR
from src.utils import psnr_torch, seed_everything


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1", "y"):
        return True
    if v.lower() in ("no", "false", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("boolean expected")


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_root", type=str, default="HW4/dataset")
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # augmentation
    p.add_argument("--aug_flip", type=_str2bool, default=True)
    p.add_argument("--aug_rot90", type=_str2bool, default=True)
    p.add_argument("--aug_rgb_shuffle", type=_str2bool, default=False)
    p.add_argument("--aug_mixup_p", type=float, default=0.0)

    # model
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--num_blocks", type=int, nargs=4, default=[4, 6, 6, 8])
    p.add_argument("--num_refinement_blocks", type=int, default=4)
    p.add_argument("--dw_expand", type=int, default=2)
    p.add_argument("--ffn_expand", type=int, default=2)
    p.add_argument("--drop_path", type=float, default=0.0)
    p.add_argument("--decoder_prompt", type=_str2bool, default=True)
    p.add_argument("--prompt_dims", type=int, nargs=3, default=[64, 128, 320])
    p.add_argument("--prompt_len", type=int, default=5)
    p.add_argument("--prompt_sizes", type=int, nargs=3, default=[64, 32, 16])

    # optimizer
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", type=_str2bool, default=True)
    p.add_argument("--warmup_epochs", type=int, default=15)

    # loss
    p.add_argument("--loss", type=str, default="charbonnier",
                   choices=["l1", "l2", "charbonnier", "psnr"])
    p.add_argument("--charbonnier_eps", type=float, default=1e-3)
    p.add_argument("--psnr_toY", type=_str2bool, default=False)
    p.add_argument("--ssim_weight", type=float, default=0.0)
    p.add_argument("--fft_weight", type=float, default=0.0)

    # ema
    p.add_argument("--use_ema", type=_str2bool, default=True)
    p.add_argument("--ema_decay", type=float, default=0.999)

    # checkpoint
    p.add_argument("--ckpt_dir", type=str, default="HW4/ckpt/run")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--val_every", type=int, default=1)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--device", type=str, default="cuda")

    # wandb
    p.add_argument("--wandb_project", type=str, default="VRDL_4",
                   help="empty disables wandb")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--wandb_tags", type=str, nargs="*", default=[])
    p.add_argument("--wandb_notes", type=str, default="")
    p.add_argument("--wandb_log_images", type=_str2bool, default=True,
                   help="log a few val triplets (degraded | pred | clean)")
    p.add_argument("--wandb_image_count", type=int, default=1,
                   help="how many val samples to visualize")
    p.add_argument("--wandb_image_every", type=int, default=10,
                   help="log images every N epochs")
    p.add_argument("--wandb_watch_freq", type=int, default=0,
                   help="wandb.watch log_freq; 0 = disabled")
    p.add_argument("--wandb_save_code", type=_str2bool, default=True)

    return p.parse_args()


def model_kwargs(args):
    return dict(
        inp_channels=3,
        out_channels=3,
        dim=args.dim,
        num_blocks=tuple(args.num_blocks),
        num_refinement_blocks=args.num_refinement_blocks,
        dw_expand=args.dw_expand,
        ffn_expand=args.ffn_expand,
        drop_path=args.drop_path,
        decoder=args.decoder_prompt,
        prompt_dims=tuple(args.prompt_dims),
        prompt_len=args.prompt_len,
        prompt_sizes=tuple(args.prompt_sizes),
    )


@torch.no_grad()
def _wandb_image_triplets(wandb, model, samples):
    """Build a list of wandb.Image showing (degraded | pred | clean) triplets.

    ``samples`` is a list of (deg_tensor[1,3,H,W], clean_tensor[1,3,H,W], de_id).
    """
    was_training = model.training
    model.eval()
    names = {0: "rain", 1: "snow"}
    images = []
    for deg, clean, de_id in samples:
        pred = model(deg).clamp(0, 1)
        triplet = torch.cat([deg[0], pred[0], clean[0]], dim=-1)  # (3, H, 3W)
        arr = (triplet.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(
            np.uint8
        )
        images.append(wandb.Image(arr, caption=f"{names.get(de_id, de_id)}: degraded | pred | clean"))
    if was_training:
        model.train()
    return images


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr_by_type = defaultdict(list)
    for deg, clean, de_id in loader:
        deg = deg.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        pred = model(deg).clamp(0, 1)
        psnr = psnr_torch(pred, clean)
        for p_, d_ in zip(psnr.tolist(), de_id.tolist()):
            psnr_by_type[int(d_)].append(p_)
    model.train()
    out = {
        "psnr_all": float(np.mean(sum(psnr_by_type.values(), []))) if psnr_by_type else 0.0,
        "psnr_rain": float(np.mean(psnr_by_type[0])) if psnr_by_type[0] else 0.0,
        "psnr_snow": float(np.mean(psnr_by_type[1])) if psnr_by_type[1] else 0.0,
    }
    return out


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)

    train_ds = HW4TrainDataset(
        args.data_root, patch_size=args.patch_size, is_train=True,
        val_ratio=args.val_ratio, seed=args.seed,
        aug_flip=args.aug_flip, aug_rot90=args.aug_rot90,
        aug_rgb_shuffle=args.aug_rgb_shuffle, aug_mixup_p=args.aug_mixup_p,
    )
    val_ds = HW4TrainDataset(
        args.data_root, patch_size=0, is_train=False,
        val_ratio=args.val_ratio, seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True,
        drop_last=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, pin_memory=True,
        num_workers=min(args.num_workers, 2),
    )
    print(f"train pairs={len(train_ds)} val pairs={len(val_ds)}")

    model = PromptIRNAF(**model_kwargs(args)).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_param/1e6:.2f}M")

    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd,
        betas=(args.beta1, args.beta2),
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs,
    )

    primary = build_primary(args.loss, charbonnier_eps=args.charbonnier_eps,
                            psnr_toY=args.psnr_toY).to(device)
    loss_fn = CompositeLoss(primary, ssim_weight=args.ssim_weight,
                            fft_weight=args.fft_weight).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # wandb (optional)
    use_wandb = bool(args.wandb_project)
    wandb = None
    viz_samples = []
    if use_wandb:
        import wandb as _wandb  # local alias so we can keep `wandb = None` semantics
        wandb = _wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.run_name or os.path.basename(args.ckpt_dir),
            tags=args.wandb_tags or None,
            notes=args.wandb_notes or None,
            config=vars(args),
            dir=args.ckpt_dir,
            save_code=args.wandb_save_code,
        )
        wandb.run.summary["params_M"] = n_param / 1e6
        wandb.run.summary["train_pairs"] = len(train_ds)
        wandb.run.summary["val_pairs"] = len(val_ds)
        if args.wandb_watch_freq > 0:
            wandb.watch(model, log="all", log_freq=args.wandb_watch_freq)
        if args.wandb_log_images and len(val_ds) > 0:
            # cache fixed val samples for deterministic image logging
            for i, (deg, clean, de_id) in enumerate(val_loader):
                if i >= args.wandb_image_count:
                    break
                viz_samples.append((
                    deg.to(device, non_blocking=True),
                    clean.to(device, non_blocking=True),
                    int(de_id[0]),
                ))

    start_epoch = 0
    best_psnr = -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "ema" in ckpt and ema is not None:
            ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", -1.0)
        print(f"resumed from {args.resume} @ epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running, n = 0.0, 0
        grad_norm_sum, grad_norm_n = 0.0, 0
        for deg, clean, _ in train_loader:
            deg = deg.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                pred = model(deg)
                loss = loss_fn(pred, clean)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # clip_grad_norm_ also returns the pre-clip total norm; capture it
            # whether or not we actually clip (use a huge max_norm if disabled).
            max_norm = args.grad_clip if args.grad_clip > 0 else 1e9
            gn = nn_utils.clip_grad_norm_(model.parameters(), max_norm)
            grad_norm_sum += float(gn)
            grad_norm_n += 1
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            running += loss.item() * deg.size(0)
            n += deg.size(0)
        scheduler.step()
        dt = time.time() - t0
        train_loss = running / max(1, n)
        lr = optimizer.param_groups[0]["lr"]
        grad_norm_avg = grad_norm_sum / max(1, grad_norm_n)
        mem_gb = (torch.cuda.max_memory_allocated() / 1024 ** 3
                  if torch.cuda.is_available() else 0.0)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        print(
            f"[epoch {epoch:3d}] loss={train_loss:.4f} lr={lr:.2e} "
            f"gnorm={grad_norm_avg:.2f} time={dt:.1f}s mem={mem_gb:.2f}GB",
            flush=True,
        )
        log = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/lr": lr,
            "train/grad_norm": grad_norm_avg,
            "train/time_s": dt,
            "train/gpu_mem_gb": mem_gb,
            "train/samples_per_s": n / max(dt, 1e-6),
        }

        improved = False
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            eval_model = ema.module if ema is not None else model
            metrics = validate(eval_model, val_loader, device)
            print(
                f"           val: PSNR all={metrics['psnr_all']:.3f} "
                f"rain={metrics['psnr_rain']:.3f} snow={metrics['psnr_snow']:.3f}",
                flush=True,
            )
            log.update({
                "val/psnr_all": metrics["psnr_all"],
                "val/psnr_rain": metrics["psnr_rain"],
                "val/psnr_snow": metrics["psnr_snow"],
            })
            if metrics["psnr_all"] > best_psnr:
                best_psnr = metrics["psnr_all"]
                improved = True
                save_ckpt(args, model, ema, optimizer, scheduler, scaler,
                          epoch, best_psnr, name="best.pt")
                if ema is not None:
                    torch.save({"model": ema.state_dict(), "args": vars(args),
                                "epoch": epoch, "psnr": best_psnr},
                               os.path.join(args.ckpt_dir, "ema_best.pt"))
        log["val/best_psnr"] = best_psnr

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_ckpt(args, model, ema, optimizer, scheduler, scaler,
                      epoch, best_psnr, name="last.pt")

        if use_wandb:
            if (
                args.wandb_log_images
                and viz_samples
                and (epoch % args.wandb_image_every == 0
                     or epoch == args.epochs - 1
                     or improved)
            ):
                eval_model = ema.module if ema is not None else model
                log["val/samples"] = _wandb_image_triplets(
                    wandb, eval_model, viz_samples,
                )
            wandb.log(log, step=epoch)
            wandb.run.summary["best_psnr"] = best_psnr

    print(f"done. best val PSNR = {best_psnr:.3f}")
    if use_wandb:
        wandb.run.summary["final_best_psnr"] = best_psnr
        wandb.finish()


def save_ckpt(args, model, ema, opt, sched, scaler, epoch, best_psnr, name):
    path = os.path.join(args.ckpt_dir, name)
    sd = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_psnr": best_psnr,
        "args": vars(args),
    }
    if ema is not None:
        sd["ema"] = ema.state_dict()
    torch.save(sd, path)


if __name__ == "__main__":
    main()
