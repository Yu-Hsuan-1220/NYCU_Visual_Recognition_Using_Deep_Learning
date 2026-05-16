"""Inference: build pred.npz for CodaBench submission.

Loads a checkpoint (.pt) saved by train.py, runs the model over every PNG in
``--data_root/test/degraded``, and writes a dict-style .npz where keys are the
input filename (e.g. ``"0.png"``) and values are uint8 arrays of shape
``(3, H, W)``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import HW4TestDataset
from src.promptir_naf import PromptIRNAF
from src.utils import restore_image


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "1", "y")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True,
                   help="path to .pt produced by train.py")
    p.add_argument("--data_root", type=str, default="HW4/dataset")
    p.add_argument("--output", type=str, default="pred.npz")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tta", type=_str2bool, default=True)
    p.add_argument("--tile", type=int, default=0,
                   help="0 = whole image; else sliding window of this size")
    p.add_argument("--overlap", type=int, default=32)
    p.add_argument("--ckpt_kind", type=str, default="auto",
                   choices=["auto", "ema", "raw"],
                   help="auto: prefer 'ema' inside the file if present")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    # model overrides — usually loaded from ckpt['args']
    p.add_argument("--config_json", type=str, default="",
                   help="optional args.json to use for model config (overrides ckpt)")
    return p.parse_args()


def load_model(args, device):
    blob = torch.load(args.ckpt, map_location=device)
    cfg = None
    if args.config_json and os.path.isfile(args.config_json):
        with open(args.config_json) as f:
            cfg = json.load(f)
    elif isinstance(blob, dict) and "args" in blob:
        cfg = blob["args"]
    else:
        sib = os.path.join(os.path.dirname(args.ckpt), "args.json")
        if os.path.isfile(sib):
            with open(sib) as f:
                cfg = json.load(f)
    if cfg is None:
        raise RuntimeError("could not find training args (pass --config_json)")

    model = PromptIRNAF(
        dim=cfg["dim"],
        num_blocks=tuple(cfg["num_blocks"]),
        num_refinement_blocks=cfg["num_refinement_blocks"],
        dw_expand=cfg["dw_expand"],
        ffn_expand=cfg["ffn_expand"],
        drop_path=0.0,  # disable dropout for inference
        decoder=cfg["decoder_prompt"],
        prompt_dims=tuple(cfg["prompt_dims"]),
        prompt_len=cfg["prompt_len"],
        prompt_sizes=tuple(cfg["prompt_sizes"]),
    ).to(device)

    # pick state dict
    if isinstance(blob, dict):
        if args.ckpt_kind == "ema" and "ema" not in blob and "model" in blob:
            print("[warn] --ckpt_kind=ema requested but no EMA in ckpt; using model")
        if args.ckpt_kind == "raw":
            sd = blob.get("model", blob)
        elif args.ckpt_kind == "ema":
            sd = blob.get("ema", blob.get("model", blob))
        else:  # auto
            sd = blob.get("ema", blob.get("model", blob))
    else:
        sd = blob

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args, device)

    ds = HW4TestDataset(args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    out_dict = {}
    with torch.no_grad():
        for names, deg in tqdm(loader, total=len(loader)):
            deg = deg.to(device, non_blocking=True)
            pred = restore_image(model, deg, tile=args.tile,
                                 overlap=args.overlap, use_tta=args.tta)
            arr = (pred.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
            for i, name in enumerate(names):
                out_dict[name] = arr[i]  # (3, H, W) uint8

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    np.savez(args.output, **out_dict)
    print(f"saved {len(out_dict)} predictions to {args.output}")


if __name__ == "__main__":
    main()
