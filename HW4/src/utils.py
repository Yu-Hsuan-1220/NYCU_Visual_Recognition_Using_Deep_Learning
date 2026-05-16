"""Misc helpers: seeding, PSNR, TTA, sliding-window inference."""
from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr_torch(pred, target, max_val=1.0):
    """PSNR of clamped float tensors in [0, max_val]."""
    pred = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
    psnr = 10.0 * torch.log10((max_val ** 2) / (mse + 1e-12))
    return psnr


def pad_to_multiple(x, m=16):
    h, w = x.shape[-2:]
    nh = (h + m - 1) // m * m
    nw = (w + m - 1) // m * m
    if nh == h and nw == w:
        return x, (0, 0, 0, 0)
    pad = (0, nw - w, 0, nh - h)
    return F.pad(x, pad, mode="reflect"), pad


def crop_to_original(x, pad):
    _, _, _, h_orig = (None, None, None, None)
    _, right, _, bottom = pad
    if right == 0 and bottom == 0:
        return x
    return x[..., : x.shape[-2] - bottom, : x.shape[-1] - right]


_TTA_OPS = [
    "i",        # identity
    "h",        # horizontal flip
    "v",        # vertical flip
    "hv",       # both flips
    "r",        # rot90
    "rh",       # rot90 + h
    "rv",       # rot90 + v
    "rhv",      # rot90 + h + v
]


def _tta_forward_one(x, op):
    if "r" in op:
        x = torch.rot90(x, 1, dims=(-2, -1))
    if "h" in op:
        x = torch.flip(x, dims=(-1,))
    if "v" in op:
        x = torch.flip(x, dims=(-2,))
    return x


def _tta_inverse_one(y, op):
    # apply inverse in reverse order: v -> h -> r
    if "v" in op:
        y = torch.flip(y, dims=(-2,))
    if "h" in op:
        y = torch.flip(y, dims=(-1,))
    if "r" in op:
        y = torch.rot90(y, -1, dims=(-2, -1))
    return y


@torch.no_grad()
def tta_forward(model, x):
    """8-way TTA average. Assumes ``model`` returns same-shape tensor."""
    outs = []
    for op in _TTA_OPS:
        xi = _tta_forward_one(x, op)
        yi = model(xi)
        yi = _tta_inverse_one(yi, op)
        outs.append(yi)
    return torch.stack(outs, 0).mean(0)


def _gauss_window(h, w, device):
    yy = torch.linspace(-1, 1, h, device=device).view(h, 1)
    xx = torch.linspace(-1, 1, w, device=device).view(1, w)
    w2 = torch.exp(-(xx ** 2 + yy ** 2) * 2.0)
    return w2 / w2.max()


@torch.no_grad()
def sliding_window_forward(model, x, tile, overlap, use_tta=False):
    """Sliding-window inference with Gaussian blending.

    ``x``: (B, C, H, W). Returns same shape.
    """
    B, C, H, W = x.shape
    if tile <= 0 or (H <= tile and W <= tile):
        return tta_forward(model, x) if use_tta else model(x)

    stride = tile - overlap
    out = torch.zeros_like(x)
    weight = torch.zeros((1, 1, H, W), device=x.device, dtype=x.dtype)
    win = _gauss_window(tile, tile, x.device)[None, None]

    ys = list(range(0, max(1, H - tile + 1), stride))
    if ys[-1] != H - tile:
        ys.append(H - tile)
    xs = list(range(0, max(1, W - tile + 1), stride))
    if xs[-1] != W - tile:
        xs.append(W - tile)

    for y in ys:
        for xx in xs:
            patch = x[..., y:y + tile, xx:xx + tile]
            pred = tta_forward(model, patch) if use_tta else model(patch)
            out[..., y:y + tile, xx:xx + tile] += pred * win
            weight[..., y:y + tile, xx:xx + tile] += win
    return out / weight.clamp(min=1e-8)


def restore_image(model, x, *, tile=0, overlap=32, use_tta=False, pad_multiple=16):
    """Top-level inference for a single batch tensor in [0, 1].

    Pads to multiple of ``pad_multiple``, optionally applies sliding window
    and / or TTA, crops back. Returns clamped output in [0, 1].
    """
    x_p, pad = pad_to_multiple(x, pad_multiple)
    if tile > 0:
        y = sliding_window_forward(model, x_p, tile, overlap, use_tta=use_tta)
    else:
        y = tta_forward(model, x_p) if use_tta else model(x_p)
    y = crop_to_original(y, pad)
    return y.clamp(0, 1)
