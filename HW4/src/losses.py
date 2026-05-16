"""Loss zoo for HW4.

Primary losses: l1, l2, charbonnier, psnr (NAFNet-style).
Auxiliary terms (added with their own weights): ssim, fft.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class PSNRLoss(nn.Module):
    """NAFNet-style: scales mse so the model directly optimizes -PSNR.

    Operates on RGB in [0, 1]. The constant scale 10 / ln(10) is taken
    from NAFNet's reference implementation.
    """

    def __init__(self, toY=False):
        super().__init__()
        self.scale = 10 / math.log(10)
        self.toY = toY  # convert to Y channel (luminance) before computing

    def forward(self, pred, target):
        if self.toY:
            coef = torch.tensor([65.481, 128.553, 24.966], device=pred.device).reshape(1, 3, 1, 1) / 255.0
            pred = (pred * coef).sum(dim=1, keepdim=True) + 16.0 / 255.0
            target = (target * coef).sum(dim=1, keepdim=True) + 16.0 / 255.0
        mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
        return -self.scale * torch.log(mse + 1e-8).mean()


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w_1d = g.unsqueeze(1)
    w_2d = (w_1d @ w_1d.t()).unsqueeze(0).unsqueeze(0)
    return w_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """1 - SSIM over RGB channels; differentiable, simple Gaussian window.

    Always evaluated in fp32 (even under AMP) — the variance term
    ``E[x²] - E[x]²`` is numerically delicate and fp16 makes the Gaussian
    window denormal, which produces NaN in the SSIM ratio.
    """

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self._cache = {}

    def _win(self, c, device):
        key = (c, device)
        if key not in self._cache:
            self._cache[key] = _gaussian_window(
                self.window_size, self.sigma, c, device, torch.float32,
            )
        return self._cache[key]

    def forward(self, pred, target):
        with torch.cuda.amp.autocast(enabled=False):
            pred = pred.float()
            target = target.float()
            c = pred.size(1)
            win = self._win(c, pred.device)
            pad = self.window_size // 2
            mu1 = F.conv2d(pred, win, padding=pad, groups=c)
            mu2 = F.conv2d(target, win, padding=pad, groups=c)
            mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
            sigma1_sq = (
                F.conv2d(pred * pred, win, padding=pad, groups=c) - mu1_sq
            ).clamp(min=0)
            sigma2_sq = (
                F.conv2d(target * target, win, padding=pad, groups=c) - mu2_sq
            ).clamp(min=0)
            sigma12 = F.conv2d(pred * target, win, padding=pad, groups=c) - mu12
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            num = (2 * mu12 + c1) * (2 * sigma12 + c2)
            den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
            ssim_map = num / den.clamp(min=1e-12)
            return 1.0 - ssim_map.mean()


class FFTLoss(nn.Module):
    """L1 in frequency domain.

    ``torch.fft.rfft2`` is not autocast-safe for fp16 inputs; force fp32.
    """

    def forward(self, pred, target):
        with torch.cuda.amp.autocast(enabled=False):
            p = torch.fft.rfft2(pred.float(), norm="ortho")
            t = torch.fft.rfft2(target.float(), norm="ortho")
            return torch.mean(torch.abs(p - t))


def build_primary(name, **kw):
    name = name.lower()
    if name == "l1":
        return nn.L1Loss()
    if name == "l2" or name == "mse":
        return nn.MSELoss()
    if name == "charbonnier":
        return CharbonnierLoss(eps=kw.get("charbonnier_eps", 1e-3))
    if name == "psnr":
        return PSNRLoss(toY=kw.get("psnr_toY", False))
    raise ValueError(f"unknown loss: {name}")


class CompositeLoss(nn.Module):
    """primary + ssim_weight * (1-SSIM) + fft_weight * FFTLoss."""

    def __init__(self, primary, ssim_weight=0.0, fft_weight=0.0):
        super().__init__()
        self.primary = primary
        self.ssim_weight = ssim_weight
        self.fft_weight = fft_weight
        self.ssim = SSIMLoss() if ssim_weight > 0 else None
        self.fft = FFTLoss() if fft_weight > 0 else None

    def forward(self, pred, target):
        loss = self.primary(pred, target)
        if self.ssim is not None:
            loss = loss + self.ssim_weight * self.ssim(pred, target)
        if self.fft is not None:
            loss = loss + self.fft_weight * self.fft(pred, target)
        return loss
