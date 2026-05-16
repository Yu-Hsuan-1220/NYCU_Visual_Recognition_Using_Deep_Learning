"""Standard shadow-copy EMA of model parameters."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.module = copy.deepcopy(self._unwrap(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = self._unwrap(model).state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)
