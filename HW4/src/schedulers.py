"""Linear warmup + cosine annealing LR scheduler."""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0.0,
                 warmup_start_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        out = []
        for base_lr in self.base_lrs:
            if e < self.warmup_epochs:
                if self.warmup_epochs == 0:
                    lr = base_lr
                else:
                    lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * e / self.warmup_epochs
            else:
                progress = (e - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
                lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            out.append(lr)
        return out
