"""Training utilities of the PyTorch networks of the standard schedule: seeding, an exponential moving average of
the weights and a learning rate with linear warm-up and cosine decay."""
import copy
import math
import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class ModelEMA:
    """Exponential moving average of the weights (decay 0.9998, warmed up over the first steps); used for validation,
    model selection and testing."""

    def __init__(self, model, decay=0.9998):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay); self.updates = 0
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.is_floating_point():
                v.mul_(d).add_(src[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(src[k])


class WarmupCosineSchedule:
    """Learning rate rising linearly over the warm-up steps, then decaying along a cosine to min_lr."""

    def __init__(self, optimizer, total_steps, warmup_steps, base_lr, min_lr):
        self.optimizer, self.total, self.warmup = optimizer, max(1, int(total_steps)), max(0, int(warmup_steps))
        self.base, self.min, self.step_index = float(base_lr), float(min_lr), 0
        self._apply(self.lr_at(0))

    def lr_at(self, step):
        if self.warmup > 0 and step < self.warmup:
            return self.base * float(step + 1) / float(self.warmup)
        progress = min(1.0, max(0.0, (step - self.warmup) / max(1, self.total - self.warmup)))
        return self.min + (self.base - self.min) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def step(self):
        self.step_index += 1
        lr = self.lr_at(self.step_index); self._apply(lr)
        return lr
