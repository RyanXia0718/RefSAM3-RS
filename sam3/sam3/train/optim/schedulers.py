# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import math


class InverseSquareRootParamScheduler:
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        cooldown_steps: int,
        timescale: int,
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.timescale = timescale

    def __call__(self, step: int, where: float):
        lr = self.base_lr

        if where > 0:
            total_steps = step / where
            progress = (step - self.warmup_steps) / float(
                total_steps - self.warmup_steps
            )
            progress = max(min(progress, 1), 0)
        else:
            progress = 0
            total_steps = 1

        shift = self.timescale - self.warmup_steps
        if self.warmup_steps < step:
            lr = lr / math.sqrt((step + shift) / self.timescale)

        if self.warmup_steps:
            lr = lr * min(1.0, step / self.warmup_steps)
        if self.cooldown_steps:
            lr = lr * min(1.0, (total_steps - step) / self.cooldown_steps)

        return lr


class HoldThenExponentialDecayParamScheduler:
    """Hold the base LR briefly, then decay exponentially to a fixed ratio."""

    def __init__(
        self,
        base_lr: float,
        final_lr_ratio: float = 0.01,
        hold_fraction: float = 0.0,
    ):
        if base_lr < 0:
            raise ValueError(f"base_lr must be non-negative, got {base_lr}")
        if not 0 < final_lr_ratio <= 1:
            raise ValueError(
                "final_lr_ratio must be in (0, 1], "
                f"got {final_lr_ratio}"
            )
        if not 0 <= hold_fraction < 1:
            raise ValueError(
                "hold_fraction must be in [0, 1), "
                f"got {hold_fraction}"
            )

        self.base_lr = base_lr
        self.final_lr_ratio = final_lr_ratio
        self.hold_fraction = hold_fraction

    def __call__(self, step: int, where: float):
        del step

        where = max(0.0, min(float(where), 1.0))
        if where <= self.hold_fraction:
            return self.base_lr

        decay_progress = (where - self.hold_fraction) / (
            1.0 - self.hold_fraction
        )
        return self.base_lr * self.final_lr_ratio**decay_progress
