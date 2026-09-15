"""Tests for the scheduler configuration."""
from __future__ import annotations

import math

from qwen_lightning.models.loader import build_scheduler


def test_scheduler_distillation_shift():
    scheduler = build_scheduler()
    assert scheduler.config.base_shift == math.log(3)
    assert scheduler.config.max_shift == math.log(3)
    assert scheduler.config.num_train_timesteps == 1000
    assert scheduler.config.use_dynamic_shifting is True
    assert scheduler.config.time_shift_type == "exponential"
