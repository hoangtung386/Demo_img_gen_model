"""Device selection helpers."""

from __future__ import annotations

import torch


def select_device(override: str | None = None) -> str:
    """Select the CUDA device to run on.

    Prefers ``override`` (e.g. ``"cuda:1"``). Otherwise defaults to
    ``cuda:1`` when two or more GPUs are present, else ``cuda:0``.
    """
    if override:
        device = override
        try:
            torch.cuda.set_device(int(device.split(":")[-1]))
        except (ValueError, IndexError):
            pass
        return device

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count >= 2:
        torch.cuda.set_device(1)
        return "cuda:1"
    return "cuda:0"
