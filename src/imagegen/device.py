"""Device selection helpers."""

from __future__ import annotations

import torch


def select_device(override: str | None = None) -> str:
    """Select the CUDA device to run on.

    Prefers ``override`` (e.g. ``"cuda:1"``). Otherwise defaults to
    ``cuda:1`` when two or more GPUs are present, else ``cuda:0``.

    Model chỉ ~10.9 GiB nên nằm trọn trên MỘT card — không còn chiến lược
    chia component qua nhiều GPU như bản Qwen trước đây (text_encoder một
    card, transformer + vae card kia). Heuristic "≥2 GPU → cuda:1" giữ lại
    chỉ để chừa cuda:0 cho tiến trình khác trên bàn dev.
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
