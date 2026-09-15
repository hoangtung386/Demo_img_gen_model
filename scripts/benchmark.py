"""Đo tốc độ sinh ảnh thật trên máy đang chạy.

Dùng để xác nhận chiến lược offload đang chọn là đúng, và để so trước/sau
mỗi lần chỉnh cấu hình.

    python scripts/benchmark.py                # 5 lượt ở 1024px
    python scripts/benchmark.py --runs 10
    python scripts/benchmark.py --area 832     # cạnh 832px

Trong container:

    docker compose exec qwen-lightning python scripts/benchmark.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from qwen_lightning.config import load_settings  # noqa: E402
from qwen_lightning.device import select_device  # noqa: E402
from qwen_lightning.inference import generate  # noqa: E402
from qwen_lightning.logging_setup import configure_logging  # noqa: E402
from qwen_lightning.models.loader import load_pipeline  # noqa: E402
from qwen_lightning.tuning import apply_gpu_tuning, warmup  # noqa: E402

PROMPT = "make the lighting warmer and more cinematic"


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--area",
        type=int,
        default=1024,
        help="Cạnh ảnh vuông tương đương; diện tích = cạnh^2.",
    )
    args = parser.parse_args()

    configure_logging()
    settings = load_settings()
    apply_gpu_tuning()
    device = select_device(settings.device_override)

    load_start = time.time()
    pipeline = load_pipeline(settings, device)
    pipeline.set_progress_bar_config(disable=True)
    load_s = time.time() - load_start

    area = args.area * args.area
    warm_s = warmup(pipeline, settings.num_steps, area)

    # Mỗi lượt dùng một ảnh khác nhau để KHÔNG ăn cache prompt-embeds —
    # nếu không thì từ lượt thứ hai trở đi ta chỉ đang đo tốc độ tra dict.
    timings: list[float] = []
    for run in range(args.runs):
        shade = 40 + run * 7
        image = Image.new("RGB", (768, 768), (shade, shade, shade))
        started = time.time()
        _, info = generate(
            pipeline, [image], PROMPT, 1.0, run, settings.num_steps,
            output_area=area,
        )
        elapsed = time.time() - started
        timings.append(elapsed)
        print(f"  lượt {run + 1}/{args.runs}: {elapsed:.2f}s | {info}")

    print()
    print("=" * 62)
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"Offload          : {settings.offload}")
    print(f"Precision/steps  : {settings.precision} / {settings.num_steps}")
    print(f"Độ phân giải     : ~{args.area}px")
    print("-" * 62)
    print(f"Load model       : {load_s:.1f}s")
    print(f"Warm-up          : {warm_s:.1f}s")
    print(f"Trung vị         : {statistics.median(timings):.2f}s/ảnh")
    print(f"Nhanh nhất       : {min(timings):.2f}s")
    print(f"Chậm nhất        : {max(timings):.2f}s")
    print(f"VRAM đỉnh        : {_vram_gb():.1f} GB")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
