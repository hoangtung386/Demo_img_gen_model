"""Đo tốc độ sinh ảnh thật trên máy đang chạy.

Dùng để xác nhận cấu hình đang chọn là đúng, và để so trước/sau mỗi lần
chỉnh. Con số ở đây là thứ phải điền vào ``rabbitmq.avg_inference_seconds``
của ``config_setup/base.yaml`` — để sai giá trị đó là cách nhanh nhất làm
backlog queue phình ra.

    python scripts/benchmark.py                 # 3 lượt ở 2048x2048
    python scripts/benchmark.py --runs 5
    python scripts/benchmark.py --width 2560 --height 1440
    python scripts/benchmark.py --compare     # so sánh các đường tối ưu
    python scripts/benchmark.py --profile     # thời gian đi vào kernel nào

Trong container:

    docker compose exec imagegen python scripts/benchmark.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from imagegen.config import load_settings  # noqa: E402
from imagegen.device import select_device  # noqa: E402
from imagegen.hidream import build_recipe, generate, load_model  # noqa: E402
from imagegen.logging_setup import configure_logging  # noqa: E402
from imagegen.tuning import apply_gpu_tuning, warmup  # noqa: E402

PROMPT = (
    "A weathered brass compass resting on a folded nautical chart, warm "
    "afternoon light from a window on the left, shallow depth of field, "
    "fine scratches visible on the metal, muted blue and ochre palette."
)


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Mặc định 3 chứ không phải 5: mỗi lượt tốn hàng chục giây đến vài phút,
    # 5 lượt là chờ mười phút cho một phép đo.
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--width", type=int, default=0, help="0 → theo config")
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="0 → theo config")
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Chạy lần lượt mọi tổ hợp attention × trọng số và in bảng so "
            "sánh. Nạp lại model cho mỗi biến thể trọng số nên TỐN THỜI "
            "GIAN — dùng khi cần chứng minh một tối ưu có thật."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "Chạy MỘT lượt ngắn dưới torch.profiler và in 15 kernel CUDA tốn "
            "nhiều thời gian nhất. Dùng khi tổng thời gian không khớp với "
            "những gì docs/PERFORMANCE.md giải thích được — bảng kernel nói "
            "thẳng thời gian đi đâu, không phải đoán."
        ),
    )
    return parser.parse_args()


def _run_profile(args, settings, device, width, height) -> int:
    """In bảng kernel CUDA của một lượt sinh ảnh ngắn.

    Dùng ít bước (mặc định 4): hồ sơ kernel là TỈ LỆ, không phải tổng — chạy
    50 bước chỉ làm trace phình lên chứ không cho thêm thông tin nào.
    """
    from torch.profiler import ProfilerActivity, profile

    model, processor = load_model(settings, device)
    steps = args.steps or 4
    recipe = build_recipe(settings, num_steps=steps)
    warmup(model, processor, settings)  # đừng để chi phí lần đầu vào hồ sơ

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        generate(
            model, processor, [], PROMPT, recipe,
            seed=0, width=width, height=height,
        )

    print()
    print("=" * 78)
    print(
        f"Hồ sơ kernel — {steps} bước, {width}x{height}, "
        f"attention={settings.attention_mode}, "
        f"dequantize={settings.dequantize}"
    )
    print("=" * 78)
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=15
        )
    )
    print(
        "Đọc bảng: kernel `*gemm*`/`*cutlass*` là tính toán thật. Kernel "
        "`*elementwise*`/`*copy*`/`*cat*` chiếm nhiều = đang trả tiền cho "
        "giải nén trọng số hoặc cho việc dựng/expand mask."
    )
    return 0


def _time_runs(model, processor, recipe, width, height, runs):
    """Chạy `runs` lượt, trả list thời gian. In từng lượt để thấy tiến độ."""
    timings = []
    for run in range(runs):
        started = time.time()
        _, info = generate(
            model, processor, [], PROMPT, recipe,
            seed=run, width=width, height=height,
        )
        elapsed = time.time() - started
        timings.append(elapsed)
        print(f"  lượt {run + 1}/{runs}: {elapsed:.2f}s | {info}")
    return timings


def _run_compare(args, settings, device, width, height) -> int:
    """So sánh trực tiếp các đường tối ưu trên chính GPU này.

    Không có bảng nào thay được phép đo tại chỗ: tỉ lệ giữa SDNQ và bf16
    phụ thuộc băng thông bộ nhớ của card, còn tỉ lệ giữa SDPA và mask 4D
    phụ thuộc backend mà torch chọn được cho shape đó.
    """
    import dataclasses

    results = []
    for dequantize in ("off", "bf16"):
        cfg = dataclasses.replace(settings, dequantize=dequantize)
        try:
            model, processor = load_model(cfg, device)
        except torch.cuda.OutOfMemoryError:
            print(f"  [dequantize={dequantize}] OOM — bỏ qua")
            torch.cuda.empty_cache()
            continue

        for attention in ("mask", "sdpa"):
            run_cfg = dataclasses.replace(cfg, attention_mode=attention)
            recipe = build_recipe(run_cfg, num_steps=args.steps)
            warmup(model, processor, run_cfg)
            print(f"\n[dequantize={dequantize} attention={attention}]")
            timings = _time_runs(
                model, processor, recipe, width, height, args.runs
            )
            results.append(
                (dequantize, attention, statistics.median(timings), _vram_gb())
            )
            torch.cuda.reset_peak_memory_stats()

        del model, processor
        torch.cuda.empty_cache()

    print()
    print("=" * 62)
    header = f"{'trọng số':<12}{'attention':<12}"
    print(f"{header}{'trung vị':>12}{'VRAM đỉnh':>14}")
    print("-" * 62)
    baseline = results[0][2] if results else 0.0
    for dequantize, attention, median, vram in results:
        speedup = f"  ({baseline / median:.2f}×)" if baseline else ""
        print(
            f"{dequantize:<12}{attention:<12}{median:>10.2f}s"
            f"{vram:>12.1f} GB{speedup}"
        )
    print("=" * 62)
    print("Dòng đầu (off/mask) là hành vi trước khi tối ưu.")
    return 0


def main() -> int:
    args = _parse_args()
    configure_logging()
    settings = load_settings()
    apply_gpu_tuning()
    device = select_device(settings.device_override)

    width = args.width or settings.width
    height = args.height or settings.height

    if args.compare:
        return _run_compare(args, settings, device, width, height)
    if args.profile:
        return _run_profile(args, settings, device, width, height)

    recipe = build_recipe(settings, num_steps=args.steps)

    load_start = time.time()
    model, processor = load_model(settings, device)
    load_s = time.time() - load_start

    warm_s = warmup(model, processor, settings)

    # Mỗi lượt đổi seed. Không cần đổi ảnh vào như bản Qwen cũ: model này
    # không có cache prompt-embeds nên lượt nào cũng là đo thật.
    timings = _time_runs(model, processor, recipe, width, height, args.runs)
    median = statistics.median(timings)
    print()
    print("=" * 62)
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"Biến thể         : {settings.model_type}")
    print(f"Bước / CFG       : {recipe.num_steps} / {recipe.guidance_scale}")
    # Đây mới là đơn vị chi phí thật: một bước CÓ CFG tốn hai forward pass.
    print(
        f"Forward pass     : {recipe.forward_passes} "
        f"({median / recipe.forward_passes:.2f}s mỗi pass)"
    )
    print(f"Attention        : {settings.attention_mode}")
    print(f"Trọng số         : dequantize={settings.dequantize}")
    print(f"Scheduler        : {recipe.scheduler_name}")
    print(f"Kích thước       : {width}x{height}")
    print("-" * 62)
    print(f"Load model       : {load_s:.1f}s")
    print(f"Warm-up          : {warm_s:.1f}s")
    print(f"Trung vị         : {median:.2f}s/ảnh")
    print(f"Nhanh nhất       : {min(timings):.2f}s")
    print(f"Chậm nhất        : {max(timings):.2f}s")
    print(f"VRAM đỉnh        : {_vram_gb():.1f} GB")
    print("-" * 62)
    print("Điền vào config_setup/base.yaml:")
    print(f"  rabbitmq.avg_inference_seconds: {median:.0f}")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
