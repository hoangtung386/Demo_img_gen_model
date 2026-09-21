"""Đo tốc độ sinh ảnh thật trên máy đang chạy.

Dùng để xác nhận cấu hình đang chọn là đúng, và để so trước/sau mỗi lần
chỉnh. Đây là công cụ nghiệm thu cho ma trận ở Phase 5 của
docs/REFACTOR_FLUX2_KLEIN_4B.md.

    python scripts/benchmark.py                   # 5 lượt, t2i + edit, 1024px
    python scripts/benchmark.py --runs 10
    python scripts/benchmark.py --area 768        # cạnh 768px
    python scripts/benchmark.py --mode t2i        # chỉ text-to-image
    python scripts/benchmark.py --mode edit2      # edit với 2 ảnh tham chiếu

Trong container:

    docker compose exec gen-image python scripts/benchmark.py
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from gen_image.config import load_settings  # noqa: E402
from gen_image.device import select_device  # noqa: E402
from gen_image.inference import clear_embed_cache, generate  # noqa: E402
from gen_image.logging_setup import configure_logging  # noqa: E402
from gen_image.models.loader import load_pipeline  # noqa: E402
from gen_image.tuning import (  # noqa: E402
    apply_gpu_tuning,
    maybe_compile,
    warmup,
)

PROMPT_T2I = "a cat holding a sign that says hello world, photorealistic, 35mm"
PROMPT_EDIT = "make the lighting warmer and more cinematic"

# Prompt dài cỡ các tab demo trong ui/prompts.py (~1800 ký tự). Tồn tại để
# ĐO một thứ dễ bị bỏ sót: text token đi vào joint-attention CÙNG 4096 image
# token ở mọi bước denoise, nên prompt dài làm đắt thêm TỪNG BƯỚC — chứ
# không chỉ đắt thêm một lần ở khâu encode.
#
# Số đo Colab L4 2026-09-21: prompt ngắn 2.46s/bước, prompt demo dài
# 4.30s/bước. Cùng model, cùng 1024², cùng 4 bước. Đó là +75% từ MỘT thứ
# duy nhất, và là chênh lệch lớn nhất đo được trong toàn bộ dự án này.
PROMPT_LONG = (
    "A photorealistic editorial portrait, 35mm lens, shallow depth of field, "
    "natural window light falling from the left at a 45 degree angle, soft "
    "shadows, warm golden hour tone, subject wearing a tailored charcoal wool "
    "coat over a cream turtleneck, standing in a minimalist concrete interior "
    "with large floor-to-ceiling windows, shot on Kodak Portra 400, fine film "
    "grain, no digital sharpening, skin texture preserved with visible pores "
    "and fine lines, catchlights in both eyes, hair lit from behind creating "
    "a subtle rim light, background softly out of focus showing a blurred "
    "city skyline at dusk, composition follows the rule of thirds with the "
    "subject positioned on the left third, gaze directed slightly off camera "
    "to the right, expression calm and contemplative, colour grading leans "
    "toward warm amber in the highlights and cool teal in the shadows, "
    "overall mood quiet and cinematic, avoid plastic skin, avoid oversaturated "
    "colours, avoid harsh direct flash, avoid distorted hands, avoid text "
    "artefacts, avoid watermarks, preserve accurate facial proportions and "
    "natural neck and ear tones matching the face, match the lighting "
    "direction of the reference image exactly, do not flatten the subject "
    "onto the background, blend the edges so no cutout seam is visible"
)

MODES = ("t2i", "edit", "edit2")


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def _inputs_for(mode: str, run: int) -> list[Image.Image]:
    """Ảnh tham chiếu cho một lượt. Mỗi lượt một sắc xám khác nhau.

    Ảnh phải khác nhau giữa các lượt để không vô tình đo lại cùng một
    đường đi trong VAE cache của pipeline.
    """
    if mode == "t2i":
        return []
    shade = 40 + run * 7
    base = Image.new("RGB", (768, 768), (shade, shade, shade))
    if mode == "edit":
        return [base]
    reference = Image.new("RGB", (640, 640), (shade // 2, shade, 200))
    return [reference, base]


_PHASE_RE = re.compile(
    r"denoise ([\d.]+)s.*?\+ text ([\d.]+)s.*?"
    r"\+ prep ([\d.]+)s \+ decode ([\d.]+)s"
)


def _parse_phases(info: str) -> dict[str, float] | None:
    """Bóc denoise/text/prep/decode từ chuỗi `info` mà generate() trả về.

    Đọc lại chuỗi thay vì đo thêm một lần nữa: đó đúng là con số service báo
    cho BE, nên nếu nó sai thì benchmark cũng phải sai theo — chứ không phải
    im lặng đúng nhờ một đường đo riêng.
    """
    m = _PHASE_RE.search(info.replace("\n", " "))
    if not m:
        return None
    denoise, text, prep, decode = (float(x) for x in m.groups())
    return {"denoise": denoise, "text": text, "prep": prep, "decode": decode}


def _print_phase_share(label: str, phases: list[dict[str, float]]) -> None:
    """In tỉ trọng từng giai đoạn — cơ sở để quyết có cần tối ưu VAE không.

    Với model 4 bước, denoise KHÔNG còn áp đảo như ở model 30 bước: các chi
    phí cố định (text encode, VAE decode) chiếm tỉ trọng lớn hơn hẳn. Đây là
    chỗ đọc xem TAEF2 (tiny VAE) có đáng công tích hợp không — nếu `decode`
    dưới ~10% thì tiết kiệm được cũng không đổi gì.
    """
    if not phases:
        return
    keys = ("denoise", "text", "prep", "decode")
    avg = {k: statistics.mean(p[k] for p in phases) for k in keys}
    total = sum(avg.values()) or 1.0
    parts = " | ".join(
        f"{k} {avg[k]:.2f}s ({avg[k] / total * 100:4.1f}%)" for k in keys
    )
    print(f"{label:<16} : {parts}")


def _run_mode(pipeline, settings, mode: str, args):
    area = args.area * args.area
    if args.long_prompt:
        prompt = PROMPT_LONG
    else:
        prompt = PROMPT_T2I if mode == "t2i" else PROMPT_EDIT
    timings: list[float] = []
    phases: list[dict[str, float]] = []

    print(f"\n--- chế độ: {mode} ---")
    for run in range(args.runs):
        # Xoá cache prompt-embed trước MỖI lượt: giữ nguyên prompt qua các
        # lượt sẽ ăn cache từ lượt thứ hai và ta chỉ còn đang đo tốc độ tra
        # dict thay vì tốc độ model.
        clear_embed_cache()
        started = time.time()
        _, info = generate(
            pipeline,
            _inputs_for(mode, run),
            prompt,
            settings.guidance_scale,
            run,
            settings.num_steps,
            output_area=area,
        )
        elapsed = time.time() - started
        timings.append(elapsed)
        parsed = _parse_phases(info)
        if parsed:
            phases.append(parsed)
        print(f"  lượt {run + 1}/{args.runs}: {elapsed:.2f}s | {info}")
    return timings, phases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--area",
        type=int,
        default=1024,
        help="Cạnh ảnh vuông tương đương; diện tích = cạnh^2.",
    )
    parser.add_argument(
        "--mode",
        choices=(*MODES, "all"),
        default="all",
        help="t2i | edit (1 ảnh) | edit2 (2 ảnh) | all",
    )
    parser.add_argument(
        "--long-prompt",
        action="store_true",
        help=(
            "Dùng prompt dài cỡ các tab demo thay vì prompt ngắn. Chạy hai "
            "lần (có/không cờ này) để đo chi phí thật của prompt dài."
        ),
    )
    args = parser.parse_args()

    configure_logging()
    settings = load_settings()
    apply_gpu_tuning()
    device = select_device(settings.device_override)

    load_start = time.time()
    pipeline = load_pipeline(settings, device)
    pipeline.set_progress_bar_config(disable=True)
    maybe_compile(pipeline, settings.compile_transformer, settings.quantization)
    load_s = time.time() - load_start

    warm_s = warmup(
        pipeline,
        settings.num_steps,
        args.area * args.area,
        settings.guidance_scale,
    )

    modes = MODES if args.mode == "all" else (args.mode,)
    results = {m: _run_mode(pipeline, settings, m, args) for m in modes}

    print()
    print("=" * 62)
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"Quantization     : {settings.quantization}")
    print(f"Text encoder     : {settings.text_encoder_quantization}")
    print(f"VAE tiling/slice : {settings.vae_tiling} / {settings.vae_slicing}")
    print(f"Attn backend     : {os.getenv('DIFFUSERS_ATTN_BACKEND', 'native')}")
    print(f"torch.compile    : {settings.compile_transformer}")
    print(f"Đặt model        : {settings.offload}")
    print(f"Steps / guidance : {settings.num_steps} / {settings.guidance_scale}")
    print(f"Độ phân giải     : ~{args.area}px")
    kind = "dài (~1800 ký tự, cỡ tab demo)" if args.long_prompt else "ngắn"
    print(f"Prompt           : {kind}")
    print("-" * 62)
    print(f"Load model       : {load_s:.1f}s")
    print(f"Warm-up          : {warm_s:.1f}s")
    for mode, (timings, _) in results.items():
        print(
            f"{mode:<16} : trung vị {statistics.median(timings):.2f}s | "
            f"nhanh {min(timings):.2f}s | chậm {max(timings):.2f}s"
        )
    print(f"VRAM đỉnh        : {_vram_gb():.1f} GB")
    print("-" * 62)
    print("Tỉ trọng từng giai đoạn (trung bình):")
    for mode, (_, phases) in results.items():
        _print_phase_share(mode, phases)
    print(
        "\n→ `decode` là phần VAE. Dưới ~10% thì tiny VAE (TAEF2) không\n"
        "  đáng công tích hợp; trên ~20% thì đáng cân nhắc.\n"
        "→ `text` cao mà prompt lặp lại nhiều → tăng `embed_cache_size`."
    )
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
