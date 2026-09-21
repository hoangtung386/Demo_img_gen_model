"""Spike xác thực FLUX.2-klein-4B trên GPU thật — Phase 0 của refactor.

Đây là CỔNG CHẶN: chạy script này trên đúng phần cứng đích (L4 24GB) và ghi
kết quả vào ``docs/adr/0001-flux2-klein-backend.md`` TRƯỚC khi tin vào bất kỳ
hằng số mặc định nào trong ``config_setup/base.yaml``.

Script cố tình KHÔNG dùng ``gen_image.config``/``models.loader``: nó phải trả
lời được câu hỏi "diffusers có nạp nổi model này không" một cách độc lập với
tầng wiring của service. Lẫn hai thứ vào nhau thì một lỗi config sẽ trông
giống hệt một lỗi model.

    python scripts/spike_flux2.py                 # bf16+gguf, t2i+edit
    python scripts/spike_flux2.py --modes bf16 fp8     # so FP8 với bf16
    python scripts/spike_flux2.py --modes bf16
    python scripts/spike_flux2.py --compile            # thêm nhánh compile
    python scripts/spike_flux2.py --runs 5 --out /tmp/spike

Tiêu chí nghiệm thu — xem docs/REFACTOR_FLUX2_KLEIN_4B.md §3.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

# Bản DISTILLED. "FLUX.2-klein-base-4B" là repo khác: không distilled, cần
# vài chục bước và guidance thật — tải nhầm là mất ~4 lần tốc độ.
REPO = "black-forest-labs/FLUX.2-klein-4B"
GGUF_URL = (
    "https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/blob/main/"
    "flux-2-klein-4b-Q8_0.gguf"
)

PROMPT = "a cat holding a sign that says hello world"
GUIDANCE = 1.0
STEPS = 4


def build(mode: str):
    """Dựng pipeline cho ``mode`` ∈ {bf16, gguf, fp8}."""
    from diffusers import (
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
        GGUFQuantizationConfig,
        QuantoConfig,
    )

    if mode == "bf16":
        return Flux2KleinPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)

    if mode == "fp8":
        # L4 là Ada (sm_89) — CÓ tensor core FP8 thật, nên khác GGUF ở chỗ
        # không phải giải nén trong forward. Đây là câu hỏi mở của spike:
        # fp8 có thắng bf16 trên L4 không, và ảnh ra có suy giảm không.
        # Cần `pip install optimum-quanto` (extra "fp8" trong pyproject).
        transformer = Flux2Transformer2DModel.from_pretrained(
            REPO,
            subfolder="transformer",
            quantization_config=QuantoConfig(weights_dtype="float8"),
            torch_dtype=torch.bfloat16,
        )
        return Flux2KleinPipeline.from_pretrained(
            REPO, transformer=transformer, torch_dtype=torch.bfloat16
        )

    # config= + subfolder= là BẮT BUỘC, không phải cho gọn: thiếu chúng,
    # from_single_file tự đoán kiến trúc từ checkpoint, nhận ra "flux2"
    # nhưng không phân biệt được klein-4B với dev, rồi nổ shape mismatch
    # (diffusers#13001). Đây chính là giả thuyết mà spike này đi kiểm.
    transformer = Flux2Transformer2DModel.from_single_file(
        GGUF_URL,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        config=REPO,
        subfolder="transformer",
    )
    return Flux2KleinPipeline.from_pretrained(
        REPO, transformer=transformer, torch_dtype=torch.bfloat16
    )


def _reference_images(kind: str) -> list[Image.Image] | None:
    if kind == "t2i":
        return None
    return [Image.new("RGB", (768, 768), (127, 127, 127))]


def _run(pipe, kind: str, runs: int, out_dir: Path, tag: str) -> list[float]:
    """Chạy ``runs`` lượt, bỏ lượt đầu (warm-up). Trả list giây."""
    timings: list[float] = []
    image = None
    for i in range(runs + 1):
        started = time.time()
        image = pipe(
            image=_reference_images(kind),
            prompt=PROMPT,
            height=1024,
            width=1024,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
            generator=torch.Generator("cuda").manual_seed(0),
        ).images[0]
        torch.cuda.synchronize()
        elapsed = time.time() - started
        label = "warm-up" if i == 0 else f"lượt {i}"
        print(f"    {label:<9} {elapsed:6.2f}s")
        if i > 0:
            timings.append(elapsed)

    if image is not None:
        path = out_dir / f"spike_{tag}_{kind}.png"
        image.save(path)
        print(f"    ảnh -> {path}  (PHẢI xem bằng mắt: tiêu chí F)")
    return timings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["bf16", "gguf"],
        choices=["bf16", "gguf", "fp8"],
    )
    parser.add_argument("--kinds", nargs="+", default=["t2i", "edit"])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/tmp"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ Không thấy CUDA — spike này phải chạy trên GPU thật.")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Repo: {REPO}\n")

    summary: dict[str, dict[str, float]] = {}
    for mode in args.modes:
        tag = mode + ("-compile" if args.compile else "")
        print(f"=== {tag} ===")
        torch.cuda.reset_peak_memory_stats()
        try:
            load_start = time.time()
            pipe = build(mode).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            print(f"  load: {time.time() - load_start:.1f}s")
        except Exception as exc:  # noqa: BLE001
            # Đây là kết quả hợp lệ của spike, không phải sự cố: tiêu chí C
            # tồn tại chính vì nhánh này có thể xảy ra (diffusers#13001).
            print(f"  ❌ KHÔNG NẠP ĐƯỢC: {type(exc).__name__}: {exc}")
            print("     → tiêu chí C TRƯỢT cho nhánh này; ghi vào ADR.")
            continue

        if args.compile:
            if mode == "gguf":
                print("  bỏ qua compile: GGUF không trace được (#10795)")
            else:
                # in-place; gán lại thuộc tính sẽ khiến
                # DiffusionPipeline.__setattr__ ghi đè config (xem
                # gen_image/tuning.py::maybe_compile).
                pipe.transformer.compile(
                    mode="max-autotune-no-cudagraphs",
                    dynamic=True,
                )

        for kind in args.kinds:
            print(f"  [{kind}]")
            timings = _run(pipe, kind, args.runs, args.out, tag)
            if timings:
                summary[f"{tag}/{kind}"] = {
                    "median": statistics.median(timings),
                    "min": min(timings),
                }

        peak = torch.cuda.max_memory_allocated() / 2**30
        summary.setdefault(f"{tag}/_peak", {})["gib"] = peak
        print(f"  VRAM đỉnh: {peak:.1f} GiB\n")

        del pipe
        torch.cuda.empty_cache()

    print("=" * 62)
    print("TỔNG KẾT — chép vào docs/adr/0001-flux2-klein-backend.md")
    print("=" * 62)
    for key, values in summary.items():
        rendered = " | ".join(f"{k}={v:.2f}" for k, v in values.items())
        print(f"  {key:<24} {rendered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
