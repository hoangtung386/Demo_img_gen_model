"""Spike xác thực FLUX.2-klein-9B GGUF trên GPU thật.

Đây là CỔNG CHẶN: chạy script này trên đúng phần cứng đích (L4 24GB) và ghi
kết quả vào ``docs/adr/0001-flux2-klein-backend.md`` TRƯỚC khi tin vào bất kỳ
hằng số mặc định nào trong ``config_setup/base.yaml``.

Script cố tình KHÔNG dùng ``gen_image.config``/``models.loader``: nó phải trả
lời được câu hỏi "diffusers có nạp nổi model này không" một cách độc lập với
tầng wiring của service. Lẫn hai thứ vào nhau thì một lỗi config sẽ trông
giống hệt một lỗi model.

    python scripts/spike_flux2.py                      # gguf+fp8, t2i+edit
    python scripts/spike_flux2.py --modes gguf fp8     # so FP8 với GGUF
    python scripts/spike_flux2.py --compile            # thêm nhánh compile
    python scripts/spike_flux2.py --runs 5 --out /tmp/spike
    python scripts/spike_flux2.py --text-encoder bf16  # cần GPU >= 40GB

⚠️ **Text encoder mặc định NF4.** Qwen3-8B ở bf16 là 16.4 GiB; cộng
transformer thì KHÔNG mode nào vừa L4 24GB — kể cả GGUF. Script này trước
đây dựng text encoder bf16 ở mọi mode, nên trên L4 nó OOM trước khi đo được
gì, tức cổng chặn không thể qua được trên đúng phần cứng đích. Xem
``gen_image/models/loader.py::build_text_encoder_quant_config``.

Tiêu chí nghiệm thu — xem docs/REFACTOR_FLUX2_KLEIN_4B.md §3.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

# Bản DISTILLED. "FLUX.2-klein-base-9B" là repo khác: không distilled, cần
# vài chục bước và guidance thật — tải nhầm là mất ~4 lần tốc độ.
REPO = "black-forest-labs/FLUX.2-klein-9B"
GGUF_URL = (
    "https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/main/"
    "flux-2-klein-9b-Q4_K_M.gguf"
)

PROMPT = "a cat holding a sign that says hello world"
GUIDANCE = 1.0
STEPS = 4


def build(mode: str, text_encoder: str = "nf4"):
    """Dựng pipeline cho ``mode`` ∈ {bf16, gguf, fp8}.

    ``text_encoder`` đi qua chính hàm của service
    (``build_text_encoder_quant_config``) — ngoại lệ có chủ đích với nguyên
    tắc "không dùng gen_image" ở đầu file. Lý do: đây là thứ quyết định
    pipeline có vừa VRAM hay không, và chép lại nó ở đây thì hai bên sẽ
    trôi khỏi nhau đúng vào lúc nguy hiểm nhất.
    """
    from diffusers import (
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
        GGUFQuantizationConfig,
        QuantoConfig,
    )

    from gen_image.models.loader import build_text_encoder_quant_config

    quant = build_text_encoder_quant_config(text_encoder)
    common: dict[str, object] = {"torch_dtype": torch.bfloat16}
    if quant is not None:
        common["quantization_config"] = quant

    if mode == "bf16":
        return Flux2KleinPipeline.from_pretrained(REPO, **common)

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
            REPO, transformer=transformer, **common
        )

    # config= + subfolder= là BẮT BUỘC, không phải cho gọn: thiếu chúng,
    # from_single_file tự đoán kiến trúc từ checkpoint, nhận ra "flux2"
    # nhưng không phân biệt được klein-9B với dev, rồi nổ shape mismatch
    # (diffusers#13001). Đây chính là giả thuyết mà spike này đi kiểm.
    transformer = Flux2Transformer2DModel.from_single_file(
        GGUF_URL,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        config=REPO,
        subfolder="transformer",
    )
    return Flux2KleinPipeline.from_pretrained(REPO, transformer=transformer, **common)


def _is_environment_error(exc: Exception) -> bool:
    """Thiếu package/toolchain trên máy, chứ không phải model nạp không nổi."""
    text = str(exc).lower()
    markers = ("ninja", "no module named", "cuda_home", "nvcc", "not installed")
    return isinstance(exc, ImportError) or any(m in text for m in markers)


def _environment_hint(exc: Exception) -> str:
    text = str(exc).lower()
    if "ninja" in text:
        return "uv pip install ninja  (quanto biên dịch extension C++ lúc nạp)"
    if "quanto" in text:
        return "uv sync --extra fp8"
    return "kiểm tra dependency của nhánh này rồi chạy lại"


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
        default=["gguf", "fp8"],
        choices=["bf16", "gguf", "fp8"],
    )
    parser.add_argument("--kinds", nargs="+", default=["t2i", "edit"])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/tmp"))
    parser.add_argument(
        "--text-encoder",
        default="nf4",
        choices=["nf4", "int8", "bf16"],
        help="Lượng tử hoá text encoder Qwen3-8B. bf16 cần GPU >= 40GB.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ Không thấy CUDA — spike này phải chạy trên GPU thật.")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Repo: {REPO}")
    print(f"Text encoder: {args.text_encoder}\n")

    summary: dict[str, dict[str, float]] = {}
    for mode in args.modes:
        tag = mode + ("-compile" if args.compile else "")
        print(f"=== {tag} ===")
        torch.cuda.reset_peak_memory_stats()
        try:
            load_start = time.time()
            pipe = build(mode, args.text_encoder).to("cuda")
            pipe.set_progress_bar_config(disable=True)
            print(f"  load: {time.time() - load_start:.1f}s")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ KHÔNG NẠP ĐƯỢC: {type(exc).__name__}: {exc}")
            # Phân biệt hai loại thất bại — gộp chung là cách ghi vào ADR
            # một kết luận sai về model. "Thiếu ninja/CUDA toolkit" nói về
            # MÁY, không nói gì về việc diffusers có nạp nổi model hay
            # không; ghi nó thành "tiêu chí C trượt" sẽ khiến người đọc ADR
            # sau này loại bỏ một nhánh hoàn toàn dùng được.
            if _is_environment_error(exc):
                print("     → LỖI MÔI TRƯỜNG, không phải kết luận về model.")
                print(f"     → {_environment_hint(exc)}")
            else:
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
