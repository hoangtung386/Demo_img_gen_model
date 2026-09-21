"""Dựng pipeline FLUX.2-klein-4B và giải đường dẫn trọng số."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from diffusers import (
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
    GGUFQuantizationConfig,
)

from ..config import Settings
from .placement import place

logger = logging.getLogger("gen-image")

# Toàn bộ pipeline chạy bf16. FLUX.2 được huấn luyện ở bf16 và VAE của nó
# (AutoencoderKLFlux2) không ổn định ở fp16 — đừng đổi sang float16 để "tiết
# kiệm", nó chỉ đổi lấy ảnh ra có artefact.
DTYPE = torch.bfloat16

# Các giá trị hợp lệ của `quantization`. "bf16" là mặc định; xem
# load_pipeline() để biết vì sao nó là mặc định dù nặng nhất trên đĩa.
QUANTIZATIONS = ("bf16", "gguf", "fp8")


def _warn_remote(what: str, target: str) -> None:
    """Cảnh báo to khi một component sắp được kéo từ Hub thay vì đọc local.

    Đây là chế độ hỏng âm thầm nguy hiểm nhất của service: chỉ cần
    .model_paths.env không được nạp (hoặc compose quên set biến) là pipeline
    vẫn chạy bình thường nhưng tải lại hàng chục GB vào cache. Đặt
    HF_HUB_OFFLINE=1 để biến nó thành lỗi dừng hẳn thay vì một lần tải im
    lặng.
    """
    logger.warning(
        "%s sẽ được tải từ HuggingFace Hub (%s) chứ KHÔNG đọc từ đĩa. "
        "HF_HOME=%s | HF_HUB_OFFLINE=%s",
        what,
        target,
        os.environ.get("HF_HOME", "<default>"),
        os.environ.get("HF_HUB_OFFLINE", "0"),
    )


def resolve_base_model(settings: Settings) -> str:
    """Trả thư mục pipeline local, hoặc Hub id nếu chưa tải về."""
    if settings.base_model_local:
        local = Path(settings.base_model_local)
        if not (local / "model_index.json").is_file():
            raise FileNotFoundError(
                f"GENIMG_BASE_MODEL_LOCAL trỏ tới {local} nhưng không thấy "
                "model_index.json. Kiểm tra mount ./models trong "
                "docker-compose.yml, hoặc chạy lại scripts/download_model.py."
            )
        logger.info("Dùng base pipeline local: %s", local)
        return str(local)

    _warn_remote("Base pipeline", settings.base_model)
    return settings.base_model


def resolve_gguf_path(settings: Settings) -> str:
    """Trả đường dẫn file .gguf local, hoặc URL blob trên Hub."""
    if settings.transformer_gguf:
        local = Path(settings.transformer_gguf)
        if not local.is_file():
            raise FileNotFoundError(
                f"GENIMG_TRANSFORMER_GGUF trỏ tới {local} nhưng file không "
                "tồn tại. Kiểm tra mount ./models trong docker-compose.yml, "
                "hoặc chạy lại scripts/download_model.py."
            )
        logger.info("Dùng transformer GGUF local: %s", local)
        return str(local)

    url = f"https://huggingface.co/{settings.gguf_repo}/blob/main/{settings.gguf_file}"
    _warn_remote("Transformer GGUF", url)
    return url


def _load_gguf_transformer(
    settings: Settings, base_model: str
) -> Flux2Transformer2DModel:
    """Nạp transformer từ checkpoint GGUF đã lượng tử hoá.

    ``config=base_model`` + ``subfolder="transformer"`` là BẮT BUỘC, không
    phải tuỳ chọn cho gọn. Thiếu hai tham số này, ``from_single_file`` gọi
    ``fetch_diffusers_config()`` để tự đoán kiến trúc từ nội dung checkpoint;
    nó nhận ra "đây là flux2" nhưng không phân biệt được **klein 4B** với
    **dev**, nên dựng state-dict rỗng theo chiều của dev rồi nổ:

        double_stream_modulation_img.linear.weight has an expected quantized
        shape of: (18432, 3072), but received shape: (18432, 6144)

    (diffusers#13001). Trỏ thẳng ``config=`` vào repo klein sẽ bỏ qua hẳn
    bước đoán đó và đọc đúng ``transformer/config.json`` của klein.
    """
    path = resolve_gguf_path(settings)
    logger.info("Nạp transformer GGUF: %s", path)
    return Flux2Transformer2DModel.from_single_file(
        path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
        torch_dtype=DTYPE,
        config=base_model,
        subfolder="transformer",
    )


def load_pipeline(settings: Settings, device: str) -> Flux2KleinPipeline:
    """Nạp FLUX.2-klein-4B. MỘT pipeline cho cả text-to-image lẫn image-edit.

    Khác hẳn backend cũ, nơi phải dựng hai pipeline (``QwenImageEditPlus`` bắt
    buộc có ``image=``, nên text-to-image cần một ``QwenImagePipeline`` thứ
    hai dùng chung weight). ``Flux2KleinPipeline.__call__`` nhận ``image=None``
    nên một object phục vụ được cả hai đường — không còn chỗ nào để hai
    pipeline lệch cấu hình khỏi nhau.

    ``quantization``:
      - ``bf16`` (mặc định): đọc thẳng transformer bf16 của repo. ~16 GiB
        tổng, vừa L4 24GB, và là nhánh DUY NHẤT ``torch.compile`` được.
      - ``gguf``: transformer lượng tử hoá (~4.3 GiB ở Q8_0). Nhỏ hơn nhưng
        diffusers giải nén về ``compute_dtype`` ngay trong forward — đổi
        dung lượng lấy băng thông, mà băng thông mới là nút cổ chai trên L4.
        Chỉ chọn khi cần chỗ cho nhiều process worker trên cùng một card.
        Xem docs/REFACTOR_FLUX2_KLEIN_4B.md §2.2.
      - ``fp8``: lượng tử hoá weight sang float8 bằng optimum-quanto. L4 là
        Ada (sm_89) nên CÓ tensor core FP8 thật — khác GGUF, đây không phải
        giải nén trong forward. **CHƯA ĐƯỢC ĐO** trên phần cứng nào; xem
        ``_load_fp8_transformer``.
    """
    base_model = resolve_base_model(settings)
    logger.info(
        "Quantization: %s | steps: %d | guidance: %s | offload: %s",
        settings.quantization,
        settings.num_steps,
        settings.guidance_scale,
        settings.offload,
    )

    kwargs: dict[str, object] = {"torch_dtype": DTYPE}
    if settings.quantization == "gguf":
        kwargs["transformer"] = _load_gguf_transformer(settings, base_model)
    elif settings.quantization == "fp8":
        kwargs["transformer"] = _load_fp8_transformer(base_model)
    elif settings.quantization != "bf16":
        raise ValueError(
            f"quantization không hợp lệ: {settings.quantization!r}. "
            f"Chọn một trong: {' | '.join(QUANTIZATIONS)}."
        )

    pipeline = Flux2KleinPipeline.from_pretrained(base_model, **kwargs)
    _warn_if_not_distilled(pipeline, base_model)
    _apply_vae_memory_options(pipeline, settings)
    place(pipeline, settings.offload, device)
    return pipeline


def _apply_vae_memory_options(pipeline: Flux2KleinPipeline, settings: Settings) -> None:
    """Bật tiling/slicing cho VAE nếu config yêu cầu.

    Cả hai mặc định TẮT trong ``AutoencoderKLFlux2`` và cũng để tắt ở đây —
    chúng đổi thời gian lấy VRAM, mà cấu hình mặc định (bf16 resident trên
    L4) đang dư VRAM chứ không thiếu.

    Lúc nào bật: khi ``decode`` trong chuỗi ``info`` cho thấy VAE đang là
    phần đáng kể của VRAM đỉnh, hoặc khi chạy ở độ phân giải cao hơn 1024²,
    hoặc khi nhồi nhiều process worker lên một card. Đo bằng
    ``make benchmark`` trước — với model 4 bước, decode chiếm tỉ trọng lớn
    hơn hẳn so với model 30 bước, nên đừng suy từ kinh nghiệm SDXL.

    ``Flux2KleinPipeline`` KHÔNG có ``enable_vae_tiling()`` ở cấp pipeline
    (khác nhiều pipeline khác của diffusers), nên phải gọi thẳng trên
    ``pipeline.vae``.
    """
    vae = getattr(pipeline, "vae", None)
    if vae is None:
        return
    if settings.vae_tiling:
        vae.enable_tiling()
        logger.info("VAE tiling: BẬT (giảm VRAM đỉnh lúc decode)")
    if settings.vae_slicing:
        vae.enable_slicing()
        logger.info("VAE slicing: BẬT (chỉ có tác dụng khi batch > 1)")


def _load_fp8_transformer(base_model: str) -> Flux2Transformer2DModel:
    """Nạp transformer với weight float8 (optimum-quanto).

    Khác GGUF ở chỗ quan trọng nhất: L4 là kiến trúc Ada (sm_89) nên có
    **tensor core FP8 thật**. GGUF phải giải nén về bf16 ngay trong forward
    — đổi dung lượng lấy băng thông; FP8 thì không.

    ⚠️ **CHƯA ĐO trên phần cứng nào.** Nhánh này thêm vào vì `quantization`
    vốn đã là một công tắc và L4 có phần cứng phù hợp, nhưng đừng bật trên
    production trước khi so bằng ``scripts/spike_flux2.py --modes bf16 fp8``.
    Hai thứ phải kiểm: (1) nó có thật sự nhanh hơn bf16 không, (2) ảnh ra
    có suy giảm nhìn thấy được không.

    ``torch.compile`` với nhánh này cũng chưa kiểm — ``maybe_compile`` chỉ
    chặn GGUF, không chặn fp8.
    """
    try:
        from diffusers import QuantoConfig
    except ImportError as exc:  # pragma: no cover - phụ thuộc phiên bản
        raise RuntimeError(
            "quantization='fp8' cần QuantoConfig của diffusers."
        ) from exc

    logger.warning(
        "quantization='fp8' là nhánh CHƯA ĐƯỢC ĐO. So với bf16 bằng "
        "scripts/spike_flux2.py trước khi dùng trên production."
    )
    return Flux2Transformer2DModel.from_pretrained(
        base_model,
        subfolder="transformer",
        quantization_config=QuantoConfig(weights_dtype="float8"),
        torch_dtype=DTYPE,
    )


def _warn_if_not_distilled(pipeline: Flux2KleinPipeline, base_model: str) -> None:
    """Kêu to nếu nạp nhầm bản KHÔNG distilled.

    ``FLUX.2-klein-4B`` (distilled) và ``FLUX.2-klein-base-4B`` khác nhau
    đúng một cờ trong model_index.json, nhưng khác nhau ~4 lần về tốc độ:
    bản base cần vài chục bước và guidance thật (CFG hai nhánh), bản distilled
    xong trong 4 bước với guidance nhúng sẵn. Tải nhầm thì service vẫn chạy,
    chỉ là mỗi ảnh lâu gấp bội và ảnh ra ở 4 bước thì nhiễu — một triệu chứng
    dễ bị đổ nhầm cho phần cứng.
    """
    if not getattr(pipeline.config, "is_distilled", False):
        logger.warning(
            "Pipeline nạp từ %s KHÔNG được đánh dấu is_distilled. Nhiều khả "
            "năng đây là bản 'klein-base' chứ không phải bản distilled 4 "
            "bước — ảnh ra ở num_steps=4 sẽ nhiễu và CFG sẽ chạy hai nhánh "
            "(chậm gấp đôi). Kiểm tra lại base_model.",
            base_model,
        )
