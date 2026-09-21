"""Dựng pipeline FLUX.2-klein-9B với transformer GGUF và giải đường dẫn."""

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
from diffusers.quantizers import PipelineQuantizationConfig

from ..config import PROJECT_ROOT, Settings
from .placement import place

logger = logging.getLogger("gen-image")

# Toàn bộ pipeline chạy bf16. FLUX.2 được huấn luyện ở bf16 và VAE của nó
# (AutoencoderKLFlux2) không ổn định ở fp16 — đừng đổi sang float16 để "tiết
# kiệm", nó chỉ đổi lấy ảnh ra có artefact.
DTYPE = torch.bfloat16

# Các giá trị hợp lệ của `quantization`. 9B mặc định chạy GGUF.
QUANTIZATIONS = ("bf16", "gguf", "fp8")

# Các giá trị hợp lệ của `text_encoder_quantization`.
#
# Ở FLUX.2-klein-9B, text encoder là **Qwen3-8B** — bf16 ≈ 16.4 GiB, tức là
# component LỚN NHẤT của pipeline, lớn hơn cả transformer GGUF. Đây là khác
# biệt then chốt so với klein-4B (Qwen3-4B, ~8 GiB), nơi transformer mới là
# phần to nhất và `quantization` một mình đã đủ.
#
# Hệ quả: trên L4 24GB, nén transformer mà để nguyên text encoder thì tổng
# weight vẫn ~22.6 GiB (16.4 + 5.9 + 0.3) — vừa đúng bằng VRAM khả dụng,
# nên OOM ngay ở activation của lượt denoise đầu tiên.
TEXT_ENCODER_QUANTIZATIONS = ("bf16", "nf4", "int8")


def _model_root(settings: Settings) -> Path:
    """Resolve ``model_root`` exactly as the downloader does."""
    root = Path(settings.model_root)
    return root if root.is_absolute() else PROJECT_ROOT / root


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
    configured = settings.base_model_local
    # ``.model_paths.env`` is an optimisation, not a prerequisite.  A source
    # checkout on a server commonly has ``models/`` copied in but not that
    # generated env file (the exact failure in the reported log).
    discovered = _model_root(settings) / "FLUX.2-klein-9B"
    if configured or (discovered / "model_index.json").is_file():
        local = Path(configured) if configured else discovered
        if not (local / "model_index.json").is_file():
            raise FileNotFoundError(
                f"GENIMG_BASE_MODEL_LOCAL trỏ tới {local} nhưng không thấy "
                "model_index.json. Kiểm tra mount ./models trong "
                "docker-compose.yml, hoặc chạy lại scripts/download_model.py."
            )
        logger.info("Dùng base pipeline local: %s", local)
        return str(local)

    if not settings.hf_token:
        raise RuntimeError(
            "Không tìm thấy base pipeline local. FLUX.2-klein-9B là repo "
            "gated nên không thể chạy với request ẩn danh. Đặt "
            "GENIMG_BASE_MODEL_LOCAL tới thư mục có model_index.json (và "
            "mount/copy models/), hoặc đặt HF_TOKEN/app.hf_token sau khi "
            "đã được cấp quyền trên Hugging Face."
        )
    _warn_remote("Base pipeline", settings.base_model)
    return settings.base_model


def resolve_gguf_path(settings: Settings) -> str:
    """Trả đường dẫn file .gguf local, hoặc URL blob trên Hub."""
    configured = settings.transformer_gguf
    discovered = _model_root(settings) / "gguf" / settings.gguf_file
    if configured or discovered.is_file():
        local = Path(configured) if configured else discovered
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


def build_text_encoder_quant_config(mode: str) -> PipelineQuantizationConfig | None:
    """Cấu hình lượng tử hoá cho **text encoder**, hoặc None nếu giữ bf16.

    Vì sao knob này tách khỏi ``quantization``: hai component được nén bằng
    hai cơ chế khác hẳn nhau và có nút cổ chai khác nhau.

      - transformer: GGUF, nạp từ MỘT file rời qua ``from_single_file``.
      - text encoder: nằm trong repo base dưới dạng safetensors, nén tại
        chỗ lúc ``from_pretrained`` bằng bitsandbytes.

    Gộp chung thành một knob sẽ buộc hai thứ phải đổi cùng nhau, trong khi
    cấu hình đáng dùng nhất trên L4 lại là một tổ hợp lệch: transformer GGUF
    Q4_K_M + text encoder NF4.

    Ngân sách VRAM cho FLUX.2-klein-9B (weight, chưa tính activation):

    | text encoder | transformer GGUF Q4_K_M | VAE | tổng |
    |---|---|---|---|
    | ``bf16`` 16.4 GiB | 5.9 GiB | 0.3 | **22.6 GiB** → OOM trên L4 |
    | ``int8``  8.5 GiB | 5.9 GiB | 0.3 | ~14.7 GiB |
    | ``nf4``   ~5.0 GiB | 5.9 GiB | 0.3 | **~11.2 GiB** ← mặc định |

    ``nf4`` là mặc định vì nó là lựa chọn DUY NHẤT chừa lại đủ chỗ cho
    activation ở 1024² mà vẫn giữ được ``offload="resident"`` — tức không
    một lần chuyển weight qua PCIe nào lúc suy luận. ``int8`` cũng vừa,
    nhưng kernel LLM.int8() của bitsandbytes chậm hơn NF4 đáng kể ở batch
    nhỏ, và ở đây batch luôn = 1.

    ⚠️ **Chưa đo chất lượng.** NF4 làm lệch embedding của prompt ở mức nào
    thì phải nhìn ảnh mới biết. Lưu ý cache embed (``inference._EMBED_CACHE``)
    giữ lại kết quả theo prompt, nên sai lệch nếu có sẽ dính trong cả phiên.
    Đặt ``text_encoder_quantization: "bf16"`` + ``offload: "model_offload"``
    để có bản đối chứng không nén.
    """
    normalized = (mode or "").strip().lower()
    if normalized not in TEXT_ENCODER_QUANTIZATIONS:
        raise ValueError(
            f"text_encoder_quantization không hợp lệ: {mode!r}. "
            f"Chọn một trong: {' | '.join(TEXT_ENCODER_QUANTIZATIONS)}."
        )
    if normalized == "bf16":
        logger.warning(
            "text_encoder_quantization='bf16': text encoder Qwen3-8B chiếm "
            "~16.4 GiB. Cộng transformer và VAE thì vượt VRAM khả dụng của "
            "L4 24GB — cấu hình này cần card lớn hơn, hoặc "
            "offload='model_offload'."
        )
        return None

    if normalized == "int8":
        backend = "bitsandbytes_8bit"
        quant_kwargs: dict[str, object] = {"load_in_8bit": True}
    else:
        backend = "bitsandbytes_4bit"
        quant_kwargs = {
            "load_in_4bit": True,
            # NF4 chứ không phải FP4: cùng dung lượng, nhưng NF4 giả định
            # weight phân phối chuẩn — đúng với weight đã qua huấn luyện —
            # nên sai số lượng tử hoá thấp hơn. Không có lý do chọn FP4.
            "bnb_4bit_quant_type": "nf4",
            # Nén thêm một lượt các hằng số lượng tử hoá. Tiết kiệm ~0.4 GiB
            # với model 8B, chi phí tính toán gần như không đo được.
            "bnb_4bit_use_double_quant": True,
            # Giải nén về đúng dtype của phần còn lại trong pipeline. Để lệch
            # sang fp16 ở đây là mời một lần ép kiểu âm thầm ở ranh giới
            # text encoder → transformer.
            "bnb_4bit_compute_dtype": DTYPE,
        }

    logger.info(
        "Text encoder: %s (bitsandbytes) — component lớn nhất của pipeline 9B.",
        normalized,
    )
    return PipelineQuantizationConfig(
        quant_backend=backend,
        quant_kwargs=quant_kwargs,
        # CHỈ text encoder. Transformer đã được dựng sẵn và truyền vào
        # ``from_pretrained`` qua kwargs; VAE thì bé (0.3 GiB) và là nơi
        # sai số lượng tử hoá hiện ra trực tiếp thành artefact trên ảnh.
        components_to_quantize=["text_encoder"],
    )


def _load_gguf_transformer(
    settings: Settings, base_model: str
) -> Flux2Transformer2DModel:
    """Nạp transformer từ checkpoint GGUF đã lượng tử hoá.

    ``config=base_model`` + ``subfolder="transformer"`` là BẮT BUỘC, không
    phải tuỳ chọn cho gọn. Thiếu hai tham số này, ``from_single_file`` gọi
    ``fetch_diffusers_config()`` để tự đoán kiến trúc từ nội dung checkpoint;
    nó nhận ra "đây là flux2" nhưng không phân biệt được đúng biến thể klein
    9B với **dev**, nên có thể dựng state-dict sai chiều rồi nổ.

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
        token=settings.hf_token,
    )


def load_pipeline(settings: Settings, device: str) -> Flux2KleinPipeline:
    """Nạp FLUX.2-klein-9B. MỘT pipeline cho cả text-to-image lẫn image-edit.

    Khác hẳn backend cũ, nơi phải dựng hai pipeline (``QwenImageEditPlus`` bắt
    buộc có ``image=``, nên text-to-image cần một ``QwenImagePipeline`` thứ
    hai dùng chung weight). ``Flux2KleinPipeline.__call__`` nhận ``image=None``
    nên một object phục vụ được cả hai đường — không còn chỗ nào để hai
    pipeline lệch cấu hình khỏi nhau.

    **Hai knob lượng tử hoá, không phải một.** ``quantization`` điều khiển
    transformer; ``text_encoder_quantization`` điều khiển text encoder
    Qwen3-8B (~16.4 GiB ở bf16 — component lớn nhất ở bản 9B). Trên L4 24GB
    phải nén CẢ HAI; nén mỗi transformer là không đủ. Xem
    ``build_text_encoder_quant_config`` để có bảng ngân sách VRAM.

    ``quantization``:
      - ``bf16``: đọc thẳng transformer bf16 của repo, ~18 GiB. Chỉ vừa L4
        khi text encoder đã nén, và ngay cả khi đó cũng rất sát trần.
      - ``gguf`` (mặc định): transformer lượng tử hoá (~5.9 GiB ở Q4_K_M).
        Nhỏ hơn nhưng diffusers giải nén về ``compute_dtype`` ngay trong
        forward — đổi dung lượng lấy băng thông. Ở bản 4B, đánh đổi đó là
        một khoản lỗ vì bf16 vốn đã vừa; ở bản 9B thì bf16 KHÔNG vừa, nên
        đây thành cấu hình mặc định chứ không còn là đường phụ.
        Cần dùng cùng config của base pipeline 9B, không phải config 4B.
      - ``fp8``: lượng tử hoá weight sang float8 bằng optimum-quanto. L4 là
        Ada (sm_89) nên CÓ tensor core FP8 thật — khác GGUF, đây không phải
        giải nén trong forward. **CHƯA ĐƯỢC ĐO** trên phần cứng nào; xem
        ``_load_fp8_transformer``.
    """
    base_model = resolve_base_model(settings)
    logger.info(
        "Quantization: transformer=%s | text_encoder=%s | steps: %d | "
        "guidance: %s | offload: %s",
        settings.quantization,
        settings.text_encoder_quantization,
        settings.num_steps,
        settings.guidance_scale,
        settings.offload,
    )

    kwargs: dict[str, object] = {"torch_dtype": DTYPE}
    text_encoder_quant = build_text_encoder_quant_config(
        settings.text_encoder_quantization
    )
    if text_encoder_quant is not None:
        kwargs["quantization_config"] = text_encoder_quant
    if settings.quantization == "gguf":
        kwargs["transformer"] = _load_gguf_transformer(settings, base_model)
    elif settings.quantization == "fp8":
        kwargs["transformer"] = _load_fp8_transformer(base_model, settings.hf_token)
    elif settings.quantization != "bf16":
        raise ValueError(
            f"quantization không hợp lệ: {settings.quantization!r}. "
            f"Chọn một trong: {' | '.join(QUANTIZATIONS)}."
        )

    pipeline = Flux2KleinPipeline.from_pretrained(
        base_model, token=settings.hf_token, **kwargs
    )
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


def _load_fp8_transformer(
    base_model: str, token: str | None
) -> Flux2Transformer2DModel:
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
        token=token,
    )


def _warn_if_not_distilled(pipeline: Flux2KleinPipeline, base_model: str) -> None:
    """Kêu to nếu nạp nhầm bản KHÔNG distilled.

    ``FLUX.2-klein-9B`` (distilled) và ``FLUX.2-klein-base-9B`` khác nhau
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
