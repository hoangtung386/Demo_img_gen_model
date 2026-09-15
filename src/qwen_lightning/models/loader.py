"""Pipeline construction, scheduler and transformer path resolution."""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImagePipeline,
)
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision

from ..config import Settings
from .offload import apply_offload

logger = logging.getLogger("qwen-image-edit-2509-lightning")


def build_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """Build the FlowMatch scheduler with the Lightning distillation shift."""
    config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    return FlowMatchEulerDiscreteScheduler.from_config(config)


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


def resolve_transformer_path(settings: Settings, precision: str) -> str:
    """Return the local or HuggingFace transformer weight path."""
    if settings.transformer_path:
        local = Path(settings.transformer_path)
        if not local.is_file():
            raise FileNotFoundError(
                f"QIE_TRANSFORMER_PATH trỏ tới {local} nhưng file không tồn "
                "tại. Kiểm tra mount ./models trong docker-compose.yml, hoặc "
                "chạy lại scripts/download_model.py."
            )
        logger.info("Using local transformer weights: %s", local)
        return str(local)

    path = (
        f"{settings.transformer_repo}/{settings.transformer_subdir}/"
        f"svdq-{precision}_r{settings.rank}-"
        f"qwen-image-edit-2509-lightning-"
        f"{settings.num_steps}steps-251115.safetensors"
    )
    _warn_remote("Transformer", path)
    return path


def resolve_base_model(settings: Settings) -> str:
    """Return the local base pipeline dir, or the Hub id as a fallback."""
    if settings.base_model_local:
        local = Path(settings.base_model_local)
        if not (local / "model_index.json").is_file():
            raise FileNotFoundError(
                f"QIE_BASE_MODEL_LOCAL trỏ tới {local} nhưng không thấy "
                "model_index.json. Kiểm tra mount ./models trong "
                "docker-compose.yml, hoặc chạy lại scripts/download_model.py."
            )
        logger.info("Using local base pipeline: %s", local)
        return str(local)

    _warn_remote("Base pipeline", settings.base_model)
    return settings.base_model


def load_pipeline(
    settings: Settings,
    device: str,
) -> QwenImageEditPlusPipeline:
    """Load the quantized transformer and assemble the pipeline."""
    precision = settings.precision or get_precision()
    logger.info(
        "Precision: %s | steps: %d | rank: %d",
        precision,
        settings.num_steps,
        settings.rank,
    )

    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        resolve_transformer_path(settings, precision)
    )
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        resolve_base_model(settings),
        transformer=transformer,
        scheduler=build_scheduler(),
        torch_dtype=torch.bfloat16,
    )

    apply_offload(
        pipeline,
        transformer,
        settings.offload,
        device,
        settings.text_encoder_device,
    )
    return pipeline


def build_t2i_pipeline(
    edit_pipeline: QwenImageEditPlusPipeline,
) -> QwenImagePipeline:
    """Pipeline text-to-image dùng CHUNG module với pipeline edit.

    Tab "Prompt to Image" cần sinh ảnh không có ảnh vào, mà
    ``QwenImageEditPlusPipeline`` bắt buộc phải có ``image=``. Hai pipeline
    nhận đúng cùng 5 component (scheduler, vae, text_encoder, tokenizer,
    transformer) và gọi transformer bằng CÙNG một chữ ký — chỉ khác ở chỗ
    bản edit nối thêm latent của ảnh điều kiện vào hidden_states và cắt lại
    output. Nên dựng thêm một pipeline trỏ vào đúng các object đó là đủ:
    **không tốn thêm một byte VRAM nào**.

    KHÔNG dùng ``QwenImagePipeline.from_pipe()``: nó mặc định
    ``torch_dtype=torch.float32`` rồi gọi ``new_pipeline.to(dtype=...)`` trên
    chính các module đang dùng chung — cast transformer INT4 và text_encoder
    bf16 sang fp32, hỏng luôn cả pipeline edit. Dựng tay thì không có đường
    nào để lọt chuyện đó.

    Lưu ý chất lượng: trọng số ở đây là Qwen-Image-**Edit**-2509 Lightning,
    tinh chỉnh cho việc sửa ảnh. Sinh ảnh từ chữ vẫn chạy nhưng không phải
    thứ nó được luyện, nên đừng kỳ vọng ngang bản Qwen-Image gốc.
    """
    t2i = QwenImagePipeline(
        scheduler=edit_pipeline.scheduler,
        vae=edit_pipeline.vae,
        text_encoder=edit_pipeline.text_encoder,
        tokenizer=edit_pipeline.tokenizer,
        transformer=edit_pipeline.transformer,
    )

    # Chiến lược offload 'split' ghim _execution_device bằng một subclass
    # riêng cho instance (xem offload._pin_execution_device). Pipeline mới
    # không thừa hưởng, nên phải gắn lại đúng property đó — thiếu bước này
    # thì latents rơi về card của text_encoder trong khi transformer nằm ở
    # card khác.
    pinned = type(edit_pipeline).__dict__.get("_execution_device")
    if isinstance(pinned, property):
        t2i.__class__ = type(
            f"{t2i.__class__.__name__}Pinned",
            (t2i.__class__,),
            {"_execution_device": pinned},
        )
        logger.info("T2I pipeline pinned to %s", t2i._execution_device)

    logger.info("Text-to-image pipeline ready (dùng chung weights, +0 VRAM)")
    return t2i
