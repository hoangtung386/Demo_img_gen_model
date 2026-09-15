"""VRAM offload strategies for the pipeline."""
from __future__ import annotations

import logging

import torch
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from diffusers import QwenImageEditPlusPipeline
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory

logger = logging.getLogger("qwen-image-edit-2509-lightning")

# text_encoder (Qwen2.5-VL bf16) ~16GB + transformer INT4 ~11GB không vừa
# một card 24GB, nên 'auto' chỉ chọn 'split' khi có từ 2 GPU trở lên.
_MIN_SPLIT_GB = 20.0

# Ngưỡng để nhét TOÀN BỘ pipeline lên MỘT card và không offload gì cả.
# Tổng weight resident đo được là 26.4 GiB (text_encoder 15.4 + transformer
# INT4 10.7 + vae 0.24); phần dư dành cho activation lúc encode ảnh và
# attention trên ~4k latent token. 32 GiB cho A100-40GB (get_gpu_memory trả
# 39) và H100 đi qua, còn 4090-24GB (trả 23) vẫn rơi về model offload đúng
# như trước.
_MIN_SINGLE_GB = 32.0


def _exclude_transformer(pipeline: QwenImageEditPlusPipeline) -> None:
    """Keep the transformer resident during sequential offload if supported."""
    exclude = getattr(pipeline, "_exclude_from_cpu_offload", None)
    if exclude is not None:
        exclude.append("transformer")


def _other_device(device: str) -> str | None:
    """Return the second CUDA device to pair with ``device``, if any."""
    if torch.cuda.device_count() < 2:
        return None
    try:
        index = int(device.split(":")[-1])
    except (ValueError, IndexError):
        index = 0
    for candidate in range(torch.cuda.device_count()):
        if candidate != index:
            return f"cuda:{candidate}"
    return None


def _pin_execution_device(
    pipeline: QwenImageEditPlusPipeline, device: str
) -> None:
    """Force the pipeline to allocate its tensors on ``device``.

    Với các component nằm trên nhiều GPU, ``_execution_device`` của
    diffusers có thể trả về card của text_encoder và làm latents lệch
    khỏi transformer. Ghim bằng một subclass riêng cho instance này để
    không ảnh hưởng class gốc.
    """
    target = torch.device(device)
    pipeline.__class__ = type(
        f"{pipeline.__class__.__name__}Pinned",
        (pipeline.__class__,),
        {"_execution_device": property(lambda self: target)},
    )


def _apply_split(
    pipeline: QwenImageEditPlusPipeline,
    main_device: str,
    text_device: str,
) -> None:
    """Keep every component resident, splitting them across two GPUs.

    Không có lần chuyển CPU<->GPU nào lúc suy luận — đây là điểm khác
    biệt lớn nhất về tốc độ so với ``enable_model_cpu_offload``.
    """
    pipeline.transformer.to(main_device)
    pipeline.vae.to(main_device)
    pipeline.text_encoder.to(text_device)

    # Hook chuyển input sang card của text_encoder rồi trả output về
    # đúng card đã gọi, nên phần còn lại của pipeline không cần biết.
    add_hook_to_module(
        pipeline.text_encoder,
        AlignDevicesHook(execution_device=text_device, io_same_device=True),
        append=True,
    )
    _pin_execution_device(pipeline, main_device)
    logger.info(
        "Split across GPUs: transformer+vae on %s, text_encoder on %s",
        main_device,
        text_device,
    )


def choose_strategy(
    strategy: str,
    gpu_mem: float | None,
    has_second_gpu: bool,
) -> str:
    """Quy 'auto' về một chiến lược cụ thể.

    Hàm thuần, không đụng tới GPU hay pipeline, nên kiểm được toàn bộ bảng
    quyết định bằng unit test thường thay vì phải có card thật.

    Thứ tự ưu tiên khi 'auto':
      1. split      — 2 GPU, mỗi card đủ chỗ. Nhanh nhất.
      2. none       — 1 GPU nhưng >= 32 GiB (A100-40GB, H100): nhét hết lên
                      card, cũng không có lần chuyển CPU<->GPU nào.
      3. model      — 1 GPU 18..32 GiB: nạp ~27GB qua PCIe mỗi lần sinh ảnh.
      4. sequential — card quá nhỏ, offload từng block.
    """
    strategy = strategy.lower()
    if strategy != "auto":
        return strategy
    if has_second_gpu and (gpu_mem is None or gpu_mem > _MIN_SPLIT_GB):
        return "split"
    if gpu_mem is not None and gpu_mem >= _MIN_SINGLE_GB:
        return "none"
    if gpu_mem is not None and gpu_mem > 18:
        return "model"
    return "sequential"


def apply_offload(
    pipeline: QwenImageEditPlusPipeline,
    transformer: NunchakuQwenImageTransformer2DModel,
    strategy: str,
    device: str,
    text_encoder_device: str | None = None,
) -> None:
    """Apply the requested offload strategy.

    strategy: ``auto`` | ``none`` | ``split`` | ``model`` | ``sequential``.
    """
    gpu_mem = None
    try:
        gpu_mem = get_gpu_memory()
        logger.info("Detected GPU memory: %.1f GB", gpu_mem)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not detect GPU memory (%s); using 'none'", exc)

    text_device = text_encoder_device or _other_device(device)
    resolved = choose_strategy(strategy, gpu_mem, text_device is not None)
    if resolved != strategy.lower():
        logger.info("Offload 'auto' resolved to '%s'", resolved)

    if resolved == "split":
        if text_device is None:
            logger.warning(
                "Strategy 'split' needs a second GPU; falling back to 'model'"
            )
            pipeline.enable_model_cpu_offload(device=device)
            return
        _apply_split(pipeline, device, text_device)
        return

    if resolved == "none":
        # Mọi component nằm lại trên GPU. Đây là chế độ dùng cho A100-40GB:
        # 26.4 GiB weight trong 39 GiB VRAM, không offload gì cả.
        logger.info("Keeping every component resident on %s", device)
        pipeline.to(device)
        return

    if resolved == "model":
        pipeline.enable_model_cpu_offload(device=device)
        return

    if resolved == "sequential":
        transformer.set_offload(
            True, use_pin_memory=False, num_blocks_on_gpu=1
        )
        _exclude_transformer(pipeline)
        pipeline.enable_sequential_cpu_offload(device=device)
        return

    raise ValueError(
        f"Chiến lược offload không hợp lệ: {strategy!r}. Chọn một trong "
        "auto | none | split | model | sequential."
    )
