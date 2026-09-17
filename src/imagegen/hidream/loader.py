"""Nạp model HiDream-O1-Image đã lượng tử hóa SDNQ.

Thay cho ``models/loader.py`` cũ (diffusers + Nunchaku). Khác biệt lớn nhất:
model này KHÔNG phải pipeline diffusers mà là một model ``transformers``
duy nhất — không VAE, không text encoder rời, không transformer tách riêng.
Vì thế cũng không còn ``build_scheduler()`` ở tầng này: scheduler do
``vendor/pipeline.py::build_scheduler()`` dựng bên trong mỗi lượt sinh ảnh
theo ``scheduler_name``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

# PHẢI import TRƯỚC from_pretrained: sdnq đăng ký quantizer "sdnq" vào
# transformers ở module level. Thiếu dòng này thì transformers đọc
# quantization_config.quant_method == "sdnq" trong config.json rồi báo
# "Unknown quantization type" — lỗi trông như file hỏng chứ không như thiếu
# thư viện, và chỉ lộ ra sau khi đã đọc xong metadata của 3 shard.
import sdnq  # noqa: F401
import torch
from transformers import AutoProcessor, PreTrainedTokenizerBase

from ..config import Settings
from ..logging_setup import LOGGER_NAME
from .artifacts import REQUIRED_FILES, SPECIAL_TOKENS
from .vendor.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

logger = logging.getLogger(LOGGER_NAME)


def _tokenizer_of(processor):
    """Processor có thể chính là tokenizer,

    hoặc bọc một tokenizer bên trong.
    """
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def attach_special_tokens(processor) -> None:
    """Gắn 5 token đặc biệt mà pipeline đọc trực tiếp trên tokenizer."""
    tokenizer = _tokenizer_of(processor)
    for attr, value in SPECIAL_TOKENS.items():
        setattr(tokenizer, attr, value)


def resolve_model_path(settings: Settings) -> str:
    """Trả về thư mục model local, hoặc repo id trên Hub nếu chưa tải."""
    if settings.model_path_local:
        local = Path(settings.model_path_local)
        missing = [f for f in REQUIRED_FILES if not (local / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"IMG_MODEL_PATH trỏ tới {local} nhưng thiếu {missing}. "
                "Kiểm tra mount ./models trong docker-compose.yml, hoặc "
                "chạy lại scripts/download_model.py."
            )
        logger.info("Dùng model local: %s", local)
        return str(local)

    # Chế độ hỏng âm thầm nguy hiểm nhất của service: chỉ cần
    # .model_paths.env không được nạp (hoặc compose quên set biến) là model
    # vẫn nạp bình thường nhưng tải lại ~10GB vào cache. Đặt HF_HUB_OFFLINE=1
    # để biến nó thành lỗi dừng hẳn thay vì một lần tải im lặng.
    logger.warning(
        "Model sẽ tải từ HuggingFace Hub (%s) chứ KHÔNG đọc từ đĩa. "
        "HF_HOME=%s | HF_HUB_OFFLINE=%s",
        settings.base_model,
        os.environ.get("HF_HOME", "<default>"),
        os.environ.get("HF_HUB_OFFLINE", "0"),
    )
    return settings.base_model


def load_model(settings: Settings, device: str):
    """Nạp ``(model, processor)``. Gọi MỘT LẦN lúc khởi động.

    ``device_map=device`` thay cho cả module ``models/offload.py`` cũ: model
    chỉ ~10.9 GiB nên vừa trọn một card 24GB, không còn bài toán chia
    text_encoder / transformer / vae qua nhiều GPU hay offload xuống CPU.
    """
    path = resolve_model_path(settings)
    logger.info(
        "Nạp %s lên %s | type=%s steps=%d cfg=%s",
        path,
        device,
        settings.model_type,
        settings.num_steps,
        settings.guidance_scale,
    )

    processor = AutoProcessor.from_pretrained(path)
    attach_special_tokens(processor)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map=device,
        # Nói rõ thay vì để transformers tự đoán. Đường sinh ảnh nhanh
        # (PATCH 4 trong vendor) gọi thẳng F.scaled_dot_product_attention
        # nên không đọc cờ này, NHƯNG vision encoder của ảnh tham chiếu và
        # đường mask dự phòng thì có — để mặc định "eager" ở đó là tự trả
        # thêm tiền cho mỗi request có ảnh vào.
        attn_implementation="sdpa",
    ).eval()

    strategy = apply_weight_tuning(model, settings, device)
    model = maybe_compile(model, settings)

    logger.info(
        "Model sẵn sàng trên %s | trọng số: %s | attention: %s",
        model.device,
        strategy,
        settings.attention_mode,
    )
    return model, processor


# ---------------------------------------------------------------------------
# Tối ưu tốc độ ở tầng trọng số
# ---------------------------------------------------------------------------
#
# SDNQ là quantizer "dequant-on-the-fly": mỗi lớp Linear GIẢI NÉN TOÀN BỘ ma
# trận trọng số của nó ở MỖI forward pass rồi mới gọi F.linear. Với model
# ~8B param và 50 bước × 2 forward pass (CFG), đó là ~100 lần giải nén lại
# cùng một bộ trọng số — chi phí băng thông bộ nhớ thuần tuý, không sinh ra
# thông tin gì mới.
#
# Hai đòn bẩy, theo thứ tự hiệu quả:
#   1. Giải nén MỘT LẦN lúc nạp (``_dequantize_to_bf16``) — xoá sạch chi phí
#      trên. Đổi lại VRAM tăng ~7GiB, nên chỉ làm khi card còn chỗ.
#   2. Nếu không đủ chỗ: ép scale/zero_point về bf16 (``_shrink_scales``).
#      Checkpoint lưu ``dequantize_fp32: true`` nên scale là fp32, kéo cả
#      phép giải nén chạy ở fp32 — gấp đôi lưu lượng bộ nhớ cho một độ chính
#      xác vô nghĩa trên trọng số 4-bit.

#: VRAM chừa lại cho activation + workspace khi quyết định có giải nén không.
#: 2048² ≈ 4150 token, batch 1, không KV-cache: đỉnh activation đo được nằm
#: quanh 2GiB. 3GiB là con số đó cộng biên an toàn.
DEQUANT_HEADROOM_BYTES = 3 * 1024**3


def _sdnq_layers(model):
    """Mọi lớp còn đang giữ trọng số ở dạng nén."""
    try:
        from sdnq.layers import SDNQLayer
    except ImportError:  # sdnq đổi layout module
        return []
    return [
        m
        for m in model.modules()
        if isinstance(m, SDNQLayer) and hasattr(m, "sdnq_dequantizer")
    ]


def _dequantize_cost(layers) -> int:
    """Số byte VRAM TĂNG THÊM nếu giải nén hết về dtype đích."""
    extra = 0
    for layer in layers:
        deq = layer.sdnq_dequantizer
        numel = 1
        for dim in deq.original_shape:
            numel *= int(dim)
        after = numel * torch.empty((), dtype=deq.result_dtype).element_size()
        before = layer.weight.numel() * layer.weight.element_size()
        for name in ("scale", "zero_point", "svd_up", "svd_down"):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                before += tensor.numel() * tensor.element_size()
        extra += after - before
    return extra


def _shrink_scales(layers) -> int:
    """scale/zero_point fp32 → dtype của kết quả (bf16). Trả về số lớp đã đổi.

    ``dequantize_asymmetric`` tính ``addcmul(zero_point, w.to(scale.dtype),
    scale)`` — tức là dtype của SCALE quyết định dtype của cả phép giải nén.
    Checkpoint này lưu ``dequantize_fp32: true`` nên scale là fp32: mỗi
    forward pass ghi ra một bản fp32 của toàn bộ trọng số rồi mới ép về
    bf16. Hạ scale xuống bf16 cắt đôi lưu lượng đó.

    Sai số thêm vào là một lần làm tròn bf16 trên tích ``w*scale`` — trong
    khi bản thân trọng số đã là uint4 (sai số ~3%). Đặt
    ``IMG_SDNQ_FP32_SCALES=true`` để quay lại fp32 nếu cần đối chiếu.
    """
    if os.environ.get("IMG_SDNQ_FP32_SCALES", "").strip().lower() == "true":
        logger.info("IMG_SDNQ_FP32_SCALES=true — giữ scale fp32.")
        return 0

    changed = 0
    for layer in layers:
        target = layer.sdnq_dequantizer.result_dtype
        if target not in (torch.bfloat16, torch.float16):
            continue
        touched = False
        for name in ("scale", "zero_point"):
            tensor = getattr(layer, name, None)
            if tensor is None or tensor.dtype != torch.float32:
                continue
            _replace_tensor_attr(layer, name, tensor.to(target))
            touched = True
        changed += int(touched)
    return changed


def _replace_tensor_attr(module, name: str, value) -> None:
    """Gán lại một tensor trên module, GIỮ NGUYÊN kiểu vỏ của nó.

    SDNQ lưu ``scale``/``zero_point`` bằng
    ``torch.nn.Parameter(..., requires_grad=False)``. ``setattr`` một
    ``Tensor`` trần lên tên đang nằm trong ``module._parameters`` thì
    ``nn.Module.__setattr__`` ném ``TypeError: cannot assign ... as
    parameter`` — và nó ném lúc NẠP MODEL, tức là sau khi đã đọc xong ~10GB
    weight. Hàm này tồn tại chỉ để chuyện đó không xảy ra.
    """
    if name in module._parameters:
        param = torch.nn.Parameter(value, requires_grad=False)
        # transformers đánh dấu này để biết tensor đã được nạp, không cần
        # init lại. Mất nó là mất luôn dấu.
        param._is_hf_initialized = getattr(
            module._parameters[name], "_is_hf_initialized", False
        )
        module._parameters[name] = param
    elif name in module._buffers:
        module._buffers[name] = value
    else:
        setattr(module, name, value)


def _dequantize_to_bf16(model) -> None:
    """Giải nén trọng số SDNQ tại chỗ, một lần, lúc nạp."""
    from sdnq.dequantizer import dequantize_sdnq_model

    dequantize_sdnq_model(model)
    # ``SDNQLayer.dequantize()`` tạo Parameter với requires_grad=True.
    # Service chỉ chạy inference: để nguyên là mọi forward pass đều dựng
    # autograd graph cho ~8B param.
    model.requires_grad_(False)


def apply_weight_tuning(model, settings: Settings, device: str) -> str:
    """Chọn và áp dụng chiến lược trọng số. Trả về nhãn để log/benchmark."""
    mode = (settings.dequantize or "auto").strip().lower()
    layers = _sdnq_layers(model)
    if not layers:
        return "bf16 (model không lượng tử hoá)"

    if mode in {"off", "false", "no", "quantized"}:
        decision, reason = False, "config dequantize=off"
    elif mode in {"bf16", "on", "true", "full"}:
        decision, reason = True, "config dequantize=bf16"
    else:
        extra = _dequantize_cost(layers)
        try:
            free, total = torch.cuda.mem_get_info(torch.device(device))
        except Exception:  # noqa: BLE001 — device lạ/không phải CUDA
            free = total = 0
        needed = extra + DEQUANT_HEADROOM_BYTES
        decision = free >= needed
        reason = (
            f"auto: cần thêm {extra / 1024**3:.1f}GiB + "
            f"{DEQUANT_HEADROOM_BYTES / 1024**3:.0f}GiB biên, còn trống "
            f"{free / 1024**3:.1f}/{total / 1024**3:.1f}GiB"
        )

    if decision:
        logger.info("Giải nén trọng số SDNQ về bf16 (%s)...", reason)
        started = time.time()
        _dequantize_to_bf16(model)
        try:
            used = torch.cuda.memory_allocated(torch.device(device)) / 1024**3
        except Exception:  # noqa: BLE001
            used = 0.0
        logger.info(
            "Giải nén xong trong %.1fs — không còn chi phí dequant mỗi "
            "forward pass. VRAM đang dùng: %.1fGiB",
            time.time() - started,
            used,
        )
        return "bf16 (giải nén lúc nạp)"

    changed = _shrink_scales(layers)
    logger.info(
        "Giữ trọng số ở dạng nén SDNQ (%s). Hạ scale về bf16 ở %d/%d lớp. "
        "Mỗi forward pass vẫn phải giải nén lại toàn bộ trọng số — đặt "
        "dequantize: bf16 trong base.yaml nếu card đủ VRAM.",
        reason,
        changed,
        len(layers),
    )
    return "SDNQ 4-bit (dequant mỗi forward pass)"


def maybe_compile(model, settings: Settings):
    """torch.compile cho phần decoder, nếu config bật.

    KHÔNG bật mặc định. Lần chạy đầu tiên tốn vài phút biên dịch và Inductor
    biên dịch lại cho MỖI hình dạng đầu vào mới — mà độ phân giải thì đổi
    theo tỉ lệ ảnh người dùng (11 giá trị). Chỉ có lãi cho worker chạy dài ở
    một kích thước cố định.
    """
    if not settings.compile_model:
        return model
    try:
        target = model.model.language_model
        target.forward = torch.compile(
            target.forward, dynamic=False, fullgraph=False
        )
        logger.info(
            "torch.compile đã bật cho decoder — lượt sinh ảnh ĐẦU TIÊN ở mỗi "
            "độ phân giải sẽ chậm hơn nhiều do biên dịch."
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Bật torch.compile không thành (%s); bỏ qua.", exc)
    return model

