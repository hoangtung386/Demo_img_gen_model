"""Tinh chỉnh tốc độ ở tầng torch cho một GPU đơn.

Tách riêng khỏi phần nạp model: ``hidream/loader.py`` quyết định model nằm
ở ĐÂU, file này quyết định torch CHẠY THẾ NÀO khi mọi thứ đã nằm đúng chỗ.
"""

from __future__ import annotations

import logging
import time

import torch

from .config import Settings
from .logging_setup import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def apply_gpu_tuning() -> None:
    """Bật các cờ torch có lợi, bỏ những cờ chỉ có hại ở đây."""
    if not torch.cuda.is_available():
        logger.warning("Không thấy CUDA; bỏ qua phần tinh chỉnh GPU.")
        return

    # TF32 cho mọi matmul/conv fp32 còn sót lại (model chạy bf16 nên phần
    # này nhỏ, nhưng miễn phí và không đổi kết quả ở mức nhìn thấy được).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Cùng ý nghĩa với hai dòng trên nhưng phủ cả những đường mới của torch
    # (F.linear, torch.compile/Inductor) vốn đọc cờ này chứ không đọc
    # allow_tf32.
    torch.set_float32_matmul_precision("high")

    # Cho phép reduction bf16 tích luỹ ở độ chính xác thấp hơn. Mặc định của
    # torch là True, nhưng một số image đặt lại; nói rõ ở đây để không phải
    # đoán vì sao cùng code lại chạy chậm hơn trên máy khác.
    _mm = torch.backends.cuda.matmul
    if hasattr(_mm, "allow_bf16_reduced_precision_reduction"):
        _mm.allow_bf16_reduced_precision_reduction = True

    # Backend của scaled_dot_product_attention. Đường sinh ảnh nhanh gọi SDPA
    # KHÔNG mask (xem PATCH 4 trong vendor/qwen3_vl_transformers.py) nên
    # backend flash dùng được — nó nhanh hơn mem-efficient và không cần
    # materialize [batch, heads, seq, seq] nào cả. Bật đủ ba để torch vẫn có
    # đường lui khi một shape nào đó flash không nhận.
    for name, enable in (
        ("enable_flash_sdp", True),
        ("enable_mem_efficient_sdp", True),
        ("enable_math_sdp", True),
    ):
        fn = getattr(torch.backends.cuda, name, None)
        if fn is not None:
            fn(enable)

    # KHÔNG bật torch.backends.cudnn.benchmark.
    #
    # benchmark=True autotune thuật toán conv cho TỪNG shape đầu vào rồi mới
    # cache lại. Model snap kích thước về 11 độ phân giải khác nhau tuỳ tỉ lệ
    # ảnh người dùng, nên shape vẫn đổi giữa các request: chi phí autotune
    # phải trả đi trả lại và không bao giờ khấu hao hết.
    torch.backends.cudnn.benchmark = False

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(
        "GPU tuning: %s (sm_%d%d, %.1fGiB) | TF32 on | flash-SDPA on | "
        "cudnn.benchmark off",
        name,
        capability[0],
        capability[1],
        total,
    )


def warmup(model, processor, settings: Settings) -> float:
    """Chạy một lượt sinh ảnh cụt để trả trước mọi chi phí khởi tạo.

    Lượt sinh ảnh đầu tiên luôn đắt hơn hẳn các lượt sau: dựng CUDA context,
    cho allocator mở rộng pool, dequant lần đầu của SDNQ, autotune kernel, và
    chi phí gọi hàm lần đầu của transformers. Không warm-up thì người dùng
    ĐẦU TIÊN gánh toàn bộ khoản đó.

    Chi phí đó là CHI PHÍ MỖI-LƯỢT-ĐẦU, không phải chi phí mỗi bước: chạy 2
    bước trả trước đúng những gì chạy 50 bước trả trước. Vì vậy warm-up dùng
    ``settings.warmup_steps`` (mặc định 2) chứ không phải ``num_steps`` —
    bản trước tốn nguyên một request đầy đủ (hàng phút) chỉ để vứt ảnh đi.

    Độ phân giải thì PHẢI giữ đúng của config: một phần chi phí lần đầu
    (allocator pool, autotune) phụ thuộc hình dạng đầu vào.

    Trả về số giây đã tốn. Không bao giờ ném lỗi ra ngoài: warm-up hỏng thì
    service vẫn phải phục vụ được.
    """
    from .hidream import build_recipe, generate

    steps = max(1, settings.warmup_steps)
    logger.info(
        "Bắt đầu warm-up: %dx%d, %d bước (chỉ để trả trước chi phí khởi "
        "tạo — ảnh ra bị vứt đi)...",
        settings.width,
        settings.height,
        steps,
    )
    started = time.time()
    try:
        generate(
            model,
            processor,
            [],
            "a warmup image, minimalist",
            build_recipe(settings, num_steps=steps),
            seed=0,
            width=settings.width,
            height=settings.height,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up thất bại (%s); bỏ qua.", exc)
        return 0.0

    elapsed = time.time() - started
    logger.info(
        "Warm-up hoàn tất trong %.1fs — request đầu tiên sẽ nhanh hơn.",
        elapsed,
    )
    return elapsed
