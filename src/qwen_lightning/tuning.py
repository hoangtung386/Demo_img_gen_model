"""Tinh chỉnh tốc độ ở tầng torch cho một GPU đơn.

Tách riêng khỏi ``offload.py``: offload quyết định component nằm ở ĐÂU, còn
file này quyết định torch CHẠY THẾ NÀO khi mọi thứ đã nằm đúng chỗ.
"""
from __future__ import annotations

import logging
import time

import torch

logger = logging.getLogger("qwen-image-edit-2509-lightning")

# Ảnh warm-up: xám trung tính, tỉ lệ vuông. Nội dung không quan trọng — mục
# tiêu là ép mọi kernel được nạp và mọi pool bộ nhớ được cấp phát.
_WARMUP_SIZE = 512


def apply_gpu_tuning() -> None:
    """Bật các cờ torch có lợi cho A100, bỏ những cờ chỉ có hại ở đây."""
    if not torch.cuda.is_available():
        logger.warning("Không thấy CUDA; bỏ qua phần tinh chỉnh GPU.")
        return

    # TF32 cho mọi matmul/conv fp32 còn sót lại (pipeline chạy bf16 nên phần
    # này nhỏ, nhưng miễn phí và không đổi kết quả ở mức nhìn thấy được).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # KHÔNG bật torch.backends.cudnn.benchmark.
    #
    # benchmark=True autotune thuật toán conv cho TỪNG shape đầu vào rồi mới
    # cache lại. Ở đây kích thước ảnh ra do calculate_dimensions() suy từ tỉ
    # lệ ảnh người dùng tải lên, nên gần như mỗi request là một shape mới:
    # chi phí autotune phải trả đi trả lại và không bao giờ khấu hao hết.
    # Với shape cố định thì bật sẽ có lợi — ở đây thì không.
    torch.backends.cudnn.benchmark = False

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    logger.info(
        "GPU tuning: %s (sm_%d%d) | TF32 on | cudnn.benchmark off",
        name,
        capability[0],
        capability[1],
    )


def warmup(pipeline, num_steps: int, output_area: int) -> float:
    """Chạy một lượt sinh ảnh giả để trả trước mọi chi phí khởi tạo.

    Lượt sinh ảnh đầu tiên luôn đắt hơn hẳn các lượt sau: nạp cubin của
    nunchaku, dựng CUDA context, cho allocator mở rộng pool, và trả chi phí
    gọi hàm lần đầu của diffusers/transformers. Không warm-up thì người dùng
    ĐẦU TIÊN phải gánh toàn bộ khoản đó.

    Trả về số giây đã tốn. Không bao giờ ném lỗi ra ngoài: warm-up hỏng thì
    service vẫn phải phục vụ được.
    """
    from PIL import Image

    from .inference import generate

    started = time.time()
    try:
        image = Image.new("RGB", (_WARMUP_SIZE, _WARMUP_SIZE), (127, 127, 127))
        generate(
            pipeline,
            [image],
            "warmup",
            1.0,
            0,
            num_steps,
            output_area=output_area,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up thất bại (%s); bỏ qua.", exc)
        return 0.0

    elapsed = time.time() - started
    logger.info(
        "Warm-up xong trong %.1fs — request đầu tiên sẽ nhanh.", elapsed
    )
    return elapsed
