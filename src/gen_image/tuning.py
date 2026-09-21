"""Tinh chỉnh tốc độ ở tầng torch.

Tách riêng khỏi ``models/placement.py``: placement quyết định component nằm
ở ĐÂU, còn file này quyết định torch CHẠY THẾ NÀO khi mọi thứ đã nằm đúng
chỗ.
"""

from __future__ import annotations

import logging
import time

import torch

logger = logging.getLogger("gen-image")

# Ảnh warm-up: xám trung tính, tỉ lệ vuông. Nội dung không quan trọng — mục
# tiêu là ép mọi kernel được nạp và mọi pool bộ nhớ được cấp phát.
_WARMUP_SIZE = 512


def apply_gpu_tuning() -> None:
    """Bật các cờ torch có lợi, bỏ những cờ chỉ có hại ở đây."""
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


def maybe_compile(pipeline, enabled: bool, quantization: str) -> None:
    """``torch.compile`` transformer, nếu được bật VÀ nhánh cho phép.

    Hai rào chắn, cả hai đều là giới hạn thật chứ không phải phòng xa:

    1. **GGUF không compile được.** Weight được giải nén trong forward bằng
       kernel của ``gguf``; Dynamo không trace qua được (diffusers#10795).
       Bật compile ở nhánh đó chỉ đổi lấy một traceback dài.
    2. **``dynamic=True`` là bắt buộc.** Chiều ảnh ra suy từ tỉ lệ ảnh người
       dùng nên gần như mỗi request một shape mới. Compile ở chế độ static
       sẽ recompile lại từ đầu cho từng shape — chi phí đó lớn hơn hẳn phần
       tiết kiệm được, biến một tối ưu thành một hồi quy.

    Để mặc định TẮT. Bật bằng ``compile_transformer: true`` sau khi đã đo
    thật trên đúng phần cứng đó với ``TORCH_LOGS=recompiles`` để xác nhận số
    lần recompile hội tụ về 0 sau vài request đầu.
    """
    if not enabled:
        return
    if quantization == "gguf":
        logger.warning(
            "Bỏ qua torch.compile: nhánh quantization='gguf' không trace "
            "được qua kernel giải nén GGUF (diffusers#10795). Dùng "
            "quantization='bf16' nếu muốn compile."
        )
        return

    logger.info("torch.compile transformer (dynamic=True) ...")
    started = time.time()
    # ``module.compile()`` (in-place) chứ KHÔNG phải
    # ``pipeline.transformer = torch.compile(pipeline.transformer)``.
    # DiffusionPipeline.__setattr__ bắt mọi phép gán lên một component đã
    # đăng ký rồi gọi register_to_config() — tức là ghi đè entry
    # "transformer" trong config thành class wrapper của Dynamo
    # (``OptimizedModule``). Config đó là thứ đi vào model_index.json khi
    # save_pretrained, và là thứ ``pipeline.components`` trả về.
    pipeline.transformer.compile(
        # max-autotune-no-cudagraphs: cudagraph giả định địa chỉ bộ nhớ vào
        # ra cố định giữa các lần gọi, mà allocator ở đây chạy
        # expandable_segments và mỗi request một shape — hai thứ đó không đi
        # cùng nhau được.
        mode="max-autotune-no-cudagraphs",
        dynamic=True,
    )
    logger.info(
        "torch.compile đã gắn sau %.1fs — chi phí biên dịch thật sẽ trả ở "
        "lượt warm-up ngay sau đây.",
        time.time() - started,
    )


def warmup(pipeline, num_steps: int, output_area: int, guidance_scale: float) -> float:
    """Chạy hai lượt sinh ảnh giả để trả trước mọi chi phí khởi tạo.

    Lượt sinh ảnh đầu tiên luôn đắt hơn hẳn các lượt sau: nạp kernel, dựng
    CUDA context, cho allocator mở rộng pool, trả chi phí gọi hàm lần đầu của
    diffusers/transformers, và — nếu bật — toàn bộ phần biên dịch của
    ``torch.compile``. Không warm-up thì người dùng ĐẦU TIÊN gánh khoản đó.

    Chạy CẢ HAI đường, vì chúng đi qua những nhánh code khác nhau:
      - text-to-image: không đụng VAE encode;
      - image-edit: có thêm lượt VAE-encode ảnh tham chiếu.

    Trả về số giây đã tốn. Không bao giờ ném lỗi ra ngoài: warm-up hỏng thì
    service vẫn phải phục vụ được.
    """
    from PIL import Image

    from .inference import generate

    started = time.time()
    try:
        image = Image.new("RGB", (_WARMUP_SIZE, _WARMUP_SIZE), (127, 127, 127))
        for images in ([], [image]):
            generate(
                pipeline,
                images,
                "warmup",
                guidance_scale,
                0,
                num_steps,
                output_area=output_area,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up thất bại (%s); bỏ qua.", exc)
        return 0.0

    elapsed = time.time() - started
    logger.info("Warm-up xong trong %.1fs — request đầu tiên sẽ nhanh.", elapsed)
    return elapsed
