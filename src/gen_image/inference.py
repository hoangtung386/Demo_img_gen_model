"""Sinh ảnh: text-to-image và image-edit qua cùng một hàm."""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import OrderedDict
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger("gen-image")

# Diện tích ảnh ra mặc định (pixel). 1024×1024 là điểm pipeline được tinh
# chỉnh quanh đó, và cũng là trần mà Flux2KleinPipeline tự thu ảnh tham
# chiếu xuống trước khi VAE-encode.
DEFAULT_OUTPUT_AREA = 1024 * 1024

# FLUX.2 dùng max_sequence_length 512 (Qwen3-8B). Backend cũ truyền 1024 —
# con số đó thuộc về Qwen2.5-VL và sẽ bị pipeline này cắt bớt.
MAX_SEQUENCE_LENGTH = 512

# Cache prompt-embed. Text encoder của klein-9B là Qwen3-8B **text-only**: ảnh
# tham chiếu đi vào model qua VAE latent, KHÔNG qua text encoder. Nên khoá
# cache chỉ cần prompt — khác hẳn backend cũ (Qwen2.5-VL encode prompt KÈM
# ảnh, buộc phải băm cả pixel của từng ảnh vào khoá). Hệ quả thực tế: đổi
# ảnh mà giữ nguyên prompt — thao tác lặp nhiều nhất trong production — giờ
# ăn cache, trước đây thì không.
#
# Cache NẰM TRONG TIẾN TRÌNH, cố ý: một lượt tra dict mất micro-giây, còn
# một vòng qua Redis mất mili-giây cộng chi phí serialize một tensor vài MB.
# Cache ngoài chỉ đáng khi có NHIỀU process cùng phục vụ và muốn chia sẻ —
# cấu hình hiện tại là một process một GPU (xem docs/MULTI_PROCESS_WORKERS.md),
# nên chia sẻ không có ai để chia.
_EMBED_CACHE: OrderedDict[str, torch.Tensor] = OrderedDict()

# Số prompt giữ lại. Mỗi entry là một tensor [1, seq, dim] bf16 — với
# max_sequence_length=512 thì cỡ vài MB, nên nâng lên hàng trăm vẫn rẻ so
# với 24GB VRAM. Đặt qua `embed_cache_size` trong base.yaml.
#
# Ở bản 9B cache này đáng giá hơn hẳn bản 4B: text encoder to gấp đôi, và
# khi chạy NF4 thì mỗi lượt encode còn phải giải nén weight trong forward.
_EMBED_CACHE_MAX = 8


def _to_pil(item: Any) -> Image.Image:
    """Chuẩn hoá một ảnh đầu vào về ``PIL.Image`` RGB.

    Gradio có thể trả về ``(ảnh, caption)`` hoặc đường dẫn file tạm thay vì
    đối tượng PIL.
    """
    if isinstance(item, (tuple, list)):
        item = item[0]
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    return Image.open(item).convert("RGB")


def calculate_dimensions(area: int, ratio: float, multiple_of: int) -> tuple[int, int]:
    """Khung ảnh ra có diện tích ≈ ``area``, tỉ lệ ``ratio``, cạnh chia hết.

    Tự viết thay vì import ``calculate_dimensions`` của diffusers. Hàm cũ nằm
    trong ``diffusers.pipelines.qwenimage`` — một module *internal*, và chính
    kiểu phụ thuộc đó là thứ từng ghim cả repo vào ``diffusers==0.36``. Mười
    dòng ở đây đổi lấy quyền nâng diffusers tự do.

    ``multiple_of`` do pipeline quyết định (``vae_scale_factor * 2``), truyền
    vào chứ không hard-code: VAE đổi thì số này đổi theo.
    """
    ratio = max(ratio, 1e-6)
    width = math.sqrt(area * ratio)
    height = width / ratio

    def _round(value: float) -> int:
        # max(...) chặn cạnh 0 khi ảnh vào có tỉ lệ cực đoan (panorama rất
        # dài): làm tròn xuống bội số sẽ ra 0 và pipeline nổ ở prepare_latents.
        return max(multiple_of, int(round(value / multiple_of)) * multiple_of)

    return _round(width), _round(height)


def _pipeline_multiple_of(pipeline) -> int:
    """Bội số mà cả hai cạnh phải chia hết, đọc từ chính pipeline."""
    return int(getattr(pipeline, "vae_scale_factor", 8)) * 2


def _encode_prompt_cached(pipeline, prompt: str) -> tuple[torch.Tensor, bool]:
    """Trả ``(prompt_embeds, đã_cache)``.

    Đổi seed mà giữ nguyên prompt — thao tác lặp nhiều nhất khi thử model —
    sẽ ăn cache và bỏ qua hẳn lượt chạy Qwen3-8B.

    Chỉ cache ``prompt_embeds``, không cache ``text_ids``: truyền
    ``prompt_embeds`` vào ``__call__`` khiến ``encode_prompt`` đi vào nhánh
    short-circuit và dựng lại ``text_ids`` bằng ``_prepare_text_ids()`` —
    một phép tạo tensor chỉ số, rẻ hơn nhiều so với việc giữ thêm một tensor
    trong cache.
    """
    key = hashlib.blake2b(prompt.encode("utf-8"), digest_size=16).hexdigest()

    cached = _EMBED_CACHE.get(key)
    if cached is not None:
        _EMBED_CACHE.move_to_end(key)
        return cached, True

    embeds, _text_ids = pipeline.encode_prompt(
        prompt=prompt,
        device=pipeline._execution_device,
        num_images_per_prompt=1,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    _EMBED_CACHE[key] = embeds
    while len(_EMBED_CACHE) > _EMBED_CACHE_MAX:
        _EMBED_CACHE.popitem(last=False)
    return embeds, False


def clear_embed_cache() -> None:
    """Xoá cache prompt-embed (dùng khi đo benchmark thuần)."""
    _EMBED_CACHE.clear()


def configure_embed_cache(max_entries: int) -> None:
    """Đặt sức chứa cache prompt-embed.

    Gọi MỘT LẦN lúc khởi động, trước khi phục vụ request. Hạ sức chứa sẽ
    loại bớt entry cũ ngay để trạng thái luôn khớp cấu hình — không để một
    cache 500 entry còn sống sau khi ai đó hạ config xuống 10.
    """
    global _EMBED_CACHE_MAX
    _EMBED_CACHE_MAX = max(1, int(max_entries))
    while len(_EMBED_CACHE) > _EMBED_CACHE_MAX:
        _EMBED_CACHE.popitem(last=False)
    logger.info("Cache prompt-embed: tối đa %d prompt", _EMBED_CACHE_MAX)


def _split_timings(
    step_marks: list[float], pipe_start: float, end: float
) -> tuple[float, float, float, float]:
    """Tách thời gian pipeline thành (denoise, mỗi bước, chuẩn bị, decode).

    ``callback_on_step_end`` chỉ chạy SAU khi một bước kết thúc, nên mốc đầu
    tiên đã nằm sau bước 1: khoảng pipe_start -> mốc đầu = chuẩn bị CỘNG một
    bước denoise, và các mốc chỉ bao được num_steps-1 bước. Phải trừ ra, nếu
    không phần chuẩn bị bị thổi lên gấp nhiều lần giá trị thật.
    """
    if len(step_marks) < 2:
        return 0.0, 0.0, end - pipe_start, 0.0
    per_step = (step_marks[-1] - step_marks[0]) / (len(step_marks) - 1)
    denoise_s = per_step * len(step_marks)
    prep_s = step_marks[0] - pipe_start - per_step
    decode_s = end - step_marks[-1]
    return denoise_s, per_step, prep_s, decode_s


def _cfg_active(pipeline, guidance_scale: float) -> bool:
    """Pipeline có thực sự chạy CFG hai nhánh với ``guidance_scale`` này không.

    Tự tính thay vì đọc ``pipeline.do_classifier_free_guidance``. Property đó
    của diffusers đọc ``self._guidance_scale``, mà field này chỉ được gán ở
    đầu ``__call__`` — đọc nó TRƯỚC lần gọi đầu tiên sẽ ném AttributeError.
    Công thức ở đây sao chép đúng property đó
    (``guidance_scale > 1 and not is_distilled``).
    """
    if guidance_scale <= 1:
        return False
    return not getattr(pipeline.config, "is_distilled", False)


def _negative_embeds(
    pipeline, negative_prompt: str | None, guidance_scale: float
) -> torch.Tensor | None:
    """Embed negative prompt, CHỈ khi pipeline thực sự chạy CFG.

    Bản klein distilled nhúng guidance thẳng vào model thay vì chạy hai
    nhánh, nên CFG luôn tắt. Trong trạng thái đó negative prompt không có
    đường nào tác động tới ảnh ra.

    Trả None và ghi log thay vì im lặng bỏ qua: BE vẫn gửi
    ``negative_prompt`` trong inbound message (hợp đồng không đổi), và người
    chỉnh prompt cần biết vì sao nó không có tác dụng.
    """
    text = (negative_prompt or "").strip()
    if not text:
        return None
    if not _cfg_active(pipeline, guidance_scale):
        logger.info(
            "negative_prompt bị bỏ qua: model distilled không chạy CFG "
            "(guidance nhúng sẵn). Đặt base_model sang bản 'klein-base' và "
            "guidance_scale > 1.0 nếu thực sự cần negative prompt."
        )
        return None
    embeds, _ = pipeline.encode_prompt(
        prompt=text,
        device=pipeline._execution_device,
        num_images_per_prompt=1,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    return embeds


def generate(
    pipeline,
    images: list[Any] | None,
    prompt: str,
    guidance_scale: float,
    seed: int,
    num_steps: int,
    negative_prompt: str | None = None,
    output_area: int = DEFAULT_OUTPUT_AREA,
    aspect_ratio: float = 1.0,
    match_input_size: bool = False,
) -> tuple[Image.Image | None, str]:
    """Chạy pipeline và trả ``(ảnh, thông điệp trạng thái)``.

    MỘT hàm cho cả hai đường, vì ``Flux2KleinPipeline`` nhận ``image=None``:

      - ``images`` rỗng → text-to-image. Khung ảnh ra suy từ ``aspect_ratio``.
      - ``images`` có phần tử → image-edit. Khung ảnh ra bám ảnh CUỐI.

    Ảnh CUỐI trong ``images`` là ảnh nền: khung hình lấy theo tỉ lệ của nó,
    và model coi ảnh cuối là ảnh chính cần chỉnh sửa. Ảnh tham chiếu (trang
    phục, phòng mẫu) phải đứng TRƯỚC nó. FLUX.2 nhận tối đa 10 ảnh.

    ``negative_prompt`` chỉ có tác dụng khi pipeline chạy CFG thật — bản
    distilled thì không, xem ``_negative_embeds``.

    ``match_input_size=True`` thu/phóng ảnh ra về ĐÚNG kích thước pixel của
    ảnh nền. Cần cho những task chỉ sửa một vùng nhỏ (face swap) mà người
    dùng muốn ảnh ra thay thế được ảnh vào. Lưu ý đây là phép resize sau khi
    sinh, KHÔNG phải sinh ở độ phân giải gốc: model vẫn chạy ở
    ``output_area``, nên ảnh 12MP không vì thế mà có thêm chi tiết thật.
    """
    if not prompt or not prompt.strip():
        return None, "Vui lòng nhập prompt."

    pil_images = [_to_pil(img) for img in (images or []) if img is not None]

    multiple_of = _pipeline_multiple_of(pipeline)
    if pil_images:
        base_w, base_h = pil_images[-1].size
        ratio = base_w / base_h
    else:
        base_w = base_h = 0
        ratio = aspect_ratio
    width, height = calculate_dimensions(output_area, ratio, multiple_of)

    if seed is None or seed < 0:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    generator = torch.Generator(device="cpu").manual_seed(seed)

    logger.info(
        "Sinh ảnh: %d ảnh vào, out=%dx%d, prompt=%r, guidance=%s, seed=%d",
        len(pil_images),
        width,
        height,
        prompt[:80],
        guidance_scale,
        seed,
    )

    start = time.time()
    with torch.inference_mode():
        prompt_embeds, was_cached = _encode_prompt_cached(pipeline, prompt)
        negative_embeds = _negative_embeds(pipeline, negative_prompt, guidance_scale)
    text_s = time.time() - start

    step_marks: list[float] = []

    def _on_step(pipe, step, timestep, kwargs):  # noqa: ANN001
        step_marks.append(time.time())
        return kwargs

    extra: dict[str, Any] = {}
    if negative_embeds is not None:
        extra["negative_prompt_embeds"] = negative_embeds

    pipe_start = time.time()
    with torch.inference_mode():
        output = pipeline(
            image=pil_images or None,
            prompt_embeds=prompt_embeds,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            callback_on_step_end=_on_step,
            **extra,
        )
    end = time.time()
    elapsed = end - start
    result = output.images[0]

    # calculate_dimensions làm tròn hai cạnh về bội của `multiple_of`, nên tỉ
    # lệ khung sinh ra lệch khỏi tỉ lệ ảnh gốc một chút. Resize thẳng về kích
    # thước gốc kéo lại đúng phần đó — mắt không thấy, và đổi lại ảnh ra khớp
    # pixel với ảnh vào. Cắt cho khỏi méo thì mất nội dung ở rìa, tệ hơn.
    if match_input_size and base_w and result.size != (base_w, base_h):
        result = result.resize((base_w, base_h), Image.LANCZOS)

    denoise_s, per_step, prep_s, decode_s = _split_timings(step_marks, pipe_start, end)
    text_label = f"{text_s:.1f}s (cache)" if was_cached else f"{text_s:.1f}s"
    # `elapsed` đo từ TRƯỚC encode_prompt đến SAU khi pipeline trả kết quả —
    # thuần thời gian tính toán trên GPU. Nó KHÔNG gồm: upload ảnh lên server,
    # hàng đợi Gradio, mã hoá PNG trả về, và độ trễ tunnel. Nhãn "GPU" ở đầu
    # dòng là để không ai nhầm con số này với round-trip.
    #
    # Cấu trúc chuỗi này đi THẲNG vào `result.info` của outbound message —
    # đổi format là đổi hợp đồng với BE.
    if result.size == (width, height):
        size_label = f"{width}×{height}"
    else:
        size_label = f"{result.width}×{result.height} (sinh ở {width}×{height})"
    info = (
        f"⚡ GPU {elapsed:.1f}s  =  denoise {denoise_s:.1f}s "
        f"({per_step:.2f}s/bước) + text {text_label} + "
        f"prep {prep_s:.1f}s + decode {decode_s:.1f}s\n"
        f"{size_label} · {num_steps} bước · guidance {guidance_scale} · "
        f"seed {seed}"
    )
    logger.info(info.replace("\n", " | "))
    return result, info
