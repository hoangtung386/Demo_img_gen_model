"""Image-editing inference helper."""
from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any

import torch
from diffusers import QwenImageEditPlusPipeline, QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    calculate_dimensions,
)
from PIL import Image

logger = logging.getLogger("qwen-image-edit-2509-lightning")

# Diện tích ảnh ra mặc định, khớp VAE_IMAGE_SIZE của pipeline.
DEFAULT_OUTPUT_AREA = 1024 * 1024

# Text encoder là Qwen2.5-VL: nó encode prompt KÈM ảnh điều kiện, nên khoá
# cache bắt buộc phải gồm cả ảnh, không chỉ text. Mỗi entry ~7MB VRAM.
_EMBED_CACHE: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = (
    OrderedDict()
)
_EMBED_CACHE_MAX = 8

# Cache riêng cho text-to-image: ở đó encode_prompt KHÔNG nhận ảnh nên khoá
# chỉ là prompt. Trộn chung một cache với bản edit sẽ cho hai embedding khác
# nhau cùng khoá khi prompt trùng mà một bên có ảnh điều kiện.
_T2I_EMBED_CACHE: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = (
    OrderedDict()
)


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


def _condition_images(
    pipeline: QwenImageEditPlusPipeline, images: list[Image.Image]
) -> list[Image.Image]:
    """Resize ảnh y hệt cách pipeline làm trước khi đưa vào text encoder.

    Phải dùng đúng ``calculate_dimensions`` + ``image_processor.resize`` của
    pipeline, nếu lệch một pixel thì embeds cache được sẽ khác embeds thật.
    """
    resized = []
    for img in images:
        width, height = img.size
        cond_w, cond_h = calculate_dimensions(
            CONDITION_IMAGE_SIZE, width / height
        )
        resized.append(pipeline.image_processor.resize(img, cond_h, cond_w))
    return resized


def _cache_key(prompt: str, condition_images: list[Image.Image]) -> str:
    """Khoá cache = prompt + nội dung mọi ảnh điều kiện."""
    digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=16)
    for img in condition_images:
        digest.update(img.tobytes())
    return digest.hexdigest()


def _encode_prompt_cached(
    pipeline: QwenImageEditPlusPipeline,
    prompt: str,
    pil_images: list[Image.Image],
) -> tuple[tuple[torch.Tensor, torch.Tensor], bool]:
    """Trả ``((embeds, mask), đã_cache)``.

    Đổi seed mà giữ nguyên ảnh và prompt — thao tác lặp nhiều nhất khi thử
    model — sẽ ăn cache và bỏ qua hẳn lượt chạy Qwen2.5-VL.
    """
    condition_images = _condition_images(pipeline, pil_images)
    key = _cache_key(prompt, condition_images)

    cached = _EMBED_CACHE.get(key)
    if cached is not None:
        _EMBED_CACHE.move_to_end(key)
        return cached, True

    embeds, mask = pipeline.encode_prompt(
        image=condition_images,
        prompt=prompt,
        device=pipeline._execution_device,
        num_images_per_prompt=1,
        max_sequence_length=1024,
    )
    _EMBED_CACHE[key] = (embeds, mask)
    while len(_EMBED_CACHE) > _EMBED_CACHE_MAX:
        _EMBED_CACHE.popitem(last=False)
    return (embeds, mask), False


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


def generate(
    pipeline: QwenImageEditPlusPipeline,
    images: list[Any],
    prompt: str,
    true_cfg_scale: float,
    seed: int,
    num_steps: int,
    negative_prompt: str | None = None,
    output_area: int = DEFAULT_OUTPUT_AREA,
    match_input_size: bool = False,
) -> tuple[Image.Image | None, str]:
    """Run the editing pipeline and return ``(image, status_message)``.

    Ảnh CUỐI trong ``images`` là ảnh nền: pipeline lấy tỉ lệ khung từ
    ``image[-1]``, và mô hình coi ảnh cuối là ảnh chính cần chỉnh sửa. Ảnh
    tham chiếu (trang phục, phòng mẫu) phải đứng TRƯỚC nó.

    ``negative_prompt`` chỉ thực sự được dùng khi ``true_cfg_scale > 1.0``;
    ở mức 1.0 (mặc định của Lightning) CFG bị tắt.

    ``match_input_size=True`` thu/phóng ảnh ra về ĐÚNG kích thước pixel của
    ảnh nền. Cần cho những task chỉ sửa một vùng nhỏ (face swap) mà người
    dùng muốn ảnh ra thay thế được ảnh vào. Lưu ý đây là phép resize sau khi
    sinh, KHÔNG phải sinh ở độ phân giải gốc: model vẫn chạy ở
    ``output_area``, nên ảnh 12MP không vì thế mà có thêm chi tiết thật.
    """
    images = [img for img in (images or []) if img is not None]
    if not images:
        return None, "Please upload at least one image."
    if not prompt or not prompt.strip():
        return None, "Please enter a prompt."

    pil_images = [_to_pil(img) for img in images]

    # Khung đầu ra bám ảnh nền. Truyền tường minh thay vì để pipeline tự suy
    # từ image[-1] — chi tiết nội bộ đó đã từng khiến ảnh ra mang tỉ lệ của
    # ảnh tham chiếu khi thứ tự ảnh bị đảo.
    base_w, base_h = pil_images[-1].size
    width, height = calculate_dimensions(output_area, base_w / base_h)

    if seed is None or seed < 0:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    generator = torch.Generator(device="cpu").manual_seed(seed)

    logger.info(
        "Generating with %d image(s), out=%dx%d, prompt=%r, cfg=%s, seed=%d",
        len(pil_images),
        width,
        height,
        prompt[:80],
        true_cfg_scale,
        seed,
    )

    start = time.time()
    with torch.inference_mode():
        (prompt_embeds, prompt_embeds_mask), was_cached = (
            _encode_prompt_cached(pipeline, prompt, pil_images)
        )
    text_s = time.time() - start

    step_marks: list[float] = []

    def _on_step(pipe, step, timestep, kwargs):  # noqa: ANN001
        step_marks.append(time.time())
        return kwargs

    extra: dict[str, Any] = {}
    if negative_prompt and negative_prompt.strip():
        extra["negative_prompt"] = negative_prompt.strip()

    pipe_start = time.time()
    with torch.inference_mode():
        output = pipeline(
            image=pil_images,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            height=height,
            width=width,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_steps,
            generator=generator,
            callback_on_step_end=_on_step,
            **extra,
        )
    end = time.time()
    elapsed = end - start
    result = output.images[0]

    # calculate_dimensions làm tròn hai cạnh về bội của 32, nên tỉ lệ khung
    # sinh ra lệch khỏi tỉ lệ ảnh gốc tối đa ~1%. Resize thẳng về kích thước
    # gốc kéo lại đúng 1% đó — mắt không thấy, và đổi lại ảnh ra khớp pixel
    # với ảnh vào. Cắt cho khỏi méo thì mất nội dung ở rìa, tệ hơn.
    if match_input_size and result.size != (base_w, base_h):
        result = result.resize((base_w, base_h), Image.LANCZOS)

    denoise_s, per_step, prep_s, decode_s = _split_timings(
        step_marks, pipe_start, end
    )
    text_label = (
        f"{text_s:.1f}s (cache)" if was_cached else f"{text_s:.1f}s"
    )
    # `elapsed` đo từ TRƯỚC encode_prompt đến SAU khi pipeline trả kết quả —
    # thuần thời gian tính toán trên GPU. Nó KHÔNG gồm: upload ảnh lên server,
    # hàng đợi Gradio, mã hoá PNG trả về, và độ trễ tunnel *.gradio.live (do
    # HuggingFace host, thường là phần lớn nhất của độ trễ cảm nhận được).
    # Nhãn "GPU" ở đầu dòng là để không ai nhầm con số này với round-trip.
    # Báo kích thước THẬT của ảnh trả về; nếu nó khác kích thước sinh thì nói
    # rõ cả hai, để không ai tưởng model chạy ở độ phân giải ảnh gốc.
    if result.size == (width, height):
        size_label = f"{width}×{height}"
    else:
        size_label = (
            f"{result.width}×{result.height} (sinh ở {width}×{height})"
        )
    info = (
        f"⚡ GPU {elapsed:.1f}s  =  denoise {denoise_s:.1f}s "
        f"({per_step:.2f}s/bước) + text {text_label} + "
        f"prep {prep_s:.1f}s + decode {decode_s:.1f}s\n"
        f"{size_label} · {num_steps} bước · cfg {true_cfg_scale} · "
        f"seed {seed}"
    )
    logger.info(info.replace("\n", " | "))
    return result, info


def _encode_t2i_cached(
    pipeline: QwenImagePipeline, prompt: str
) -> tuple[tuple[torch.Tensor, torch.Tensor], bool]:
    """Như ``_encode_prompt_cached`` nhưng cho text-to-image (không có ảnh)."""
    key = hashlib.blake2b(
        prompt.encode("utf-8"), digest_size=16
    ).hexdigest()

    cached = _T2I_EMBED_CACHE.get(key)
    if cached is not None:
        _T2I_EMBED_CACHE.move_to_end(key)
        return cached, True

    embeds, mask = pipeline.encode_prompt(
        prompt=prompt,
        device=pipeline._execution_device,
        num_images_per_prompt=1,
        max_sequence_length=1024,
    )
    _T2I_EMBED_CACHE[key] = (embeds, mask)
    while len(_T2I_EMBED_CACHE) > _EMBED_CACHE_MAX:
        _T2I_EMBED_CACHE.popitem(last=False)
    return (embeds, mask), False


def generate_t2i(
    pipeline: QwenImagePipeline,
    prompt: str,
    true_cfg_scale: float,
    seed: int,
    num_steps: int,
    negative_prompt: str | None = None,
    output_area: int = DEFAULT_OUTPUT_AREA,
    aspect_ratio: float = 1.0,
) -> tuple[Image.Image | None, str]:
    """Sinh ảnh từ chữ, không có ảnh đầu vào.

    Khác ``generate()`` ở ba chỗ, đều do không có ảnh điều kiện:
      - khung ảnh ra suy từ ``aspect_ratio`` người dùng chọn chứ không bám
        theo ảnh nền;
      - ``encode_prompt`` không nhận ``image``, nên cache khoá theo prompt;
      - pipeline là ``QwenImagePipeline`` (xem ``models.loader``).

    ``negative_prompt`` chỉ có tác dụng khi ``true_cfg_scale > 1.0``.
    """
    if not prompt or not prompt.strip():
        return None, "Vui lòng nhập prompt mô tả ảnh cần sinh."

    width, height = calculate_dimensions(output_area, aspect_ratio)

    if seed is None or seed < 0:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
    generator = torch.Generator(device="cpu").manual_seed(seed)

    logger.info(
        "T2I out=%dx%d, prompt=%r, cfg=%s, seed=%d",
        width,
        height,
        prompt[:80],
        true_cfg_scale,
        seed,
    )

    start = time.time()
    with torch.inference_mode():
        (prompt_embeds, prompt_embeds_mask), was_cached = _encode_t2i_cached(
            pipeline, prompt
        )
    text_s = time.time() - start

    step_marks: list[float] = []

    def _on_step(pipe, step, timestep, kwargs):  # noqa: ANN001
        step_marks.append(time.time())
        return kwargs

    extra: dict[str, Any] = {}
    if negative_prompt and negative_prompt.strip():
        extra["negative_prompt"] = negative_prompt.strip()

    pipe_start = time.time()
    with torch.inference_mode():
        output = pipeline(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            height=height,
            width=width,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_steps,
            generator=generator,
            callback_on_step_end=_on_step,
            **extra,
        )
    end = time.time()
    elapsed = end - start
    result = output.images[0]

    denoise_s, per_step, prep_s, decode_s = _split_timings(
        step_marks, pipe_start, end
    )
    text_label = f"{text_s:.1f}s (cache)" if was_cached else f"{text_s:.1f}s"
    info = (
        f"⚡ GPU {elapsed:.1f}s  =  denoise {denoise_s:.1f}s "
        f"({per_step:.2f}s/bước) + text {text_label} + "
        f"prep {prep_s:.1f}s + decode {decode_s:.1f}s\n"
        f"{width}×{height} · {num_steps} bước · cfg {true_cfg_scale} · "
        f"seed {seed}"
    )
    logger.info(info.replace("\n", " | "))
    return result, info
