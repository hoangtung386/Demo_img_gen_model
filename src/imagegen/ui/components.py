"""Widget dùng chung giữa các tab + chỉ mục ảnh kết quả dựng sẵn.

Tách khỏi ``app.py`` để file đó chỉ còn phần định nghĩa tab. Mọi thứ ở đây
độc lập với model: không import torch, không biết ``hidream``.
"""

from __future__ import annotations

import hashlib
import logging
import time

import gradio as gr
from PIL import Image

from ..hidream.resolutions import MIN_SHORT_SIDE, PREDEFINED_RESOLUTIONS
from ..logging_setup import LOGGER_NAME
from . import examples_spec as spec
from . import prompts

logger = logging.getLogger(LOGGER_NAME)


def _label_for(width: int, height: int) -> str:
    """Nhãn dropdown: tỉ lệ rút gọn + kích thước thật."""
    from math import gcd

    divisor = gcd(width, height)
    ratio = f"{width // divisor}:{height // divisor}"
    shape = (
        "vuông" if width == height else ("ngang" if width > height else "dọc")
    )
    return f"{ratio} {shape} — {width}×{height}"


# Độ phân giải là thứ DUY NHẤT người dùng chọn được về khung hình, và chỉ
# chọn được trong đúng 11 giá trị: find_closest_resolution() snap mọi yêu cầu
# khác về danh sách này. Không còn "preset nhanh/chậm" theo diện tích như bản
# Qwen cũ — model không sinh được ảnh nhỏ hơn, nên hạ độ phân giải KHÔNG còn
# là đòn bẩy tốc độ.
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    _label_for(w, h): (w, h) for w, h in PREDEFINED_RESOLUTIONS
}

DEFAULT_RESOLUTION_LABEL = next(iter(RESOLUTION_PRESETS))

# Thời gian tối thiểu một lượt "chạy" ở chế độ demo. Không trả ảnh tức thì vì
# như vậy trông như lỗi giao diện chứ không như một lượt sinh ảnh.
DEMO_MIN_SECONDS = 0.3

_STEP_RANGE = (8, 60)
_CFG_RANGE = (0.0, 10.0)
_SHIFT_RANGE = (1.0, 6.0)


# --------------------------------------------------------------------------
# Ảnh mẫu + chỉ mục kết quả dựng sẵn
# --------------------------------------------------------------------------
def sample_image(name: str) -> str | None:
    """Đường dẫn ảnh mẫu, hoặc ``None`` nếu chưa tải về."""
    path = spec.image_path(name)
    return str(path) if path.is_file() else None


def _image_key(image) -> bytes:
    """Vân tay nội dung một ảnh, không phụ thuộc đường dẫn file tạm."""
    if image is None:
        return b""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    return hashlib.blake2b(
        image.convert("RGB").tobytes(), digest_size=16
    ).digest()


def demo_cache_key(images, *texts) -> str:
    """Khoá tra ảnh dựng sẵn: nội dung ảnh đầu vào + phần chữ đi kèm."""
    digest = hashlib.blake2b(digest_size=16)
    for image in images:
        digest.update(_image_key(image))
    for text in texts:
        digest.update(str(text or "").strip().encode("utf-8"))
    return digest.hexdigest()


def build_demo_cache() -> dict[str, str]:
    """Lập chỉ mục {khoá đầu vào -> ảnh kết quả dựng sẵn} từ examples_spec."""
    index: dict[str, str] = {}

    def add(tab, cases, key_of):
        for position, case in enumerate(cases):
            output = spec.output_path(tab, position)
            if not output.is_file():
                continue
            try:
                index[key_of(case)] = str(output)
            except (OSError, ValueError):
                continue

    add(
        "vto",
        spec.VTO_CASES,
        lambda c: demo_cache_key(
            [spec.image_path(c[0]), spec.image_path(c[1])]
        ),
    )
    add(
        "home",
        spec.HOME_CASES,
        lambda c: demo_cache_key([spec.image_path(c[0])], c[1]),
    )
    add(
        "cartoon",
        spec.CARTOON_CASES,
        lambda c: demo_cache_key(
            [spec.image_path(c[0])], prompts.build_cartoon_prompt(c[1])
        ),
    )
    add(
        "hug",
        spec.HUG_CASES,
        lambda c: demo_cache_key(
            [spec.image_path(c[0]), spec.image_path(c[1])]
        ),
    )
    add(
        "faceswap",
        spec.FACESWAP_CASES,
        lambda c: demo_cache_key(
            [spec.image_path(c[0]), spec.image_path(c[1])],
            prompts.build_faceswap_prompt(c[2]),
        ),
    )
    return index


def serve_cached(path: str, started: float) -> tuple[Image.Image, str]:
    """Trả ảnh dựng sẵn, giữ nhịp tối thiểu cho giống một lượt chạy thật."""
    image = Image.open(path)
    image.load()
    remaining = DEMO_MIN_SECONDS - (time.time() - started)
    if remaining > 0:
        time.sleep(remaining)
    elapsed = time.time() - started
    logger.info("Demo cache hit: %s (%.2fs)", path, elapsed)
    # TUYỆT ĐỐI không định dạng giống một lượt chạy thật. Trước đây chỗ này in
    # "Done in 0.3s" y hệt lượt chạy thật, khiến người đo tưởng model sinh ảnh
    # trong 0.3s trong khi không có bước denoise nào chạy cả.
    return image, (
        f"{spec.CACHED_STATUS}\n"
        f"{image.width}×{image.height} · ảnh đọc từ đĩa trong {elapsed:.1f}s"
    )


def example_rows(cases, to_row) -> list[list]:
    """Các dòng cho ``gr.Examples`` — chỉ gồm cột đầu vào."""
    rows = []
    for case in cases:
        row = to_row(case)
        if any(value is None for value in row):
            continue
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Khối widget
# --------------------------------------------------------------------------
def params_block(
    default_steps: int, default_cfg: float, default_shift: float
) -> tuple[gr.Slider, gr.Number, gr.Slider, gr.Slider, gr.Dropdown]:
    """Các tham số lấy mẫu.

    Không còn ô negative prompt: HiDream-O1 không nhận negative prompt —
    nhánh uncond của CFG dùng prompt " " cố định. Giữ một ô không có tác
    dụng là bẫy cho người test.
    """
    with gr.Row():
        cfg = gr.Slider(
            label="guidance_scale (CFG)",
            minimum=_CFG_RANGE[0],
            maximum=_CFG_RANGE[1],
            value=default_cfg,
            step=0.5,
            info=(
                "Bản full chạy tốt nhất ở 5.0. Mỗi bước chạy HAI forward "
                "pass khi > 1.0 — đặt 0.0 để nhanh gấp đôi (bản dev)."
            ),
        )
        seed = gr.Number(label="Seed (-1 = ngẫu nhiên)", value=-1, precision=0)
    with gr.Row():
        steps = gr.Slider(
            label="Số bước",
            minimum=_STEP_RANGE[0],
            maximum=_STEP_RANGE[1],
            value=default_steps,
            step=1,
            info="50 cho bản full, 28 cho bản dev.",
        )
        shift = gr.Slider(
            label="shift",
            minimum=_SHIFT_RANGE[0],
            maximum=_SHIFT_RANGE[1],
            value=default_shift,
            step=0.5,
            info="Độ lệch lịch nhiễu. 3.0 cho bản full, 1.0 cho bản dev.",
        )
    resolution = gr.Dropdown(
        label="Độ phân giải đầu ra",
        choices=list(RESOLUTION_PRESETS),
        value=DEFAULT_RESOLUTION_LABEL,
        info=(
            f"Chỉ 11 giá trị này. Cạnh ngắn không xuống dưới "
            f"{MIN_SHORT_SIDE}px — hạ độ phân giải KHÔNG làm model chạy "
            "nhanh hơn đáng kể."
        ),
    )
    return cfg, seed, steps, shift, resolution


def params_accordion(
    default_steps: int, default_cfg: float, default_shift: float
) -> tuple[gr.Slider, gr.Number, gr.Slider, gr.Slider, gr.Dropdown]:
    """``params_block`` gói trong accordion — cho tab prompt tự do."""
    with gr.Accordion("Tuỳ chỉnh nâng cao", open=False):
        return params_block(default_steps, default_cfg, default_shift)


def advanced_block(
    default_prompt: str,
    default_steps: int,
    default_cfg: float,
    default_shift: float,
) -> tuple[
    gr.Textbox,
    gr.Slider,
    gr.Number,
    gr.Slider,
    gr.Slider,
    gr.Dropdown,
    gr.Button,
]:
    """Khối nâng cao cho các tab có prompt hệ thống sinh sẵn."""
    with gr.Accordion("Tuỳ chỉnh nâng cao", open=False):
        prompt_box = gr.Textbox(
            label="Prompt gửi cho model",
            value=default_prompt,
            lines=10,
            info=(
                "Được sinh sẵn theo task. Có thể sửa trực tiếp; bấm "
                "'Khôi phục prompt mặc định' để lấy lại bản gốc."
            ),
        )
        reset_btn = gr.Button("Khôi phục prompt mặc định", size="sm")
        cfg, seed, steps, shift, resolution = params_block(
            default_steps, default_cfg, default_shift
        )
    return prompt_box, cfg, seed, steps, shift, resolution, reset_btn


def output_block() -> tuple[gr.Image, gr.Textbox]:
    """Khối kết quả dùng chung cho mọi tab."""
    output_image = gr.Image(label="Kết quả", type="pil", height=560)
    status = gr.Textbox(
        label="Thời gian chạy thật (GPU)",
        info=(
            "Chỉ tính thời gian model chạy trên GPU. KHÔNG gồm upload ảnh, "
            "hàng đợi Gradio, mã hoá ảnh trả về và độ trễ tunnel "
            "*.gradio.live — đó mới là phần khiến bạn thấy lâu hơn con số này."
        ),
        lines=2,
        interactive=False,
    )
    return output_image, status
