"""Ảnh kết quả dựng sẵn cho khối ví dụ của UI.

Tách khỏi ``app.py`` vì đây là một mối quan tâm riêng: nó không dựng widget
nào, chỉ ánh xạ {nội dung đầu vào → ảnh kết quả trên đĩa}. ``app.py`` chỉ gọi
``build_index()`` một lần rồi tra ``serve_cached()`` khi trúng khoá.

Ảnh kết quả do ``scripts/warm_examples.py`` sinh ra trước; ở đây chỉ nạp lại
nên tester bấm ví dụ là thấy ngay, không phải đợi model chạy. Thiếu file nào
thì case đó bị bỏ qua — demo vẫn chạy bình thường.
"""

from __future__ import annotations

import hashlib
import logging
import time

from PIL import Image

from . import examples_spec as spec
from . import prompts

logger = logging.getLogger("gen-image")

# Thời gian tối thiểu một lượt "chạy" ở chế độ demo. Không trả ảnh tức thì vì
# như vậy trông như lỗi giao diện chứ không như một lượt sinh ảnh.
DEMO_MIN_SECONDS = 0.3


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
    return hashlib.blake2b(image.convert("RGB").tobytes(), digest_size=16).digest()


def cache_key(images, *texts) -> str:
    """Khoá tra ảnh dựng sẵn: nội dung ảnh đầu vào + phần chữ đi kèm."""
    digest = hashlib.blake2b(digest_size=16)
    for image in images:
        digest.update(_image_key(image))
    for text in texts:
        digest.update(str(text or "").strip().encode("utf-8"))
    return digest.hexdigest()


def build_index() -> dict[str, str]:
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
        lambda c: cache_key([spec.image_path(c[0]), spec.image_path(c[1])]),
    )
    add("home", spec.HOME_CASES, lambda c: cache_key([spec.image_path(c[0])], c[1]))
    add(
        "cartoon",
        spec.CARTOON_CASES,
        lambda c: cache_key(
            [spec.image_path(c[0])], prompts.build_cartoon_prompt(c[1])
        ),
    )
    add(
        "hug",
        spec.HUG_CASES,
        lambda c: cache_key([spec.image_path(c[0]), spec.image_path(c[1])]),
    )
    add(
        "faceswap",
        spec.FACESWAP_CASES,
        lambda c: cache_key(
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
