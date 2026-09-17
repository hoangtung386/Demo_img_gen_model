"""Thứ mỗi tab cần để dựng chính nó.

Bảy tab trong ``ui/tabs/`` đều là biến thể của cùng một khuôn: vài ô nhập,
một khối tham số, một khối kết quả, một handler tra cache rồi gọi model.
``TabContext`` là toàn bộ phần chúng dùng chung — gom lại một chỗ để mỗi
module tab chỉ nhận đúng một đối số và không đụng tới ``Settings``, model
hay processor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

# (image | None, status_message)
Result = tuple[Image.Image | None, str]


@dataclass(frozen=True)
class TabContext:
    """Dependency mà một tab cần, đã đóng sẵn model + settings."""

    #: Chạy model với 0..N ảnh điều kiện. Chữ ký khớp ``app._run``.
    run: Callable[..., Result]

    #: Chạy model không ảnh (text-to-image). Chữ ký khớp ``app._run_t2i``.
    run_t2i: Callable[..., Result]

    #: {khoá đầu vào -> đường dẫn ảnh dựng sẵn}. Rỗng khi tắt demo cache.
    cache: dict[str, str] = field(default_factory=dict)

    #: Giá trị khởi tạo cho các widget tham số.
    num_steps: int = 50
    default_cfg: float = 5.0
    default_shift: float = 3.0

    def cached(self, key: str) -> str | None:
        """Ảnh dựng sẵn cho khoá này, nếu có."""
        return self.cache.get(key)


#: Chữ ký của mọi module tab: ``build(ctx)`` dựng widget vào Blocks hiện tại.
TabBuilder = Callable[[TabContext], Any]
