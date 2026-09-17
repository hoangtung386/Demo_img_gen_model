"""Bảy tab của demo, mỗi tab một module.

Tách khỏi ``app.py`` vì chúng là bảy biến thể của cùng một khuôn (~120 dòng
mỗi cái) và gộp chung làm file đó dài gần 1000 dòng — không ai đọc hết để
sửa một ô nhập.

Mỗi module export đúng một hàm ``build(ctx: TabContext) -> None`` tự dựng cả
``gr.Tab`` của nó. Thêm tab mới: tạo module, thêm vào ``ALL`` bên dưới.
"""

from __future__ import annotations

from . import (
    face_swap,
    home_design,
    hug_two_people,
    image_and_prompt,
    image_to_cartoon,
    prompt_to_image,
    virtual_tryon,
)

#: Thứ tự hiển thị trên UI.
ALL = (
    virtual_tryon,
    home_design,
    image_to_cartoon,
    hug_two_people,
    face_swap,
    prompt_to_image,
    image_and_prompt,
)

__all__ = [
    "ALL",
    "face_swap",
    "home_design",
    "hug_two_people",
    "image_and_prompt",
    "image_to_cartoon",
    "prompt_to_image",
    "virtual_tryon",
]
