"""UI phải dựng được mà KHÔNG cần model.

Đây là bất biến giữ cho ``ui/app.py`` (dựng widget) tách khỏi ``ui/launch.py``
(nạp model, mở cổng, dựng tunnel). Mất nó thì không còn cách nào kiểm giao
diện trên máy không có GPU, và hai mối quan tâm sẽ từ từ trộn lại.
"""

from __future__ import annotations

import os

import pytest

# Gradio gọi về api.gradio.app lúc import/khởi tạo. Tắt trước khi import để
# test không phụ thuộc mạng.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# Số Block mà giao diện 7 tab dựng ra. Con số chính xác không quan trọng bằng
# việc nó ỔN ĐỊNH: đổi UI thì cập nhật, nhưng một thay đổi ngoài ý muốn (mất
# tab, mất khối ví dụ) sẽ làm nó lệch và test đỏ.
EXPECTED_BLOCKS = 183


@pytest.fixture(scope="module")
def demo():
    from gen_image.ui.app import build_ui

    # pipeline=None: build_ui chỉ đóng gói nó vào closure của nút bấm, không
    # gọi tới lúc dựng. Đây chính là tính chất test này bảo vệ.
    return build_ui(None, num_steps=4, demo_cache=False)


def test_build_ui_needs_no_model(demo):
    import gradio as gr

    assert isinstance(demo, gr.Blocks)


def test_block_count_is_stable(demo):
    assert len(demo.blocks) == EXPECTED_BLOCKS, (
        f"UI dựng ra {len(demo.blocks)} block thay vì {EXPECTED_BLOCKS}. "
        "Nếu bạn cố ý đổi giao diện thì cập nhật EXPECTED_BLOCKS; nếu không "
        "thì có thứ gì đó đã biến mất khỏi UI."
    )


def test_app_module_does_not_import_torch():
    """``ui/app.py`` không được kéo theo torch.

    Phần nạp model nằm ở ``launch.py``. Nếu ``app.py`` import torch trở lại
    thì ranh giới đã vỡ, và test dựng UI ở trên sẽ chỉ còn đúng do may mắn.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    source = (root / "src/gen_image/ui/app.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "load_pipeline" not in source
