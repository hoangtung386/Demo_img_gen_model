"""Snap độ phân giải — hàm thuần, kiểm được không cần GPU."""

from __future__ import annotations

import pytest

from imagegen.hidream.resolutions import (
    MIN_SHORT_SIDE,
    PREDEFINED_RESOLUTIONS,
    find_closest_resolution,
    size_for_aspect_ratio,
)


@pytest.mark.parametrize(
    "width,height,expected",
    [
        # Yêu cầu 1024² của config cũ → model sinh 2048²: gấp 4 lần pixel.
        (1024, 1024, (2048, 2048)),
        (2048, 2048, (2048, 2048)),
        (1920, 1080, (2560, 1440)),
        (1080, 1920, (1440, 2560)),
        # Không có đường nào ra ảnh nhỏ hơn danh sách cứng.
        (512, 512, (2048, 2048)),
    ],
)
def test_snapping(width, height, expected):
    assert find_closest_resolution(width, height) == expected


def test_snapping_is_ratio_only():
    """Chỉ tỉ lệ quyết định kết quả; nhân đôi cả hai cạnh không đổi gì."""
    for w, h in [(800, 450), (1600, 900), (3200, 1800)]:
        assert find_closest_resolution(w, h) == find_closest_resolution(
            w * 2, h * 2
        )


def test_no_resolution_below_2048_long_side():
    """Chống hồi quy cho giả định "output_area điều khiển được kích thước".

    Config cũ có output_area=1024². Với model này nó vô nghĩa, nên field đó
    đã bị xoá khỏi ProcessorConfig. Test này giữ cho ai đó không thêm lại
    một preset "nhanh, 704px" vốn không thể tồn tại.
    """
    assert all(max(r) >= 2048 for r in PREDEFINED_RESOLUTIONS)
    assert MIN_SHORT_SIDE == min(min(r) for r in PREDEFINED_RESOLUTIONS)


def test_table_matches_vendor():
    """Bảng chép tay phải khớp bản upstream trong vendor/.

    Bỏ qua khi thiếu torch (vendor/utils.py import nó) — trên máy có đủ
    dependency thì test này là chốt chặn chống trôi sau mỗi lần chạy
    scripts/vendor_hidream.sh.
    """
    torch = pytest.importorskip("torch")
    assert torch is not None
    from imagegen.hidream.vendor import utils as vendor

    assert list(PREDEFINED_RESOLUTIONS) == [
        tuple(r) for r in vendor.PREDEFINED_RESOLUTIONS
    ]
    for wh in ((1024, 1024), (1920, 1080), (1080, 1920), (3000, 1300)):
        assert find_closest_resolution(*wh) == tuple(
            vendor.find_closest_resolution(*wh)
        )


@pytest.mark.parametrize(
    "ratio,expected",
    [
        (1.0, (2048, 2048)),
        (16 / 9, (2560, 1440)),
        (9 / 16, (1440, 2560)),
        (4 / 3, (2304, 1728)),
        (3 / 4, (1728, 2304)),
    ],
)
def test_size_for_aspect_ratio(ratio, expected):
    assert size_for_aspect_ratio(ratio) == expected


@pytest.mark.parametrize("bad", [0, -1.0, -0.5, None])
def test_bad_aspect_ratio_falls_back_to_square(bad):
    """aspect_ratio hỏng trong payload không được làm chết cả request.

    Message đến từ BE/App; một giá trị 0 hoặc âm lọt qua validate sẽ gây
    ZeroDivisionError tận trong pipeline nếu không chặn ở đây.
    """
    assert size_for_aspect_ratio(bad) == (2048, 2048)


def test_base_does_not_change_result():
    """`base` chỉ là số quy chiếu — đổi nó không đổi độ phân giải chọn ra."""
    for ratio in (1.0, 16 / 9, 3 / 4):
        assert size_for_aspect_ratio(ratio, base=512) == size_for_aspect_ratio(
            ratio, base=4096
        )
