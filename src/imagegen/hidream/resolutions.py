"""Bảng độ phân giải cố định của HiDream-O1-Image.

Bản sao thuần-stdlib của ``PREDEFINED_RESOLUTIONS`` và
``find_closest_resolution`` trong ``vendor/utils.py``. Tồn tại vì hai lý do:

1. ``vendor/utils.py`` import ``torch``. UI, queue config và test cần biết
   model sinh ra kích thước nào mà không phải kéo theo ~3GB wheel CUDA —
   nếu không, một test kiểm tra "1024 snap về 2048" sẽ không chạy nổi trên
   máy CI không GPU.
2. Đây là hằng số quyết định giá mỗi request (2048² gấp 4 lần pixel so với
   1024² của model cũ). Nó đáng được đặt ở chỗ đọc được, không chôn trong
   thư mục vendor mà quy ước là không ai sửa.

``tests/test_resolution.py`` so bảng này với bản vendor và fail nếu upstream
đổi — không có chuyện hai bên trôi khỏi nhau trong im lặng.
"""

from __future__ import annotations

# Nguồn: HiDream-O1-Image models/utils.py::PREDEFINED_RESOLUTIONS
PREDEFINED_RESOLUTIONS: tuple[tuple[int, int], ...] = (
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2560, 1440),
    (1440, 2560),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
)

# Cạnh ngắn nhỏ nhất model sinh được. KHÔNG có đường nào ra ảnh nhỏ hơn: mọi
# (width, height) đều bị snap về bảng trên. Hằng số này tồn tại để UI và
# test khỏi phải tự suy ra.
MIN_SHORT_SIDE = min(min(r) for r in PREDEFINED_RESOLUTIONS)


def find_closest_resolution(width: int, height: int) -> tuple[int, int]:
    """Độ phân giải trong bảng có tỉ lệ gần ``width/height`` nhất.

    CHỈ tỉ lệ được tính đến — kích thước yêu cầu bị bỏ qua hoàn toàn. Yêu
    cầu 512×512 và 4096×4096 đều cho ra 2048×2048.
    """
    target = width / height
    return min(
        PREDEFINED_RESOLUTIONS, key=lambda wh: abs(wh[0] / wh[1] - target)
    )


def size_for_aspect_ratio(ratio: float, base: int = 2048) -> tuple[int, int]:
    """Tỉ lệ khung (rộng/cao) → một độ phân giải trong bảng.

    Message của queue chỉ gửi ``aspect_ratio``, không gửi kích thước — và
    cũng không cần, vì kích thước không phải thứ chọn được. ``base`` chỉ là
    chiều cao quy chiếu để dựng cặp (w, h) trước khi snap; đổi nó không đổi
    kết quả, chỉ đổi độ chính xác làm tròn.

    Giá trị không hợp lệ (0, âm, None) coi như vuông thay vì nổ: một
    ``aspect_ratio`` sai trong payload không đáng làm hỏng cả request.
    """
    if not ratio or ratio <= 0:
        ratio = 1.0
    return find_closest_resolution(max(1, int(base * ratio)), base)
