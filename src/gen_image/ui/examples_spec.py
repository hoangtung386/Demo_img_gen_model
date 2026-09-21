"""Danh sách test case mẫu, dùng chung cho UI và script dựng sẵn kết quả.

``scripts/warm_examples.py`` chạy hết các case ở đây và ghi ảnh kết quả vào
``examples/outputs/``. UI đọc lại đúng những file đó, nên khi tester bấm vào
một ví dụ thì ảnh kết quả hiện ra ngay, không phải đợi model chạy.

Đổi danh sách này thì phải chạy lại script, nếu không ví dụ mới sẽ không có
ảnh dựng sẵn (UI vẫn hoạt động, chỉ là ô kết quả để trống).
"""

from __future__ import annotations

from ..config import PROJECT_ROOT

EXAMPLES_DIR = PROJECT_ROOT / "examples"
OUTPUTS_DIR = EXAMPLES_DIR / "outputs"

# Thông báo hiện ở ô Trạng thái khi nạp một ví dụ dựng sẵn — phải nói rõ đây
# không phải kết quả vừa chạy, tránh tester tưởng model chạy tức thời.
CACHED_STATUS = (
    "Kết quả dựng sẵn cho ví dụ này (không phải vừa chạy). "
    "Bấm nút chạy để model sinh lại ảnh mới."
)

# (ảnh trang phục, ảnh người mẫu)
VTO_CASES: list[tuple[str, str]] = [
    ("vto_garment.jpg", "vto_person.jpg"),
    ("vto_garment_2.jpg", "vto_person_2.jpg"),
    ("vto_garment_3.jpg", "vto_person_3.jpg"),
]

# (ảnh căn phòng, mô tả thiết kế)
HOME_CASES: list[tuple[str, str]] = [
    (
        "home_your_room.jpg",
        "Scandinavian style: light grey linen sofa, round oak coffee table, "
        "cream wool rug, tall green plants in ceramic pots, a warm floor "
        "lamp, open bookshelf, white and beige palette with light wood "
        "accents",
    ),
    (
        "home_room_2.jpg",
        "Cosy modern bedroom: low wooden bed with white linen, soft beige "
        "headboard, two rattan bedside tables with warm lamps, large woven "
        "rug, sheer curtains, a few framed prints, muted earth tones",
    ),
    (
        "home_room_3.jpg",
        "Warm industrial dining room: reclaimed wood table with black metal "
        "legs, six leather chairs, hanging Edison pendant lights, exposed "
        "brick accent, dark green plants, brass details",
    ),
]

# (ảnh chụp, nhãn phong cách trong CARTOON_STYLES)
CARTOON_CASES: list[tuple[str, str]] = [
    ("cartoon_people.jpg", "3D Pixar"),
    ("cartoon_people_2.jpg", "Anime Nhật Bản"),
    ("cartoon_people_3.jpg", "Disney vẽ tay"),
]

# (ảnh mặt đã crop, ảnh đầy đủ, nhãn phạm vi trong FACE_SWAP_SCOPES)
FACESWAP_CASES: list[tuple[str, str, str]] = [
    ("faceswap_face.jpg", "faceswap_photo.jpg", "Chỉ khuôn mặt (giữ tóc của ảnh gốc)"),
    (
        "faceswap_face_2.jpg",
        "faceswap_photo_2.jpg",
        "Chỉ khuôn mặt (giữ tóc của ảnh gốc)",
    ),
    ("faceswap_face_3.jpg", "faceswap_photo_3.jpg", "Khuôn mặt + tóc"),
]

# (ảnh người thứ nhất, ảnh người thứ hai)
HUG_CASES: list[tuple[str, str]] = [
    ("hug_person_a.jpg", "hug_person_b.jpg"),
    ("hug_person_c.jpg", "hug_person_d.jpg"),
    ("hug_person_b.jpg", "hug_person_c.jpg"),
]


def output_path(tab: str, index: int):
    """Đường dẫn ảnh dựng sẵn của case thứ ``index`` trong tab ``tab``."""
    return OUTPUTS_DIR / f"{tab}_{index + 1}.webp"


def image_path(name: str):
    """Đường dẫn tuyệt đối tới một ảnh mẫu."""
    return EXAMPLES_DIR / name
