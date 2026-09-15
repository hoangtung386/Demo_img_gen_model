"""Prompt templates cho các bài toán demo.

Mỗi task có một "system prompt" riêng được viết sẵn bằng tiếng Anh — text
encoder của Qwen-Image-Edit bám sát chỉ dẫn tiếng Anh tốt hơn tiếng Việt, và
prompt được viết theo cùng một khung ba phần:

1. Hành động chính (đổi cái gì)
2. Ràng buộc bảo toàn (tuyệt đối không đổi cái gì) — phần quyết định chất lượng
3. Yêu cầu về độ chân thực (ánh sáng, bóng đổ, phối cảnh)

``negative prompt`` chỉ có tác dụng khi ``true_cfg_scale > 1.0``; ở mức 1.0
mặc định của Lightning, CFG bị tắt nên nhánh negative không được tính.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Tab 1 — Virtual Try-On
# --------------------------------------------------------------------------

# Ánh xạ nhãn tiếng Việt trên UI sang cụm danh từ tiếng Anh nhét vào prompt.
GARMENT_TYPES: dict[str, str] = {
    "Tự động (toàn bộ trang phục)": "outfit",
    "Áo (top / áo thun / sơ mi)": "upper garment",
    "Quần": "trousers",
    "Váy / chân váy": "skirt",
    "Đầm liền": "dress",
    "Áo khoác": "outer jacket or coat",
    "Bộ đồ nguyên bộ": "full outfit",
}

VTO_PROMPT_TEMPLATE = (
    "Image 1 shows a garment. Image 2 is a photo of a person.\n"
    "Edit image 2: replace ONLY the {garment} worn by that person with the "
    "exact {garment} from image 1. Image 2 is the base photo and must stay "
    "the output frame.\n"
    "Keep the person of image 2 completely unchanged: identical face and "
    "facial features, identical facial expression, identical hairstyle and "
    "hair colour, identical skin tone and skin texture, identical body shape, "
    "proportions and height, identical hands and fingers, identical pose, "
    "body orientation and camera angle. Keep the original background, "
    "lighting direction, colour temperature and shadows of image 2 intact.\n"
    "Transfer the garment from image 1 with faithful fidelity: same colour, "
    "same fabric texture, same pattern, print, logo and graphic placement, "
    "same neckline and collar, same sleeve length, same hem length, same "
    "buttons, zippers and overall cut. Do not redesign, restyle, recolour or "
    "simplify the garment. Ignore the background, the pose and any model "
    "shown in image 1 — take the clothing only.\n"
    "Fit the garment naturally onto the body with correct perspective and "
    "scale, realistic folds, wrinkles and drape following the person's pose, "
    "and soft contact shadows where the fabric meets the body.\n"
    "Do not change any other clothing item, footwear, accessory or object in "
    "the photo. Photorealistic result, sharp fabric detail, natural "
    "transitions at the neck, wrists and waist.\n"
    "The result must be the person of image 2 wearing the new {garment} — "
    "never a copy of image 1."
)

VTO_NEGATIVE = (
    "different face, changed identity, altered skin tone, whitened skin, "
    "changed hairstyle, distorted body, wrong proportions, extra limbs, "
    "deformed hands, extra fingers, changed pose, changed camera angle, "
    "changed background, redesigned garment, wrong garment colour, missing "
    "print, floating clothes, blurry, low quality, watermark, text, cartoon"
)


def build_vto_prompt(garment_label: str) -> str:
    """Prompt thử đồ ảo cho loại trang phục đã chọn."""
    garment = GARMENT_TYPES.get(garment_label, "outfit")
    return VTO_PROMPT_TEMPLATE.format(garment=garment)


# --------------------------------------------------------------------------
# Tab 2 — Home Design
# --------------------------------------------------------------------------

ROOM_TYPES: dict[str, str] = {
    "Tự động (giữ nguyên công năng phòng)": "room",
    "Phòng khách": "living room",
    "Phòng ngủ": "bedroom",
    "Bếp": "kitchen",
    "Phòng ăn": "dining room",
    "Phòng làm việc": "home office",
    "Phòng tắm": "bathroom",
    "Phòng trẻ em": "kid's room",
}

# Prompt NGẮN có chủ đích. Bản dài trước đây (12 câu ràng buộc kiến trúc) làm
# model 4 bước lạc và trả về gần như nguyên ảnh tham chiếu — kiểm chứng bằng
# ảnh phòng thật. Tab này giờ chỉ nhận MỘT ảnh (phòng gốc); phong cách do
# người dùng mô tả bằng chữ, nên không còn ảnh nào để model copy nhầm.
HOME_PROMPT_TEMPLATE = (
    "Keep this photo of a {room} exactly as it is: same walls, same windows "
    "and doors, same floor, same ceiling, same camera angle and "
    "perspective.\n"
    "Remove the old furniture, decor and clutter, then furnish and decorate "
    "that same room following the design brief below.\n"
    "Only the furniture and decor may change, never the room itself. Place "
    "everything at realistic scale on the existing floor, with shadows "
    "matching the daylight already in the photo.\n"
    "Photorealistic interior photograph."
)

# Mô tả mẫu, dùng cho ô ví dụ của tab Home Design.
HOME_DESIGN_EXAMPLE = (
    "Scandinavian style: light grey linen sofa, round oak coffee table, "
    "cream wool rug, tall green plants in ceramic pots, a warm floor lamp, "
    "open bookshelf, white and beige palette with light wood accents"
)

HOME_NEGATIVE = (
    "changed room layout, moved or resized windows, added windows, removed "
    "doors, warped walls, curved straight lines, tilted horizon, different "
    "camera angle, different room shape, distorted perspective, wrong scale "
    "furniture, floating furniture, furniture clipping through walls, "
    "duplicated room, cluttered, blurry, low quality, watermark, text"
)


def build_home_prompt(room_label: str) -> str:
    """Khung hệ thống cho tab Home Design (chưa gồm mô tả của người dùng)."""
    room = ROOM_TYPES.get(room_label, "room")
    return HOME_PROMPT_TEMPLATE.format(room=room)


def compose_home_prompt(system_prompt: str, design: str) -> str:
    """Ghép khung hệ thống với bản mô tả thiết kế người dùng tự viết."""
    design = (design or "").strip()
    if not design:
        return system_prompt
    return f"{system_prompt}\nDesign brief: {design}"


# --------------------------------------------------------------------------
# Tab 4 — Hai người ôm nhau
# --------------------------------------------------------------------------

HUG_PROMPT_TEMPLATE = (
    "Image 1 shows one person. Image 2 shows another person.\n"
    "Create one single natural photo of these two people standing together "
    "and hugging each other warmly.\n"
    "Keep both people exactly recognisable: same face and facial features, "
    "same hairstyle, hair colour and length, same skin tone, same body build "
    "and height, same clothing, clothing colours and accessories as in their "
    "own photo.\n"
    "Natural friendly hug with arms wrapped around each other, correct human "
    "anatomy, hands with five fingers, both faces clearly visible and "
    "undistorted, feet on the ground, consistent lighting, shadows and colour "
    "temperature on both people so they belong to the same photo.\n"
    "Exactly two people — nobody added, removed or duplicated. Clean simple "
    "background, photorealistic result."
)

HUG_NEGATIVE = (
    "different faces, unrecognisable people, changed hairstyle, changed "
    "clothing, distorted anatomy, deformed hands, extra fingers, extra arms, "
    "missing limbs, three or more people, only one person, merged bodies, "
    "fused faces, floating limbs, mismatched lighting, blurry, low quality, "
    "watermark, text"
)


def build_hug_prompt() -> str:
    """Prompt ghép hai người vào cùng một ảnh, đang ôm nhau."""
    return HUG_PROMPT_TEMPLATE


# --------------------------------------------------------------------------
# Tab 3 — Image to Cartoon
# --------------------------------------------------------------------------

CARTOON_STYLES: dict[str, str] = {
    "3D Pixar": "3D Pixar-style animated movie",
    "Anime Nhật Bản": "Japanese anime",
    "Disney vẽ tay": "Disney-style hand-drawn animation",
    "2D vector phẳng": "2D flat vector illustration",
    "Chibi dễ thương": "cute chibi cartoon",
    "Truyện tranh Mỹ": "American comic book",
    "Màu nước hoạt hình": "watercolour storybook cartoon",
}

CARTOON_PROMPT_TEMPLATE = (
    "Convert this photograph into a {style} artwork.\n"
    "Turn every person in the photo into a {style} cartoon character. For "
    "each person preserve their identity as closely as possible: same face "
    "shape and recognisable facial features, same eye shape and eye colour, "
    "same hairstyle, hair colour, hair length and parting, same facial hair, "
    "same skin tone, same age and gender, same body build and height, same "
    "pose, gesture and gaze direction, same clothing design, clothing colours "
    "and accessories such as glasses, hat, bag, watch or jewellery. Keep the "
    "exact number of people and their exact positions, scale and overlapping "
    "order in the frame — do not add, remove, merge or swap anyone.\n"
    "Re-render the surrounding scenery in the same {style} so the picture "
    "reads as one single coherent artwork: keep the original composition, "
    "camera angle, depth and horizon, keep the landmarks, buildings, "
    "vegetation, ground, sky, weather and lighting direction recognisable, "
    "only restyled — never replaced by a different place.\n"
    "Appealing character design, clean confident line work, natural and "
    "pleasant colour palette, soft consistent shading, high quality "
    "illustration, sharp and clean rendering."
)

CARTOON_NEGATIVE = (
    "different face, unrecognisable person, changed hairstyle, changed hair "
    "colour, changed skin tone, changed clothing, changed accessories, extra "
    "people, missing people, merged faces, distorted anatomy, deformed hands, "
    "extra fingers, changed background, different location, empty background, "
    "photorealistic face on cartoon body, inconsistent style, blurry, low "
    "quality, watermark, text, signature"
)


def build_cartoon_prompt(style_label: str) -> str:
    """Prompt chuyển ảnh chụp thành tranh hoạt hình."""
    style = CARTOON_STYLES.get(style_label, "3D Pixar-style animated movie")
    return CARTOON_PROMPT_TEMPLATE.format(style=style)


# --------------------------------------------------------------------------
# Tab 5 — Face Swap
# --------------------------------------------------------------------------

# Phạm vi hoán đổi. Mặc định CHỈ khuôn mặt: tóc thuộc về ảnh nền, đổi cả tóc
# là cách nhanh nhất để lộ đường ghép ở chân tóc khi ảnh nền có tóc che trán,
# mũ hoặc tóc bay.
FACE_SWAP_SCOPES: dict[str, str] = {
    "Chỉ khuôn mặt (giữ tóc của ảnh gốc)": (
        "Swap the facial region only - forehead, eyes, eyebrows, nose, "
        "mouth, cheeks, chin and jawline. Keep the hairstyle, hair colour "
        "and hairline of image 2 exactly as they are."
    ),
    "Khuôn mặt + tóc": (
        "Swap the face together with the hair: take the hairstyle, hair "
        "colour and hairline from image 1 as well, and fit them to the head "
        "shape, head angle and lighting of image 2."
    ),
}

# Prompt viết theo khung ba phần của module, nhưng ở task này phần 2 (bảo
# toàn) nặng hơn hẳn: người dùng muốn MỘT vùng đổi và toàn bộ phần còn lại
# đứng yên, nên mọi thứ không phải khuôn mặt đều được liệt kê tường minh.
#
# Ba câu giữa là phần hay bị bỏ sót và cũng là phần quyết định ảnh có trông
# như ghép hay không:
#   - "do not paste image 1 flat": model rất hay dán thẳng ảnh crop vào,
#     giữ nguyên góc mặt của ảnh crop trong khi đầu ở ảnh nền đang nghiêng.
#   - relight: ảnh crop thường chụp ở ánh sáng khác hẳn ảnh nền.
#   - khớp tông da với cổ/tai/tay: chỗ lộ đường ghép rõ nhất.
FACESWAP_PROMPT_TEMPLATE = (
    "Image 1 is a close-up crop of a face. Image 2 is a full photograph of "
    "a person in a scene.\n"
    "Edit image 2: replace the face of the person in image 2 with the face "
    "from image 1. Image 2 is the base photo and must stay the output "
    "frame.\n"
    "{scope}\n"
    "Carry over the identity of image 1 faithfully: same face shape and "
    "bone structure, same eye shape, eye colour and eye spacing, same "
    "eyebrows, same nose, same mouth and lips, same chin and jawline, same "
    "skin texture, freckles and moles, same apparent age and gender. The "
    "result must be recognisable as the person of image 1.\n"
    "Do not paste image 1 flat - rebuild that face inside image 2: match "
    "the head orientation, tilt and gaze direction of image 2, match its "
    "facial expression, match the camera perspective and distance, and "
    "scale the face to fit the head of image 2 exactly.\n"
    "Relight the new face with the lighting already in image 2 - same light "
    "direction, same shadows, same colour temperature, same contrast, same "
    "grain and depth of field - and match its skin tone to the neck, ears "
    "and hands of image 2 so there is no visible seam or colour break at "
    "the jawline, hairline, ears and neck.\n"
    "Everything else in image 2 stays exactly as it is: the background and "
    "scenery, every object, the clothing and its colours, the body, "
    "shoulders, arms, hands and posture, the neck, the framing and crop, "
    "the camera angle and the overall colour grading. Do not move, resize "
    "or re-render the person or the scene. Keep the same number of people - "
    "add nobody, remove nobody.\n"
    "Photorealistic result that reads as one single untouched photograph: "
    "natural skin, sharp facial detail, no mask edge, no halo around the "
    "head, no ghost of the original face."
)

# Không liệt kê "changed hairstyle" ở đây: phạm vi hoán đổi có tuỳ chọn đổi
# cả tóc, nên câu đó sẽ chống lại chính prompt khi người dùng chọn nó.
FACESWAP_NEGATIVE = (
    "different person, unrecognisable face, original face still visible, "
    "ghost face, double face, two faces on one head, blended identity, "
    "distorted face, warped features, asymmetric eyes, deformed mouth, "
    "mismatched skin tone, visible seam, mask edge, halo around head, "
    "blurry face, waxy plastic skin, changed clothing, changed background, "
    "different scene, moved person, changed pose, changed body, different "
    "camera angle, different crop, extra people, missing people, blurry, "
    "low quality, watermark, text, cartoon"
)


def build_faceswap_prompt(scope_label: str) -> str:
    """Prompt hoán đổi khuôn mặt cho phạm vi đã chọn."""
    scope = FACE_SWAP_SCOPES.get(
        scope_label, next(iter(FACE_SWAP_SCOPES.values()))
    )
    return FACESWAP_PROMPT_TEMPLATE.format(scope=scope)


# --------------------------------------------------------------------------


def append_note(prompt: str, extra_note: str) -> str:
    """Nối ghi chú tự do của người dùng vào cuối prompt task."""
    note = (extra_note or "").strip()
    if not note:
        return prompt
    return f"{prompt}\nAdditional requirement: {note}"
