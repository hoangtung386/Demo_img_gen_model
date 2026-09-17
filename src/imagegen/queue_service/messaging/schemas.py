from dataclasses import dataclass, field
from typing import Any

from .status import StatusCode

# Schema bắt buộc cho mọi inbound. Mỗi entry: (field_name, expected_type,
# must_be_truthy).
#
# Domain gen-image: prompt là bắt buộc; ảnh đầu vào (`images`) là OPTIONAL —
# có ảnh → chế độ editing, không có → text-to-image. HiDream-O1 là model hợp
# nhất nên cả hai đi qua cùng một lời gọi. Vì thế `images` KHÔNG nằm trong
# required fields.
_REQUIRED_FIELDS: tuple[tuple[str, type, bool], ...] = (
    ("_id", str, True),
    ("os", str, True),
    ("firebase_token", str, True),
    ("appid", str, True),
    ("country", str, False),  # country có thể rỗng (vd test)
    ("device_id", str, True),
    ("prompt", str, True),
)

_OPTIONAL_FIELDS_TYPES: dict[str, type] = {
    "negative_prompt": str,
    "seed": int,
    "num_steps": int,
    "storage_index": int,
    "aspect_ratio": float,
    "match_input_size": int,
}

_VALID_OS = {"android", "ios"}
_URL_PREFIXES = ("http://", "https://")


@dataclass(frozen=True)
class ValidationError:
    """Detailed validation failure. Caller maps to a StatusCode."""

    code: StatusCode
    field: str
    reason: str


@dataclass
class InboundMessage:
    id: str
    os: str
    firebase_token: str
    appid: str
    country: str
    device_id: str
    prompt: str
    # 0..N ảnh điều kiện (URL). Rỗng → text-to-image.
    #
    # HiDream-O1 khuyến nghị ĐÚNG MỘT ảnh cho editing: khi đó ảnh ra giữ
    # khung của ảnh vào (keep_original_aspect). Nhiều ảnh vẫn chạy nhưng là
    # bài toán subject-driven khác — KHÔNG còn ngữ nghĩa "ảnh cuối là ảnh
    # nền, ảnh trước là reference" của pipeline Qwen cũ.
    images: list[str] = field(default_factory=list)
    # KHÔNG CÒN TÁC DỤNG với HiDream-O1 (model không nhận negative prompt).
    # Giữ field để BE/App không phải đổi payload; processor log debug rồi bỏ.
    negative_prompt: str = ""
    seed: int = -1
    num_steps: int = 0  # 0 → dùng default của processor config
    # Chỉ chọn được TỈ LỆ khung. Kích thước do model quyết định: mọi yêu cầu
    # bị snap về một trong 11 độ phân giải cứng, nhỏ nhất 2048x2048.
    aspect_ratio: float = 1.0
    match_input_size: int = 0  # 1 → resize ảnh ra về đúng kích thước ảnh vào
    # Index trong storage.backends[]; default 0. BE/App optional gửi để route
    # output tới destination khác (vd serve qua CDN public).
    storage_index: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InboundMessage":
        raw_images = data.get("images")
        if raw_images is None:
            # Cho phép field đơn `image` (1 URL) như alias tiện lợi.
            single = data.get("image")
            images = [str(single)] if single else []
        elif isinstance(raw_images, (list, tuple)):
            images = [str(u) for u in raw_images if u]
        else:
            images = [str(raw_images)]

        return cls(
            id=str(data.get("_id", "")),
            os=str(data.get("os", "")).lower(),
            firebase_token=str(data.get("firebase_token", "")),
            appid=str(data.get("appid", "")),
            country=str(data.get("country", "")),
            device_id=str(data.get("device_id", "")),
            prompt=str(data.get("prompt", "")),
            images=images,
            negative_prompt=str(data.get("negative_prompt", "")),
            seed=int(
                data.get("seed", -1) if data.get("seed") is not None else -1
            ),
            num_steps=int(data.get("num_steps", 0) or 0),
            aspect_ratio=float(data.get("aspect_ratio", 1.0) or 1.0),
            match_input_size=int(data.get("match_input_size", 0) or 0),
            storage_index=int(data.get("storage_index", 0) or 0),
            raw=data,
        )


@dataclass
class OutboundMessage:
    id: str
    os: str
    firebase_token: str
    status_code: int = int(StatusCode.OK)
    message: str = "success"
    result: Any = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "_id": self.id,
            "os": self.os,
            "firebase_token": self.firebase_token,
            "status_code": self.status_code,
            "message": self.message,
            "result": self.result,
        }

    def set_status(self, code: StatusCode, **fmt: Any) -> int:
        """Apply a StatusCode entry from the registry onto this reply."""
        from .status import make_reply

        status_int, message, _kind = make_reply(code, **fmt)
        self.status_code = status_int
        self.message = message
        return status_int


# ───────────────────── validation ─────────────────────


def validate_inbound(data: dict[str, Any]) -> ValidationError | None:
    """
    Strict validation. Checks:
      • Required fields exist + correct type
      • _id / appid / device_id / firebase_token / prompt non-empty
      • os ∈ {android, ios}
      • images (if present) is a list of http(s) URLs
      • optional typed fields convertible to their declared type

    Returns None on pass, ValidationError on fail.
    """
    if not isinstance(data, dict):
        return ValidationError(
            StatusCode.BAD_REQUEST, "_root", "payload must be a JSON object"
        )

    for name, expected_type, must_be_truthy in _REQUIRED_FIELDS:
        if name not in data:
            return ValidationError(
                StatusCode.MISSING_REQUIRED_FIELD, name, "missing"
            )
        value = data[name]
        if not isinstance(value, expected_type):
            return ValidationError(
                StatusCode.INVALID_FIELD_VALUE,
                name,
                f"expected {expected_type.__name__}, got "
                f"{type(value).__name__}",
            )
        if must_be_truthy and not value.strip():
            return ValidationError(
                StatusCode.INVALID_FIELD_VALUE, name, "empty string"
            )

    # OS whitelist
    os_value = data["os"].lower()
    if os_value not in _VALID_OS:
        return ValidationError(
            StatusCode.INVALID_FIELD_VALUE,
            "os",
            f"must be one of {sorted(_VALID_OS)}, got {data['os']!r}",
        )

    # Optional image list — nếu có phải là list URL http(s). Chấp nhận cả
    # field đơn `image` (1 URL). Ảnh rỗng là hợp lệ (text-to-image).
    images_err = _validate_images(data)
    if images_err is not None:
        return images_err

    # Optional typed fields
    for name, expected_type in _OPTIONAL_FIELDS_TYPES.items():
        if name not in data:
            continue
        value = data[name]
        if expected_type in (int, float):
            try:
                expected_type(value)
            except (TypeError, ValueError):
                return ValidationError(
                    StatusCode.INVALID_FIELD_VALUE,
                    name,
                    f"expected {expected_type.__name__}, got "
                    f"{type(value).__name__}",
                )
        elif not isinstance(value, expected_type):
            return ValidationError(
                StatusCode.INVALID_FIELD_VALUE,
                name,
                f"expected {expected_type.__name__}, got "
                f"{type(value).__name__}",
            )

    return None


def _validate_images(data: dict[str, Any]) -> ValidationError | None:
    raw_images = data.get("images")
    if raw_images is None:
        single = data.get("image")
        if single is None:
            return None
        urls = [single]
    elif isinstance(raw_images, (list, tuple)):
        urls = list(raw_images)
    else:
        return ValidationError(
            StatusCode.INVALID_FIELD_VALUE, "images", "expected a list of URLs"
        )

    for i, url in enumerate(urls):
        if not isinstance(url, str) or not url.startswith(_URL_PREFIXES):
            return ValidationError(
                StatusCode.INVALID_FIELD_VALUE,
                f"images[{i}]",
                "must start with http:// or https://",
            )
    return None
