from dataclasses import dataclass, field
from typing import Any

from .status import StatusCode

# Schema bắt buộc cho mọi inbound. Mỗi entry: (field_name, expected_type,
# must_be_truthy).
#
# Domain gen-image: prompt là bắt buộc; ảnh đầu vào (`images`) là OPTIONAL —
# có ảnh → chế độ image-edit, không có → text-to-image. Cùng một pipeline
# phục vụ cả hai (Flux2KleinPipeline nhận `image=None`), nên `images` KHÔNG
# nằm trong required fields.
_REQUIRED_FIELDS: tuple[tuple[str, type, bool], ...] = (
    ("_id", str, True),
    # os KHÔNG bắt buộc non-empty / whitelist: BE (aichat-service) gửi ""
    # khi client không truyền os (image.service.ts: `os: body.os || ''`).
    # Downstream (object_naming._normalize_os) đã tự fallback giá trị lạ về
    # "unknown" khi đặt tên file GCS — validate strict ở đây chỉ chặn sớm
    # hơn mức cần thiết, không có gì crash nếu bỏ qua.
    ("os", str, False),
    # firebase_token KHÔNG bắt buộc non-empty: BE gửi "" khi client không
    # truyền (image.service.ts: `firebase_token: body.firebase_token || ''`).
    # model-gen-img chỉ echo lại nguyên văn trong OutboundMessage (xem
    # pipelines/gen_image.py, workers/ai_worker.py) — không dùng để rẽ nhánh
    # hay validate gì ở tầng này. Token rỗng chỉ ảnh hưởng khả năng BE push
    # FCM cho user, đó là việc của BE/app, không phải lý do reject ở đây.
    ("firebase_token", str, False),
    ("appid", str, True),
    ("country", str, False),  # country có thể rỗng (vd test)
    # device_id KHÔNG bắt buộc non-empty — cùng lý do với os: BE
    # (aichat-service) gửi "" khi client không truyền field này
    # (image.service.ts: `device_id: body.device_id || ''`). Downstream
    # (object_naming._sanitize) đã tự fallback "unknown-device" khi đặt tên
    # file GCS; audit chỉ hiển thị giá trị chứ không rẽ nhánh theo nó.
    ("device_id", str, False),
    # prompt KHÔNG bắt buộc non-empty: prompt rỗng đi thẳng vào text encoder
    # của pipeline (khác os/device_id/firebase_token, không có fallback nào
    # downstream) — kết quả là ảnh sinh không theo hướng dẫn nào, nhưng vẫn
    # CHẠY ĐƯỢC, không crash. Chấp nhận đánh đổi tốn 1 lượt GPU cho request
    # prompt rỗng thay vì reject cứng ở đây.
    ("prompt", str, False),
)

_OPTIONAL_FIELDS_TYPES: dict[str, type] = {
    "negative_prompt": str,
    "seed": int,
    "num_steps": int,
    "storage_index": int,
    "aspect_ratio": float,
    "match_input_size": int,
    "provider": str,
}

# "" (default) → giữ format result cũ {"url", "info"}. "openai" → thêm
# result.openai theo cấu trúc OpenAI Images API (xem OutboundMessage.result
# trong pipelines/gen_image.py). Provider khác (vd "google") sẽ nối vào đây
# sau này — validate_inbound từ chối provider lạ để lỗi sớm thay vì lặng lẽ
# rơi về format cũ.
_VALID_PROVIDERS = {"", "openai"}

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
    # 0..N ảnh điều kiện (URL). Rỗng → text-to-image. Có ảnh → image-edit;
    # ảnh CUỐI là ảnh nền (pipeline lấy tỉ lệ + coi là ảnh chính), ảnh trước
    # đó là reference (trang phục, phòng mẫu…). Xem inference.generate().
    images: list[str] = field(default_factory=list)
    negative_prompt: str = ""
    seed: int = -1
    num_steps: int = 0  # 0 → dùng default của processor config
    aspect_ratio: float = 1.0  # chỉ dùng cho text-to-image (không có ảnh nền)
    match_input_size: int = 0  # 1 → resize ảnh ra về đúng kích thước ảnh nền
    # Index trong storage.backends[]; default 0. BE/App optional gửi để route
    # output tới destination khác (vd serve qua CDN public).
    storage_index: int = 0
    # "" (default) → result giữ format cũ {"url", "info"}. "openai" → thêm
    # result.openai theo cấu trúc OpenAI Images API. Xem _VALID_PROVIDERS.
    provider: str = ""
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
            seed=int(data.get("seed", -1) if data.get("seed") is not None else -1),
            num_steps=int(data.get("num_steps", 0) or 0),
            aspect_ratio=float(data.get("aspect_ratio", 1.0) or 1.0),
            match_input_size=int(data.get("match_input_size", 0) or 0),
            storage_index=int(data.get("storage_index", 0) or 0),
            provider=str(data.get("provider", "") or "").lower(),
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
      • _id / appid non-empty
      • os / device_id: type-checked only, may be empty (normalized
        downstream, see storage.object_naming)
      • firebase_token: type-checked only, may be empty (echoed as-is,
        never used to branch in this service)
      • prompt: type-checked only, may be empty (goes straight to the text
        encoder — no downstream fallback, but doesn't crash; empty prompt
        just wastes one GPU pass on an unguided image)
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
            return ValidationError(StatusCode.MISSING_REQUIRED_FIELD, name, "missing")
        value = data[name]
        if not isinstance(value, expected_type):
            return ValidationError(
                StatusCode.INVALID_FIELD_VALUE,
                name,
                f"expected {expected_type.__name__}, got {type(value).__name__}",
            )
        if must_be_truthy and not value.strip():
            return ValidationError(StatusCode.INVALID_FIELD_VALUE, name, "empty string")

    # os không còn validate whitelist — xem ghi chú tại _REQUIRED_FIELDS.

    # Optional image list — nếu có phải là list URL http(s). Chấp nhận cả
    # field đơn `image` (1 URL). Ảnh rỗng là hợp lệ (text-to-image).
    images_err = _validate_images(data)
    if images_err is not None:
        return images_err

    # Provider whitelist — rỗng (format cũ) hoặc "openai" hiện tại.
    if "provider" in data:
        provider_value = str(data["provider"] or "").lower()
        if provider_value not in _VALID_PROVIDERS:
            return ValidationError(
                StatusCode.INVALID_FIELD_VALUE,
                "provider",
                f"must be one of {sorted(_VALID_PROVIDERS)}, got {data['provider']!r}",
            )

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
                    f"expected {expected_type.__name__}, got {type(value).__name__}",
                )
        elif not isinstance(value, expected_type):
            return ValidationError(
                StatusCode.INVALID_FIELD_VALUE,
                name,
                f"expected {expected_type.__name__}, got {type(value).__name__}",
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
