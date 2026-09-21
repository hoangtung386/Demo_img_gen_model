"""Tests for InboundMessage.provider và validate_inbound liên quan."""

from __future__ import annotations

from gen_image.queue_service.messaging.schemas import (
    InboundMessage,
    validate_inbound,
)
from gen_image.queue_service.messaging.status import StatusCode

_BASE = {
    "_id": "req-1",
    "os": "android",
    "firebase_token": "tok",
    "appid": "app",
    "country": "vn",
    "device_id": "dev",
    "prompt": "a cat",
}


def test_provider_default_empty():
    msg = InboundMessage.from_dict(_BASE)
    assert msg.provider == ""


def test_provider_openai_parsed_lowercase():
    msg = InboundMessage.from_dict({**_BASE, "provider": "OpenAI"})
    assert msg.provider == "openai"


def test_validate_inbound_accepts_missing_provider():
    assert validate_inbound(_BASE) is None


def test_validate_inbound_accepts_openai_provider():
    assert validate_inbound({**_BASE, "provider": "openai"}) is None


def test_validate_inbound_rejects_unknown_provider():
    err = validate_inbound({**_BASE, "provider": "bedrock"})
    assert err is not None
    assert err.code == StatusCode.INVALID_FIELD_VALUE
    assert err.field == "provider"


def test_validate_inbound_accepts_empty_os():
    """aichat-service gửi os="" khi client không truyền os (`body.os || ''`)
    — downstream tự fallback 'unknown' khi đặt tên file GCS, nên không cần
    reject ở đây (xem ghi chú tại _REQUIRED_FIELDS)."""
    assert validate_inbound({**_BASE, "os": ""}) is None


def test_validate_inbound_accepts_unknown_os_value():
    assert validate_inbound({**_BASE, "os": "windows"}) is None


def test_validate_inbound_accepts_empty_device_id():
    """aichat-service gửi device_id="" khi client không truyền field này
    (`device_id: body.device_id || ''`) — downstream tự fallback
    'unknown-device' khi đặt tên file GCS (xem ghi chú ở _REQUIRED_FIELDS)."""
    assert validate_inbound({**_BASE, "device_id": ""}) is None


def test_validate_inbound_accepts_empty_firebase_token():
    """aichat-service gửi firebase_token="" khi client không truyền field
    này (`firebase_token: body.firebase_token || ''`) — model-gen-img chỉ
    echo lại nguyên văn, không rẽ nhánh theo giá trị này."""
    assert validate_inbound({**_BASE, "firebase_token": ""}) is None


def test_validate_inbound_accepts_empty_prompt():
    """prompt rỗng đi thẳng vào text encoder — không crash, chỉ ra ảnh
    không theo hướng dẫn nào. Chấp nhận đánh đổi tốn GPU thay vì reject."""
    assert validate_inbound({**_BASE, "prompt": ""}) is None
