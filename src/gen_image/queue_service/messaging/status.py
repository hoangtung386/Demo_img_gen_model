"""
Service-wide status code registry.

Every reply published to RabbitMQ carries a status_code and a message.
This module is the single source of truth for:
  • Which codes exist
  • Their human-readable message templates
  • How the worker should treat them (PERMANENT → DLQ, TRANSIENT → retry)
  • Severity (for log filtering / alerting)

Code ranges:
  2xx  Success
  4xx  Input / client fault (PERMANENT — DLQ immediately, retry won't help)
  5xx  Internal / service fault (TRANSIENT — retry; might be a one-off bug)
  6xx  Upstream / dependency fault (TRANSIENT — retry; usually transient)
  9xx  Unknown

Use:
    code, message, kind = make_reply(StatusCode.INPUT_DOWNLOAD_FAILED, url=msg.image)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any


class FailureKind(str, Enum):
    """How the worker should route the message after this status."""

    NONE = "none"  # success → ack
    PERMANENT = "permanent"  # bad input / unrecoverable → DLQ immediately
    TRANSIENT = "transient"  # network blip, GPU OOM, etc. → retry queue


class Severity(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class StatusCode(IntEnum):
    # ─── 2xx Success ─────────────────────────────────────────
    OK = 200

    # ─── 4xx Input / client fault → PERMANENT ────────────────
    BAD_REQUEST = 400
    MISSING_REQUIRED_FIELD = 401
    INVALID_FIELD_VALUE = 402
    INPUT_DOWNLOAD_NOT_FOUND = 410
    INPUT_INVALID_FORMAT = 411
    INPUT_TOO_LARGE = 412
    INPUT_DIMENSION_TOO_LARGE = 413  # width/height vượt trần pixel cho phép
    PROMPT_EMPTY = 416  # prompt rỗng / chỉ khoảng trắng
    PROMPT_TOO_LONG = 417  # vượt max_sequence_length của text encoder
    AI_NO_RESULT = 420  # model chạy xong nhưng không có ảnh ra
    DUPLICATE_REQUEST = 425

    # ─── 5xx Internal / service fault → TRANSIENT ────────────
    INTERNAL_ERROR = 500
    AI_INFERENCE_ERROR = 501
    CUDA_OUT_OF_MEMORY = 510
    MODEL_NOT_LOADED = 511
    AUDIT_WRITE_ERROR = 520

    # ─── 6xx Upstream / dependency fault → TRANSIENT ─────────
    UPSTREAM_ERROR = 600
    INPUT_DOWNLOAD_TIMEOUT = 610
    INPUT_DOWNLOAD_FAILED = 611
    STORAGE_UPLOAD_FAILED = 620
    STORAGE_AUTH_FAILED = 621
    RABBITMQ_PUBLISH_FAILED = 630

    # ─── 9xx Unknown ─────────────────────────────────────────
    UNKNOWN_ERROR = 900


@dataclass(frozen=True)
class StatusInfo:
    code: StatusCode
    message_template: str
    failure_kind: FailureKind
    severity: Severity


# Single source of truth. Any code in StatusCode MUST be present here
# (enforced by tests).
STATUSES: dict[StatusCode, StatusInfo] = {
    StatusCode.OK: StatusInfo(
        StatusCode.OK, "success", FailureKind.NONE, Severity.INFO
    ),
    # ─── 4xx ────────────────────────────────────────────────
    StatusCode.BAD_REQUEST: StatusInfo(
        StatusCode.BAD_REQUEST,
        "bad request: {detail}",
        FailureKind.PERMANENT,
        Severity.WARN,
    ),
    StatusCode.MISSING_REQUIRED_FIELD: StatusInfo(
        StatusCode.MISSING_REQUIRED_FIELD,
        "missing required field: {field}",
        FailureKind.PERMANENT,
        Severity.WARN,
    ),
    StatusCode.INVALID_FIELD_VALUE: StatusInfo(
        StatusCode.INVALID_FIELD_VALUE,
        "invalid value for {field}: {reason}",
        FailureKind.PERMANENT,
        Severity.WARN,
    ),
    StatusCode.INPUT_DOWNLOAD_NOT_FOUND: StatusInfo(
        StatusCode.INPUT_DOWNLOAD_NOT_FOUND,
        "input image not found at URL ({url})",
        FailureKind.PERMANENT,
        Severity.WARN,
    ),
    StatusCode.INPUT_INVALID_FORMAT: StatusInfo(
        StatusCode.INPUT_INVALID_FORMAT,
        "input image format not supported or corrupt",
        FailureKind.PERMANENT,
        Severity.WARN,
    ),
    StatusCode.INPUT_TOO_LARGE: StatusInfo(
        StatusCode.INPUT_TOO_LARGE,
        "input image exceeds size limit ({size_mb}MB > {limit_mb}MB)",
        FailureKind.PERMANENT,
        Severity.WARN,
    ),
    StatusCode.INPUT_DIMENSION_TOO_LARGE: StatusInfo(
        StatusCode.INPUT_DIMENSION_TOO_LARGE,
        "input image dimensions exceed limit ({width}x{height} > {limit}px per side)",
        FailureKind.PERMANENT,
        Severity.WARN,
    ),
    StatusCode.PROMPT_EMPTY: StatusInfo(
        StatusCode.PROMPT_EMPTY,
        "prompt is empty",
        FailureKind.PERMANENT,
        Severity.WARN,
    ),
    StatusCode.PROMPT_TOO_LONG: StatusInfo(
        StatusCode.PROMPT_TOO_LONG,
        "prompt exceeds max length ({length} > {limit})",
        FailureKind.PERMANENT,
        Severity.WARN,
    ),
    StatusCode.AI_NO_RESULT: StatusInfo(
        StatusCode.AI_NO_RESULT,
        "model could not produce a result for this input",
        FailureKind.PERMANENT,
        Severity.WARN,
    ),
    StatusCode.DUPLICATE_REQUEST: StatusInfo(
        StatusCode.DUPLICATE_REQUEST,
        "request already processed recently (idempotency cache hit)",
        FailureKind.PERMANENT,
        Severity.INFO,
    ),
    # ─── 5xx ────────────────────────────────────────────────
    StatusCode.INTERNAL_ERROR: StatusInfo(
        StatusCode.INTERNAL_ERROR,
        "internal error: {detail}",
        FailureKind.TRANSIENT,
        Severity.ERROR,
    ),
    StatusCode.AI_INFERENCE_ERROR: StatusInfo(
        StatusCode.AI_INFERENCE_ERROR,
        "model inference failed: {detail}",
        # TRANSIENT theo range convention (5xx → TRANSIENT). Service org dùng
        # `retry_max_attempts: 0` canonical → mọi TRANSIENT DLQ ngay sau
        # attempt đầu. Service nào muốn bật retry > 0 phải audit pipeline đảm
        # bảo 501 chỉ raise cho transient thật (OOM tạm thời, model lock).
        FailureKind.TRANSIENT,
        Severity.ERROR,
    ),
    StatusCode.CUDA_OUT_OF_MEMORY: StatusInfo(
        StatusCode.CUDA_OUT_OF_MEMORY,
        "GPU out of memory during inference",
        FailureKind.TRANSIENT,
        Severity.ERROR,
    ),
    StatusCode.MODEL_NOT_LOADED: StatusInfo(
        StatusCode.MODEL_NOT_LOADED,
        "model not initialized",
        FailureKind.TRANSIENT,
        Severity.ERROR,
    ),
    StatusCode.AUDIT_WRITE_ERROR: StatusInfo(
        StatusCode.AUDIT_WRITE_ERROR,
        "could not persist audit data: {detail}",
        FailureKind.TRANSIENT,
        Severity.ERROR,
    ),
    # ─── 6xx ────────────────────────────────────────────────
    StatusCode.UPSTREAM_ERROR: StatusInfo(
        StatusCode.UPSTREAM_ERROR,
        "upstream dependency error: {detail}",
        FailureKind.TRANSIENT,
        Severity.ERROR,
    ),
    StatusCode.INPUT_DOWNLOAD_TIMEOUT: StatusInfo(
        StatusCode.INPUT_DOWNLOAD_TIMEOUT,
        "timed out fetching input image ({url})",
        FailureKind.TRANSIENT,
        Severity.WARN,
    ),
    StatusCode.INPUT_DOWNLOAD_FAILED: StatusInfo(
        StatusCode.INPUT_DOWNLOAD_FAILED,
        "failed to fetch input image: {detail}",
        FailureKind.TRANSIENT,
        Severity.WARN,
    ),
    StatusCode.STORAGE_UPLOAD_FAILED: StatusInfo(
        StatusCode.STORAGE_UPLOAD_FAILED,
        "failed to upload result to storage: {detail}",
        FailureKind.TRANSIENT,
        Severity.ERROR,
    ),
    StatusCode.STORAGE_AUTH_FAILED: StatusInfo(
        StatusCode.STORAGE_AUTH_FAILED,
        "storage authentication failed",
        FailureKind.TRANSIENT,
        Severity.ERROR,
    ),
    StatusCode.RABBITMQ_PUBLISH_FAILED: StatusInfo(
        StatusCode.RABBITMQ_PUBLISH_FAILED,
        "could not publish reply to broker",
        FailureKind.TRANSIENT,
        Severity.ERROR,
    ),
    # ─── 9xx ────────────────────────────────────────────────
    StatusCode.UNKNOWN_ERROR: StatusInfo(
        StatusCode.UNKNOWN_ERROR,
        "unknown error: {detail}",
        FailureKind.TRANSIENT,
        Severity.ERROR,
    ),
}


def info_for(code: StatusCode) -> StatusInfo:
    """Look up the StatusInfo for a code. Falls back to UNKNOWN_ERROR."""
    return STATUSES.get(code, STATUSES[StatusCode.UNKNOWN_ERROR])


def make_reply(code: StatusCode, **fmt: Any) -> tuple[int, str, FailureKind]:
    """
    Build (status_code, formatted_message, failure_kind) for a code.

    Format kwargs are interpolated into the template (str.format). Missing
    kwargs degrade gracefully — the raw template stays in the message so
    we never crash on a formatting bug in production.
    """
    info = info_for(code)
    try:
        message = info.message_template.format(**fmt) if fmt else info.message_template
    except (KeyError, IndexError, ValueError):
        message = info.message_template
    return int(info.code), message, info.failure_kind


def severity_of(code: StatusCode) -> Severity:
    return info_for(code).severity
