"""
Standard logging helpers.

Use these instead of raw logger.* calls so every log line follows the same
format ([TAG] [EVENT] key=value ...) and request_id appears on every line
that involves a specific message.
"""

from __future__ import annotations

from typing import Any

from .logging import logger

# Module-level config (mutable; set by setup_logging from ObservabilityConfig).
_payload_max_bytes: int = 500
_strip_url_query: bool = True


def configure_helpers(
    *, payload_max_bytes: int = 500, strip_url_query: bool = True
) -> None:
    """Set helper-wide options.

    Called once at startup from ObservabilityConfig.
    """
    global _payload_max_bytes, _strip_url_query
    _payload_max_bytes = max(0, payload_max_bytes)
    _strip_url_query = strip_url_query


# ───────────────────── safe formatters ─────────────────────


def safe_url(url: str) -> str:
    """Strip query string with secrets (signed URL tokens, etc)."""
    if not _strip_url_query or not url:
        return url
    if "?" in url:
        return url.split("?", 1)[0] + "?<truncated>"
    return url


def preview_body(body: bytes | str | None) -> str:
    """Truncate body to log_payload_max_bytes.

    Returns '' if disabled or empty.
    """
    if _payload_max_bytes <= 0 or not body:
        return ""
    if isinstance(body, bytes):
        try:
            s = body.decode("utf-8", errors="replace")
        except Exception:
            return f"<{len(body)} bytes binary>"
    else:
        s = body
    total = len(s)
    if total <= _payload_max_bytes:
        return s
    return s[:_payload_max_bytes] + f"...(truncated, total={total}B)"


def _kv(items: dict[str, Any]) -> str:
    """Format dict as 'k1=v1 k2=v2'. Skip None values."""
    parts = []
    for k, v in items.items():
        if v is None:
            continue
        if isinstance(v, str) and " " in v:
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


# ───────────────────── inbound / outbound ─────────────────────


def log_inbound(
    *, tier: str, request_id: str, body: bytes, queue: str | None = None
) -> None:
    size = len(body) if body else 0
    fields = _kv(
        {"tier": tier, "id": request_id, "queue": queue, "size": f"{size}B"}
    )
    preview = preview_body(body)
    suffix = f" preview={preview}" if preview else ""
    logger.info(f"[consumer] [INBOUND] {fields}{suffix}")


def log_validate_fail(
    *, tier: str, request_id: str, missing_field: str
) -> None:
    fields = _kv(
        {"tier": tier, "id": request_id, "missing_field": missing_field}
    )
    logger.warning(f"[consumer] [VALIDATE_FAIL] {fields}")


def log_outbound(*, request_id: str, status_code: int, queue: str) -> None:
    logger.info(
        f"[publisher] [OUTBOUND] "
        f"{_kv({'id': request_id, 'status': status_code, 'queue': queue})}"
    )


def log_publish_ok(*, request_id: str, status_code: int) -> None:
    logger.debug(
        f"[publisher] [PUBLISH_OK] "
        f"{_kv({'id': request_id, 'status': status_code})}"
    )


def log_publish_fail(
    *,
    request_id: str | None,
    status_code: int | None,
    exception: BaseException | None = None,
) -> None:
    fields = _kv({"id": request_id, "status": status_code})
    msg = f"[publisher] [PUBLISH_FAIL] {fields}"
    if exception is not None:
        logger.opt(exception=exception).error(msg)
    else:
        logger.error(msg)


# ───────────────────── worker / pipeline ─────────────────────


def log_process_start(
    *, worker_id: int, tier: str, request_id: str, attempt: int
) -> None:
    logger.info(
        f"[worker {worker_id}] [PROCESS_START] "
        f"{_kv({'tier': tier, 'id': request_id, 'attempt': attempt})}"
    )


def log_process_ok(
    *, worker_id: int, request_id: str, status_code: int, elapsed_ms: float
) -> None:
    fields = _kv(
        {
            "id": request_id,
            "status": status_code,
            "elapsed_ms": f"{elapsed_ms:.0f}",
        }
    )
    logger.info(f"[worker {worker_id}] [PROCESS_OK] {fields}")


def log_process_fail(
    *,
    worker_id: int,
    request_id: str,
    status_code: int,
    message: str,
    elapsed_ms: float,
) -> None:
    fields = _kv(
        {
            "id": request_id,
            "status": status_code,
            "elapsed_ms": f"{elapsed_ms:.0f}",
            "msg": message,
        }
    )
    logger.warning(f"[worker {worker_id}] [PROCESS_FAIL] {fields}")


def log_process_error(
    *,
    worker_id: int,
    request_id: str,
    status_code: int,
    exception: BaseException,
    elapsed_ms: float | None = None,
) -> None:
    """ERROR + full traceback via loguru.opt(exception=...)."""
    fields = _kv(
        {
            "id": request_id,
            "status": status_code,
            "elapsed_ms": f"{elapsed_ms:.0f}"
            if elapsed_ms is not None
            else None,
            "exc": type(exception).__name__,
        }
    )
    logger.opt(exception=exception).error(
        f"[worker {worker_id}] [PROCESS_ERROR] {fields}"
    )


# ───────────────────── routing decisions ─────────────────────


def log_retry(
    *,
    worker_id: int,
    request_id: str,
    attempt: int,
    max_attempts: int,
    reason: str,
) -> None:
    fields = _kv(
        {
            "id": request_id,
            "attempt": f"{attempt}/{max_attempts}",
            "reason": reason,
        }
    )
    logger.warning(f"[worker {worker_id}] [RETRY] {fields}")


def log_dlq(*, worker_id: int, request_id: str, reason: str) -> None:
    logger.error(
        f"[worker {worker_id}] [DLQ] "
        f"{_kv({'id': request_id, 'reason': reason})}"
    )


# ───────────────────── infrastructure ─────────────────────


def log_connection_lost(
    *,
    component: str,
    label: str,
    queue: str | None = None,
    exception: BaseException,
) -> None:
    fields = _kv(
        {"label": label, "queue": queue, "exc": type(exception).__name__}
    )
    logger.opt(exception=exception).error(
        f"[{component}] [CONNECTION_LOST] {fields}"
    )


def log_topology_declared(*, exchanges: list[str], queues: list[str]) -> None:
    logger.info(
        f"[topology] [TOPOLOGY_DECLARED] "
        f"{_kv({'exchanges': len(exchanges), 'queues': len(queues)})}"
    )


def log_ready(*, service_version: str) -> None:
    logger.info(f"[app] [READY] {_kv({'service_version': service_version})}")


def log_shutdown(*, signal_name: str) -> None:
    logger.info(f"[app] [SHUTDOWN] {_kv({'signal': signal_name})}")
