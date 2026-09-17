"""
Persist RabbitMQ-level errors to disk.

These are events that don't belong to any single request (connection drops,
publisher retries exhausted, etc.). Stored separately under:

    data_predict/<service>/rmq_errors/<Y>/<M>/<D>/<event>_<ts>.json

Auto-cleaned by the same retention policy as request folders.

DEDUP: connection-loss events are deduplicated per (event, identity_key) —
ghi file LẦN ĐẦU broker fail; các fail tiếp theo (cùng identity, broker chưa
recovery) chỉ tăng counter trong RAM, không tạo file mới. Khi component
reconnect thành công → ghi 1 file `connection_recovered` kèm counter của lần
down này. Tránh spam 600+ file khi broker down vài giờ.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import AuditConfig
from ..observability import logger

_audit_cfg: AuditConfig | None = None

# Dedup state: key = (event, identity) → dict với first_ts, last_ts, count,
# exception_type. Khi recover: emit `connection_recovered` + clear key.
_dedup_state: dict[tuple[str, str], dict[str, Any]] = {}
_dedup_lock = threading.Lock()

# Các event được dedup. Event khác (vd publish_all_retries_failed —
# per-message) không dedup vì mỗi message là độc lập.
_DEDUP_EVENTS = {
    "consumer_connection_lost",
    "publisher_connection_error",
    "topology_declare_failed",
}


def init_rmq_error_logger(cfg: AuditConfig) -> None:
    global _audit_cfg
    _audit_cfg = cfg
    with _dedup_lock:
        _dedup_state.clear()


def log_rmq_error(
    event: str,
    exception: BaseException | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """
    Append-style log: writes one JSON file per event (with dedup for
    connection-loss events).

    Args:
        event: Short identifier, e.g. "consumer_connection_lost".
        exception: Optional exception (str + traceback captured).
        context: Free-form dict (tier, queue, attempt, etc.).
    """
    if _audit_cfg is None or not _audit_cfg.enabled:
        return

    # Dedup: cùng event + identity (vd consumer 'premium') → chỉ ghi lần đầu.
    if event in _DEDUP_EVENTS:
        identity = _identity_key(event, context or {})
        with _dedup_lock:
            existing = _dedup_state.get((event, identity))
            if existing is not None:
                existing["count"] += 1
                existing["last_ts"] = time.time()
                return  # skip writing duplicate file
            # First occurrence — record state + fall through to write file.
            _dedup_state[(event, identity)] = {
                "first_ts": time.time(),
                "last_ts": time.time(),
                "count": 1,
                "exception_type": type(exception).__name__
                if exception
                else None,
            }

    _write_event_file(event, exception, context)


def notify_connection_recovered(
    component: str, label: str, **context_kv: Any
) -> None:
    """
    Gọi khi consumer/publisher reconnect thành công sau khi đã từng fail.

    Nếu trước đó có dedup state cho component này → emit 1 file
    `connection_recovered` với count + duration của lần down. Nếu không có
    state (lần connect đầu tiên hoặc không từng fail) → no-op.
    """
    if _audit_cfg is None or not _audit_cfg.enabled:
        return

    # Map component → event đã được dedup
    event_map = {
        "consumer": "consumer_connection_lost",
        "publisher": "publisher_connection_error",
        "topology": "topology_declare_failed",
    }
    failed_event = event_map.get(component)
    if failed_event is None:
        return

    context = {"component": component, "label": label, **context_kv}
    identity = _identity_key(failed_event, context)
    with _dedup_lock:
        state = _dedup_state.pop((failed_event, identity), None)

    if state is None:
        return  # never failed → nothing to recover

    duration_s = time.time() - state["first_ts"]
    _write_event_file(
        "connection_recovered",
        exception=None,
        context={
            **context,
            "previous_event": failed_event,
            "fail_count_during_outage": state["count"],
            "first_failure_at": datetime.fromtimestamp(
                state["first_ts"]
            ).isoformat(),
            "outage_duration_seconds": round(duration_s, 1),
            "exception_type": state["exception_type"],
        },
    )
    logger.info(
        f"[rmq] [RECOVERED] component={component} label={label} "
        f"failed_count={state['count']} duration={duration_s:.1f}s"
    )


def _identity_key(event: str, context: dict[str, Any]) -> str:
    """
    Stable string identifying component instance — for dedup.

    Same identity key MUST be produced from log_rmq_error(...) and
    notify_connection_recovered(...).

    Consumer dedup theo `tier`/`label`. Publisher dedup theo `queue_out`.
    Topology dedup theo `exchange`.
    """

    def pick(*keys: str) -> str:
        for key in keys:
            value = context.get(key)
            if value:
                return str(value)
        return "default"

    if event == "consumer_connection_lost":
        return f"consumer={pick('tier', 'label')}"
    if event == "publisher_connection_error":
        return f"publisher={pick('queue_out', 'label')}"
    if event == "topology_declare_failed":
        return f"topology={pick('exchange', 'label')}"
    return "default"


def _write_event_file(
    event: str,
    exception: BaseException | None,
    context: dict[str, Any] | None,
) -> None:
    ts = time.time()
    d = datetime.fromtimestamp(ts)
    base = (
        Path(_audit_cfg.root_dir)  # type: ignore[union-attr]
        / _audit_cfg.service_name  # type: ignore[union-attr]
        / "rmq_errors"
        / str(d.year)
        / str(d.month)
        / str(d.day)
    )
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(f"[rmq_error_logger] mkdir {base}: {e}")
        return

    payload: dict[str, Any] = {
        "timestamp": d.isoformat(),
        "timestamp_epoch": ts,
        "event": event,
    }
    if exception is not None:
        payload["error"] = str(exception)
        payload["error_type"] = type(exception).__name__
        payload["traceback"] = "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        )
    if context:
        payload["context"] = context

    # Filename collision: 2 events trong cùng millisecond. Microsecond
    # resolution + counter để tránh ghi đè khi nhiều consumer fail cùng lúc.
    fname = f"{event}_{int(ts * 1_000_000)}.json"
    target = base / fname
    counter = 0
    while target.exists() and counter < 100:
        counter += 1
        target = base / f"{event}_{int(ts * 1_000_000)}_{counter}.json"
    try:
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    except OSError as e:
        logger.warning(f"[rmq_error_logger] write {target.name}: {e}")
