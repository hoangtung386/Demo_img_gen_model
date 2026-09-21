"""
Retry counting + exponential backoff utilities.

> **CANONICAL implementation: KHÔNG dùng retry.** `AIWorker._route_failure`
> hard-code DLQ ngay mọi failure. `should_retry` + `compute_backoff_ms` ở
> file này giữ làm utility cho service mới fork muốn implement retry logic —
> phải custom `_route_failure` để gọi 2 hàm này, đảm bảo audit classifier kỹ
> trước khi bật.

RabbitMQ writes an `x-death` header each time a message is dead-lettered. We
count entries with reason="expired" (TTL-driven retry) to decide whether to
keep retrying or send the message to the DLQ.
"""

from __future__ import annotations

from typing import Any

X_DEATH_HEADER = "x-death"


def attempts_so_far(headers: dict[str, Any] | None) -> int:
    """
    Number of times this message has been through the retry queue already.
    First delivery returns 0; after first retry returns 1; etc.
    """
    if not headers:
        return 0
    deaths = headers.get(X_DEATH_HEADER) or []
    total = 0
    for entry in deaths:
        if entry.get("reason") == "expired":
            total += int(entry.get("count", 1))
    return total


def should_retry(headers: dict[str, Any] | None, max_attempts: int) -> bool:
    return attempts_so_far(headers) < max_attempts


def compute_backoff_ms(
    attempt: int, *, base_ms: int, multiplier: float, max_ms: int
) -> int:
    """
    attempt: 0-indexed (first retry = attempt 0).
    Returns capped exponential backoff.
    """
    delay = base_ms * (multiplier**attempt)
    return min(int(delay), max_ms)
