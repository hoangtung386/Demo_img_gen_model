"""Đếm số lần một message đã đi qua retry queue.

RabbitMQ ghi header `x-death` mỗi lần message bị dead-letter. Ta đếm các
entry có reason="expired" (retry theo TTL) để biết đây là lần thử thứ mấy.

Con số đó hiện chỉ dùng để GHI AUDIT (`AIWorker` truyền vào
`pipeline.execute(attempt=...)`), không dùng để quyết định retry: cấu hình
canonical là `retry_max_attempts: 0` — mọi failure đi thẳng DLQ.

Trước đây file này còn `should_retry()` và `compute_backoff_ms()` giữ sẵn
"cho service nào fork muốn bật retry". Không ai gọi chúng, và hai knob config
đi kèm (`retry_backoff_multiplier`, `retry_max_delay_ms`) làm ops tưởng
exponential backoff đang chạy trong khi retry queue chỉ có TTL cố định. Đã
xoá cả ba. Muốn bật backoff thật thì viết lại cùng lúc với `_route_failure`.
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
