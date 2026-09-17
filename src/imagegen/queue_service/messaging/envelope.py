"""
Generic Envelope: wraps a delivered RabbitMQ message + delivery context.

The consumer hands an Envelope to the worker via the in-process queue. The
worker is responsible for calling exactly one of: ack(), retry(),
dead_letter().
This guarantees the broker keeps the message until the worker is done.

Why a callable indirection: the consumer thread owns the pika Channel (pika is
NOT thread-safe). The worker thread cannot call `channel.basic_ack` directly.
Instead it pushes the action onto the connection via
`connection.add_callback_threadsafe` that the consumer thread drains.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Reasons logged when a message is dropped/dead-lettered. Free-form string.
Reason = str

AckCallable = Callable[[], None]
NackCallable = Callable[[bool], None]  # arg: requeue
PublishCallable = Callable[[str, str, bytes, dict[str, Any]], None]
# (exchange, routing_key, body, headers) → publish via consumer thread


@dataclass
class Envelope:
    """
    A delivered message + the actions you can take on it.

    Exactly one of ack(), retry(), dead_letter() must be called. After that,
    further actions are no-ops (the wrapped channel call would raise).
    """

    body: dict[str, Any]
    raw_bytes: bytes
    delivery_tag: int
    routing_key: str
    headers: dict[str, Any] = field(default_factory=dict)
    message_id: str | None = None

    # Wired by RabbitMQConsumer at construction. Workers should not touch.
    _ack: AckCallable = field(repr=False, default=lambda: None)
    _nack: NackCallable = field(repr=False, default=lambda _requeue: None)
    _publish: PublishCallable = field(
        repr=False, default=lambda _ex, _rk, _body, _hdrs: None
    )
    _dlq_exchange: str = field(repr=False, default="")
    # `_passive_mode` = True khi topology KHÔNG có DLX exchange / .dlq queue
    # (modes `passive` và `declare_minimal`). retry() requeue ngay vào cùng
    # queue thay vì dead-letter, dead_letter() chỉ ack — audit log trên disk
    # (`data_predict/<service>/{fail,error}/...`) là trail để inspect.
    _passive_mode: bool = field(repr=False, default=False)
    _resolved: bool = field(default=False, repr=False)

    # ─────────────────── public actions ───────────────────

    def ack(self) -> None:
        """Successfully processed. Tell broker to drop the message."""
        if self._resolved:
            return
        self._resolved = True
        self._ack()

    def retry(self, reason: Reason = "") -> None:
        """
        Transient failure. Normally: nack without requeue → message goes to the
        retry queue via the queue's x-dead-letter-exchange, comes back after
        TTL.

        Passive mode (queue has no DLX): nack with requeue=True → broker puts
        the message back in the same queue immediately. Re-processing happens
        without TTL backoff but the message is not lost.
        """
        if self._resolved:
            return
        self._resolved = True
        self._nack(self._passive_mode)

    def dead_letter(self, reason: Reason = "") -> None:
        """
        Permanent failure or retries exhausted. Normally: publish to DLQ
        exchange and ack the original.

        Passive mode (no DLQ exchange): just ack — the audit trail on disk
        (data_predict/<service>/{fail,error}/...) is the inspection log.
        """
        if self._resolved:
            return
        self._resolved = True
        if self._passive_mode:
            self._ack()
            return
        body_with_reason = dict(self.body)
        body_with_reason.setdefault("_dlq_reason", reason)
        payload = json.dumps(body_with_reason).encode()
        self._publish(
            self._dlq_exchange, self.routing_key, payload, self.headers
        )
        self._ack()
