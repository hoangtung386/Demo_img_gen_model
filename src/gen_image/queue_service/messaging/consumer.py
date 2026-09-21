"""
Generic RabbitMQ consumer.

Pattern:
  • Consumer thread owns the pika connection + channel (pika is NOT thread-safe).
  • For each delivery, build an Envelope. The Envelope's ack/retry/dead_letter
    methods are wired to dispatch back to the consumer thread via
    `connection.add_callback_threadsafe`, so the worker thread can call them
    safely.
  • The consumer pushes the Envelope into the in-process handler (typically a
    backpressure-bounded Queue) and immediately returns to receive more.
  • The broker keeps the message until the worker resolves the envelope, so
    if the worker (or the whole container) crashes, the message will be
    redelivered — exactly the safety the multi-container deployment needs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import suppress

import pika

from ..audit import log_rmq_error, notify_connection_recovered
from ..config import RabbitMQConfig
from ..observability import log_connection_lost, log_inbound, logger
from .connection import BackoffPolicy, build_connection_params
from .envelope import Envelope
from .shutdown import is_shutting_down
from .topology import dlq_exchange_name, setup_topology

EnvelopeHandler = Callable[[Envelope], None]


class RabbitMQConsumer:
    """One connection + one channel + one queue. Reconnects on failure."""

    def __init__(
        self,
        cfg: RabbitMQConfig,
        queue_name: str,
        handler: EnvelopeHandler,
        label: str = "consumer",
    ):
        self.cfg = cfg
        self.queue_name = queue_name
        self.handler = handler
        self.label = label
        self._dlq_exchange = dlq_exchange_name(cfg)

    def run_forever(self) -> None:
        params = build_connection_params(self.cfg)
        backoff = BackoffPolicy(
            initial_seconds=self.cfg.reconnect_initial_seconds,
            max_seconds=self.cfg.reconnect_max_seconds,
            jitter=self.cfg.reconnect_jitter,
        )
        while not is_shutting_down():
            conn: pika.BlockingConnection | None = None
            try:
                conn = pika.BlockingConnection(params)
                channel = conn.channel()
                prefetch = self.cfg.effective_prefetch()
                channel.basic_qos(prefetch_count=prefetch)
                # Idempotent: ensures topology even if a fresh broker is started.
                setup_topology(channel, self.cfg)

                consumer_tag = channel.basic_consume(
                    queue=self.queue_name,
                    on_message_callback=self._make_on_message(conn, channel),
                    auto_ack=False,
                )
                logger.info(
                    f"[{self.label}] consuming queue={self.queue_name} "
                    f"prefetch={prefetch} (sla={self.cfg.target_sla_seconds}s, "
                    f"avg={self.cfg.avg_inference_seconds}s, "
                    f"cap={self.cfg.prefetch_max}) tag={consumer_tag}"
                )
                # Connect thành công — reset backoff để lần fail kế tiếp bắt
                # đầu lại từ initial.
                backoff.reset()
                notify_connection_recovered(
                    "consumer", self.label, queue=self.queue_name
                )
                self._consume_until_shutdown(channel)
            except Exception as e:
                if is_shutting_down():
                    break
                log_connection_lost(
                    component="consumer",
                    label=self.label,
                    queue=self.queue_name,
                    exception=e,
                )
                log_rmq_error(
                    "consumer_connection_lost",
                    e,
                    {"tier": self.label, "queue": self.queue_name},
                )
            finally:
                self._close(conn)
                if not is_shutting_down():
                    delay = backoff.next_delay()
                    logger.debug(
                        f"[{self.label}] reconnect attempt={backoff.attempt} "
                        f"sleep={delay:.1f}s"
                    )
                    time.sleep(delay)
        logger.info(f"[{self.label}] stopped")

    # ───────────────────── delivery ─────────────────────

    def _make_on_message(self, conn: pika.BlockingConnection, channel):
        """Return a closure capturing this connection/channel for ack dispatch."""

        def on_message(_ch, method, properties, body) -> None:
            start = time.time()
            try:
                payload = json.loads(body)
            except Exception as e:
                logger.opt(exception=e).error(
                    f"[consumer] [INBOUND_BAD_JSON] tier={self.label} "
                    f"size={len(body)}B → DLQ"
                )
                channel.basic_publish(
                    exchange=self._dlq_exchange,
                    routing_key=method.routing_key,
                    body=body,
                )
                channel.basic_ack(method.delivery_tag)
                return

            request_id = payload.get("_id", "<missing>")
            log_inbound(
                tier=self.label, request_id=request_id, body=body, queue=self.queue_name
            )

            envelope = self._build_envelope(
                conn, channel, method, properties, body, payload
            )
            try:
                self.handler(envelope)
            except Exception as e:
                # Hard-code DLQ — KHÔNG retry. Handler exception là bug code
                # hoặc backpressure panic; retry vô ích → loop vô tận.
                logger.opt(exception=e).error(
                    f"[consumer] [HANDLER_ERROR] tier={self.label} id={request_id} → "
                    f"DLQ"
                )
                envelope.dead_letter(reason=f"handler exception: {e}")
            logger.debug(
                f"[consumer] dispatched id={request_id} in {time.time() - start:.3f}s"
            )

        return on_message

    def _build_envelope(
        self, conn, channel, method, properties, raw_bytes, payload
    ) -> Envelope:
        delivery_tag = method.delivery_tag
        routing_key = method.routing_key
        headers = (properties.headers or {}) if properties else {}
        message_id = properties.message_id if properties else None
        dlq_exchange = self._dlq_exchange

        # Pika is not thread-safe; the worker calls add_callback_threadsafe
        # to ask the consumer's I/O loop to perform the actual basic_ack/nack.
        def _ack() -> None:
            conn.add_callback_threadsafe(lambda: _safe_ack(channel, delivery_tag))

        def _nack(requeue: bool) -> None:
            conn.add_callback_threadsafe(
                lambda: _safe_nack(channel, delivery_tag, requeue)
            )

        def _publish(exchange: str, rkey: str, body: bytes, hdrs: dict) -> None:
            conn.add_callback_threadsafe(
                lambda: _safe_publish(channel, exchange, rkey, body, hdrs)
            )

        return Envelope(
            body=payload,
            raw_bytes=raw_bytes,
            delivery_tag=delivery_tag,
            routing_key=routing_key,
            headers=headers,
            message_id=message_id,
            _ack=_ack,
            _nack=_nack,
            _publish=_publish,
            _dlq_exchange=dlq_exchange,
            # passive + declare_minimal đều KHÔNG có DLX exchange / .dlq queue;
            # Envelope.retry()/dead_letter() phải fallback (requeue / plain ack).
            _passive_mode=(self.cfg.topology_mode in ("passive", "declare_minimal")),
        )

    # ───────────────────── lifecycle ─────────────────────

    def _consume_until_shutdown(self, channel) -> None:
        """
        Replace the simple `start_consuming()` blocking call with a polling
        loop so that we can react to the shutdown event between deliveries.
        """
        while not is_shutting_down():
            channel.connection.process_data_events(time_limit=1.0)
        logger.info(f"[{self.label}] draining: stop_consuming + close")
        with suppress(Exception):
            channel.stop_consuming()

    @staticmethod
    def _close(conn: pika.BlockingConnection | None) -> None:
        if conn is None:
            return
        with suppress(Exception):
            conn.close()


# ───────────────────── thread-safe channel ops ─────────────────────


def _safe_ack(channel, delivery_tag: int) -> None:
    with suppress(Exception):
        channel.basic_ack(delivery_tag)


def _safe_nack(channel, delivery_tag: int, requeue: bool) -> None:
    with suppress(Exception):
        channel.basic_nack(delivery_tag, requeue=requeue)


def _safe_publish(
    channel, exchange: str, routing_key: str, body: bytes, headers: dict
) -> None:
    with suppress(Exception):
        channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2, headers=headers),
        )
