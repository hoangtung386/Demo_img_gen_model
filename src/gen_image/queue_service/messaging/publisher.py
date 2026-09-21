"""
Publisher thread.

Reads from an in-process queue, publishes to RabbitMQ with publisher confirms
+ mandatory routing. Reconnects on failure. Drains the queue when shutdown is
requested before exiting.
"""

from __future__ import annotations

import json
import queue
import time
from contextlib import suppress
from typing import Any

import pika
from pika.exceptions import UnroutableError

from ..audit import log_rmq_error, notify_connection_recovered
from ..config import RabbitMQConfig
from ..observability import (
    log_connection_lost,
    log_publish_fail,
    log_publish_ok,
    logger,
)
from .connection import BackoffPolicy, build_connection_params
from .shutdown import is_shutting_down

_PUBLISH_RETRIES = 3
# Inter-attempt backoff between consecutive publish() retries on the SAME
# message (network blip mid-publish). Different from reconnect_delay which
# governs how long to wait before retrying a CONNECTION after it failed.
_PUBLISH_RETRY_BACKOFF_SECONDS = 1.0
_DRAIN_POLL_SECONDS = 0.1


class RabbitMQPublisher:
    def __init__(self, cfg: RabbitMQConfig, publish_queue: queue.Queue):
        self.cfg = cfg
        self.publish_queue = publish_queue

    # ───────────────────── public API ─────────────────────

    def submit(self, message: dict[str, Any]) -> bool:
        """Non-blocking. Returns False if local queue is full."""
        try:
            self.publish_queue.put_nowait(message)
            return True
        except queue.Full:
            logger.warning("[publisher] queue full → message dropped")
            return False

    def run_forever(self) -> None:
        params = build_connection_params(self.cfg, heartbeat=60)
        backoff = BackoffPolicy(
            initial_seconds=self.cfg.reconnect_initial_seconds,
            max_seconds=self.cfg.reconnect_max_seconds,
            jitter=self.cfg.reconnect_jitter,
        )
        while not is_shutting_down():
            conn: pika.BlockingConnection | None = None
            try:
                conn = pika.BlockingConnection(params)
                channel = self._setup_channel(conn)
                logger.info("[publisher] ready")
                backoff.reset()
                notify_connection_recovered(
                    "publisher", "publisher", queue_out=self.cfg.queue_out
                )
                conn = self._publish_loop(conn, channel, params)
            except Exception as e:
                if is_shutting_down():
                    break
                log_connection_lost(
                    component="publisher",
                    label="publisher",
                    queue=self.cfg.queue_out,
                    exception=e,
                )
                log_rmq_error(
                    "publisher_connection_error", e, {"queue_out": self.cfg.queue_out}
                )
            finally:
                self._close(conn)
                if not is_shutting_down():
                    delay = backoff.next_delay()
                    logger.debug(
                        f"[publisher] reconnect attempt={backoff.attempt} "
                        f"sleep={delay:.1f}s"
                    )
                    time.sleep(delay)

        # Drain phase: try one final flush before exit so workers' last results
        # don't disappear with the process.
        self._drain_on_shutdown(params)
        logger.info("[publisher] stopped")

    # ───────────────────── internals ─────────────────────

    def _setup_channel(self, conn: pika.BlockingConnection):
        # Topology should already be declared by the bootstrap call in app.py;
        # we still ensure the outbound queue exists in case publisher is run
        # standalone. In passive mode we do NOT redeclare — we only verify.
        # declare_minimal & declare cùng path (cùng dùng exchange direct + bind);
        # outbound queue KHÔNG có DLX bất kể mode nào nên fork không cần thiết.
        channel = conn.channel()
        if self.cfg.topology_mode == "passive":
            channel.queue_declare(queue=self.cfg.queue_out, passive=True)
        else:
            channel.exchange_declare(
                exchange=self.cfg.exchange, exchange_type="direct", durable=True
            )
            channel.queue_declare(queue=self.cfg.queue_out, durable=True)
            channel.queue_bind(
                exchange=self.cfg.exchange,
                queue=self.cfg.queue_out,
                routing_key=self.cfg.queue_out_key,
            )
        channel.confirm_delivery()
        return channel

    def _publish_loop(self, conn, channel, params) -> pika.BlockingConnection:
        while not is_shutting_down():
            try:
                msg = self.publish_queue.get(timeout=_DRAIN_POLL_SECONDS)
            except queue.Empty:
                continue
            success, conn, channel = self._publish_with_retry(
                conn, channel, params, msg
            )
            if not success:
                logger.error("[publisher] requeue after exhausted retries")
                log_rmq_error(
                    "publish_all_retries_failed",
                    None,
                    {"queue_out": self.cfg.queue_out, "msg_id": msg.get("_id")},
                )
                self.publish_queue.put(msg)
                return conn
            self.publish_queue.task_done()
        return conn

    def _publish_with_retry(self, conn, channel, params, msg: dict[str, Any]):
        body = json.dumps(msg).encode()
        request_id = msg.get("_id")
        status_code = msg.get("status_code")
        for attempt in range(_PUBLISH_RETRIES):
            try:
                channel.basic_publish(
                    exchange=self.cfg.exchange,
                    routing_key=self.cfg.queue_out_key,
                    body=body,
                    properties=pika.BasicProperties(delivery_mode=2),
                    mandatory=True,
                )
                log_publish_ok(request_id=request_id, status_code=status_code)
                return True, conn, channel
            except UnroutableError as e:
                log_publish_fail(
                    request_id=request_id, status_code=status_code, exception=e
                )
                logger.error(
                    f"[publisher] [UNROUTABLE] exchange={self.cfg.exchange} "
                    f"key={self.cfg.queue_out_key} id={request_id}"
                )
                return True, conn, channel  # treated as handled so we don't requeue
            except Exception as e:
                logger.warning(
                    f"[publisher] publish fail id={request_id} "
                    f"attempt={attempt + 1}/{_PUBLISH_RETRIES}: {e}"
                )
                time.sleep(_PUBLISH_RETRY_BACKOFF_SECONDS)
                self._close(conn)
                conn = pika.BlockingConnection(params)
                channel = self._setup_channel(conn)
        log_publish_fail(request_id=request_id, status_code=status_code, exception=None)
        return False, conn, channel

    def _drain_on_shutdown(self, params) -> None:
        if self.publish_queue.empty():
            return
        logger.info(
            f"[publisher] draining {self.publish_queue.qsize()} pending messages"
        )
        conn: pika.BlockingConnection | None = None
        try:
            conn = pika.BlockingConnection(params)
            channel = self._setup_channel(conn)
            while not self.publish_queue.empty():
                msg = self.publish_queue.get_nowait()
                self._publish_with_retry(conn, channel, params, msg)
                self.publish_queue.task_done()
        except Exception as e:
            logger.error(f"[publisher] drain failed: {e}")
        finally:
            self._close(conn)

    @staticmethod
    def _close(conn: pika.BlockingConnection | None) -> None:
        if conn is None:
            return
        with suppress(Exception):
            conn.close()
