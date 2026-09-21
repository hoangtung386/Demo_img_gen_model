import time
from datetime import datetime

from ..config import Settings
from ..messaging import (
    InboundMessage,
    OutboundMessage,
    RabbitMQPublisher,
    StatusCode,
    make_reply,
)
from ..messaging.envelope import Envelope
from ..messaging.retry import attempts_so_far
from ..messaging.shutdown import is_shutting_down
from ..observability import (
    log_dlq,
    log_outbound,
    log_process_error,
    log_process_fail,
    log_process_ok,
    log_process_start,
    logger,
)
from ..pipelines import FailureKind, Pipeline
from ..queueing import PriorityRouter, RecentRequestCache


class AIWorker:
    """
    Drain priority queues of Envelopes, run pipeline, ack/retry/DLQ via
    Envelope. Submits successful replies to publisher.
    """

    def __init__(
        self,
        worker_id: int,
        settings: Settings,
        router: PriorityRouter,
        pipeline: Pipeline,
        publisher: RabbitMQPublisher,
        recent_requests: RecentRequestCache | None = None,
    ):
        self.worker_id = worker_id
        self.settings = settings
        self.router = router
        self.pipeline = pipeline
        self.publisher = publisher
        # Shared per-process LRU cache; None = idempotency check disabled.
        self.recent_requests = recent_requests

    def run_forever(self) -> None:
        loop = 0
        while not is_shutting_down():
            loop += 1
            if loop % 10 == 0:
                self._log_queue_sizes()

            batch = self.router.get_next_batch()
            if not batch:
                time.sleep(0.05)
                continue

            for label, envelope in batch:
                self._handle(label, envelope)
        logger.info(f"[worker {self.worker_id}] [STOPPED]")

    # ───────────────────── per-message handling ─────────────────────

    def _handle(self, label: str, envelope: Envelope) -> None:
        attempt = attempts_so_far(envelope.headers)
        request_id = envelope.body.get("_id", "<missing>")
        try:
            msg = InboundMessage.from_dict(envelope.body)
        except Exception as e:
            logger.opt(exception=e).error(
                f"[worker {self.worker_id}] [BAD_PAYLOAD] tier={label} id={request_id} "
                f"→ DLQ"
            )
            envelope.dead_letter(reason=f"bad payload: {e}")
            return

        # Idempotency: chỉ check ở attempt 0. Duplicate = BE/App publish 2 lần
        # cùng request_id, cả 2 đều attempt 0 — không xử lý cái thứ hai vì cái
        # đầu đã reply.
        if attempt == 0 and self.recent_requests and self.recent_requests.seen(msg.id):
            logger.info(
                f"[worker {self.worker_id}] [DUPLICATE] tier={label} id={msg.id} "
                f"— already processed recently, ack + skip GPU"
            )
            self._reply_duplicate(msg, label)
            envelope.ack()
            return

        log_process_start(
            worker_id=self.worker_id, tier=label, request_id=msg.id, attempt=attempt + 1
        )
        start = time.time()
        try:
            result = self.pipeline.execute(
                msg,
                worker_id=self.worker_id,
                attempt=attempt,
                rmq_headers=envelope.headers,
            )
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000.0
            log_process_error(
                worker_id=self.worker_id,
                request_id=msg.id,
                status_code=500,
                exception=e,
                elapsed_ms=elapsed_ms,
            )
            # Hard-code DLQ — KHÔNG retry. Worker exception (pipeline crash,
            # OOM …) là bug code, retry vô ích → loop vô tận. AI reply 500 đã
            # publish về BE; BE re-publish thủ công nếu cần sau khi fix.
            envelope.dead_letter(reason=f"worker exception: {e}")
            return

        elapsed_ms = (time.time() - start) * 1000.0
        reply_dict = result.reply.to_dict()

        if result.succeeded:
            log_process_ok(
                worker_id=self.worker_id,
                request_id=msg.id,
                status_code=result.reply.status_code,
                elapsed_ms=elapsed_ms,
            )
            self._publish_reply(reply_dict, msg.id)
            self._mark_processed(msg.id)
            envelope.ack()
            return

        # Failure path: log, publish reply, then route message.
        log_process_fail(
            worker_id=self.worker_id,
            request_id=msg.id,
            status_code=result.reply.status_code,
            message=result.reply.message,
            elapsed_ms=elapsed_ms,
        )

        self._publish_reply(reply_dict, msg.id)
        # Mark as processed cho cả permanent failures (4xx) — không xử lý lại
        # duplicate. TRANSIENT (5xx/6xx) vẫn để cache trống → retry message mới
        # của BE vẫn được xử lý.
        if result.failure is FailureKind.PERMANENT:
            self._mark_processed(msg.id)
        self._route_failure(label, envelope, msg, result.failure, attempt)

    def _mark_processed(self, request_id: str) -> None:
        if self.recent_requests is not None:
            self.recent_requests.add(request_id)

    def _reply_duplicate(self, msg: InboundMessage, label: str) -> None:
        """Send a DUPLICATE_REQUEST reply để BE biết request đã bị skip."""
        out = OutboundMessage(id=msg.id, os=msg.os, firebase_token=msg.firebase_token)
        code, message, _ = make_reply(StatusCode.DUPLICATE_REQUEST)
        out.status_code = int(code)
        out.message = message
        log_outbound(
            request_id=msg.id,
            status_code=int(code),
            queue=self.settings.rabbitmq.queue_out,
        )
        self.publisher.submit(out.to_dict())
        logger.info(
            f"[worker {self.worker_id}] [DUPLICATE] tier={label} id={msg.id} replied"
        )

    def _publish_reply(self, reply_dict: dict, request_id: str) -> None:
        log_outbound(
            request_id=request_id,
            status_code=reply_dict.get("status_code", 0),
            queue=self.settings.rabbitmq.queue_out,
        )
        self.publisher.submit(reply_dict)

    def _route_failure(
        self,
        label: str,
        envelope: Envelope,
        msg: InboundMessage,
        kind: FailureKind,
        attempt: int,
    ) -> None:
        """
        Mọi failure → DLQ NGAY, KHÔNG bao giờ retry.

        Hard-code policy này tại CODE thay vì config — bypass
        `retry_max_attempts` của base.yaml để đảm bảo dù bất kỳ environment
        nào đều KHÔNG có vector loop vô hạn.

        - PERMANENT (4xx) → input fault, retry vô ích.
        - TRANSIENT (5xx/6xx) → bug code hoặc network blip. Bug retry vô ích;
          network blip BE re-publish thủ công sau khi infra recover.

        AI reply vẫn đi về BE qua publisher.submit(reply) ở caller — BE biết
        kết quả ngay sau attempt đầu.
        """
        reason = (
            "permanent failure"
            if kind is FailureKind.PERMANENT
            else "transient failure (no retry)"
        )
        log_dlq(worker_id=self.worker_id, request_id=msg.id, reason=reason)
        envelope.dead_letter(reason=reason)

    # ───────────────────── helpers ─────────────────────

    def _log_queue_sizes(self) -> None:
        sizes = self.router.queues.sizes()
        logger.info(
            f"[worker {self.worker_id}] [QUEUE_SIZES] "
            f"{datetime.now():%Y-%m-%d %H:%M:%S} "
            f"version={self.settings.service_version} "
            + " ".join(f"{k}:{v}" for k, v in sizes.items())
        )
