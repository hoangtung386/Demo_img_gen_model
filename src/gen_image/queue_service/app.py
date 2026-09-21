import queue
import threading

from .audit import (
    init_rmq_error_logger,
    init_storage_manager,
    log_rmq_error,
    start_cleanup_thread,
)
from .config import Settings
from .health import HealthServer
from .messaging import (
    BrokerHealthMonitor,
    Envelope,
    OutboundMessage,
    RabbitMQConsumer,
    RabbitMQPublisher,
    declare_topology_standalone,
    install_handlers,
    shutdown_event,
    validate_inbound,
)
from .observability import (
    configure_helpers,
    log_ready,
    log_shutdown,
    logger,
    setup_logging,
)
from .pipelines import GenImagePipeline
from .processors import create_processor
from .queueing import (
    PREMIUM_LABEL,
    PriorityRouter,
    RecentRequestCache,
    TieredQueues,
    basic_label,
    put_blocking,
)
from .storage import create_storage
from .workers import AIWorker, ResourceMonitor

_HEALTH_PORT = 8395
_THREAD_JOIN_TIMEOUT_SECONDS = 30.0


class Application:
    """
    DI container + lifecycle.

    Bootstrap order:
      1. install signal handlers (SIGTERM/SIGINT → shutdown event)
      2. init audit subsystem (storage manager + rmq error logger + cleanup thread)
      3. declare RabbitMQ topology (idempotent)
      4. start health server (/healthz=200, /readyz=503 until ready)
      5. build storage / processor / pipeline (model load happens here)
      6. spawn consumer threads, publisher thread, worker threads, monitor
      7. flip /readyz to 200
      8. wait on shutdown event
      9. on shutdown: drain naturally, join threads with timeout
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._threads: list[threading.Thread] = []

    def run(self) -> None:
        obs = self.settings.observability
        setup_logging(level=obs.log_level)
        configure_helpers(
            payload_max_bytes=obs.log_payload_max_bytes,
            strip_url_query=obs.strip_url_query,
        )
        install_handlers()
        logger.info(
            f"######## starting service {self.settings.service_version} ########"
        )

        self._init_audit()
        self._declare_topology()

        health = HealthServer(port=_HEALTH_PORT)
        health.start()

        storage = create_storage(self.settings.storage)
        processor = create_processor(self.settings.processor)  # model load ở đây
        audit_storage = init_storage_manager(self.settings.audit)
        pipeline = GenImagePipeline(self.settings, processor, storage, audit_storage)

        tiered, router = self._build_router()
        publisher = self._build_publisher()

        self._start_consumers(tiered, publisher)
        self._spawn(publisher.run_forever, name="publisher")
        self._start_workers(router, pipeline, publisher)
        self._spawn(ResourceMonitor().run_forever, name="monitor")

        # Audit cleanup — deletes by retention policy.
        start_cleanup_thread(self.settings.audit)

        # Proactive broker healthcheck — flip /readyz to 503 when broker
        # unreachable so load balancer/k8s can stop routing traffic here.
        broker_health = BrokerHealthMonitor(
            self.settings.rabbitmq,
            on_state_change=self._on_broker_state_change(health),
        )
        broker_health.start()

        health.set_ready(True)
        log_ready(service_version=self.settings.service_version)

        shutdown_event().wait()
        health.set_ready(False)
        log_shutdown(signal_name="SIGTERM/SIGINT")
        self._join_threads()
        logger.info("[app] [STOPPED]")

    # ───────────────────── builders ─────────────────────

    def _init_audit(self) -> None:
        # Initialize the rmq error logger early so messaging connection failures
        # during topology setup are captured.
        init_rmq_error_logger(self.settings.audit)

    def _declare_topology(self) -> None:
        try:
            declare_topology_standalone(self.settings.rabbitmq)
        except Exception as e:
            logger.error(
                f"[topology] initial declare failed (will retry inside consumers): {e}"
            )
            log_rmq_error(
                "topology_declare_failed",
                e,
                {"exchange": self.settings.rabbitmq.exchange},
            )

    def _build_router(self) -> tuple[TieredQueues, PriorityRouter]:
        wcfg = self.settings.worker
        cap = wcfg.in_queue_capacity
        tiered = TieredQueues(
            premium=queue.Queue(maxsize=cap),
            basic=[queue.Queue(maxsize=cap) for _ in wcfg.basic_tier_weights],
            weights=tuple(wcfg.basic_tier_weights),
        )
        return tiered, PriorityRouter(tiered)

    def _build_publisher(self) -> RabbitMQPublisher:
        publish_queue: queue.Queue = queue.Queue(
            maxsize=self.settings.worker.publish_queue_capacity
        )
        return RabbitMQPublisher(self.settings.rabbitmq, publish_queue)

    # ───────────────────── thread starters ─────────────────────

    def _start_consumers(
        self, tiered: TieredQueues, publisher: RabbitMQPublisher
    ) -> None:
        rmq = self.settings.rabbitmq
        tier_to_queue: dict[str, queue.Queue] = {PREMIUM_LABEL: tiered.premium}
        for idx, q in enumerate(tiered.basic):
            tier_to_queue[basic_label(idx)] = q

        for tier_name, in_queue in tier_to_queue.items():
            queue_name = rmq.queues_in.get(tier_name)
            if not queue_name:
                logger.warning(
                    f"no rabbitmq queue configured for tier {tier_name}; skipping"
                )
                continue
            handler = self._make_envelope_handler(in_queue, tier_name, publisher)
            consumer = RabbitMQConsumer(rmq, queue_name, handler, label=tier_name)
            self._spawn(consumer.run_forever, name=f"consumer-{tier_name}")

    def _start_workers(
        self, router: PriorityRouter, pipeline, publisher: RabbitMQPublisher
    ) -> None:
        # Shared per-process idempotency cache — all workers see same set.
        recent = RecentRequestCache(
            maxsize=self.settings.worker.recent_requests_capacity
        )
        for i in range(self.settings.worker.num_ai_workers):
            worker = AIWorker(
                i, self.settings, router, pipeline, publisher, recent_requests=recent
            )
            self._spawn(worker.run_forever, name=f"ai-worker-{i}")

    @staticmethod
    def _make_envelope_handler(
        in_queue: queue.Queue, label: str, publisher: RabbitMQPublisher
    ):
        """Validate the body; bad payload → reply with proper status code + DLQ."""

        def handler(envelope: Envelope) -> None:
            error = validate_inbound(envelope.body)
            if error is not None:
                logger.warning(
                    f"[{label}] invalid message → DLQ: code={int(error.code)} "
                    f"field={error.field} reason={error.reason}"
                )
                body = envelope.body if isinstance(envelope.body, dict) else {}
                reply = OutboundMessage(
                    id=str(body.get("_id", "")),
                    os=str(body.get("os", "")),
                    firebase_token=str(body.get("firebase_token", "")),
                )
                # Apply the specific code chosen by validate_inbound.
                reply.set_status(
                    error.code,
                    field=error.field,
                    reason=error.reason,
                    detail=error.reason,
                )
                if not publisher.submit(reply.to_dict()):
                    log_rmq_error(
                        "validation_error_send_failed",
                        None,
                        {"tier": label, "field": error.field, "reason": error.reason},
                    )
                envelope.dead_letter(
                    reason=f"validation: {error.field} ({error.reason})"
                )
                return

            put_blocking(in_queue, envelope)

        return handler

    def _spawn(self, target, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    @staticmethod
    def _on_broker_state_change(health: HealthServer):
        """Callback factory: broker UP → /readyz=200; DOWN → /readyz=503."""

        def callback(broker_alive: bool) -> None:
            health.set_ready(broker_alive)

        return callback

    def _join_threads(self) -> None:
        for t in self._threads:
            t.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
            if t.is_alive():
                logger.warning(
                    f"[shutdown] thread {t.name} did not stop within "
                    f"{_THREAD_JOIN_TIMEOUT_SECONDS}s"
                )


def _cli() -> None:
    """Console-script entry (`run-queue`). Load config từ base.yaml + chạy."""
    from .config import load_config

    Application(load_config()).run()
