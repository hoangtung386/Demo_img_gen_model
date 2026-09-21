"""
Idempotent RabbitMQ topology setup.

Declares for each consumer queue:
  • The main queue with x-dead-letter-exchange → retry exchange (routed by queue name)
  • A retry queue (TTL=N ms) that dead-letters back to the main exchange
  • A DLQ (no TTL) where exhausted-retry messages land for inspection

Topology overview (per inbound queue Q):

    main exchange ──── routing_key=Q ────▶ Q
        ▲                                  │ on nack/expire
        │                                  ▼
        │                         retry exchange ── routing_key=Q ─▶ Q.retry (TTL)
        │                                                             │
        └─────────── dead-lettered after TTL ─────────────────────────┘

    Q.dlq fed manually via `Envelope.dead_letter()` (publisher publishes to
    DLQ exchange routing_key=Q.dlq). Inspected manually by oncall.

Run once at service startup; safe to re-run.
"""

from __future__ import annotations

import pika

from ..config import RabbitMQConfig
from ..observability import logger


def retry_exchange_name(cfg: RabbitMQConfig) -> str:
    return f"{cfg.exchange}{cfg.retry_exchange_suffix}"


def dlq_exchange_name(cfg: RabbitMQConfig) -> str:
    return f"{cfg.exchange}{cfg.dlq_suffix}"


def retry_queue_name(cfg: RabbitMQConfig, queue: str) -> str:
    return f"{queue}{cfg.retry_exchange_suffix}"


def dlq_queue_name(cfg: RabbitMQConfig, queue: str) -> str:
    return f"{queue}{cfg.dlq_suffix}"


def setup_topology(channel, cfg: RabbitMQConfig) -> None:
    """Declare exchanges + per-queue retry/DLQ topology. Idempotent.

    3 modes:

    - `"declare"` (default): full topology — main exchange + `.retry` exchange
      + `.dlq` exchange, each inbound queue gets `x-dead-letter-exchange` args
      + `.retry` sibling (TTL) + `.dlq` sibling. Failure routing dùng broker
      DLX + retry queue. Yêu cầu BE/DevOps chấp nhận args trên queue.

    - `"declare_minimal"`: chỉ exchange + queue + bind, KHÔNG DLX args, KHÔNG
      `.retry`/`.dlq` siblings. Tương thích broker do BE pre-create queue
      legacy không có args. Failure path → service publish reply qua
      `queue_out` (BE biết kết quả) + ack message gốc. Audit trail trên disk.

    - `"passive"`: queue đã tồn tại trên broker do bên khác quản lý. Service
      chỉ verify existence (passive declare) — KHÔNG redeclare bất kỳ thứ gì.
      Failure routing giống `declare_minimal` (ack + audit log).
    """
    if cfg.topology_mode == "passive":
        _verify_passive(channel, cfg)
        return

    if cfg.topology_mode == "declare_minimal":
        _declare_minimal(channel, cfg)
        return

    main = cfg.exchange
    retry_x = retry_exchange_name(cfg)
    dlq_x = dlq_exchange_name(cfg)

    # Exchanges
    channel.exchange_declare(exchange=main, exchange_type="direct", durable=True)
    channel.exchange_declare(exchange=retry_x, exchange_type="direct", durable=True)
    channel.exchange_declare(exchange=dlq_x, exchange_type="direct", durable=True)

    # Outbound (publisher) queue — no DLX; replies are terminal.
    channel.queue_declare(queue=cfg.queue_out, durable=True)
    channel.queue_bind(
        exchange=main, queue=cfg.queue_out, routing_key=cfg.queue_out_key
    )

    # Inbound queues + their retry/DLQ siblings.
    for queue in cfg.queues_in.values():
        _declare_queue_with_retry(
            channel, cfg, queue, main_x=main, retry_x=retry_x, dlq_x=dlq_x
        )

    from ..observability import log_topology_declared

    log_topology_declared(
        exchanges=[main, retry_x, dlq_x], queues=list(cfg.queues_in.values())
    )
    logger.debug(
        f"[topology] details: exchanges={[main, retry_x, dlq_x]} "
        f"queues={list(cfg.queues_in.values())}"
    )


def _declare_minimal(channel, cfg: RabbitMQConfig) -> None:
    """Tối giản: exchange + N inbound queue + 1 outbound queue + (N+1) bind.

    KHÔNG args DLX trên queue → tương thích với queue legacy do BE pre-create
    không có DLX. Idempotent — restart service không lỗi.

      • exchange_declare(main, direct, durable=True)
      • for each Q in queues_in.values(): queue_declare(Q, durable=True, no args)
      •                                     queue_bind(main, Q, routing_key=Q)
      • queue_declare(queue_out, durable=True)
      • queue_bind(main, queue_out, routing_key=queue_out_key)
    """
    main = cfg.exchange
    channel.exchange_declare(exchange=main, exchange_type="direct", durable=True)

    # Outbound queue + bind.
    channel.queue_declare(queue=cfg.queue_out, durable=True)
    channel.queue_bind(
        exchange=main, queue=cfg.queue_out, routing_key=cfg.queue_out_key
    )

    # Inbound queues + bind (routing_key = queue name, direct exchange).
    for queue in cfg.queues_in.values():
        channel.queue_declare(queue=queue, durable=True)
        channel.queue_bind(exchange=main, queue=queue, routing_key=queue)

    from ..observability import log_topology_declared

    queues = [cfg.queue_out, *cfg.queues_in.values()]
    log_topology_declared(exchanges=[main], queues=list(cfg.queues_in.values()))
    logger.info(
        f"[topology] declare_minimal — exchange={main} queues={queues} (no DLX, no "
        f"retry, no DLQ)"
    )


def _verify_passive(channel, cfg: RabbitMQConfig) -> None:
    """Verify queues exist without modifying broker state. Raises if missing."""
    queues = [cfg.queue_out, *cfg.queues_in.values()]
    for q in queues:
        channel.queue_declare(queue=q, passive=True)
    logger.info(
        f"[topology] passive mode — verified {len(queues)} queues exist; "
        f"retry/DLQ siblings NOT created (managed externally)"
    )


def _declare_queue_with_retry(
    channel, cfg: RabbitMQConfig, queue: str, *, main_x: str, retry_x: str, dlq_x: str
) -> None:
    retry_q = retry_queue_name(cfg, queue)
    dlq_q = dlq_queue_name(cfg, queue)

    # Main queue: failed messages → retry exchange routed by queue name.
    channel.queue_declare(
        queue=queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": retry_x,
            "x-dead-letter-routing-key": queue,
        },
    )
    channel.queue_bind(exchange=main_x, queue=queue, routing_key=queue)

    # Retry queue: TTL on the queue; after expiry, message dead-letters back
    # into the main exchange routed by the original queue name.
    channel.queue_declare(
        queue=retry_q,
        durable=True,
        arguments={
            "x-dead-letter-exchange": main_x,
            "x-dead-letter-routing-key": queue,
            "x-message-ttl": cfg.retry_initial_delay_ms,
        },
    )
    channel.queue_bind(exchange=retry_x, queue=retry_q, routing_key=queue)

    # DLQ: terminal. No TTL.
    channel.queue_declare(queue=dlq_q, durable=True)
    channel.queue_bind(exchange=dlq_x, queue=dlq_q, routing_key=queue)


def declare_topology_standalone(cfg: RabbitMQConfig) -> None:
    """Open a short-lived connection just to declare topology, then close."""
    from .connection import build_connection_params

    params = build_connection_params(cfg, heartbeat=30)
    conn = pika.BlockingConnection(params)
    try:
        channel = conn.channel()
        setup_topology(channel, cfg)
    finally:
        conn.close()
