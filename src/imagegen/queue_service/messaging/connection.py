import random

import pika

from ..config import RabbitMQConfig


def build_connection_params(
    cfg: RabbitMQConfig, heartbeat: int | None = None
) -> pika.ConnectionParameters:
    return pika.ConnectionParameters(
        host=cfg.host,
        port=cfg.port,
        virtual_host=cfg.vhost,
        credentials=pika.PlainCredentials(
            username=cfg.user, password=cfg.password
        ),
        heartbeat=heartbeat if heartbeat is not None else cfg.heartbeat,
        blocked_connection_timeout=60,
    )


class BackoffPolicy:
    """
    Exponential backoff với jitter cho reconnect khi broker down.

    Thay vì retry flat mỗi N giây (broker down 1h = 360 reconnect attempt
    spam log), tăng dần delay theo công thức:

        delay = min(initial * 2^attempt, max) * (1 + jitter * random[-1, 1])

    Mỗi lần connect thành công → reset attempt = 0.

    Defaults:
      - initial=1s, max=60s, jitter=0.2 (±20%)
      - attempt 0 → ~1s, 1 → ~2s, 2 → ~4s, ..., 6+ → ~60s (capped)

    Jitter giúp avoid thundering herd khi N container cùng reconnect lúc
    broker recover — tránh spike connection burst.
    """

    def __init__(
        self,
        initial_seconds: float = 1.0,
        max_seconds: float = 60.0,
        jitter: float = 0.2,
    ) -> None:
        if initial_seconds <= 0:
            raise ValueError("initial_seconds must be > 0")
        if max_seconds < initial_seconds:
            raise ValueError("max_seconds must be >= initial_seconds")
        if not 0 <= jitter < 1:
            raise ValueError("jitter must be in [0, 1)")
        self._initial = initial_seconds
        self._max = max_seconds
        self._jitter = jitter
        self._attempt = 0

    def next_delay(self) -> float:
        """Compute delay cho lần reconnect hiện tại + tăng attempt counter."""
        base = min(self._initial * (2**self._attempt), self._max)
        noise = base * self._jitter * (2 * random.random() - 1)  # nosec - non-cryptographic
        delay = max(0.0, base + noise)
        self._attempt += 1
        return delay

    def reset(self) -> None:
        """Gọi sau khi connect thành công."""
        self._attempt = 0

    @property
    def attempt(self) -> int:
        return self._attempt
