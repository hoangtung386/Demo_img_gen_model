"""
BrokerHealthMonitor — proactive broker reachability check.

Vấn đề: consumer + publisher chỉ biết broker down khi đang dùng (publish/
consume). `/readyz` endpoint trả 200 mặc dù broker đã down → load balancer
+ k8s không biết để route traffic đi container khác.

Solution: thread chạy ngầm, ping TCP socket tới broker mỗi N giây. Update
1 flag `broker_alive` mà health server đọc → flip `/readyz` → 503 khi
broker không reach.

Tách biệt với consumer/publisher reconnect loop (chúng vẫn tự retry với
exponential backoff). Healthcheck chỉ là _signal_ cho LB/k8s.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable

from ..config import RabbitMQConfig
from ..observability import logger
from .shutdown import is_shutting_down


class BrokerHealthMonitor:
    """
    Background thread checking broker TCP reachability.

    Usage:
        monitor = BrokerHealthMonitor(cfg, on_state_change=health.set_ready)
        monitor.start()  # spawns daemon thread
        ...
        monitor.is_alive_now()  # current state, thread-safe read
    """

    def __init__(
        self,
        cfg: RabbitMQConfig,
        on_state_change: Callable[[bool], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._interval = cfg.broker_healthcheck_interval_seconds
        self._timeout = cfg.broker_healthcheck_timeout_seconds
        self._on_state_change = on_state_change
        self._alive = True  # optimistic until first check
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # Chưa có chỗ gọi: health endpoint nhận trạng thái qua callback
    # `on_state_change` chứ không hỏi ngược. Giữ vì đây là cách duy nhất
    # đọc trạng thái an toàn giữa các thread. Xem docs/QUEUE_SERVICE.md.
    def is_alive_now(self) -> bool:
        with self._lock:
            return self._alive

    def start(self) -> None:
        """Spawn daemon thread. No-op nếu interval <= 0 (tắt healthcheck)."""
        if self._interval <= 0:
            logger.info("[broker-health] disabled (interval <= 0)")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_forever,
            name="broker-health",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            f"[broker-health] started interval={self._interval}s "
            f"timeout={self._timeout}s"
        )

    def _run_forever(self) -> None:
        while not is_shutting_down():
            alive = self._probe_once()
            self._set_alive(alive)
            time.sleep(self._interval)

    def _probe_once(self) -> bool:
        """Open + close TCP socket. True nếu connect được trong timeout."""
        try:
            with socket.create_connection(
                (self._cfg.host, self._cfg.port), timeout=self._timeout
            ):
                return True
        except (TimeoutError, OSError):
            return False

    def _set_alive(self, alive: bool) -> None:
        with self._lock:
            changed = alive != self._alive
            self._alive = alive
        if changed:
            state = "UP" if alive else "DOWN"
            logger.warning(
                f"[broker-health] state change → {state} "
                f"({self._cfg.host}:{self._cfg.port})"
            )
            if self._on_state_change is not None:
                try:
                    self._on_state_change(alive)
                except Exception as e:
                    logger.warning(
                        f"[broker-health] on_state_change callback failed: {e}"
                    )
