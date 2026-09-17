"""
Process-wide shutdown signal. SIGTERM (Docker stop, k8s) and SIGINT (Ctrl+C)
both flip a single threading.Event that consumers/workers/publisher poll.
"""

from __future__ import annotations

import signal
import threading

from ..observability import logger

_shutdown = threading.Event()


def install_handlers() -> None:
    """Register SIGTERM + SIGINT to set the shutdown event. Idempotent."""
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)


def shutdown_event() -> threading.Event:
    return _shutdown


def is_shutting_down() -> bool:
    return _shutdown.is_set()


def request_shutdown() -> None:
    """Programmatic trigger (e.g. from healthcheck failure)."""
    _shutdown.set()


def _on_signal(signum, _frame) -> None:
    name = signal.Signals(signum).name
    if _shutdown.is_set():
        logger.warning(f"[shutdown] {name} received again — forcing exit")
        # Second signal: bypass graceful drain. Re-raise default handler.
        signal.signal(signum, signal.SIG_DFL)
        return
    logger.info(f"[shutdown] {name} received — initiating graceful shutdown")
    _shutdown.set()
