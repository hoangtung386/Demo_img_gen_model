"""
Tiny HTTP healthcheck server (stdlib only). Exposes /healthz and /readyz.

Usage:
    server = HealthServer(port=8395)
    server.start()
    server.set_ready(True)   # once model + RMQ ready
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .observability import logger


class _State:
    ready: bool = False


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self._respond(200, b"ok\n")
        elif self.path == "/readyz":
            if _State.ready:
                self._respond(200, b"ready\n")
            else:
                self._respond(503, b"not ready\n")
        else:
            self._respond(404, b"not found\n")

    def _respond(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):  # silence default access log
        return


class HealthServer:
    def __init__(self, port: int = 8395, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="healthcheck", daemon=True
        )
        self._thread.start()
        logger.info(f"[health] serving {self.host}:{self.port}")

    def set_ready(self, ready: bool) -> None:
        _State.ready = ready
        logger.info(f"[health] ready={ready}")
