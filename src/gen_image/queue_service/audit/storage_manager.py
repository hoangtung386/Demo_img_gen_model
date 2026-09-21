"""
Per-request audit storage.

Layout:
    <root_dir>/<service_name>/<Y>/<M>/<D>/
        success/<request_id>/[attempt_N/]
            input.json     full payload + checksums + dimensions
            input.png      first input image (if save_input_image + có ảnh)
            output.json    reply + status + failure_kind + attempt info
            output.png     generated image (if save_output_image)
        fail/<request_id>/[attempt_N/]
            input.json + input.png + output.json + error.log
        error/<request_id>/[attempt_N/]
            input.json + output.json + error.log + traceback.txt

Per-request order in pipeline:
    download → save_input → generate → save_output_* → upload (deletes local)
    → update_output_url → publisher.submit reply

Multi-attempt: when the same request_id is processed more than once (RMQ retry
after transient failure), each attempt writes into its own
<request_id>/attempt_<N>/ subfolder so previous attempts are preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from ..config import AuditConfig
from ..observability import logger

OutcomeKind = Literal["success", "fail", "error"]

# Top-level fields surfaced from the message payload for quick scanning. The
# COMPLETE payload is also stored under "raw_message" so service-specific
# fields are never lost.
_COMMON_INPUT_FIELDS = (
    "os",
    "appid",
    "device_id",
    "firebase_token",
    "country",
    "prompt",
    "negative_prompt",
    "seed",
    "num_steps",
    "aspect_ratio",
)


@dataclass
class RequestRecord:
    """Handle to one request's audit folder. Created by save_input()."""

    request_dir: Path
    request_id: str
    outcome: OutcomeKind
    started_at: float
    attempt: int


class StorageManager:
    """Per-request audit writer. Each call writes its own files (thread-safe)."""

    def __init__(self, cfg: AuditConfig):
        self.cfg = cfg
        self.service_root = Path(cfg.root_dir) / cfg.service_name

    # ───────────────────── public API ─────────────────────

    def save_input(
        self,
        request_id: str,
        message: dict[str, Any],
        input_image_path: str | None,
        outcome: OutcomeKind,
        started_at: float | None = None,
        *,
        attempt: int = 0,
        rmq_headers: dict[str, Any] | None = None,
    ) -> RequestRecord:
        """
        Create the request's attempt folder, write input.json, copy the first
        input image (if any). Returns a RequestRecord the caller passes back.
        """
        if not self.cfg.enabled:
            return RequestRecord(
                Path(os.devnull), request_id, outcome, time.time(), attempt
            )

        started_at = started_at or time.time()
        safe_id = _sanitize_id(request_id)
        req_dir = self._request_dir(outcome, started_at, safe_id, attempt)
        req_dir.mkdir(parents=True, exist_ok=True)

        input_meta = (
            self._maybe_copy_image(input_image_path, req_dir, "input")
            if self.cfg.save_input_image
            else None
        )
        self._write_input_json(
            req_dir / "input.json",
            request_id=request_id,
            message=message,
            started_at=started_at,
            attempt=attempt,
            rmq_headers=rmq_headers,
            input_meta=input_meta,
        )
        return RequestRecord(req_dir, request_id, outcome, started_at, attempt)

    def save_output_success(
        self,
        record: RequestRecord,
        reply: dict[str, Any],
        output_image_paths: list[str] | str | None,
        elapsed_ms: float,
        worker_id: int,
        model_version: str = "",
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.cfg.enabled:
            return
        self._move_to_outcome(record, "success")
        output_meta = self._copy_output_images(output_image_paths, record.request_dir)
        self._write_output_json(
            record,
            reply=reply,
            elapsed_ms=elapsed_ms,
            worker_id=worker_id,
            model_version=model_version,
            failure_kind="none",
            output_meta=output_meta,
            extra=extra,
        )

    def save_output_fail(
        self,
        record: RequestRecord,
        reply: dict[str, Any],
        error_message: str,
        elapsed_ms: float,
        worker_id: int,
        message: dict[str, Any] | None = None,
        *,
        failure_kind: str = "permanent",
        routing: str = "",
        max_attempts: int = 0,
    ) -> None:
        if not self.cfg.enabled:
            return
        self._move_to_outcome(record, "fail")
        self._write_output_json(
            record,
            reply=reply,
            elapsed_ms=elapsed_ms,
            worker_id=worker_id,
            failure_kind=failure_kind,
            extra={"routing_decision": routing, "max_attempts": max_attempts},
        )
        self._write_error_log(
            record,
            kind="FAIL",
            error_message=error_message,
            reply=reply,
            elapsed_ms=elapsed_ms,
            worker_id=worker_id,
            message=message,
            failure_kind=failure_kind,
            routing=routing,
        )

    def save_output_error(
        self,
        record: RequestRecord,
        reply: dict[str, Any],
        exception: BaseException,
        elapsed_ms: float,
        worker_id: int,
        message: dict[str, Any] | None = None,
        *,
        failure_kind: str = "transient",
        routing: str = "",
    ) -> None:
        if not self.cfg.enabled:
            return
        self._move_to_outcome(record, "error")
        self._write_output_json(
            record,
            reply=reply,
            elapsed_ms=elapsed_ms,
            worker_id=worker_id,
            failure_kind=failure_kind,
            extra={
                "error_type": type(exception).__name__,
                "error_message": str(exception),
                "routing_decision": routing,
            },
        )
        self._write_error_log(
            record,
            kind="ERROR",
            error_message=f"{type(exception).__name__}: {exception}",
            reply=reply,
            elapsed_ms=elapsed_ms,
            worker_id=worker_id,
            message=message,
            failure_kind=failure_kind,
            routing=routing,
        )
        self._write_traceback(record, exception)

    def update_output_url(self, record: RequestRecord, url: str) -> None:
        """Patch output.json after upload completes (URL not known earlier)."""
        if not self.cfg.enabled:
            return
        path = record.request_dir / "output.json"
        try:
            data = json.loads(path.read_text()) if path.exists() else {}
            result = data.setdefault("result", {})
            result["url"] = url
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except OSError as e:
            logger.warning(f"[audit] update_output_url failed for {path}: {e}")

    # ───────────────────── path helpers ─────────────────────

    def _request_dir(
        self, outcome: OutcomeKind, ts: float, safe_id: str, attempt: int
    ) -> Path:
        d = datetime.fromtimestamp(ts)
        base = (
            self.service_root
            / str(d.year)
            / str(d.month)
            / str(d.day)
            / outcome
            / safe_id
        )
        # Always nest under attempt_<N> so retries don't clobber.
        return base / f"attempt_{attempt}"

    def _move_to_outcome(self, record: RequestRecord, new_outcome: OutcomeKind) -> None:
        """If save_input was called with a guess, move the folder to the actual "
        "outcome."""
        if record.outcome == new_outcome:
            return
        old_dir = record.request_dir
        safe_id = _sanitize_id(record.request_id)
        new_dir = self._request_dir(
            new_outcome, record.started_at, safe_id, record.attempt
        )
        try:
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            if old_dir.exists():
                if new_dir.exists():
                    shutil.rmtree(new_dir)
                shutil.move(str(old_dir), str(new_dir))
            # Remove empty parent dirs from the old outcome side.
            old_request_root = old_dir.parent  # <old_outcome>/<request_id>/
            with suppress(OSError):
                if old_request_root.exists() and not any(old_request_root.iterdir()):
                    old_request_root.rmdir()
            record.request_dir = new_dir
            record.outcome = new_outcome
        except OSError as e:
            logger.warning(f"[audit] move {old_dir} → {new_dir} failed: {e}")

    # ───────────────────── writers ─────────────────────

    def _write_input_json(
        self,
        path: Path,
        *,
        request_id: str,
        message: dict[str, Any],
        started_at: float,
        attempt: int,
        rmq_headers: dict[str, Any] | None,
        input_meta: dict[str, Any] | None,
    ) -> None:
        data: dict[str, Any] = {
            "request_id": request_id,
            "timestamp": datetime.fromtimestamp(started_at).isoformat(),
            "timestamp_epoch": started_at,
            "attempt": attempt,
            "service": self.cfg.service_name,
        }
        # Surface common fields at top-level for quick scanning.
        for key in _COMMON_INPUT_FIELDS:
            if key in message:
                data[key] = message[key]
        # Ảnh điều kiện (0..N URL). `images` mới, `image` cũ (1 URL) — cả hai.
        if "images" in message:
            data["image_urls"] = message["images"]
        elif "image" in message:
            data["image_urls"] = [message["image"]]
        if rmq_headers:
            data["rmq_headers"] = rmq_headers
        if input_meta:
            data["input_image"] = input_meta
        # Always include the full original payload so service-specific fields
        # are never lost, even if not whitelisted above.
        data["raw_message"] = message
        try:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except OSError as e:
            logger.warning(f"[audit] write input.json {path}: {e}")

    def _maybe_copy_image(
        self, src: str | None, req_dir: Path, name: str
    ) -> dict[str, Any] | None:
        """Copy src into req_dir/<name>.<ext>; return metadata dict or None."""
        if not src:
            return None
        src_path = Path(src)
        if not src_path.exists():
            return None
        ext = src_path.suffix or ".png"
        dst = req_dir / f"{name}{ext}"
        try:
            shutil.copy2(src_path, dst)
        except OSError as e:
            logger.warning(f"[audit] copy {src} → {dst}: {e}")
            return None
        return self._image_metadata(dst)

    def _copy_output_images(
        self, srcs: list[str] | str | None, req_dir: Path
    ) -> list[dict[str, Any]]:
        if not srcs or not self.cfg.save_output_image:
            return []
        items = [srcs] if isinstance(srcs, str) else list(srcs)
        out_meta: list[dict[str, Any]] = []
        for idx, src in enumerate(items):
            if not src:
                continue
            sp = Path(src)
            if not sp.exists():
                continue
            ext = sp.suffix or ".png"
            name = "output" if len(items) == 1 else f"output_{idx}"
            dst = req_dir / f"{name}{ext}"
            try:
                shutil.copy2(sp, dst)
            except OSError as e:
                logger.warning(f"[audit] copy output {src} → {dst}: {e}")
                continue
            out_meta.append(self._image_metadata(dst))
        return out_meta

    def _image_metadata(self, path: Path) -> dict[str, Any]:
        meta: dict[str, Any] = {"path": str(path.name), "size_bytes": _safe_size(path)}
        if self.cfg.compute_checksum:
            meta["sha256"] = _sha256(path)
        try:
            with Image.open(path) as im:
                meta["width"], meta["height"] = im.size
                meta["format"] = im.format
        except Exception:
            pass
        return meta

    def _write_output_json(
        self,
        record: RequestRecord,
        *,
        reply: dict[str, Any],
        elapsed_ms: float,
        worker_id: int,
        failure_kind: str,
        model_version: str = "",
        output_meta: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        data: dict[str, Any] = {
            "request_id": record.request_id,
            "attempt": record.attempt,
            "outcome": record.outcome,
            "failure_kind": failure_kind,
            "status_code": reply.get("status_code", 0),
            "message": reply.get("message", ""),
            "result": reply.get("result", {}),
            "processing_time_ms": round(elapsed_ms, 2),
            "worker_id": worker_id,
            "model_version": model_version,
            "service": self.cfg.service_name,
            "completed_at": datetime.now().isoformat(),
        }
        if output_meta:
            data["output_images"] = output_meta
        if extra:
            data.update(extra)
        try:
            (record.request_dir / "output.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False)
            )
        except OSError as e:
            logger.warning(f"[audit] write output.json: {e}")

    def _write_error_log(
        self,
        record: RequestRecord,
        *,
        kind: str,
        error_message: str,
        reply: dict[str, Any],
        elapsed_ms: float,
        worker_id: int,
        message: dict[str, Any] | None,
        failure_kind: str,
        routing: str,
    ) -> None:
        msg = message or {}
        images = msg.get("images") or ([msg["image"]] if msg.get("image") else [])
        lines = [
            "=" * 60,
            f"  ERROR LOG - {kind}",
            "=" * 60,
            "",
            f"Time:            {datetime.now().isoformat()}",
            f"Processing:      {elapsed_ms / 1000:.3f}s",
            f"Request ID:      {record.request_id}",
            f"Attempt:         {record.attempt}",
            f"Worker:          {worker_id}",
            f"Status Code:     {reply.get('status_code', '')}",
            f"Message:         {reply.get('message', '')}",
            f"Failure Kind:    {failure_kind}",
            f"Routing:         {routing}",
            f"OS:              {msg.get('os', '')}",
            f"App ID:          {msg.get('appid', '')}",
            f"Device ID:       {msg.get('device_id', '')}",
            f"Country:         {msg.get('country', '')}",
            f"Prompt:          {msg.get('prompt', '')}",
            f"Image URLs:      {images}",
            "",
            f"Error: {error_message}",
            "=" * 60,
            "",
        ]
        try:
            (record.request_dir / "error.log").write_text("\n".join(lines))
        except OSError as e:
            logger.warning(f"[audit] write error.log: {e}")

    @staticmethod
    def _write_traceback(record: RequestRecord, exception: BaseException) -> None:
        try:
            tb = "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )
            stamp = datetime.now().isoformat()
            header = f"[{stamp}] {type(exception).__name__}: {exception}\n\n"
            (record.request_dir / "traceback.txt").write_text(header + tb)
        except OSError as e:
            logger.warning(f"[audit] write traceback.txt: {e}")


# ───────────────────── helpers ─────────────────────

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_id(request_id: str) -> str:
    """Filesystem-safe folder name. Keeps the first 100 chars."""
    safe = _SAFE_ID_RE.sub("_", request_id or "unknown")
    return safe[:100]


def _safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


# ───────────────────── module-level singleton ─────────────────────

_storage_manager: StorageManager | None = None


def init_storage_manager(cfg: AuditConfig) -> StorageManager:
    global _storage_manager
    _storage_manager = StorageManager(cfg)
    return _storage_manager


def get_storage_manager() -> StorageManager:
    if _storage_manager is None:
        raise RuntimeError(
            "StorageManager not initialized; call init_storage_manager() first"
        )
    return _storage_manager
