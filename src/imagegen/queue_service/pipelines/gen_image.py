import gc
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ..audit import RequestRecord, StorageManager
from ..config import Settings
from ..messaging import (
    FailureKind,
    InboundMessage,
    OutboundMessage,
    StatusCode,
    info_for,
)
from ..observability import logger
from ..processors import ImageGenerator
from ..storage import (
    DownloadEmpty,
    DownloadFailed,
    DownloadNotFound,
    DownloadTimeout,
    StorageBackend,
    UploadAuthFailed,
    UploadFailed,
)
from .base import Pipeline, PipelineResult

# Model version dùng cho audit. Đọc từ package version của imagegen.
try:
    from ... import __version__ as _PKG_VERSION  # type: ignore
except Exception:  # pragma: no cover - __version__ có thể không tồn tại
    _PKG_VERSION = "unknown"
MODEL_VERSION = f"hidream-o1-image-sdnq-uint4@{_PKG_VERSION}"


@dataclass(frozen=True)
class _JobPaths:
    inputs: list[str]
    predict: str


class GenImagePipeline(Pipeline):
    """
    Download 0..N ảnh điều kiện → chạy HiDream-O1 (editing nếu có ảnh,
    text-to-image nếu không) → upload kết quả → reply URL.

    Status codes đi qua `messaging.status` để cùng 1 code map cùng message +
    failure_kind ở mọi nơi.

    Storage order:
      download → save_input → generate → save_output_* → upload (xóa local)
      → update_output_url → reply.
    """

    def __init__(
        self,
        settings: Settings,
        processor: ImageGenerator,
        storage: StorageBackend,
        audit: StorageManager,
    ):
        self.settings = settings
        self.processor = processor
        self.storage = storage
        self.audit = audit
        Path(settings.audit.staging_dir).mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        msg: InboundMessage,
        worker_id: int = 0,
        *,
        attempt: int = 0,
        rmq_headers: dict | None = None,
    ) -> PipelineResult:
        out = OutboundMessage(
            id=msg.id, os=msg.os, firebase_token=msg.firebase_token
        )
        started_at = time.time()
        record: RequestRecord | None = None
        staging = self._make_staging()
        paths = self._build_paths(staging, len(msg.images))

        try:
            # 1. Download ảnh điều kiện (nếu có). Text-to-image → bỏ qua bước
            # này.
            for i, url in enumerate(msg.images):
                try:
                    self.storage.download(url, paths.inputs[i])
                except (
                    DownloadNotFound,
                    DownloadTimeout,
                    DownloadFailed,
                    DownloadEmpty,
                ) as e:
                    code = self._classify_input_download_error(e)
                    record = self.audit.save_input(
                        msg.id,
                        msg.raw,
                        None,
                        "fail",
                        started_at,
                        attempt=attempt,
                        rmq_headers=rmq_headers,
                    )
                    return self._fail(
                        out,
                        msg,
                        code,
                        record,
                        started_at,
                        worker_id,
                        url=url,
                        detail=str(e),
                    )

            # 2. save_input (ảnh đầu tiên nếu có) — guess outcome=success.
            first_input = paths.inputs[0] if paths.inputs else None
            record = self.audit.save_input(
                msg.id,
                msg.raw,
                first_input,
                "success",
                started_at,
                attempt=attempt,
                rmq_headers=rmq_headers,
            )

            # 3. Load ảnh đầu vào + chạy model.
            try:
                pil_inputs = self._load_inputs(paths.inputs)
            except UnidentifiedImageError as e:
                logger.warning(f"[pipeline] decode failed id={msg.id}: {e}")
                return self._fail_with_audit(
                    out,
                    msg,
                    StatusCode.INPUT_INVALID_FORMAT,
                    record,
                    started_at,
                    worker_id,
                )

            try:
                result_image, info = self.processor.generate(
                    prompt=msg.prompt,
                    images=pil_inputs,
                    negative_prompt=msg.negative_prompt,
                    seed=msg.seed,
                    num_steps=msg.num_steps,
                    aspect_ratio=msg.aspect_ratio,
                    match_input_size=bool(msg.match_input_size),
                )
            except MemoryError as e:
                logger.exception(f"[pipeline] OOM id={msg.id}: {e}")
                return self._error_with_audit(
                    out,
                    msg,
                    StatusCode.CUDA_OUT_OF_MEMORY,
                    e,
                    record,
                    started_at,
                    worker_id,
                )
            except Exception as e:
                if self._is_cuda_oom(e):
                    logger.exception(f"[pipeline] CUDA OOM id={msg.id}: {e}")
                    return self._error_with_audit(
                        out,
                        msg,
                        StatusCode.CUDA_OUT_OF_MEMORY,
                        e,
                        record,
                        started_at,
                        worker_id,
                    )
                logger.exception(
                    f"[pipeline] inference error id={msg.id}: {e}"
                )
                return self._error_with_audit(
                    out,
                    msg,
                    StatusCode.AI_INFERENCE_ERROR,
                    e,
                    record,
                    started_at,
                    worker_id,
                    detail=str(e),
                )

            if result_image is None:
                logger.warning(f"[pipeline] no result id={msg.id}: {info}")
                return self._fail_with_audit(
                    out,
                    msg,
                    StatusCode.AI_NO_RESULT,
                    record,
                    started_at,
                    worker_id,
                )

            # 4. Save processed image then save_output_success (BEFORE upload).
            result_image.save(paths.predict)
            out.set_status(StatusCode.OK)
            out.result = {"url": "", "info": info}
            self.audit.save_output_success(
                record,
                out.to_dict(),
                paths.predict,
                self._elapsed_ms(started_at),
                worker_id,
                MODEL_VERSION,
            )

            # 5. Upload + update_output_url
            try:
                url = self.storage.upload(
                    paths.predict,
                    os_name=msg.os,
                    app_id=msg.appid,
                    device_id=msg.device_id,
                    storage_index=msg.storage_index,
                )
            except UploadAuthFailed as e:
                return self._error_with_audit(
                    out,
                    msg,
                    StatusCode.STORAGE_AUTH_FAILED,
                    e,
                    record,
                    started_at,
                    worker_id,
                )
            except UploadFailed as e:
                return self._fail(
                    out,
                    msg,
                    StatusCode.STORAGE_UPLOAD_FAILED,
                    record,
                    started_at,
                    worker_id,
                    detail=str(e),
                )

            out.result = {"url": url, "info": info}
            self.audit.update_output_url(record, url)
            return PipelineResult(reply=out, failure=FailureKind.NONE)

        except Exception as e:
            logger.exception(f"[pipeline] FATAL id={msg.id}: {e}")
            if record is None:
                record = self.audit.save_input(
                    msg.id,
                    msg.raw,
                    None,
                    "error",
                    started_at,
                    attempt=attempt,
                    rmq_headers=rmq_headers,
                )
            return self._error_with_audit(
                out,
                msg,
                StatusCode.INTERNAL_ERROR,
                e,
                record,
                started_at,
                worker_id,
                detail=str(e),
            )
        finally:
            self._delete_staging(staging)
            self._cleanup_runtime()

    # ───────────────────── classification ─────────────────────

    @staticmethod
    def _classify_input_download_error(e: Exception) -> StatusCode:
        if isinstance(e, DownloadNotFound):
            return StatusCode.INPUT_DOWNLOAD_NOT_FOUND
        if isinstance(e, DownloadEmpty):
            return (
                StatusCode.INPUT_INVALID_FORMAT
            )  # 0-byte body = invalid input
        if isinstance(e, DownloadTimeout):
            return StatusCode.INPUT_DOWNLOAD_TIMEOUT
        return StatusCode.INPUT_DOWNLOAD_FAILED

    @staticmethod
    def _is_cuda_oom(e: Exception) -> bool:
        try:
            import torch

            if isinstance(e, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        return "out of memory" in str(e).lower() and "cuda" in str(e).lower()

    # ───────────────────── stages ─────────────────────

    def _make_staging(self) -> Path:
        root = Path(self.settings.audit.staging_dir)
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="job_", dir=root))

    @staticmethod
    def _delete_staging(d: Path) -> None:
        with suppress(FileNotFoundError, OSError):
            for p in d.glob("*"):
                with suppress(OSError):
                    p.unlink()
            d.rmdir()

    @staticmethod
    def _build_paths(staging: Path, n_inputs: int) -> _JobPaths:
        inputs = [str(staging / f"input_{i}.png") for i in range(n_inputs)]
        return _JobPaths(inputs=inputs, predict=str(staging / "output.png"))

    @staticmethod
    def _load_inputs(paths: list[str]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for p in paths:
            with Image.open(p) as im:
                images.append(im.convert("RGB").copy())
        return images

    # ───────────────────── reply helpers ─────────────────────

    def _fail(
        self, out, msg, code, record, started_at, worker_id, **fmt
    ) -> PipelineResult:
        return self._reply(
            out, msg, code, record, started_at, worker_id, fmt, exception=None
        )

    def _fail_with_audit(
        self, out, msg, code, record, started_at, worker_id, **fmt
    ) -> PipelineResult:
        return self._reply(
            out, msg, code, record, started_at, worker_id, fmt, exception=None
        )

    def _error_with_audit(
        self, out, msg, code, exception, record, started_at, worker_id, **fmt
    ) -> PipelineResult:
        return self._reply(
            out,
            msg,
            code,
            record,
            started_at,
            worker_id,
            fmt,
            exception=exception,
        )

    def _reply(
        self, out, msg, code, record, started_at, worker_id, fmt, *, exception
    ) -> PipelineResult:
        out.set_status(code, **fmt)
        failure_kind = info_for(code).failure_kind

        routing = "dlq" if failure_kind is FailureKind.PERMANENT else "retry"
        elapsed_ms = self._elapsed_ms(started_at)

        if record is not None:
            if exception is not None:
                self.audit.save_output_error(
                    record,
                    out.to_dict(),
                    exception,
                    elapsed_ms,
                    worker_id,
                    msg.raw,
                    failure_kind=failure_kind.value,
                    routing=routing,
                )
            else:
                self.audit.save_output_fail(
                    record,
                    out.to_dict(),
                    error_message=out.message,
                    elapsed_ms=elapsed_ms,
                    worker_id=worker_id,
                    message=msg.raw,
                    failure_kind=failure_kind.value,
                    routing=routing,
                    max_attempts=self.settings.rabbitmq.retry_max_attempts,
                )
        return PipelineResult(reply=out, failure=failure_kind)

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (time.time() - started_at) * 1000.0

    @staticmethod
    def _cleanup_runtime() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
