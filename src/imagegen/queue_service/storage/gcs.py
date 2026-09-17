import os
import urllib.error
import urllib.request
from contextlib import suppress
from datetime import datetime, timedelta, timezone

from google.api_core import exceptions as gax_exceptions
from google.cloud import storage

from ..config import StorageBackendConfig
from ..observability import logger
from .base import (
    DownloadEmpty,
    DownloadFailed,
    DownloadNotFound,
    DownloadTimeout,
    StorageBackend,
    UploadAuthFailed,
    UploadFailed,
)
from .object_naming import build_object_key

# Python 3.10 compat — datetime.UTC was added in 3.11.
UTC = timezone.utc

_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".mp4": "video/mp4",
    ".webp": "image/webp",
}
_DEFAULT_CONTENT_TYPE = "application/octet-stream"
_UPLOAD_RETRIES = 2


class GCSStorage(StorageBackend):
    """Single-backend GCS storage. Thường dùng qua `MultiBackendStorage`
    factory dispatch theo `storage_index` thay vì instantiate trực tiếp."""

    def __init__(self, cfg: StorageBackendConfig):
        self.cfg = cfg
        self._client = storage.Client.from_service_account_json(
            cfg.credentials_path
        )
        self._bucket = self._client.bucket(cfg.bucket)

    def download(self, url: str, destination: str) -> None:
        try:
            urllib.request.urlretrieve(url, destination)
        except urllib.error.HTTPError as e:
            # HTTP 4xx (kể cả 400, 401, 403, 404, 410, 416) đều là PERMANENT
            # — signed URL hết hạn / sai key / object bị xóa — retry vô ích.
            # Treat as DownloadNotFound → status 410 → DLQ ngay.
            # HTTP 5xx là TRANSIENT → DownloadFailed → retry hợp lý.
            if 400 <= e.code < 500:
                raise DownloadNotFound(
                    f"HTTP {e.code} for {url}: {e.reason}"
                ) from e
            raise DownloadFailed(f"HTTP {e.code} for {url}: {e.reason}") from e
        except TimeoutError as e:
            raise DownloadTimeout(str(e)) from e
        except (urllib.error.URLError, OSError) as e:
            inner = getattr(e, "reason", None)
            if isinstance(inner, TimeoutError):
                raise DownloadTimeout(str(e)) from e
            raise DownloadFailed(str(e)) from e

        # Empty body: upstream returned 200 OK but file is 0 bytes. PERMANENT —
        # retry won't help, treat as invalid input.
        if os.path.getsize(destination) == 0:
            with suppress(OSError):
                os.unlink(destination)
            raise DownloadEmpty(f"empty body (0 bytes) returned for {url}")

    def upload(
        self,
        source_path: str,
        *,
        os_name: str,
        app_id: str,
        device_id: str,
        feature: str | None = None,
        storage_index: int | None = None,
    ) -> str:
        destination = build_object_key(
            folder=self.cfg.folder,
            app_id=app_id,
            feature=feature or self.cfg.feature,
            os_name=os_name,
            device_id=device_id,
            source_path=source_path,
        )

        last_error: Exception | None = None
        for attempt in range(_UPLOAD_RETRIES):
            try:
                blob = self._bucket.blob(destination)
                blob.upload_from_filename(
                    source_path, content_type=_content_type(source_path)
                )
                logger.info(
                    f"[gcs] uploaded {destination} bucket={self.cfg.bucket}"
                )

                url = self._build_url(blob, destination)
                self._delete_local(source_path)
                return url
            except (
                gax_exceptions.Unauthenticated,
                gax_exceptions.PermissionDenied,
            ) as e:
                logger.error(f"[gcs] auth failed: {e}")
                raise UploadAuthFailed(str(e)) from e
            except Exception as e:
                logger.error(f"[gcs] upload attempt {attempt + 1} failed: {e}")
                last_error = e
        raise UploadFailed(
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _build_url(self, blob: storage.Blob, destination: str) -> str:
        """CDN-first nếu config có `cdn_base_url`, fallback signed URL."""
        if self.cfg.cdn_base_url:
            base = self.cfg.cdn_base_url.rstrip("/")
            return f"{base}/{destination}"
        return blob.generate_signed_url(
            expiration=datetime.now(UTC)
            + timedelta(minutes=self.cfg.signed_url_ttl_minutes),
            method="GET",
            version="v4",
        )

    @staticmethod
    def _delete_local(path: str) -> None:
        with suppress(FileNotFoundError):
            os.remove(path)


def _content_type(path: str) -> str:
    return _CONTENT_TYPES.get(
        os.path.splitext(path)[1].lower(), _DEFAULT_CONTENT_TYPE
    )
