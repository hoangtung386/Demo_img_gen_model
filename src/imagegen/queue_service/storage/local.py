import os
import shutil
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path

from ..config import StorageBackendConfig
from ..observability import logger
from .base import (
    DownloadEmpty,
    DownloadFailed,
    DownloadNotFound,
    DownloadTimeout,
    StorageBackend,
    UploadFailed,
)
from .object_naming import build_object_key


class LocalStorage(StorageBackend):
    """Local filesystem backend cho dev/test."""

    def __init__(self, cfg: StorageBackendConfig):
        self.cfg = cfg
        self.root = Path(cfg.folder)
        self.root.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, destination: str) -> None:
        try:
            if url.startswith(("http://", "https://")):
                urllib.request.urlretrieve(url, destination)
            else:
                src = Path(url)
                if not src.exists():
                    raise DownloadNotFound(f"local file not found: {url}")
                shutil.copy(src, destination)
        except urllib.error.HTTPError as e:
            # HTTP 4xx → PERMANENT (xem `storage/gcs.py` để biết lý do).
            if 400 <= e.code < 500:
                raise DownloadNotFound(f"HTTP {e.code} for {url}") from e
            raise DownloadFailed(f"HTTP {e.code} for {url}") from e
        except TimeoutError as e:
            raise DownloadTimeout(str(e)) from e
        except (urllib.error.URLError, OSError) as e:
            inner = getattr(e, "reason", None)
            if isinstance(inner, TimeoutError):
                raise DownloadTimeout(str(e)) from e
            raise DownloadFailed(str(e)) from e

        # Empty body: same rule as GCS backend — 0 bytes = PERMANENT.
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
        # Mirror the production object key under the local root.
        object_key = build_object_key(
            folder=self.cfg.folder,
            app_id=app_id,
            feature=feature or self.cfg.feature,
            os_name=os_name,
            device_id=device_id,
            source_path=source_path,
        )
        # `folder` is already part of object_key; root strips it via relpath.
        relative = Path(object_key).relative_to(self.cfg.folder.strip("/"))
        target = self.root / relative
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source_path, target)
            return f"file://{target.absolute()}"
        except Exception as e:
            logger.error(f"[local] upload failed: {e}")
            raise UploadFailed(str(e)) from e
