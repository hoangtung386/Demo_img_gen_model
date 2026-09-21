from abc import ABC, abstractmethod


class StorageError(Exception):
    """Base for storage errors. Distinct subclasses → different status codes."""


class DownloadNotFound(StorageError):
    """Remote returned 404 (or equivalent). PERMANENT — retry won't help."""


class DownloadTimeout(StorageError):
    """Network/read timeout fetching the URL. TRANSIENT."""


class DownloadFailed(StorageError):
    """Other download failure (network, DNS, etc). TRANSIENT."""


class DownloadEmpty(StorageError):
    """
    Remote returned 200 OK but body is empty (0 bytes).

    PERMANENT — retry won't help; the upstream stored an empty file.
    Distinct from DownloadFailed (TRANSIENT) so pipeline can classify it as
    INPUT_INVALID_FORMAT instead of retrying.
    """


class UploadFailed(StorageError):
    """Upload to storage failed (5xx, network, etc). TRANSIENT."""


class UploadAuthFailed(StorageError):
    """Storage backend rejected our credentials. TRANSIENT (likely needs ops)."""


class StorageBackend(ABC):
    """
    Abstract storage backend. Implementations: GCS, local.

    download() raises a StorageError subclass on failure (caller can map to a
    specific status code) or returns silently on success.

    upload() returns the signed/public URL on success, or raises an
    UploadFailed/UploadAuthFailed subclass on failure.
    """

    @abstractmethod
    def download(self, url: str, destination: str) -> None:
        """Download `url` to `destination`. Raises StorageError on failure."""

    @abstractmethod
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
        """
        Upload `source_path`. Returns the URL or raises StorageError.

        `feature` overrides the default feature from StorageBackendConfig.feature.
        Most callers leave it None so every upload from this service goes under
        the same feature label.

        `storage_index` chọn destination khi service có nhiều backend (xem
        `MultiBackendStorage`). Single-backend implementation bỏ qua param này.
        None → dùng default.
        """
