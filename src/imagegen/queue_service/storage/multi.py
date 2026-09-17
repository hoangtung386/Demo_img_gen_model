from ..config import StorageConfig
from ..observability import logger
from .base import StorageBackend


class MultiBackendStorage(StorageBackend):
    """
    Routes mỗi `upload(...)` tới 1 backend con dựa trên `storage_index`.
    `download` không phụ thuộc backend (input URL đã absolute) → dùng backend
    đầu tiên.

    Concrete backends thường là `GCSStorage` (1 instance / entry trong
    `storage.backends[]`); `factory.create_storage` build danh sách này.
    """

    def __init__(self, cfg: StorageConfig, backends: list[StorageBackend]):
        if not backends:
            raise ValueError("MultiBackendStorage requires ≥ 1 backend")
        self.cfg = cfg
        self._backends = backends
        logger.info(
            f"[storage] {len(backends)} backend(s) ready, "
            f"default_index={cfg.default_index}"
        )

    def download(self, url: str, destination: str) -> None:
        # Download không cần backend cụ thể (input URL đã absolute) — dùng
        # default để có client cấu hình sẵn (timeout, proxy …).
        self._pick(None).download(url, destination)

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
        backend = self._pick(storage_index)
        return backend.upload(
            source_path,
            os_name=os_name,
            app_id=app_id,
            device_id=device_id,
            feature=feature,
        )

    def _pick(self, storage_index: int | None) -> StorageBackend:
        i = self.cfg.default_index if storage_index is None else storage_index
        if not 0 <= i < len(self._backends):
            raise IndexError(
                f"storage_index {i} out of range (have {len(self._backends)} "
                f"backend(s))"
            )
        return self._backends[i]
