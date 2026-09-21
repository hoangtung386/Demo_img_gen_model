from ..config import StorageConfig
from .base import StorageBackend
from .multi import MultiBackendStorage


def create_storage(cfg: StorageConfig) -> StorageBackend:
    """Build StorageBackend từ config. Luôn wrap qua MultiBackendStorage —
    1 backend cũng wrap để pipeline gọi cùng API có `storage_index`."""

    if cfg.backend == "gcs":
        from .gcs import GCSStorage

        backends: list[StorageBackend] = [GCSStorage(b) for b in cfg.backends]
    elif cfg.backend == "local":
        from .local import LocalStorage

        backends = [LocalStorage(b) for b in cfg.backends]
    else:
        raise ValueError(f"Unknown storage backend: {cfg.backend}")

    return MultiBackendStorage(cfg, backends)
