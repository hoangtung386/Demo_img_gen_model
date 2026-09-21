import os
import time

import psutil

from ..messaging.shutdown import is_shutting_down
from ..observability import logger


class ResourceMonitor:
    def __init__(self, interval_seconds: int = 5):
        self.interval = interval_seconds

    def run_forever(self) -> None:
        proc = psutil.Process(os.getpid())
        # torch import lười — service có thể chạy diagnostics không cần GPU.
        try:
            import torch
        except Exception:
            torch = None

        # Loop exit on shutdown — consistent với mọi daemon thread khác.
        while not is_shutting_down():
            try:
                ram_mb = proc.memory_info().rss // 1024 // 1024
                if torch is not None and torch.cuda.is_available():
                    gpu_alloc = torch.cuda.memory_allocated() // 1024 // 1024
                    gpu_reserved = torch.cuda.memory_reserved() // 1024 // 1024
                    gpu_total = (
                        torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
                    )
                else:
                    gpu_alloc = gpu_reserved = gpu_total = 0
                logger.info(
                    f"[mem] RAM: {ram_mb} MB | GPU alloc: {gpu_alloc}/{gpu_total} MB | "
                    f"GPU reserved: {gpu_reserved} MB"
                )
            except Exception as e:
                logger.warning(f"[mem] monitor error: {e}")
            time.sleep(self.interval)
