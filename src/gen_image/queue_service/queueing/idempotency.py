"""
RecentRequestCache — LRU set of request_id đã xử lý gần đây.

Mục đích: bắt duplicate request từ cùng request_id để tránh waste GPU vài giây
xử lý lại. Production case: BE/App retry khi network blip → cùng 1 message
được gửi 2 lần vào broker → 2 container (hoặc cùng container) pick up cả 2.

Design choice:
- Per-process (in-memory) — đủ tốt cho 99% case (BE retry trong vài giây).
  KHÔNG bảo vệ duplicate giữa nhiều container (cần Redis cho việc đó).
- LRU bounded → memory không bao giờ grow. Default 1000 entries (~1KB).
- Thread-safe: dùng Lock vì worker thread + consumer thread có thể truy cập.
"""

from __future__ import annotations

import threading
from collections import OrderedDict


class RecentRequestCache:
    """
    Thread-safe bounded LRU cache of request_ids đã xử lý.

    Usage:
        cache = RecentRequestCache(maxsize=1000)
        if cache.seen(request_id):
            return  # duplicate — skip processing
        cache.add(request_id)
        # ... process normally
    """

    def __init__(self, maxsize: int = 1000) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self._maxsize = maxsize
        self._items: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def seen(self, request_id: str) -> bool:
        """Returns True nếu request_id đã có trong cache (LRU touch)."""
        if not request_id:
            return False
        with self._lock:
            if request_id in self._items:
                self._items.move_to_end(request_id)  # mark as MRU
                return True
            return False

    def add(self, request_id: str) -> None:
        """Add request_id vào cache. Evict oldest nếu vượt maxsize."""
        if not request_id:
            return
        with self._lock:
            if request_id in self._items:
                self._items.move_to_end(request_id)
                return
            self._items[request_id] = None
            while len(self._items) > self._maxsize:
                self._items.popitem(last=False)  # evict LRU

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
