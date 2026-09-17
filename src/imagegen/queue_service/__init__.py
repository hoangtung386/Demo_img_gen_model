"""
RabbitMQ queue-based worker service cho HiDream-O1-Image (SDNQ 4-bit).

Kiến trúc mirror `edit_any_image`: consumer đa tier (premium + basic 1/2/3)
→ priority router → AI worker chạy pipeline (download → generate → upload) →
publisher trả kết quả qua queue_out. Config nạp từ `config_setup/base.yaml`.

Entry point: `main_queue.py` ở project root, hoặc:
    from imagegen.queue_service import Application, load_config
    Application(load_config()).run()
"""

from .app import Application
from .config import Settings, load_config

__all__ = ["Application", "Settings", "load_config"]
