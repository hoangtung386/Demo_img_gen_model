"""
Service entry point cho RabbitMQ queue worker.

Giữ mỏng — toàn bộ wiring nằm trong gen_image.queue_service.app.Application.
Config đọc từ config_setup/base.yaml.

Chạy:
    python main_queue.py
"""

import sys
from pathlib import Path

# Cho phép chạy trực tiếp `python main_queue.py` mà không cần pip install.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gen_image.queue_service import Application, load_config


def main() -> None:
    settings = load_config()
    Application(settings).run()


if __name__ == "__main__":
    main()
