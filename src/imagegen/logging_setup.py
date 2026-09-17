"""Logging configuration.

``LOGGER_NAME`` là tên logger dùng chung cho mọi module của service, kể cả
các patch trong ``hidream/vendor/pipeline.py``. Giữ một hằng số thay vì rải
chuỗi literal khắp nơi: đổi tên ở đây là đổi ở mọi chỗ, và không có module
nào lặng lẽ log vào một logger khác rồi biến mất khỏi output.
"""

from __future__ import annotations

import logging

LOGGER_NAME = "hidream-o1"

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging and return the service logger."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
    return logging.getLogger(LOGGER_NAME)
