"""Logging configuration for the demo."""
from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging and return the demo logger."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("qwen-image-edit-2509-lightning")
