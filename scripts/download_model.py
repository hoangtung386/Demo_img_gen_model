"""Thin CLI wrapper to download model weights.

Works both via ``uv run download-model`` and a direct
``python scripts/download_model.py`` invocation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen_lightning.download import main  # noqa: E402

if __name__ == "__main__":
    main()
