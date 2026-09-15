"""Thin CLI wrapper to launch the demo.

Works both via ``uv run run-app`` and a direct
``python scripts/serve.py`` invocation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen_lightning.serve import main  # noqa: E402

if __name__ == "__main__":
    main()
