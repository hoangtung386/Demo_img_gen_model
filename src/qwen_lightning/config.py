"""Centralized configuration loaded from environment variables.

All runtime and download settings live here so that ``serve`` and
``download`` share a single source of truth and never drift apart.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATHS_ENV = PROJECT_ROOT / ".model_paths.env"


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    num_steps: int
    rank: int
    precision: str | None
    model_root: str
    base_model: str
    transformer_repo: str
    transformer_subdir: str
    offload: str
    server_name: str
    server_port: int
    demo_cache: bool
    device_override: str | None
    text_encoder_device: str | None
    base_model_local: str | None
    transformer_path: str | None
    download_base_transformer: bool
    warmup: bool

    @property
    def base_model_id(self) -> str:
        """Local base pipeline directory if present, else the HF id."""
        return self.base_model_local or self.base_model


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def load_settings() -> Settings:
    """Load ``.env`` then the resolved ``.model_paths.env`` if present."""
    load_dotenv(PROJECT_ROOT / ".env")
    if MODEL_PATHS_ENV.exists():
        load_dotenv(MODEL_PATHS_ENV)

    return Settings(
        num_steps=_as_int(os.getenv("QIE_NUM_STEPS"), 4),
        rank=_as_int(os.getenv("QIE_RANK"), 32),
        precision=os.getenv("QIE_PRECISION") or None,
        model_root=os.getenv("MODEL_ROOT", "models"),
        base_model=os.getenv("QIE_BASE_MODEL", "Qwen/Qwen-Image-Edit-2509"),
        transformer_repo=os.getenv(
            "QIE_TRANSFORMER_REPO",
            "nunchaku-tech/nunchaku-qwen-image-edit-2509",
        ),
        transformer_subdir=os.getenv(
            "QIE_TRANSFORMER_SUBDIR", "lightning-251115"
        ),
        offload=os.getenv("QIE_OFFLOAD", "auto").lower(),
        server_name=os.getenv("QIE_SERVER_NAME", "0.0.0.0"),
        server_port=_as_int(os.getenv("QIE_PORT"), 7860),
        # Chế độ demo: ví dụ mẫu trả ảnh dựng sẵn thay vì chạy model. Đặt
        # QIE_DEMO_CACHE=false khi cần đo hiệu năng thật.
        demo_cache=os.getenv("QIE_DEMO_CACHE", "true").strip().lower()
        != "false",
        device_override=os.getenv("QIE_DEVICE") or None,
        text_encoder_device=os.getenv("QIE_TEXT_ENCODER_DEVICE") or None,
        base_model_local=os.getenv("QIE_BASE_MODEL_LOCAL") or None,
        transformer_path=os.getenv("QIE_TRANSFORMER_PATH") or None,
        # 5 shard transformer BF16 (~39GB) của repo gốc không bao giờ được
        # pipeline đọc — xem ghi chú trong download.py. Mặc định bỏ qua.
        download_base_transformer=_as_bool(
            os.getenv("QIE_DOWNLOAD_BASE_TRANSFORMER")
        ),
        # Trả trước chi phí lượt sinh ảnh đầu tiên lúc khởi động thay vì bắt
        # người dùng đầu tiên gánh. Tắt khi cần đo thời gian load thuần.
        warmup=os.getenv("QIE_WARMUP", "true").strip().lower() != "false",
    )
