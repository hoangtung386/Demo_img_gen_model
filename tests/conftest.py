"""Shared pytest fixtures."""
from __future__ import annotations

import pytest

from qwen_lightning.config import Settings


@pytest.fixture
def base_settings() -> Settings:
    """A fully populated default Settings instance for unit tests.

    Settings là frozen dataclass không có giá trị mặc định, nên mọi field
    phải được liệt kê ở đây. Thêm field mới vào Settings mà quên cập nhật
    fixture này thì toàn bộ test dùng nó sẽ đổ TypeError.
    """
    return Settings(
        num_steps=4,
        rank=32,
        precision=None,
        model_root="models",
        base_model="Qwen/Qwen-Image-Edit-2509",
        transformer_repo="nunchaku-tech/nunchaku-qwen-image-edit-2509",
        transformer_subdir="lightning-251115",
        offload="auto",
        server_name="0.0.0.0",
        server_port=7860,
        demo_cache=True,
        device_override=None,
        text_encoder_device=None,
        base_model_local=None,
        transformer_path=None,
        download_base_transformer=False,
        warmup=False,
    )
