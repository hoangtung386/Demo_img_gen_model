"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from gen_image.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_GGUF_FILE,
    DEFAULT_GGUF_REPO,
    Settings,
)


@pytest.fixture
def base_settings() -> Settings:
    """A fully populated default Settings instance for unit tests.

    Settings là frozen dataclass không có giá trị mặc định, nên mọi field
    phải được liệt kê ở đây. Thêm field mới vào Settings mà quên cập nhật
    fixture này thì toàn bộ test dùng nó sẽ đổ TypeError.
    """
    return Settings(
        num_steps=4,
        guidance_scale=1.0,
        quantization="bf16",
        compile_transformer=False,
        vae_tiling=False,
        vae_slicing=False,
        embed_cache_size=8,
        model_root="models",
        base_model=DEFAULT_BASE_MODEL,
        gguf_repo=DEFAULT_GGUF_REPO,
        gguf_file=DEFAULT_GGUF_FILE,
        offload="resident",
        server_name="0.0.0.0",
        server_port=7860,
        demo_cache=True,
        device_override=None,
        base_model_local=None,
        transformer_gguf=None,
        warmup=False,
    )
