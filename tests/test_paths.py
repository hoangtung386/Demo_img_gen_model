"""Tests for base model and GGUF path resolution."""

from __future__ import annotations

import dataclasses

import pytest

from gen_image.config import DEFAULT_BASE_MODEL
from gen_image.models.loader import resolve_base_model, resolve_gguf_path


def test_resolve_base_model_local(base_settings, tmp_path):
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    settings = dataclasses.replace(base_settings, base_model_local=str(tmp_path))
    assert resolve_base_model(settings) == str(tmp_path)


def test_resolve_base_model_missing_index_raises(base_settings, tmp_path):
    """Thư mục có tồn tại nhưng thiếu model_index.json phải dừng ngay.

    Đây là hình dạng của một volume mount sai: thư mục có đó (docker tự tạo
    khi mount đường dẫn không tồn tại trên host) nhưng rỗng. Không chặn ở đây
    thì diffusers im lặng rơi về Hub và tải lại hàng chục GB.
    """
    settings = dataclasses.replace(base_settings, base_model_local=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        resolve_base_model(settings)


def test_resolve_base_model_falls_back_to_hub(base_settings):
    assert resolve_base_model(base_settings) == DEFAULT_BASE_MODEL


def test_resolve_gguf_local(base_settings, tmp_path):
    weights = tmp_path / "flux-2-klein-9b-Q4_K_M.gguf"
    weights.write_bytes(b"")
    settings = dataclasses.replace(base_settings, transformer_gguf=str(weights))
    assert resolve_gguf_path(settings) == str(weights)


def test_resolve_gguf_missing_file_raises(base_settings, tmp_path):
    """Đường dẫn local trỏ vào hư không phải dừng ngay.

    Nếu chỉ trả về đường dẫn không tồn tại thì lỗi nổ sâu trong
    from_single_file, hoặc tệ hơn là im lặng tải lại vài GB từ Hub.
    """
    settings = dataclasses.replace(
        base_settings, transformer_gguf=str(tmp_path / "khong-co.gguf")
    )
    with pytest.raises(FileNotFoundError):
        resolve_gguf_path(settings)


def test_resolve_gguf_falls_back_to_hub_blob_url(base_settings):
    url = resolve_gguf_path(base_settings)
    assert url.startswith("https://huggingface.co/")
    assert base_settings.gguf_repo in url
    assert url.endswith(base_settings.gguf_file)


# --------------------------------------------------------------------------
# lựa chọn quantization + tuỳ chọn bộ nhớ VAE
# --------------------------------------------------------------------------


def test_quantizations_are_the_documented_three():
    """Danh sách hợp lệ phải khớp tài liệu — sai chính tả thì fail-fast."""
    from gen_image.models.loader import QUANTIZATIONS

    assert QUANTIZATIONS == ("bf16", "gguf", "fp8")


class _FakeVae:
    def __init__(self) -> None:
        self.tiling = False
        self.slicing = False

    def enable_tiling(self) -> None:
        self.tiling = True

    def enable_slicing(self) -> None:
        self.slicing = True


class _FakePipe:
    def __init__(self) -> None:
        self.vae = _FakeVae()


def test_vae_options_off_by_default(base_settings):
    from gen_image.models.loader import _apply_vae_memory_options

    pipe = _FakePipe()
    _apply_vae_memory_options(pipe, base_settings)
    assert not pipe.vae.tiling
    assert not pipe.vae.slicing


def test_vae_options_applied_when_enabled(base_settings):
    from gen_image.models.loader import _apply_vae_memory_options

    settings = dataclasses.replace(base_settings, vae_tiling=True, vae_slicing=True)
    pipe = _FakePipe()
    _apply_vae_memory_options(pipe, settings)
    assert pipe.vae.tiling
    assert pipe.vae.slicing
