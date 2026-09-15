"""Tests for transformer and base model path resolution."""
from __future__ import annotations

import dataclasses

import pytest

from qwen_lightning.models.loader import (
    resolve_base_model,
    resolve_transformer_path,
)


def test_resolve_local_path(base_settings, tmp_path):
    weights = tmp_path / "transformer.safetensors"
    weights.write_bytes(b"")
    settings = dataclasses.replace(
        base_settings, transformer_path=str(weights)
    )
    assert resolve_transformer_path(settings, "int4") == str(weights)


def test_resolve_local_path_missing_file_raises(base_settings, tmp_path):
    """Đường dẫn local trỏ vào hư không phải dừng ngay.

    Trước đây hàm trả về đường dẫn không tồn tại và lỗi chỉ nổ sâu bên trong
    nunchaku, hoặc tệ hơn là im lặng tải lại 11GB từ Hub.
    """
    settings = dataclasses.replace(
        base_settings, transformer_path=str(tmp_path / "khong-co.safetensors")
    )
    with pytest.raises(FileNotFoundError):
        resolve_transformer_path(settings, "int4")


def test_resolve_hf_path(base_settings):
    path = resolve_transformer_path(base_settings, "int4")
    assert path.startswith(
        "nunchaku-tech/nunchaku-qwen-image-edit-2509/lightning-251115/"
    )
    assert "svdq-int4_r32" in path
    assert "4steps-251115.safetensors" in path


def test_resolve_honors_rank_and_steps(base_settings):
    path = resolve_transformer_path(base_settings, "fp4")
    assert "svdq-fp4_r32" in path
    assert "4steps" in path


def test_resolve_base_model_local(base_settings, tmp_path):
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    settings = dataclasses.replace(
        base_settings, base_model_local=str(tmp_path)
    )
    assert resolve_base_model(settings) == str(tmp_path)


def test_resolve_base_model_missing_index_raises(base_settings, tmp_path):
    settings = dataclasses.replace(
        base_settings, base_model_local=str(tmp_path)
    )
    with pytest.raises(FileNotFoundError):
        resolve_base_model(settings)


def test_resolve_base_model_falls_back_to_hub(base_settings):
    assert resolve_base_model(base_settings) == "Qwen/Qwen-Image-Edit-2509"
