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


def test_resolve_base_model_without_local_or_token_fails_fast(base_settings):
    with pytest.raises(RuntimeError, match="gated"):
        resolve_base_model(base_settings)


def test_resolve_base_model_falls_back_to_hub_with_token(base_settings):
    settings = dataclasses.replace(base_settings, hf_token="test-token")
    assert resolve_base_model(settings) == DEFAULT_BASE_MODEL


def test_resolve_base_model_discovers_default_local_path(
    base_settings, tmp_path, monkeypatch
):
    base = tmp_path / "models" / "FLUX.2-klein-9B"
    base.mkdir(parents=True)
    (base / "model_index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("gen_image.models.loader.PROJECT_ROOT", tmp_path)
    assert resolve_base_model(base_settings) == str(base)


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


def test_resolve_gguf_discovers_default_local_path(
    base_settings, tmp_path, monkeypatch
):
    gguf = tmp_path / "models" / "gguf" / base_settings.gguf_file
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"")
    monkeypatch.setattr("gen_image.models.loader.PROJECT_ROOT", tmp_path)
    assert resolve_gguf_path(base_settings) == str(gguf)


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


# --------------------------------------------------------------------------
# lượng tử hoá text encoder (knob riêng, tách khỏi `quantization`)
# --------------------------------------------------------------------------


def test_text_encoder_quantizations_are_the_documented_three():
    """Danh sách hợp lệ phải khớp base.example.yaml — sai chính tả fail-fast."""
    from gen_image.models.loader import TEXT_ENCODER_QUANTIZATIONS

    assert TEXT_ENCODER_QUANTIZATIONS == ("bf16", "nf4", "int8")


def test_text_encoder_bf16_means_no_quantization_config():
    """bf16 → None, tức không truyền quantization_config vào from_pretrained."""
    from gen_image.models.loader import build_text_encoder_quant_config

    assert build_text_encoder_quant_config("bf16") is None


def test_text_encoder_nf4_quantizes_only_the_text_encoder():
    """Rào chắn thật: transformer đã là GGUF, VAE nén sẽ ra artefact.

    Nếu `components_to_quantize` lỡ bị mở rộng, bitsandbytes sẽ giẫm lên
    transformer GGUF vừa dựng bằng from_single_file.
    """
    from gen_image.models.loader import build_text_encoder_quant_config

    cfg = build_text_encoder_quant_config("nf4")
    assert cfg is not None
    assert cfg.components_to_quantize == ["text_encoder"]
    assert cfg.quant_backend == "bitsandbytes_4bit"
    assert cfg.quant_kwargs["bnb_4bit_quant_type"] == "nf4"


def test_text_encoder_compute_dtype_matches_pipeline_dtype():
    """Lệch dtype ở đây là một lần ép kiểu âm thầm giữa encoder và transformer."""
    from gen_image.models.loader import DTYPE, build_text_encoder_quant_config

    cfg = build_text_encoder_quant_config("nf4")
    assert cfg.quant_kwargs["bnb_4bit_compute_dtype"] is DTYPE


def test_text_encoder_quantization_rejects_typos():
    """Giá trị lạ phải dừng hẳn, không im lặng rơi về bf16 rồi OOM lúc chạy."""
    from gen_image.models.loader import build_text_encoder_quant_config

    with pytest.raises(ValueError, match="text_encoder_quantization"):
        build_text_encoder_quant_config("int4")


def test_text_encoder_quantization_tolerates_whitespace_and_case():
    from gen_image.models.loader import build_text_encoder_quant_config

    assert build_text_encoder_quant_config("  NF4 ").quant_backend == (
        "bitsandbytes_4bit"
    )
