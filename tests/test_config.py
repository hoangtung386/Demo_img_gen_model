"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from gen_image.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_GGUF_FILE,
    DEFAULT_GGUF_REPO,
    load_settings,
)

_ENV_VARS = [
    "GENIMG_NUM_STEPS",
    "GENIMG_GUIDANCE_SCALE",
    "GENIMG_QUANTIZATION",
    "GENIMG_COMPILE",
    "GENIMG_VAE_TILING",
    "GENIMG_VAE_SLICING",
    "GENIMG_EMBED_CACHE_SIZE",
    "GENIMG_OFFLOAD",
    "GENIMG_SERVER_NAME",
    "GENIMG_PORT",
    "GENIMG_DEVICE",
    "GENIMG_BASE_MODEL_LOCAL",
    "GENIMG_TRANSFORMER_GGUF",
    "MODEL_ROOT",
    "GENIMG_BASE_MODEL",
    "GENIMG_GGUF_REPO",
    "GENIMG_GGUF_FILE",
]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Prevent the real .env AND base.yaml from leaking into isolated tests.

    Mock cả `load_dotenv` lẫn `_load_app_yaml` để test default kiểm tra ĐÚNG
    default hard-code trong config.py, độc lập với giá trị thực trong
    config_setup/base.yaml.
    """
    monkeypatch.setattr("gen_image.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("gen_image.config._load_app_yaml", lambda: {})
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_default_settings():
    """Default hard-code khi không có base.yaml lẫn env (fixture mock cả 2)."""
    settings = load_settings()
    assert settings.num_steps == 4
    assert settings.guidance_scale == 1.0
    assert settings.compile_transformer is False
    assert settings.quantization == "gguf"
    assert settings.base_model == "black-forest-labs/FLUX.2-klein-9B"
    assert settings.gguf_repo == "unsloth/FLUX.2-klein-9B-GGUF"
    assert settings.gguf_file == "flux-2-klein-9b-Q4_K_M.gguf"
    assert settings.vae_tiling is False
    assert settings.vae_slicing is False
    assert settings.embed_cache_size == 8
    assert settings.offload == "resident"
    assert settings.server_port == 7860
    assert settings.base_model_id == settings.base_model


def test_default_base_model_is_the_distilled_repo():
    """Mặc định phải là bản distilled, KHÔNG phải 'klein-base'.

    Hai repo khác nhau đúng một chữ trong tên nhưng khác ~4 lần về tốc độ:
    bản base không distilled, cần vài chục bước và CFG hai nhánh. Rơi nhầm
    sang nó là một hồi quy hiệu năng rất khó nhận ra từ log.
    """
    assert load_settings().base_model == DEFAULT_BASE_MODEL
    assert "klein-base" not in DEFAULT_BASE_MODEL
    assert DEFAULT_GGUF_REPO == "unsloth/FLUX.2-klein-9B-GGUF"
    assert DEFAULT_GGUF_FILE == "flux-2-klein-9b-Q4_K_M.gguf"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("GENIMG_NUM_STEPS", "8")
    monkeypatch.setenv("GENIMG_QUANTIZATION", "gguf")
    monkeypatch.setenv("GENIMG_GUIDANCE_SCALE", "4.0")
    settings = load_settings()
    assert settings.num_steps == 8
    assert settings.quantization == "gguf"
    assert settings.guidance_scale == 4.0


def test_yaml_provides_values(monkeypatch):
    """base.yaml khối app: cấp giá trị khi env không set (env vẫn thắng)."""
    monkeypatch.setattr(
        "gen_image.config._load_app_yaml",
        lambda: {
            "num_steps": 8,
            "quantization": "gguf",
            "offload": "model_offload",
        },
    )
    settings = load_settings()
    assert settings.num_steps == 8
    assert settings.quantization == "gguf"
    assert settings.offload == "model_offload"


def test_env_beats_yaml(monkeypatch):
    """Env override THẮNG giá trị trong base.yaml."""
    monkeypatch.setattr(
        "gen_image.config._load_app_yaml",
        lambda: {"num_steps": 8, "quantization": "gguf"},
    )
    monkeypatch.setenv("GENIMG_NUM_STEPS", "4")
    monkeypatch.setenv("GENIMG_QUANTIZATION", "bf16")
    settings = load_settings()
    assert settings.num_steps == 4
    assert settings.quantization == "bf16"


def test_no_stale_qie_prefix_is_honoured(monkeypatch):
    """Tiền tố QIE_ cũ KHÔNG được còn tác dụng.

    Giữ QIE_ sống song song sẽ để một biến cũ còn sót trong môi trường deploy
    âm thầm trỏ pipeline về trọng số Qwen không còn tồn tại — và vì env thắng
    YAML, nó sẽ ghi đè cả cấu hình đúng trong base.yaml.
    """
    monkeypatch.setenv("QIE_NUM_STEPS", "99")
    monkeypatch.setenv("QIE_BASE_MODEL_LOCAL", "/tmp/qwen-cu")
    settings = load_settings()
    assert settings.num_steps == 4
    assert settings.base_model_local is None


def test_share_is_not_configurable():
    """Link công khai là bắt buộc — không được để lọt lại một cái nút tắt.

    QIE_SHARE từng tồn tại và mặc định false, khiến demo trên Colab chạy mà
    không đưa ra link nào. Test này giữ cho nó không quay lại.
    """
    assert not hasattr(load_settings(), "share")

    root = Path(__file__).resolve().parents[1]
    # launch.py, không phải app.py: phần mở cổng + tunnel đã tách khỏi phần
    # dựng widget (app.py giờ dựng được mà không cần model).
    launch = (root / "src/gen_image/ui/launch.py").read_text(encoding="utf-8")
    assert "share=True" in launch
    assert "settings.share" not in launch


def test_base_model_local_priority(monkeypatch):
    monkeypatch.setenv("GENIMG_BASE_MODEL_LOCAL", "/tmp/local")
    monkeypatch.setenv("GENIMG_BASE_MODEL", "some/Remote")
    settings = load_settings()
    assert settings.base_model_id == "/tmp/local"
