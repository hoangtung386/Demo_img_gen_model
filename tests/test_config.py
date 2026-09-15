"""Tests for configuration loading."""
from __future__ import annotations

from pathlib import Path

import pytest

from qwen_lightning.config import load_settings

_ENV_VARS = [
    "QIE_NUM_STEPS",
    "QIE_RANK",
    "QIE_PRECISION",
    "QIE_OFFLOAD",
    "QIE_SERVER_NAME",
    "QIE_PORT",
    "QIE_DEVICE",
    "QIE_BASE_MODEL_LOCAL",
    "QIE_TRANSFORMER_PATH",
    "MODEL_ROOT",
    "QIE_BASE_MODEL",
    "QIE_TRANSFORMER_REPO",
    "QIE_TRANSFORMER_SUBDIR",
]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Prevent the real .env from leaking into isolated config tests."""
    monkeypatch.setattr(
        "qwen_lightning.config.load_dotenv", lambda *a, **k: None
    )
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_default_settings():
    settings = load_settings()
    assert settings.num_steps == 4
    assert settings.rank == 32
    assert settings.precision is None
    assert settings.offload == "auto"
    assert settings.server_port == 7860
    assert settings.base_model_id == settings.base_model


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("QIE_NUM_STEPS", "8")
    monkeypatch.setenv("QIE_RANK", "128")
    monkeypatch.setenv("QIE_PRECISION", "fp4")
    settings = load_settings()
    assert settings.num_steps == 8
    assert settings.rank == 128
    assert settings.precision == "fp4"


def test_share_is_not_configurable():
    """Link công khai là bắt buộc — không được để lọt lại một cái nút tắt.

    QIE_SHARE từng tồn tại và mặc định false, khiến demo trên Colab chạy mà
    không đưa ra link nào. Test này giữ cho nó không quay lại.
    """
    assert not hasattr(load_settings(), "share")

    root = Path(__file__).resolve().parents[1]
    app = (root / "src/qwen_lightning/ui/app.py").read_text(
        encoding="utf-8"
    )
    assert "share=True" in app
    assert "settings.share" not in app


def test_base_model_local_priority(monkeypatch):
    monkeypatch.setenv("QIE_BASE_MODEL_LOCAL", "/tmp/local")
    monkeypatch.setenv("QIE_BASE_MODEL", "Qwen/Remote")
    settings = load_settings()
    assert settings.base_model_id == "/tmp/local"
