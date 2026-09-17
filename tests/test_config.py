"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from imagegen.config import DEFAULT_MODEL_REPO, load_settings

_ENV_VARS = [
    "IMG_NUM_STEPS",
    "IMG_MODEL_TYPE",
    "IMG_GUIDANCE_SCALE",
    "IMG_SHIFT",
    "IMG_SCHEDULER",
    "IMG_WIDTH",
    "IMG_HEIGHT",
    "IMG_SERVER_NAME",
    "IMG_PORT",
    "IMG_DEVICE",
    "IMG_WARMUP",
    "IMG_DEMO_CACHE",
    "IMG_MODEL_PATH",
    "IMG_BASE_MODEL",
    "MODEL_ROOT",
]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Prevent the real .env AND base.yaml from leaking into isolated tests.

    Mock cả `load_dotenv` lẫn `_load_app_yaml` để test default kiểm tra ĐÚNG
    default hard-code trong config.py, độc lập với giá trị thực trong
    config_setup/base.yaml.
    """
    monkeypatch.setattr("imagegen.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("imagegen.config._load_app_yaml", lambda: {})
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_default_settings():
    """Default hard-code khi không có base.yaml lẫn env (fixture mock cả 2)."""
    settings = load_settings()
    assert settings.model_type == "full"
    assert settings.num_steps == 50
    assert settings.guidance_scale == 5.0
    assert settings.shift == 3.0
    assert settings.scheduler_name == "default"
    assert settings.width == 2048
    assert settings.height == 2048
    assert settings.server_port == 7860
    assert settings.base_model == DEFAULT_MODEL_REPO
    assert settings.model_id == settings.base_model


def test_dev_model_type_lowers_default_steps(monkeypatch):
    """Đặt model_type=dev mà quên num_steps thì KHÔNG được giữ 50 bước.

    Bản dev đã chưng cất xuống 28 bước; chạy nó ở 50 bước vừa chậm gấp đôi
    vừa không đúng lịch nhiễu mà nó được luyện.
    """
    monkeypatch.setenv("IMG_MODEL_TYPE", "dev")
    assert load_settings().num_steps == 28


def test_explicit_steps_beat_model_type_default(monkeypatch):
    monkeypatch.setenv("IMG_MODEL_TYPE", "dev")
    monkeypatch.setenv("IMG_NUM_STEPS", "40")
    assert load_settings().num_steps == 40


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("IMG_NUM_STEPS", "28")
    monkeypatch.setenv("IMG_GUIDANCE_SCALE", "0")
    monkeypatch.setenv("IMG_SHIFT", "1.0")
    monkeypatch.setenv("IMG_SCHEDULER", "FLASH")
    settings = load_settings()
    assert settings.num_steps == 28
    assert settings.guidance_scale == 0.0
    assert settings.shift == 1.0
    # scheduler_name luôn được hạ về chữ thường trước khi so khớp.
    assert settings.scheduler_name == "flash"


def test_yaml_provides_values(monkeypatch):
    """base.yaml khối app: cấp giá trị khi env không set (env vẫn thắng)."""
    monkeypatch.setattr(
        "imagegen.config._load_app_yaml",
        lambda: {"num_steps": 28, "guidance_scale": 0.0, "width": 2560},
    )
    settings = load_settings()
    assert settings.num_steps == 28
    assert settings.guidance_scale == 0.0
    assert settings.width == 2560


def test_env_beats_yaml(monkeypatch):
    """Env override THẮNG giá trị trong base.yaml."""
    monkeypatch.setattr(
        "imagegen.config._load_app_yaml",
        lambda: {"num_steps": 28, "shift": 1.0},
    )
    monkeypatch.setenv("IMG_NUM_STEPS", "50")
    monkeypatch.setenv("IMG_SHIFT", "3.0")
    settings = load_settings()
    assert settings.num_steps == 50
    assert settings.shift == 3.0


def test_warmup_and_demo_cache_are_true_by_default(monkeypatch):
    settings = load_settings()
    assert settings.warmup is True
    assert settings.demo_cache is True
    monkeypatch.setenv("IMG_WARMUP", "false")
    monkeypatch.setenv("IMG_DEMO_CACHE", "FALSE")
    settings = load_settings()
    assert settings.warmup is False
    assert settings.demo_cache is False


def test_share_is_not_configurable():
    """Link công khai là bắt buộc — không được để lọt lại một cái nút tắt.

    IMG_SHARE (trước là QIE_SHARE) từng tồn tại và mặc định false, khiến demo
    trên Colab chạy mà không đưa ra link nào. Test này giữ cho nó không quay
    lại.
    """
    assert not hasattr(load_settings(), "share")

    root = Path(__file__).resolve().parents[1]
    launcher = (root / "src/imagegen/ui/launcher.py").read_text("utf-8")
    assert "share=True" in launcher
    assert "settings.share" not in launcher


def test_model_path_local_priority(monkeypatch):
    """Thư mục local THẮNG repo id — nếu không thì service tải lại 10GB."""
    monkeypatch.setenv("IMG_MODEL_PATH", "/tmp/local")
    monkeypatch.setenv("IMG_BASE_MODEL", "SomeOrg/Remote")
    settings = load_settings()
    assert settings.model_id == "/tmp/local"


def test_no_qie_prefix_left_in_source():
    """Tiền tố QIE_ đã bị bỏ hẳn; sót lại một biến là sót một đường chết.

    Trong lần thay lõi, một biến QIE_* còn sót sẽ im lặng không có tác dụng
    (loader chỉ đọc IMG_*), và lỗi chỉ lộ ra khi cấu hình production không
    được áp dụng.
    """
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in (root / "src").rglob("*.py"):
        if "hidream/vendor" in path.as_posix():
            continue
        text = path.read_text(encoding="utf-8")
        if "QIE_" in text:
            offenders.append(path.relative_to(root).as_posix())
    assert not offenders, f"còn tiền tố QIE_ trong: {offenders}"
