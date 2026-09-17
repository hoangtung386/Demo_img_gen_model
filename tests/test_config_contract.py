"""`base.example.yaml` phải khớp dataclass — chống drift im lặng.

Hai chế độ hỏng mà test này bắt, cả hai đều **không báo lỗi lúc chạy**:

* Key thừa trong YAML: loader chỉ đọc key nó biết, phần còn lại rơi vào hư
  không. Ops sửa một giá trị rồi tự hỏi vì sao service không đổi hành vi.
* Field mới trong dataclass mà quên thêm vào file mẫu: người deploy copy
  template, thiếu key, service im lặng dùng default.

Đây là lỗi đã xảy ra thật trong lần thay lõi (`output_area`, `rank`,
`precision` còn nằm lại trong YAML sau khi bị xoá khỏi `ProcessorConfig`).
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest
import yaml

from imagegen.queue_service.config import schema

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "config_setup" / "base.example.yaml"

# Tên key trong khối ``app:``. Chúng KHÔNG trùng tên field của Settings
# (``model_path`` ↔ ``model_path_local``, ``device`` ↔ ``device_override``)
# nên phải liệt kê tường minh; load_settings() là nơi ánh xạ.
APP_KEYS = frozenset(
    {
        "model_type",
        "num_steps",
        "guidance_scale",
        "shift",
        "scheduler_name",
        "width",
        "height",
        "device",
        "warmup",
        "warmup_steps",
        "demo_cache",
        "attention_mode",
        "dequantize",
        "compile_model",
        "cfg_interval_start",
        "cfg_interval_end",
        "snap_resolution",
        "model_root",
        "base_model",
        "model_path",
        "server_name",
        "server_port",
        "hf_token",
        "models_uri",
        "gcs_key_file",
        "models_force",
    }
)


@pytest.fixture(scope="module")
def example() -> dict:
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "section,cls",
    [
        ("processor", schema.ProcessorConfig),
        ("worker", schema.WorkerConfig),
        ("observability", schema.ObservabilityConfig),
        ("audit", schema.AuditConfig),
        ("rabbitmq", schema.RabbitMQConfig),
    ],
)
def test_no_unknown_keys(example, section, cls):
    """Key trong YAML mà dataclass không có = một dòng config chết."""
    fields = {f.name for f in dataclasses.fields(cls)}
    extra = set(example.get(section) or {}) - fields
    assert not extra, (
        f"base.example.yaml [{section}] có key không tồn tại trong "
        f"{cls.__name__}: {sorted(extra)}"
    )


def test_app_section_has_no_unknown_keys(example):
    extra = set(example.get("app") or {}) - APP_KEYS
    assert not extra, f"[app] có key lạ: {sorted(extra)}"


def test_config_env_map_matches_app_section():
    """``scripts/config_env.py`` xuất khối app: sang env cho script bash.

    Thiếu một map thì `fetch_models.sh` / `pack_models.sh` lặng lẽ dùng
    default của chính nó thay vì giá trị trong base.yaml.
    """
    source = (ROOT / "scripts" / "config_env.py").read_text(encoding="utf-8")
    mapped = set(re.findall(r'^\s*"(\w+)":\s*"[A-Z_]+"', source, re.M))
    assert mapped == APP_KEYS, (
        f"config_env.py thiếu: {sorted(APP_KEYS - mapped)} | "
        f"thừa: {sorted(mapped - APP_KEYS)}"
    )


def test_processor_type_is_a_known_alias(example):
    """`processor.type` trong file mẫu phải được factory nhận."""
    from imagegen.queue_service.processors import factory

    known = factory._HIDREAM_ALIASES | factory._ECHO_ALIASES
    assert example["processor"]["type"] in known


def test_smoke_config_uses_echo_processor():
    """smoke.yaml tồn tại để chạy KHÔNG có GPU — đừng để ai đổi sang model."""
    smoke = yaml.safe_load(
        (ROOT / "config_setup" / "smoke.yaml").read_text(encoding="utf-8")
    )
    assert smoke["processor"]["type"] == "echo"
    assert smoke["storage"]["backend"] == "local"
