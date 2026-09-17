#!/usr/bin/env python3
"""Xuất khối ``app:`` của config_setup/base.yaml ra dạng ``IMG_KEY=value``.

Cho script bash (fetch_models.sh, pack_models.sh) đọc config từ base.yaml —
nguồn config DUY NHẤT — thay vì trùng lặp trong nhiều nơi. Chỉ IN các key
mà biến môi trường tương ứng CHƯA được set (env vẫn thắng, giữ nguyên hành
vi docker-compose).

Dùng trong bash:
    eval "$(python3 scripts/config_env.py)"

Không phụ thuộc gì ngoài PyYAML (đã là dependency). Nếu base.yaml thiếu →
in ra rỗng, script bash rơi về default của chính nó.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit(0)  # không có yaml → không export gì, bash tự lo default

_ROOT = Path(__file__).resolve().parent.parent
_BASE_YAML = _ROOT / "config_setup" / "base.yaml"

# Map yaml key (trong khối app:) → tên biến môi trường script bash mong đợi.
_YAML_TO_ENV = {
    # Tham số sinh ảnh
    "num_steps": "IMG_NUM_STEPS",
    "model_type": "IMG_MODEL_TYPE",
    "guidance_scale": "IMG_GUIDANCE_SCALE",
    "shift": "IMG_SHIFT",
    "scheduler_name": "IMG_SCHEDULER",
    "width": "IMG_WIDTH",
    "height": "IMG_HEIGHT",
    # Runtime
    "device": "IMG_DEVICE",
    "warmup": "IMG_WARMUP",
    "warmup_steps": "IMG_WARMUP_STEPS",
    "demo_cache": "IMG_DEMO_CACHE",
    # Tốc độ
    "attention_mode": "IMG_ATTENTION_MODE",
    "dequantize": "IMG_DEQUANTIZE",
    "compile_model": "IMG_COMPILE",
    "cfg_interval_start": "IMG_CFG_INTERVAL_START",
    "cfg_interval_end": "IMG_CFG_INTERVAL_END",
    "snap_resolution": "IMG_SNAP_RESOLUTION",
    # Nguồn model
    "model_root": "MODEL_ROOT",
    "base_model": "IMG_BASE_MODEL",
    "model_path": "IMG_MODEL_PATH",
    # Server
    "server_name": "IMG_SERVER_NAME",
    "server_port": "IMG_PORT",
    # Tải trọng số
    "hf_token": "HF_TOKEN",
    "models_uri": "IMG_MODELS_URI",
    "gcs_key_file": "IMG_GCS_KEY_FILE",
    "models_force": "IMG_MODELS_FORCE",
}


def _fmt(value: object) -> str:
    """YAML value → chuỗi shell. bool → true/false (khớp convention script)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def main() -> None:
    import os

    if not _BASE_YAML.exists():
        return
    try:
        with open(_BASE_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return
    app = cfg.get("app")
    if not isinstance(app, dict):
        return

    for yaml_key, env_key in _YAML_TO_ENV.items():
        if yaml_key not in app:
            continue
        value = app[yaml_key]
        if value in (None, ""):
            continue
        # Env đã set (non-empty) → KHÔNG override, env thắng.
        existing = os.getenv(env_key)
        if existing is not None and existing.strip() != "":
            continue
        print(f"export {env_key}={shlex.quote(_fmt(value))}")


if __name__ == "__main__":
    main()
