#!/usr/bin/env python3
"""Xuất khối ``app:`` của config_setup/base.yaml ra dạng ``GENIMG_KEY=value``.

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
    "num_steps": "GENIMG_NUM_STEPS",
    "guidance_scale": "GENIMG_GUIDANCE_SCALE",
    "quantization": "GENIMG_QUANTIZATION",
    "text_encoder_quantization": "GENIMG_TEXT_ENCODER_QUANTIZATION",
    "reference_area": "GENIMG_REFERENCE_AREA",
    "compile_transformer": "GENIMG_COMPILE",
    "device": "GENIMG_DEVICE",
    "offload": "GENIMG_OFFLOAD",
    "warmup": "GENIMG_WARMUP",
    "demo_cache": "GENIMG_DEMO_CACHE",
    "model_root": "MODEL_ROOT",
    "base_model": "GENIMG_BASE_MODEL",
    "gguf_repo": "GENIMG_GGUF_REPO",
    "gguf_file": "GENIMG_GGUF_FILE",
    "base_model_local": "GENIMG_BASE_MODEL_LOCAL",
    "transformer_gguf": "GENIMG_TRANSFORMER_GGUF",
    "server_name": "GENIMG_SERVER_NAME",
    "server_port": "GENIMG_PORT",
    "hf_token": "HF_TOKEN",
    "models_uri": "GENIMG_MODELS_URI",
    "gcs_key_file": "GENIMG_GCS_KEY_FILE",
    "models_force": "GENIMG_MODELS_FORCE",
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
