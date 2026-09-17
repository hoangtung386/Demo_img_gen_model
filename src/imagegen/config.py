"""Centralized configuration.

Nguồn config DUY NHẤT là ``config_setup/base.yaml`` (khối ``app:``).
``serve.py``, ``download.py`` và các script model đều đọc từ đây nên không
bao giờ lệch nhau.

Thứ tự ưu tiên (cao → thấp):
  1. Biến môi trường ``IMG_* / HF_TOKEN / MODEL_ROOT`` (docker-compose set
     env nên env THẮNG — giữ tương thích deploy hiện tại).
  2. ``.model_paths.env`` (download.py ghi ra: IMG_MODEL_PATH).
  3. ``config_setup/base.yaml`` khối ``app:``.
  4. Default hard-code trong file này (nếu base.yaml thiếu key hoặc vắng mặt).

``.env`` vẫn được nạp (nếu có) để backward-compat, nhưng khuyến nghị chuyển
mọi giá trị sang base.yaml.

Tiền tố biến môi trường là ``IMG_``. Bản trước dùng một tiền tố khác
(viết tắt của Qwen Image Edit) — mất nghĩa từ khi thay lõi sang
HiDream-O1-Image. Bảng chuyển đổi: ``docs/MODEL_HIDREAM_O1.md``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATHS_ENV = PROJECT_ROOT / ".model_paths.env"
BASE_YAML = PROJECT_ROOT / "config_setup" / "base.yaml"

# Repo mặc định: bản HiDream-O1-Image đã lượng tử hóa SDNQ uint4 động.
# MIT, không gated — khác hẳn Qwen trước đây, không cần HF_TOKEN để tải.
DEFAULT_MODEL_REPO = "WaveCut/HiDream-O1-Image-SDNQ-4bit-dynamic-uint4-th1e-2"

# Số bước mặc định theo từng biến thể, dùng khi yaml/env không nói gì.
# "full" là bản undistilled (repo trên); "dev" là bản chưng cất 28 bước.
STEPS_BY_MODEL_TYPE = {"full": 50, "dev": 28}


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    # --- Nguồn model ---
    model_root: str
    base_model: str
    model_path_local: str | None

    # --- Tham số sinh ảnh ---
    model_type: str
    num_steps: int
    guidance_scale: float
    shift: float
    scheduler_name: str
    width: int
    height: int

    # --- Runtime ---
    device_override: str | None
    warmup: bool

    # --- Gradio ---
    server_name: str
    server_port: int
    demo_cache: bool

    # --- Tốc độ (xem docs/PERFORMANCE.md) --------------------------------
    # Mọi field dưới đây CÓ default: chúng được thêm sau khi service đã chạy
    # thật, nên base.yaml cũ thiếu key vẫn phải nạp được.
    #
    # Đường attention. "auto"/"sdpa" = tách hai lượt SDPA (mặc định, nhanh
    # nhất mà không cần wheel nào). "mask" = đường 4D-mask gốc của upstream,
    # giữ lại để đối chiếu numerics. "flash" = wheel flash-attn (phải tự cài).
    attention_mode: str = "auto"

    # Giải nén trọng số SDNQ về bf16 NGAY lúc nạp thay vì giải nén lại ở mỗi
    # forward pass. "auto" = bật khi card đủ VRAM. Đây là đòn bẩy tốc độ lớn
    # nhất trên card 24GB; đổi lại VRAM tăng từ ~11GiB lên ~18GiB.
    dequantize: str = "auto"

    # torch.compile cho decoder. Lời đầu tiên tốn vài phút biên dịch và mỗi
    # độ phân giải mới lại biên dịch lại — chỉ bật cho worker chạy dài với
    # MỘT kích thước cố định.
    compile_model: bool = False

    # Khoảng bước (chuẩn hoá 0..1) CÒN chạy nhánh uncond của CFG. Ngoài
    # khoảng này mỗi bước chỉ tốn MỘT forward pass thay vì hai.
    # (0.0, 1.0) = giữ nguyên hành vi gốc.
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0

    # False → bỏ snap về PREDEFINED_RESOLUTIONS, sinh đúng kích thước yêu
    # cầu (làm tròn bội số 32). Rẻ hơn nhiều nhưng NGOÀI phân bố huấn luyện.
    snap_resolution: bool = True

    # Số bước của lượt warm-up. Warm-up chỉ cần trả trước chi phí khởi tạo
    # (CUDA context, dequant lần đầu, autotune) — không cần một tấm ảnh đẹp.
    warmup_steps: int = 2

    @property
    def model_id(self) -> str:
        """Thư mục model local nếu có, ngược lại là repo id trên Hub."""
        return self.model_path_local or self.base_model


def _load_app_yaml() -> dict[str, Any]:
    """Đọc khối ``app:`` trong base.yaml.

    Vắng file → dict rỗng (dùng default).
    """
    if not BASE_YAML.exists():
        return {}
    try:
        with open(BASE_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    app = cfg.get("app")
    return app if isinstance(app, dict) else {}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() != "false"


def _str_or_none(value: Any) -> str | None:
    """Coi rỗng/None là None (giữ semantic cũ của os.getenv(...) or None)."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def load_settings() -> Settings:
    """Resolve Settings từ base.yaml + env override.

    Cũng nạp .env và .model_paths.env nếu có, để tương thích ngược.
    """
    # .env vẫn nạp để backward-compat; base.yaml là nguồn chính.
    load_dotenv(PROJECT_ROOT / ".env")
    if MODEL_PATHS_ENV.exists():
        load_dotenv(MODEL_PATHS_ENV)

    app = _load_app_yaml()

    def pick(env_key: str, yaml_key: str, default: Any) -> Any:
        """Env THẮNG YAML THẮNG default. Env rỗng ("") coi như chưa set."""
        env_val = os.getenv(env_key)
        if env_val is not None and env_val.strip() != "":
            return env_val
        if yaml_key in app and app[yaml_key] not in (None, ""):
            return app[yaml_key]
        return default

    def pick_optional(env_key: str, yaml_key: str) -> str | None:
        """Cho field 'rỗng → None' (device, path, token…)."""
        env_val = os.getenv(env_key)
        if env_val is not None and env_val.strip() != "":
            return env_val.strip()
        return _str_or_none(app.get(yaml_key))

    model_type = str(pick("IMG_MODEL_TYPE", "model_type", "full")).lower()
    # Số bước mặc định bám theo biến thể: đặt model_type: dev mà quên sửa
    # num_steps thì 50 bước của bản full sẽ chạy trên scheduler 28 bước.
    default_steps = STEPS_BY_MODEL_TYPE.get(model_type, 50)

    return Settings(
        model_root=str(pick("MODEL_ROOT", "model_root", "models")),
        base_model=str(
            pick("IMG_BASE_MODEL", "base_model", DEFAULT_MODEL_REPO)
        ),
        model_path_local=pick_optional("IMG_MODEL_PATH", "model_path"),
        model_type=model_type,
        num_steps=_as_int(
            pick("IMG_NUM_STEPS", "num_steps", default_steps), default_steps
        ),
        # CFG bật thật (khác Lightning trước đây): mỗi bước chạy HAI forward
        # pass, cond + uncond. Đặt 0.0 cho bản dev đã chưng cất.
        guidance_scale=_as_float(
            pick("IMG_GUIDANCE_SCALE", "guidance_scale", 5.0), 5.0
        ),
        shift=_as_float(pick("IMG_SHIFT", "shift", 3.0), 3.0),
        scheduler_name=str(
            pick("IMG_SCHEDULER", "scheduler_name", "default")
        ).lower(),
        # Kích thước yêu cầu. Pipeline snap về PREDEFINED_RESOLUTIONS nên
        # giá trị nhỏ hơn 2048 KHÔNG cho ảnh nhỏ hơn — xem hidream/inference.
        width=_as_int(pick("IMG_WIDTH", "width", 2048), 2048),
        height=_as_int(pick("IMG_HEIGHT", "height", 2048), 2048),
        device_override=pick_optional("IMG_DEVICE", "device"),
        # Trả trước chi phí lượt sinh ảnh đầu tiên lúc khởi động thay vì bắt
        # người dùng đầu tiên gánh. Tắt khi cần đo thời gian load thuần.
        warmup=_as_bool(pick("IMG_WARMUP", "warmup", True), default=True),
        server_name=str(pick("IMG_SERVER_NAME", "server_name", "0.0.0.0")),
        server_port=_as_int(pick("IMG_PORT", "server_port", 7860), 7860),
        # Chế độ demo: ví dụ mẫu trả ảnh dựng sẵn thay vì chạy model. Đặt
        # IMG_DEMO_CACHE=false (hoặc demo_cache: false trong yaml) để đo thật.
        demo_cache=_as_bool(
            pick("IMG_DEMO_CACHE", "demo_cache", True), default=True
        ),
        attention_mode=str(
            pick("IMG_ATTENTION_MODE", "attention_mode", "auto")
        ).lower(),
        dequantize=str(pick("IMG_DEQUANTIZE", "dequantize", "auto")).lower(),
        compile_model=_as_bool(
            pick("IMG_COMPILE", "compile_model", False), default=False
        ),
        cfg_interval_start=_as_float(
            pick("IMG_CFG_INTERVAL_START", "cfg_interval_start", 0.0), 0.0
        ),
        cfg_interval_end=_as_float(
            pick("IMG_CFG_INTERVAL_END", "cfg_interval_end", 1.0), 1.0
        ),
        snap_resolution=_as_bool(
            pick("IMG_SNAP_RESOLUTION", "snap_resolution", True), default=True
        ),
        warmup_steps=_as_int(pick("IMG_WARMUP_STEPS", "warmup_steps", 2), 2),
    )
