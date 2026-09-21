"""Centralized configuration.

Nguồn config DUY NHẤT là ``config_setup/base.yaml`` (khối ``app:``).
``serve.py``, ``download.py`` và các script model đều đọc từ đây nên không
bao giờ lệch nhau.

Thứ tự ưu tiên (cao → thấp):
  1. Biến môi trường ``GENIMG_* / HF_TOKEN / MODEL_ROOT`` (docker-compose set
     env nên env THẮNG — giữ tương thích deploy hiện tại).
  2. ``.model_paths.env`` (download.py ghi ra: GENIMG_BASE_MODEL_LOCAL
     và GENIMG_TRANSFORMER_GGUF).
  3. ``config_setup/base.yaml`` khối ``app:``.
  4. Default hard-code trong file này (nếu base.yaml thiếu key hoặc vắng mặt).

``.env`` vẫn được nạp (nếu có) để backward-compat, nhưng khuyến nghị chuyển
mọi giá trị sang base.yaml.

Tiền tố biến môi trường đã đổi ``QIE_`` → ``GENIMG_`` cùng lúc với việc đổi
backend sang FLUX.2. Đây là một cú cắt SẠCH có chủ đích: giữ ``QIE_`` sống
song song sẽ để một ``QIE_TRANSFORMER_PATH`` cũ còn sót trong môi trường
deploy âm thầm trỏ pipeline về trọng số Qwen không còn tồn tại — đúng cái
chế độ hỏng mà ``pick()`` (env thắng YAML) khiến khó lần nhất.
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

# Repo pipeline đầy đủ 9B: tokenizer, Qwen3-8B text encoder, VAE, scheduler
# và ``transformer/config.json``. Transformer được thay bằng file GGUF bên
# dưới; repo GGUF không chứa các component còn lại mà Flux2KleinPipeline cần.
# Đây là bản distilled 4 bước, không phải ``klein-base-9B``.
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.2-klein-9B"

# Chạy 9B qua GGUF là cấu hình mặc định. Q4_K_M (~5.9 GB) là điểm cân bằng
# thực tế hơn Q8_0 (~10 GB) khi còn phải giữ text encoder 8B và VAE trên GPU.
DEFAULT_GGUF_REPO = "unsloth/FLUX.2-klein-9B-GGUF"
DEFAULT_GGUF_FILE = "flux-2-klein-9b-Q4_K_M.gguf"


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    num_steps: int
    guidance_scale: float
    quantization: str
    compile_transformer: bool
    vae_tiling: bool
    vae_slicing: bool
    embed_cache_size: int
    model_root: str
    base_model: str
    gguf_repo: str
    gguf_file: str
    offload: str
    server_name: str
    server_port: int
    demo_cache: bool
    device_override: str | None
    base_model_local: str | None
    transformer_gguf: str | None
    warmup: bool

    @property
    def base_model_id(self) -> str:
        """Local base pipeline directory if present, else the HF id."""
        return self.base_model_local or self.base_model


def _load_app_yaml() -> dict[str, Any]:
    """Đọc khối ``app:`` trong base.yaml. Vắng file → dict rỗng (dùng default).

    In WARNING ra stderr khi thiếu/lỗi file: mọi giá trị sẽ rơi về default
    hard-code trong ``load_settings()`` bên dưới. Im lặng ở đây từng khiến
    một lần thiếu ``base.yaml`` (do quên mount, quên tạo file, hay mount sai
    đường dẫn) không để lại dấu vết nào trong log — chỉ lộ ra qua hành vi
    runtime sai.
    """
    if not BASE_YAML.exists():
        print(
            f"[gen_image.config] CẢNH BÁO: không thấy {BASE_YAML} — "
            "mọi key trong base.yaml sẽ dùng default hard-code (vd "
            "quantization='gguf'). Nếu service này cần mount config_setup/, "
            "kiểm tra lại volumes: trong docker-compose.yml.",
            flush=True,
        )
        return {}
    try:
        with open(BASE_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(
            f"[gen_image.config] CẢNH BÁO: lỗi đọc {BASE_YAML} ({exc}) "
            "— mọi key trong base.yaml sẽ dùng default hard-code.",
            flush=True,
        )
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
    return str(value).strip().lower() == "true"


def _str_or_none(value: Any) -> str | None:
    """Coi rỗng/None là None (giữ semantic cũ của os.getenv(...) or None)."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def load_settings() -> Settings:
    """Resolve Settings từ base.yaml + env override.

    Cũng nạp ``.env`` và ``.model_paths.env`` nếu có.
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

    return Settings(
        num_steps=_as_int(pick("GENIMG_NUM_STEPS", "num_steps", 4), 4),
        # Klein distilled: guidance đi vào model như một điều kiện, KHÔNG
        # phải CFG hai nhánh. diffusers tắt hẳn CFG khi is_distilled=True
        # (model_index.json của repo distilled set cờ này), nên >1.0 ở đây
        # chỉ đổi điều kiện chứ không nhân đôi chi phí denoise.
        guidance_scale=_as_float(
            pick("GENIMG_GUIDANCE_SCALE", "guidance_scale", 1.0), 1.0
        ),
        # 9B mặc định GGUF; nạp BF16 cần ~29 GB VRAM và không phù hợp L4 24GB.
        quantization=str(pick("GENIMG_QUANTIZATION", "quantization", "gguf"))
        .strip()
        .lower(),
        compile_transformer=_as_bool(
            pick("GENIMG_COMPILE", "compile_transformer", False),
            default=False,
        ),
        # VAE tiling: giải mã latent theo ô thay vì cả ảnh một lần. Đổi một
        # chút thời gian lấy VRAM đỉnh thấp hơn hẳn ở bước decode. Với model
        # 4 bước, decode chiếm tỉ trọng LỚN hơn nhiều so với model 30 bước —
        # đo bằng `make benchmark` trước khi bật đại.
        vae_tiling=_as_bool(
            pick("GENIMG_VAE_TILING", "vae_tiling", False), default=False
        ),
        # VAE slicing: decode từng ảnh một khi batch > 1. Vô nghĩa ở
        # batch=1 (cấu hình hiện tại), để sẵn cho lúc bật batching.
        vae_slicing=_as_bool(
            pick("GENIMG_VAE_SLICING", "vae_slicing", False), default=False
        ),
        embed_cache_size=_as_int(
            pick("GENIMG_EMBED_CACHE_SIZE", "embed_cache_size", 8), 8
        ),
        model_root=str(pick("MODEL_ROOT", "model_root", "models")),
        base_model=str(pick("GENIMG_BASE_MODEL", "base_model", DEFAULT_BASE_MODEL)),
        gguf_repo=str(pick("GENIMG_GGUF_REPO", "gguf_repo", DEFAULT_GGUF_REPO)),
        gguf_file=str(pick("GENIMG_GGUF_FILE", "gguf_file", DEFAULT_GGUF_FILE)),
        # "resident" (mặc định, L4 24GB) | "model_offload" (đường lùi OOM).
        offload=str(pick("GENIMG_OFFLOAD", "offload", "resident")).lower(),
        server_name=str(pick("GENIMG_SERVER_NAME", "server_name", "0.0.0.0")),
        server_port=_as_int(pick("GENIMG_PORT", "server_port", 7860), 7860),
        # Chế độ demo: ví dụ mẫu trả ảnh dựng sẵn thay vì chạy model. Đặt
        # GENIMG_DEMO_CACHE=false để đo thật.
        demo_cache=str(pick("GENIMG_DEMO_CACHE", "demo_cache", "true")).strip().lower()
        != "false",
        device_override=pick_optional("GENIMG_DEVICE", "device"),
        base_model_local=pick_optional("GENIMG_BASE_MODEL_LOCAL", "base_model_local"),
        transformer_gguf=pick_optional("GENIMG_TRANSFORMER_GGUF", "transformer_gguf"),
        # Trả trước chi phí lượt sinh ảnh đầu tiên lúc khởi động thay vì bắt
        # người dùng đầu tiên gánh. Tắt khi cần đo thời gian load thuần.
        warmup=str(pick("GENIMG_WARMUP", "warmup", "true")).strip().lower() != "false",
    )
