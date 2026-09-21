"""Tải trọng số model và ghi lại đường dẫn local đã giải được.

Mọi thứ tải về đều nằm trong ``models/`` của project. Cache của
huggingface_hub cũng đã bị ghim vào project bởi ``gen_image/__init__`` nên
không có gì rơi ra ``~/.cache/huggingface``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .config import (
    PROJECT_ROOT,
    Settings,
    _load_app_yaml,
    _stored_hf_token,
    load_settings,
)

logger = logging.getLogger("download")

# Tên thư mục local cho repo pipeline. Giữ khớp với base_model_local trong
# config_setup/base.example.yaml và docker-compose.yml.
BASE_DIR_NAME = "FLUX.2-klein-9B"

# Repo GGUF có ~15 bản lượng tử hoá (Q2_K … BF16), tổng vài chục GB. Chỉ kéo
# ĐÚNG một file đang dùng — `snapshot_download` cả repo ở đây là một lỗi tốn
# kém và rất dễ mắc.
_GGUF_NOTE = (
    "Chỉ tải đúng một file GGUF đang cấu hình. Repo này có ~15 bản lượng tử "
    "hoá; tải cả repo là vài chục GB không dùng đến."
)


def _resolve_model_root(settings: Settings) -> Path:
    root = Path(settings.model_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def _resolve_token() -> str | None:
    """HF token: env THẮNG base.yaml THẮNG token đã ``login()`` ra đĩa.

    Truyền token thẳng vào từng lời gọi thay vì tự gọi ``login()``: login()
    ghi token ra ``$HF_HOME/token`` + stored_tokens, tức là tạo state trên
    đĩa ngoài ý muốn và làm token dính lại trên máy sau khi tải xong.

    Nhưng NẾU người dùng đã tự gọi ``login()`` thì phải tôn trọng — và
    không thể trông chờ hub tự tìm thấy, vì ``HF_HOME`` đã bị ghim sang
    ``models/.hf``. ``_stored_hf_token()`` đọc cả hai chỗ; xem docstring
    của nó để biết vì sao đây từng là một chế độ hỏng rất khó nhìn ra.
    """
    token = os.getenv("HF_TOKEN", "").strip() or None
    if token:
        return token
    raw = _load_app_yaml().get("hf_token", "")
    return str(raw or "").strip() or _stored_hf_token()


def download(settings: Settings) -> None:
    """Tải base pipeline, và file GGUF nếu đang chạy nhánh đó."""
    hf_token = _resolve_token()
    if not hf_token:
        logger.warning(
            "Không tìm thấy HF token ở BẤT KỲ nguồn nào (env HF_TOKEN, "
            "app.hf_token trong base.yaml, hay file token do login() ghi "
            "ra). %s là repo GATED — request ẩn danh sẽ nhận 401. Lưu ý "
            "repo GGUF chỉ chứa transformer; text encoder/VAE/scheduler "
            "vẫn phải lấy từ repo gated này. Cách chắc ăn nhất: "
            "HF_TOKEN=hf_xxx uv run download-model",
            settings.base_model,
        )

    model_root = _resolve_model_root(settings)
    logger.info("HF cache ghim tại: %s", os.environ.get("HF_HOME"))
    logger.info("Quantization: %s", settings.quantization)

    base_dir = model_root / BASE_DIR_NAME
    logger.info("Tải base pipeline %s -> %s", settings.base_model, base_dir)
    # Repo 9B có transformer BF16 ~18GB, nhưng runtime thay nó bằng GGUF.
    # Chỉ kéo các component còn lại và transformer/config.json (để
    # from_single_file biết đúng kiến trúc 9B); tránh tải rồi bỏ đi BF16.
    snapshot_download(
        repo_id=settings.base_model,
        local_dir=str(base_dir),
        token=hf_token,
        allow_patterns=[
            "model_index.json",
            "scheduler/**",
            "tokenizer/**",
            "text_encoder/**",
            "vae/**",
            "transformer/config.json",
        ],
    )
    logger.info("Base pipeline đã tải xong.")

    gguf_path: str | None = None
    if settings.quantization == "gguf":
        logger.info(
            "Tải transformer GGUF %s/%s. %s",
            settings.gguf_repo,
            settings.gguf_file,
            _GGUF_NOTE,
        )
        gguf_path = hf_hub_download(
            repo_id=settings.gguf_repo,
            filename=settings.gguf_file,
            local_dir=str(model_root / "gguf"),
            token=hf_token,
        )
        logger.info("Transformer GGUF -> %s", gguf_path)
    else:
        logger.info(
            "quantization='bf16' — bỏ qua file GGUF. Transformer bf16 đã "
            "nằm trong base pipeline ở trên."
        )

    paths_env = PROJECT_ROOT / ".model_paths.env"
    lines = [f"GENIMG_BASE_MODEL_LOCAL={base_dir}\n"]
    if gguf_path:
        lines.append(f"GENIMG_TRANSFORMER_GGUF={gguf_path}\n")
    with open(paths_env, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    logger.info("Đã ghi đường dẫn đã giải vào %s", paths_env)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = load_settings()
    download(settings)


if __name__ == "__main__":
    main()
