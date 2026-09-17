"""Download model weights and persist the resolved local path.

Mọi thứ tải về đều nằm trong ``models/`` của project. Cache của
huggingface_hub cũng đã bị ghim vào project bởi ``imagegen/__init__`` nên
không có gì rơi ra ``~/.cache/huggingface``.

So với bản Qwen trước đây: chỉ còn MỘT lời gọi tải. Model mới là một repo
transformers phẳng (~9.9GB) chứ không phải cặp "base pipeline diffusers +
file transformer Nunchaku rời", nên không còn ``hf_hub_download`` thứ hai,
không còn ``ignore_patterns`` cho 39GB shard BF16 không dùng tới, và không
còn bước dò precision int4/fp4.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import PROJECT_ROOT, Settings, _load_app_yaml, load_settings
from .hidream.artifacts import MODEL_DIR_NAME

logger = logging.getLogger("download")

# Ảnh minh hoạ của model card: ~vài chục MB, không bao giờ được đọc lúc
# chạy. Bỏ qua để archive đẩy lên GCS gọn hơn.
IGNORE_PATTERNS = ["comparison/*", "assets/*", "*.md"]


def _resolve_model_root(settings: Settings) -> Path:
    root = Path(settings.model_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def download(settings: Settings) -> Path:
    """Tải snapshot model về ``models/`` và trả về thư mục đích."""
    # Truyền token thẳng vào lời gọi thay vì login(): login() ghi token ra
    # $HF_HOME/token + stored_tokens, tức là tạo state trên đĩa ngoài ý muốn
    # và làm token dính lại trên máy sau khi tải xong.
    #
    # Repo HiDream-O1-Image (cả bản gốc lẫn bản quant của WaveCut) là MIT và
    # KHÔNG gated, nên token là tuỳ chọn — khác hẳn Qwen trước đây vốn bắt
    # accept license. Không cảnh báo khi thiếu token nữa.
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    if not hf_token:
        hf_token = (
            str(_load_app_yaml().get("hf_token", "") or "").strip() or None
        )

    model_root = _resolve_model_root(settings)
    model_dir = model_root / MODEL_DIR_NAME

    logger.info("HF cache pinned to: %s", os.environ.get("HF_HOME"))
    logger.info("Tải %s -> %s (~9.9GB)", settings.base_model, model_dir)
    snapshot_download(
        repo_id=settings.base_model,
        local_dir=str(model_dir),
        ignore_patterns=IGNORE_PATTERNS,
        token=hf_token,
    )
    logger.info("Tải xong.")

    paths_env = PROJECT_ROOT / ".model_paths.env"
    paths_env.write_text(f"IMG_MODEL_PATH={model_dir}\n", encoding="utf-8")
    logger.info("Ghi đường dẫn đã resolve vào %s", paths_env)
    return model_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    download(load_settings())


if __name__ == "__main__":
    main()
