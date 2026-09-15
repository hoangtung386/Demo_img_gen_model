"""Download model weights and persist resolved local paths.

Mọi thứ tải về đều nằm trong ``models/`` của project. Cache của
huggingface_hub cũng đã bị ghim vào project bởi ``qwen_lightning/__init__``
nên không có gì rơi ra ``~/.cache/huggingface``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .config import PROJECT_ROOT, Settings, load_settings

logger = logging.getLogger("download")

# Transformer BF16 gốc của repo Qwen (5 shard, ~39GB). Pipeline KHÔNG BAO GIỜ
# đọc nó: loader.py truyền transformer=<bản Nunchaku INT4> vào
# from_pretrained, và diffusers dùng thẳng object được truyền vào thay vì đọc
# thư mục tương ứng (pipeline_utils.py, nhánh `if name in passed_class_obj`).
# Bỏ qua 5 shard này giảm dung lượng phải tải và phải copy lên server từ
# 65GB xuống 26GB. Giữ lại config.json + index.json (~200KB) cho nguyên vẹn
# cấu trúc repo.
_BASE_TRANSFORMER_SHARDS = "transformer/*.safetensors"


def _resolve_model_root(settings: Settings) -> Path:
    root = Path(settings.model_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root


def detect_precision(settings: Settings) -> str:
    """Resolve the quantization precision.

    Prefers an explicit ``QIE_PRECISION`` override, then Nunchaku's own
    auto-detection, and finally a lightweight GPU-name heuristic.
    """
    if settings.precision:
        return settings.precision
    try:
        from nunchaku.utils import get_precision

        return get_precision()
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            if "50" in name or "blackwell" in name:
                return "fp4"
    except Exception:  # noqa: BLE001
        pass
    return "int4"


def download(settings: Settings) -> None:
    """Download the base pipeline and the quantized transformer."""
    # Truyền token thẳng vào từng lời gọi thay vì login(): login() ghi token
    # ra $HF_HOME/token + stored_tokens, tức là tạo state trên đĩa ngoài ý
    # muốn và làm token dính lại trên máy sau khi tải xong.
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    if not hf_token:
        logger.warning(
            "HF_TOKEN is empty. Qwen/Qwen-Image-Edit-2509 is gated and may "
            "require a token after accepting the license."
        )

    precision = detect_precision(settings)
    logger.info(
        "Precision: %s | steps: %d | rank: %d",
        precision,
        settings.num_steps,
        settings.rank,
    )

    model_root = _resolve_model_root(settings)
    logger.info("HF cache pinned to: %s", os.environ.get("HF_HOME"))

    ignore_patterns = None
    if not settings.download_base_transformer:
        ignore_patterns = [_BASE_TRANSFORMER_SHARDS]
        logger.info(
            "Skipping the unused BF16 transformer shards (~39GB). Set "
            "QIE_DOWNLOAD_BASE_TRANSFORMER=true to fetch the full repo."
        )

    base_dir = model_root / "Qwen-Image-Edit-2509"
    logger.info(
        "Downloading base model %s -> %s", settings.base_model, base_dir
    )
    snapshot_download(
        repo_id=settings.base_model,
        local_dir=str(base_dir),
        ignore_patterns=ignore_patterns,
        token=hf_token,
    )
    logger.info("Base model downloaded.")

    filename = (
        f"svdq-{precision}_r{settings.rank}-"
        f"qwen-image-edit-2509-lightning-"
        f"{settings.num_steps}steps-251115.safetensors"
    )
    logger.info(
        "Downloading transformer %s/%s/%s",
        settings.transformer_repo,
        settings.transformer_subdir,
        filename,
    )
    transformer_path = hf_hub_download(
        repo_id=settings.transformer_repo,
        filename=filename,
        subfolder=settings.transformer_subdir,
        local_dir=str(model_root),
        token=hf_token,
    )
    logger.info("Transformer downloaded -> %s", transformer_path)

    paths_env = PROJECT_ROOT / ".model_paths.env"
    with open(paths_env, "w", encoding="utf-8") as handle:
        handle.write(f"QIE_BASE_MODEL_LOCAL={base_dir}\n")
        handle.write(f"QIE_TRANSFORMER_PATH={transformer_path}\n")
    logger.info("Wrote resolved paths to %s", paths_env)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = load_settings()
    download(settings)


if __name__ == "__main__":
    main()
