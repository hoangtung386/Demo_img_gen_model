"""FLUX.2-klein-4B image generation package.

Sinh ảnh từ chữ và sửa ảnh theo tham chiếu bằng FLUX.2-klein-4B (bản
step-distilled 4 bước) qua diffusers. Ship hai entrypoint dùng chung một
tầng model: Gradio demo (``serve.py``) và RabbitMQ consumer
(``queue_service``).
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Đi vào MODEL_VERSION của audit trail (queue_service/pipelines/gen_image.py).
# Đọc từ metadata thay vì hard-code để không bao giờ lệch pyproject.toml;
# "unknown" chỉ xảy ra khi chạy từ source tree mà package chưa được cài —
# đúng cách container đang chạy, nên fallback về version ghi trong pyproject.
_FALLBACK_VERSION = "0.2.0"
try:
    __version__ = version("gen-image-flux2-klein")
except PackageNotFoundError:  # chạy thẳng từ src/, chưa pip install
    __version__ = _FALLBACK_VERSION

# --- Ghim toàn bộ cache HuggingFace vào trong project ---------------------
#
# Mặc định huggingface_hub ghi vào ~/.cache/huggingface. Bất cứ đường nào
# lọt xuống hub (fallback repo id trong loader.py, login(), một thư viện phụ
# gọi from_pretrained bằng repo id) đều kéo hàng chục GB ra ngoài project mà
# không ai thấy.
#
# PHẢI set trước khi huggingface_hub được import lần đầu: constants.py đọc
# các biến này ở module level, set sau đó là vô nghĩa. __init__.py của package
# là chỗ duy nhất chắc chắn chạy trước mọi submodule — kể cả download.py, nơi
# `from huggingface_hub import ...` nằm ngay dòng đầu.
#
# setdefault chứ không ghi đè: Docker set HF_HOME riêng (kèm HF_HUB_OFFLINE=1)
# và phải được tôn trọng.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_HF_HOME = _PROJECT_ROOT / "models" / ".hf"

os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))

# --- Allocator CUDA ------------------------------------------------------
#
# Kích thước ảnh ra suy từ tỉ lệ ảnh người dùng nên mỗi request cấp phát một
# shape khác nhau. Allocator mặc định chia pool thành các segment cố định và
# phân mảnh dần; trên L4-24GB đã có sẵn ~16 GiB weight nằm thường trú thì
# phân mảnh là thứ đẩy service tới OOM chứ không phải thiếu VRAM thật.
# expandable_segments cho phép segment co giãn, gần như xoá hẳn vấn đề này.
#
# Phải set trước lần cấp phát CUDA đầu tiên, nên đặt cạnh phần HF_HOME ở đây.
#
# Set CẢ HAI tên biến: torch 2.9 đổi sang PYTORCH_ALLOC_CONF và in
# DeprecationWarning cho tên cũ, nhưng tên cũ vẫn là tên duy nhất torch <2.9
# đọc. Giữ cả hai cho tới khi trần phiên bản torch trong pyproject vượt qua
# mốc đó.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
