"""HiDream-O1-Image (SDNQ 4-bit) — Gradio demo + RabbitMQ worker.

Một model hợp nhất (Pixel-level Unified Transformer) chạy cả text-to-image
lẫn image editing. Xem ``docs/MODEL_HIDREAM_O1.md`` về lần thay lõi từ
Qwen-Image-Edit-2509 Lightning.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Ghim toàn bộ cache HuggingFace vào trong project ---------------------
#
# Mặc định huggingface_hub ghi vào ~/.cache/huggingface. Bất cứ đường nào
# lọt xuống hub (fallback repo id trong hidream/loader.py, một thư viện phụ
# gọi from_pretrained bằng repo id) đều kéo hàng GB ra ngoài project mà
# không ai thấy.
#
# PHẢI set trước khi huggingface_hub được import lần đầu: constants.py đọc
# các biến này ở module level, set sau đó là vô nghĩa. __init__.py của
# package là chỗ duy nhất chắc chắn chạy trước mọi submodule — kể cả
# download.py, nơi `from huggingface_hub import ...` nằm ngay dòng đầu.
#
# setdefault chứ không ghi đè: Docker set HF_HOME riêng (kèm HF_HUB_OFFLINE=1)
# và phải được tôn trọng.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_HF_HOME = _PROJECT_ROOT / "models" / ".hf"

os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))

# --- Allocator CUDA ------------------------------------------------------
#
# Model snap kích thước ảnh ra về một trong 11 độ phân giải cứng, nên shape
# cấp phát vẫn thay đổi giữa các request. Allocator mặc định chia pool thành
# segment cố định và phân mảnh dần; expandable_segments cho phép segment co
# giãn, gần như xoá hẳn vấn đề này.
#
# Phải set trước lần cấp phát CUDA đầu tiên, nên đặt cạnh phần HF_HOME.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.2.0"
