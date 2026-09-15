"""Qwen-Image-Edit-2509 Lightning demo package.

A Gradio demo for running the Qwen-Image-Edit-2509 Lightning (4-bit, 4-step)
model via Nunchaku and diffusers, optimized for interactive inference.
"""
from __future__ import annotations

import os
from pathlib import Path

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
# phân mảnh dần; trên A100-40GB đã có sẵn 26.4 GiB weight nằm thường trú thì
# phân mảnh là thứ đẩy service tới OOM chứ không phải thiếu VRAM thật.
# expandable_segments cho phép segment co giãn, gần như xoá hẳn vấn đề này.
#
# Phải set trước lần cấp phát CUDA đầu tiên, nên đặt cạnh phần HF_HOME ở đây.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
