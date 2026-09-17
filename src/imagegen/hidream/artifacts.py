"""Hợp đồng về hình dạng của snapshot model trên đĩa.

Thuần stdlib — KHÔNG import torch/transformers/sdnq. Đây là điều kiện để
``scripts/preflight.py`` giữ đúng lời hứa của nó: kiểm tra cây model trong
vài giây, trên máy không GPU, trước khi ai đó tốn hàng chục giây nạp model
rồi mới biết thiếu file.

Bốn chỗ phải khớp nhau về tên thư mục, và tất cả đều đọc từ đây hoặc được
kiểm tra bằng đây:

* ``download.py``            — ghi snapshot vào ``models/<MODEL_DIR_NAME>``
* ``docker-compose.yml``     — ``IMG_MODEL_PATH=/app/models/<MODEL_DIR_NAME>``
* ``scripts/fetch_models.sh``— ``IMG_MODELS_REQUIRE`` sau khi giải nén archive
* ``scripts/preflight.py``   — kiểm tra trước khi nạp
"""

from __future__ import annotations

# Tên thư mục đích trong models/.
MODEL_DIR_NAME = "HiDream-O1-Image-SDNQ-uint4"

# File bắt buộc của một snapshot repo transformers hợp lệ.
#
# Cây này PHẲNG — khác hẳn cây diffusers của model cũ (model_index.json +
# thư mục con vae/ text_encoder/ transformer/). Kiểm cả tokenizer chứ không
# chỉ config + weight: thiếu tokenizer thì model nạp xong mới nổ, ở tận bước
# build sample.
REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)

# Giá trị config.json phải khai báo. Nạp nhầm một repo BF16 chưa lượng tử
# hoá vẫn chạy được nhưng tốn ~17 GiB VRAM thay vì ~11 — im lặng cho tới lúc
# OOM trên card nhỏ.
QUANT_METHOD = "sdnq"
ARCHITECTURE = "Qwen3VLForConditionalGeneration"

# Dung lượng weight mong đợi. Lệch nhiều gần như chắc chắn là snapshot tải
# dở hoặc archive giải nén thiếu.
EXPECTED_WEIGHT_GB = 9.9
WEIGHT_SIZE_TOLERANCE = 0.25

# 5 token đặc biệt mà ``vendor/pipeline.py`` đọc THẲNG trên tokenizer. Chúng
# KHÔNG nằm trong tokenizer_config.json — inference.py của HiDream gắn tay
# lúc chạy. Thiếu bước này thì lỗi là AttributeError giữa lúc build sample,
# sau khi đã nạp xong ~10GB weight lên VRAM.
SPECIAL_TOKENS: dict[str, str] = {
    "boi_token": "<|boi_token|>",
    "bor_token": "<|bor_token|>",
    "eor_token": "<|eor_token|>",
    "bot_token": "<|bot_token|>",
    "tms_token": "<|tms_token|>",
}
