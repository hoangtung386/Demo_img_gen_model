# syntax=docker/dockerfile:1

# =============================================================================
# Builder — dựng virtualenv đầy đủ bằng uv rồi copy nguyên khối sang runtime.
#
# Python 3.13: bản mới nhất mà torch 2.9.0 có wheel. Trần này từng bị ghim
# chặt hơn bởi nunchaku (wheel dựng riêng cho từng tổ hợp python × torch ×
# CUDA); package đó đã bị gỡ hẳn nên ràng buộc duy nhất còn lại là torch.
# =============================================================================
FROM python:3.13-slim-bookworm AS builder

# uv lấy từ image chính thức, ghim phiên bản để build lặp lại được.
COPY --from=ghcr.io/astral-sh/uv:0.12.4 /uv /uvx /usr/local/bin/

# UV_PROJECT_ENVIRONMENT trỏ ra ngoài /app: runtime stage chỉ copy /opt/venv,
# không kéo theo source tree của builder.
# UV_LINK_MODE=copy vì cache uv nằm trên mount khác với /opt/venv, hardlink
# sẽ hỏng.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/local/bin/python3.13

WORKDIR /app

# CHỈ copy metadata trước `uv sync` — KHÔNG copy src ở đây. `uv sync` resolve
# dependency thuần từ pyproject.toml + uv.lock, không đọc nội dung src/. Copy
# src TRƯỚC bước này (như cũ) khiến MỌI lần sửa 1 dòng Python invalidate
# cache layer COPY → kéo theo cài lại torch/CUDA (network + build nặng
# nhất, ~30 phút) dù dependency không đổi gì. Đẩy `COPY src ./src`
# xuống sau uv sync (bên dưới) để sửa code không đụng lại layer cài đặt.
COPY pyproject.toml uv.lock README.md ./

# `uv sync` cài TOÀN BỘ dependency theo uv.lock. torch==2.9.0 từ PyPI mang
# sẵn CUDA 12.8.
#
# --frozen: dùng nguyên uv.lock, không được tự giải lại. Lock lệch pyproject
# thì build fail ở đây chứ không âm thầm cài phiên bản khác.
# --no-install-project: chưa có src/ ở bước này nên KHÔNG cài package của
# chính repo — vô hại, vì runtime stage chạy trực tiếp từ source tree copy
# đè (xem ghi chú "Chạy từ source tree" ở stage runtime bên dưới), không
# bao giờ import gen_image qua bản cài trong site-packages.
# `--no-dev` bỏ nhóm dev; extra "fp8" cũng KHÔNG vào đây — đó là nhánh chưa
# được đo (xem docs/PERFORMANCE.md §8) nên image production không mang theo
# một dependency chưa ai chứng minh là có lợi. Muốn dựng image để thử FP8:
#     ... uv sync --frozen --no-dev --extra fp8 --no-install-project
# Hoặc chạy scripts/spike_flux2.py thẳng trên host, không cần Docker.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src

# Chốt tổ hợp phiên bản NGAY LÚC BUILD. Logic nằm trong scripts/verify_build.py
# (đọc metadata, không import torch/diffusers — builder không có GPU).
COPY scripts/verify_build.py ./scripts/verify_build.py
RUN /opt/venv/bin/python scripts/verify_build.py

# =============================================================================
# Runtime
# =============================================================================
FROM python:3.13-slim-bookworm

# libgomp1  : OpenMP runtime các kernel torch cần.
# libstdc++6: extension C của torch link trực tiếp vào. Base image có sẵn,
#             khai báo tường minh để một bản slim tương lai gỡ đi thì lỗi
#             lộ ra lúc build chứ không phải lúc import.
# curl      : dùng cho HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# PYTHONUNBUFFERED là bắt buộc, không phải trang trí: Gradio in link share
# bằng print(), stdout của container bị block-buffer 8KB thì link kẹt trong
# buffer và `docker logs` không bao giờ thấy nó.
# HF_HOME KHÔNG trỏ vào /app/models: thư mục đó được mount read-only, thư
# viện nào lỡ ghi metadata vào đấy sẽ nổ. Để riêng một chỗ ghi được, và khoá
# HF_HUB_OFFLINE=1 để mọi đường rò xuống Hub thành lỗi dừng hẳn thay vì một
# lần tải 65GB im lặng. Trọng số do model-fetcher kéo từ GCS về /app/models
# nên container này KHÔNG cần mạng tới HuggingFace và KHÔNG cần HF_TOKEN.
# Build sẽ cảnh báo SecretsUsedInArgOrEnv cho HF_TOKEN bên dưới. Đó là báo
# nhầm và ĐỪNG gỡ dòng đó ra: nó đặt token thành RỖNG chứ không nhét bí mật
# vào image — chốt chặn để một HF_TOKEN lỡ có trong môi trường host không đi
# vào tiến trình phục vụ. Chốt thật sự là HF_HUB_OFFLINE=1 ngay dưới.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/.hf_cache \
    HF_HUB_OFFLINE=1 \
    HF_TOKEN="" \
    GENIMG_SERVER_NAME=0.0.0.0 \
    GENIMG_PORT=7860

WORKDIR /app

RUN mkdir -p /app/.hf_cache

# Chạy từ source tree chứ không dùng package đã cài trong site-packages:
# config.py suy ra PROJECT_ROOT bằng Path(__file__).parent×3, nên phải nằm ở
# /app/src/gen_image/ thì .env và models/ mới resolve về /app.
#
# PYTHONPATH=/app/src ở trên là thứ làm source tree đó import được. Các
# entrypoint (scripts/*.py, main_queue.py) tự chèn sys.path nên vẫn chạy
# không cần biến này — nhưng thiếu nó thì `docker compose exec ... python -c
# "import gen_image"` lại hỏng, đúng lúc người ta cần nó nhất là khi đang
# debug trong `make shell`.
COPY src ./src
COPY scripts ./scripts
# Ảnh mẫu + ảnh kết quả dựng sẵn (examples/outputs/) cho khối gr.Examples.
# KHÔNG bake vào image: thư mục này chưa bao giờ có trong repo nên `COPY
# examples` làm mọi lần build từ một bản clone sạch fail ngay tại đây. Image
# chỉ tạo sẵn chỗ trống; compose mount ./examples:ro từ host lúc chạy (xem
# service gen-image). Thiếu ảnh thì UI tự bỏ qua case đó
# (ui/app.py::_img, _build_demo_cache) — demo chạy bình thường, chỉ là khối
# ví dụ trống. Dựng lại ảnh kết quả: chạy demo rồi gọi
# scripts/warm_examples.py trên host, không cần build lại image.
RUN mkdir -p /app/examples/outputs

# --- Queue worker (RabbitMQ consumer) ------------------------------------
# main_queue.py là entrypoint riêng cho service gen-image-queue (compose,
# profile "queue"). Chỉ base.example.yaml được bake (mẫu); base.yaml thật
# (chứa secret) + credentials/*.json bị .dockerignore chặn — compose mount
# ./config_setup:ro từ host lúc chạy để inject config + key thật.
COPY main_queue.py ./main_queue.py
COPY config_setup ./config_setup

EXPOSE 7860

# start-period phải phủ được thời gian nạp ~16GB weight lên VRAM cộng
# warm-up. Nhẹ hơn backend cũ (26.4GB) nhưng giữ nguyên 600s: khi server
# thuê để weight trên ổ mạng thì thời gian nạp do I/O quyết định.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

CMD ["python", "scripts/serve.py"]
