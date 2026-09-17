# syntax=docker/dockerfile:1

# =============================================================================
# Builder — dựng virtualenv đầy đủ bằng uv rồi copy nguyên khối sang runtime.
#
# Python 3.13: torch >= 2.10 và transformers 4.57.1 đều có wheel cp313.
# Không còn ràng buộc ABI cứng như thời nunchaku (wheel build riêng cho từng
# tổ hợp python × torch × CUDA) — đổi minor Python giờ chỉ cần `uv lock` lại.
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

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# `uv sync` cài TOÀN BỘ dependency theo uv.lock.
#
# --frozen: dùng nguyên uv.lock, không được tự giải lại. Lock lệch pyproject
# thì build fail ở đây chứ không âm thầm cài phiên bản khác.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Chốt lại tổ hợp phiên bản NGAY LÚC BUILD. Đọc metadata chứ KHÔNG import:
# `import torch` / `import sdnq` nạp extension cần CUDA runtime, mà builder
# không có GPU.
#
# transformers ghim cứng 4.57.1: hidream/vendor/qwen3_vl_transformers.py là
# bản copy-sửa của modeling code Qwen3-VL nên bám vào nội bộ transformers.
# Một lần `uv lock` vô ý nâng lên 5.x sẽ chỉ lộ ra sau vài chục giây nạp
# model, bằng một traceback khó lần.
RUN /opt/venv/bin/python -c "\
from importlib.metadata import version; \
assert version('transformers') == '4.57.1', version('transformers'); \
_v = tuple(int(x) for x in version('torch').split('.')[:2]); \
assert _v >= (2, 10), version('torch'); \
print('OK — torch', version('torch'), '| transformers', version('transformers'), \
      '| sdnq', version('sdnq'), '| diffusers', version('diffusers'))"

# =============================================================================
# Runtime
# =============================================================================
FROM python:3.13-slim-bookworm

# libgomp1: OpenMP runtime các kernel torch cần.
# curl    : dùng cho HEALTHCHECK.
#
# libstdc++6 đã được gỡ khỏi danh sách: nó chỉ cần cho nunchaku/_C.so, thứ
# không còn tồn tại sau khi thay lõi. SDNQ thuần PyTorch, không có extension
# C nào của riêng nó.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
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
# lần tải ~10GB im lặng. Trọng số do model-fetcher kéo từ GCS về /app/models
# nên container này KHÔNG cần mạng tới HuggingFace và KHÔNG cần HF_TOKEN.
# Build sẽ cảnh báo SecretsUsedInArgOrEnv cho HF_TOKEN bên dưới. Đó là báo
# nhầm và ĐỪNG gỡ dòng đó ra: nó đặt token thành RỖNG chứ không nhét bí mật
# vào image — chốt chặn để một HF_TOKEN lỡ có trong môi trường host không đi
# vào tiến trình phục vụ. Chốt thật sự là HF_HUB_OFFLINE=1 ngay dưới.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.hf_cache \
    HF_HUB_OFFLINE=1 \
    HF_TOKEN="" \
    IMG_SERVER_NAME=0.0.0.0 \
    IMG_PORT=7860 \
    IMG_USE_FLASH_ATTN=false

WORKDIR /app

RUN mkdir -p /app/.hf_cache

# Chạy từ source tree chứ không dùng package đã cài trong site-packages:
# config.py suy ra PROJECT_ROOT bằng Path(__file__).parent×3, nên phải nằm ở
# /app/src/imagegen/ thì .env và models/ mới resolve về /app.
# Kèm cả src/imagegen/hidream/vendor/ — code inference của HiDream không
# có trên PyPI nên phải đi theo image (xem vendor/VENDOR.md).
COPY src ./src
COPY scripts ./scripts
# Ảnh mẫu + ảnh kết quả dựng sẵn (examples/outputs/) cho khối gr.Examples.
# KHÔNG bake vào image: thư mục này chưa bao giờ có trong repo nên `COPY
# examples` làm mọi lần build từ một bản clone sạch fail ngay tại đây. Image
# chỉ tạo sẵn chỗ trống; compose mount ./examples:ro từ host lúc chạy (xem
# service imagegen). Thiếu ảnh thì UI tự bỏ qua case đó
# (ui/components.py::sample_image, build_demo_cache) — demo chạy bình
# thường, chỉ là khối ví dụ trống. Dựng lại ảnh kết quả: chạy demo rồi gọi
# scripts/warm_examples.py trên host, không cần build lại image.
RUN mkdir -p /app/examples/outputs

# --- Queue worker (RabbitMQ consumer) ------------------------------------
# main_queue.py là entrypoint riêng cho service imagegen-queue (docker-compose,
# profile "queue"). Chỉ base.example.yaml được bake (mẫu); base.yaml thật
# (chứa secret) + credentials/*.json bị .dockerignore chặn — compose mount
# ./config_setup:ro từ host lúc chạy để inject config + key thật.
COPY main_queue.py ./main_queue.py
COPY config_setup ./config_setup

EXPOSE 7860

# start-period phải phủ được CẢ nạp ~9.9GB weight LẪN warm-up. Warm-up giờ
# là một lượt sinh ảnh thật ở 2048² với 50 bước + CFG — hàng chục giây đến
# vài phút, chứ không phải 4 bước ở 512² như bản Lightning cũ. 900s là mức
# an toàn; hạ xuống là container bị giết giữa lúc warm-up rồi restart vô tận.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

CMD ["python", "scripts/serve.py"]
