# syntax=docker/dockerfile:1

# =============================================================================
# Builder — dựng virtualenv đầy đủ bằng uv rồi copy nguyên khối sang runtime.
#
# Python 3.13 chứ không phải 3.11: wheel nunchaku được build riêng cho từng
# tổ hợp python × torch × CUDA, và bản đang dùng là cp313 + torch2.9 +
# cu12.8. Lệch một thành phần là lỗi ABI ngay lúc import.
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

# `uv sync` cài TOÀN BỘ dependency theo uv.lock, KỂ CẢ nunchaku: pyproject
# khai báo nunchaku như dependency thật và [tool.uv.sources] ghim URL wheel
# cp313 khớp đúng python của image này. torch==2.9.0 từ PyPI mang sẵn CUDA
# 12.8 nên khớp luôn với wheel cu12.8.
#
# --frozen: dùng nguyên uv.lock, không được tự giải lại. Lock lệch pyproject
# thì build fail ở đây chứ không âm thầm cài phiên bản khác.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Cài lại đúng wheel nunchaku một cách tường minh. `uv sync` ở trên đã cài
# đúng bản này rồi nên đây là bước thừa về mặt kết quả — giữ lại vì nó ghim
# rõ ràng tổ hợp đang dùng ngay trong Dockerfile, và vì nếu ai đó gỡ
# [tool.uv.sources] ra thì bước này vẫn cứu được.
#
# THỨ TỰ BẮT BUỘC: phải chạy SAU `uv sync`. nunchaku khai báo `diffusers`
# KHÔNG ghim phiên bản; chạy trước thì uv kéo diffusers mới nhất (>=0.37),
# nơi chữ ký QwenEmbedRope.forward() đã đổi và transformer nunchaku gọi sai.
# Chạy sau thì diffusers==0.36.0 đã có sẵn và thoả requirement nên uv không
# đụng tới.
ARG NUNCHAKU_WHEEL=https://github.com/nunchaku-tech/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.9-cp313-cp313-linux_x86_64.whl
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python "${NUNCHAKU_WHEEL}"

# Chốt lại tổ hợp phiên bản NGAY LÚC BUILD. Đọc metadata chứ không import:
# `import nunchaku` nạp extension C cần CUDA runtime, mà builder không có GPU.
# Không có bước này thì diffusers bị nâng nhầm chỉ lộ ra sau 2 phút nạp model
# bằng một TypeError khó lần.
RUN /opt/venv/bin/python - <<'PY'
from importlib.metadata import version
checks = {
    "diffusers": "0.36.0",
    "torch": "2.9.0",
    "transformers": "5.15.0",
}
for name, want in checks.items():
    got = version(name)
    assert got == want, f"{name}: cần {want}, đang là {got}"
nunchaku = version("nunchaku")
assert nunchaku.startswith("1.2.1+cu12.8torch2.9"), f"nunchaku: {nunchaku}"
print(f"OK — diffusers {checks['diffusers']}, torch {checks['torch']}, "
      f"nunchaku {nunchaku}")
PY

# =============================================================================
# Runtime
# =============================================================================
FROM python:3.13-slim-bookworm

# libgomp1  : OpenMP runtime các kernel torch/nunchaku cần.
# libstdc++6: nunchaku/_C.so link trực tiếp vào (readelf -d cho thấy NEEDED).
#             Base image có sẵn, khai báo tường minh để một bản slim tương
#             lai gỡ đi thì lỗi lộ ra lúc build chứ không phải lúc import.
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
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.hf_cache \
    HF_HUB_OFFLINE=1 \
    HF_TOKEN="" \
    QIE_SERVER_NAME=0.0.0.0 \
    QIE_PORT=7860

WORKDIR /app

RUN mkdir -p /app/.hf_cache

# Chạy từ source tree chứ không dùng package đã cài trong site-packages:
# config.py suy ra PROJECT_ROOT bằng Path(__file__).parent×3, nên phải nằm ở
# /app/src/qwen_lightning/ thì .env và models/ mới resolve về /app.
COPY src ./src
COPY scripts ./scripts
# Ảnh mẫu + ảnh kết quả dựng sẵn (examples/outputs/) cho khối gr.Examples.
# Bake thẳng vào image nên tester bấm ví dụ là thấy kết quả ngay sau khi
# container khởi động, không phải chờ warm-up lại. Đổi ảnh hoặc đổi prompt thì
# chạy lại scripts/warm_examples.py rồi build lại.
COPY examples ./examples

EXPOSE 7860

# start-period phải phủ được thời gian nạp 26.4GB weight lên VRAM. 300s vừa
# đủ trên NVMe cục bộ nhưng quá sát nếu server thuê để weight trên ổ mạng.
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -fsS http://localhost:7860/ >/dev/null || exit 1

CMD ["python", "scripts/serve.py"]
