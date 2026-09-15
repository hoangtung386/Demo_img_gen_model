# syntax=docker/dockerfile:1

# =============================================================================
# Builder — dựng virtualenv đầy đủ rồi copy nguyên khối sang runtime stage.
# =============================================================================
FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

RUN python -m venv /opt/venv

# Torch phải cài từ index cu128 TRƯỚC nunchaku. Wheel nunchaku được build
# riêng cho từng tổ hợp python × torch × CUDA (ở đây cp311 + torch2.9 +
# cu12.8); lệch một thành phần là lỗi ABI ngay lúc import.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.9.0 torchvision==0.24.0

# Nunchaku thật chỉ phát hành trên GitHub releases. Package tên "nunchaku"
# trên PyPI là một thư viện thống kê sinh học khác hoàn toàn — xem ghi chú
# dài trong pyproject.toml. Đổi torch thì phải đổi wheel tương ứng.
ARG NUNCHAKU_WHEEL=https://github.com/nunchaku-tech/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.9-cp311-cp311-linux_x86_64.whl
RUN pip install "${NUNCHAKU_WHEEL}"

# Phần còn lại lấy từ pyproject. torch==2.9.0 đã thoả bởi bản +cu128 ở trên
# nên pip không kéo lại bản PyPI. `accelerate` (offload.py cần) đi kèm
# nunchaku, không phải khai báo thêm.
#
# THỨ TỰ BẮT BUỘC: nunchaku phải được cài TRƯỚC dòng này. pyproject khai báo
# `nunchaku` như dependency thật, nhưng URL wheel nằm trong [tool.uv.sources]
# mà pip KHÔNG đọc. Wheel cài ở trên đã thoả requirement nên pip bỏ qua;
# đảo thứ tự lại thì pip sẽ đi tìm trên PyPI và lôi về package thống kê sinh
# học trùng tên. Dùng `uv sync` thì không có vấn đề này.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# =============================================================================
# Runtime
# =============================================================================
FROM python:3.11-slim-bookworm

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
# lần tải 65GB im lặng. Toàn bộ weight đã nằm ở /app/models nên không cần mạng.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.hf_cache \
    HF_HUB_OFFLINE=1 \
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
