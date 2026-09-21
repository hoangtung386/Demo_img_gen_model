"""Wrapper đơn giản quanh loadtest_queue.py — chỉ hỏi 2 thứ: bắn nhanh cỡ nào
(--rate) và bắn bao lâu (--duration). Mọi thứ khác (host, port, user,
password, vhost, ảnh test, drain-out để đo end-to-end...) fix cứng sẵn bên
dưới, khớp đúng lệnh đã dùng thủ công trước đó.

    python scripts/run_loadtest.py --rate 5 --duration 60
    python scripts/run_loadtest.py --rate 2 --duration 30

Đổi số connection song song hoặc ảnh test thì sửa hằng số ở đầu file này —
không cần nhớ lại cú pháp CLI đầy đủ của loadtest_queue.py mỗi lần chạy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import loadtest_queue  # noqa: E402

# ────────────────────────── fix cứng ──────────────────────────
# Đổi ở đây nếu vhost/ảnh test đổi — KHÔNG cần sửa gì trong loadtest_queue.py.

RMQ_HOST = "<broker-host>"
RMQ_PORT = 5672
RMQ_USER = "<user>"
RMQ_PASSWORD = ""
RMQ_VHOST = "<vhost>"

# Ảnh thật từ tests/images_test/, đã upload sẵn — xem examples/queue_payloads/.
IMAGE_URL = (
    "https://storage.googleapis.com/<your-bucket>/images_test/"
    "304dcbb3-db8c-4440-9e84-1d245a77ce63-HiepNT.jpg"
)
EDIT_RATIO = 1.0  # 1.0 = mọi message đều kèm ảnh (image-edit, không phải text-to-image)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Bắn tải vào qwen-queue (cấu hình broker + ảnh fix cứng sẵn).",
    )
    p.add_argument(
        "--rate", type=float, default=2.0, help="msg/s tổng (0 = bắn hết sức)"
    )
    p.add_argument("--duration", type=float, default=60.0, help="giây chạy test")
    args = p.parse_args()

    forwarded = [
        "--host",
        RMQ_HOST,
        "--port",
        str(RMQ_PORT),
        "--user",
        RMQ_USER,
        "--password",
        RMQ_PASSWORD,
        "--vhost",
        RMQ_VHOST,
        "--image-url",
        IMAGE_URL,
        "--edit-ratio",
        str(EDIT_RATIO),
        "--rate",
        str(args.rate),
        "--duration",
        str(args.duration),
        "--drain-out",
    ]
    return loadtest_queue.main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
