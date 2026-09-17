"""Mở Gradio demo kèm link công khai *.gradio.live.

Tách khỏi ``app.py``: phần này không biết gì về model hay về các tab, nó chỉ
lo tunnel và vòng thử lại. Gộp chung làm ``app.py`` phình ra và che mất phần
thật sự cần đọc khi sửa giao diện.
"""

from __future__ import annotations

import hashlib
import time
import urllib.request
from pathlib import Path

import gradio as gr

from ..config import Settings
from ..logging_setup import LOGGER_NAME

logger = __import__("logging").getLogger(LOGGER_NAME)


# Link công khai là BẮT BUỘC, không phải tuỳ chọn — không có biến để tắt.
# Đường chạy chính của repo là Colab / máy thuê, nơi cổng 7860 không tiếp cận
# được từ ngoài: service chạy mà không có link thì coi như không chạy. Vì vậy
# thiếu link được coi là lỗi khởi động, không phải cảnh báo.
SHARE_ATTEMPTS = 3
SHARE_RETRY_WAIT = 5.0
FRPC_ATTEMPTS = 3
FRPC_RETRY_WAIT = 3.0


def ensure_frpc_binary() -> None:
    """Đảm bảo binary frpc mà Gradio cần để dựng tunnel đã nằm trên đĩa.

    Đây là kiểu hỏng hay gặp nhất của share link, và nó hỏng ÂM THẦM: Gradio
    tự tải binary lúc launch(), tải không được thì chỉ in vài dòng text rồi
    phục vụ tiếp ở local như không có chuyện gì. Tải trước ở đây để lỗi lộ ra
    ngay và còn thử lại được, thay vì mất 3 phút nạp model rồi mới biết.

    Đọc hằng số từ chính ``gradio.tunneling`` nên nâng gradio (kéo theo đổi
    phiên bản frpc) không phải sửa gì ở đây.
    """
    from gradio import tunneling

    path = Path(tunneling.BINARY_PATH)
    if path.exists():
        logger.info("Binary tunnel frpc đã có sẵn: %s", path)
        return

    url = tunneling.BINARY_URL
    expected = tunneling.CHECKSUMS.get(url)
    if expected is None:
        # Không có checksum để đối chiếu thì không tự ghi file thực thi —
        # để Gradio tự lo phần tải như mặc định.
        logger.warning(
            "gradio.tunneling.CHECKSUMS không có mục cho %s — để Gradio tự "
            "tải thay vì ghi một file thực thi không kiểm chứng được.",
            url,
        )
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, FRPC_ATTEMPTS + 1):
        logger.info(
            "Đang tải binary tunnel frpc (lần %d/%d): %s",
            attempt,
            FRPC_ATTEMPTS,
            url,
        )
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                blob = resp.read()
            digest = hashlib.sha256(blob).hexdigest()
            if digest != expected:
                raise RuntimeError(f"sha256 lệch: {digest} != {expected}")
        except Exception as exc:  # noqa: BLE001 - báo lại sau vòng lặp
            last_error = exc
            logger.error("Tải frpc hỏng: %s", exc)
            if attempt < FRPC_ATTEMPTS:
                time.sleep(FRPC_RETRY_WAIT)
            continue

        path.write_bytes(blob)
        path.chmod(0o755)
        logger.info("Binary frpc sẵn sàng: %s", path)
        return

    raise RuntimeError(
        f"Không tải được binary tunnel frpc sau {FRPC_ATTEMPTS} lần thử, "
        "nên không thể tạo link công khai. Tải tay rồi đặt vào "
        f"{path}:\n    curl -L {url} -o {path}\n    chmod +x {path}"
    ) from last_error


def launch_with_public_link(demo: gr.Blocks, settings: Settings) -> None:
    """Mở demo kèm link *.gradio.live, thử lại rồi nổ nếu không dựng được.

    ``prevent_thread_lock=True`` để lấy được share_url trả về mà kiểm tra —
    ``launch()`` mặc định block nên mọi dòng sau nó không bao giờ chạy. Gradio
    đặt share_url=None khi tunnel hỏng (không tải được frpc, bị chặn mạng,
    relay quá tải) và vẫn phục vụ tiếp ở local, đúng kiểu hỏng âm thầm cần
    chặn ở đây. Giữ thread sống lại bằng block_thread() ở cuối.
    """
    ensure_frpc_binary()

    last_error: Exception | None = None
    for attempt in range(1, SHARE_ATTEMPTS + 1):
        logger.info(f"Đang tạo public link (lần {attempt}/{SHARE_ATTEMPTS})...")
        try:
            _, local_url, share_url = demo.launch(
                server_name=settings.server_name,
                server_port=settings.server_port,
                share=True,
                prevent_thread_lock=True,
                show_error=True,
                quiet=False,
            )
        except Exception as exc:  # noqa: BLE001 - báo lại ở cuối vòng lặp
            last_error = exc
            share_url = None
            local_url = None
            logger.error("launch() hỏng (lần %d): %s", attempt, exc)

        if share_url:
            logger.info("-" * 60)
            logger.info("Gradio app đã khởi động thành công!")
            logger.info("Public URL: %s (sống 1 tuần)", share_url)
            logger.info("Local URL : %s", local_url)
            logger.info("-" * 60)
            logger.info("Nhấn Ctrl+C để dừng app")
            demo.block_thread()
            return

        logger.error(
            "Không dựng được tunnel share (lần %d/%d). Thử lại sau %.0fs.",
            attempt,
            SHARE_ATTEMPTS,
            SHARE_RETRY_WAIT,
        )
        # Đóng hẳn để lần sau bind lại được cổng 7860.
        demo.close()
        if attempt < SHARE_ATTEMPTS:
            logger.info(f"⏳ Đợi {SHARE_RETRY_WAIT}s trước khi thử lại...")
            time.sleep(SHARE_RETRY_WAIT)

    raise RuntimeError(
        f"Không tạo được link công khai sau {SHARE_ATTEMPTS} lần thử — "
        "dừng hẳn thay vì phục vụ ở một cổng không ai với tới được. "
        "Kiểm tra kết nối ra Internet của máy và xem "
        "https://api.gradio.app còn sống không."
    ) from last_error
