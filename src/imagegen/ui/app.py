"""Điểm ráp của demo Gradio: nạp model, dựng Blocks, mở link.

File này CHỈ giữ phần chung. Bảy tab nằm ở ``ui/tabs/`` (mỗi tab một
module), widget dùng chung ở ``components.py``, phần dựng link công khai ở
``launcher.py``.

Hai closure ``_run`` / ``_run_t2i`` là chỗ duy nhất trong tầng UI chạm tới
model; chúng được đóng gói vào :class:`TabContext` rồi truyền xuống, nên
không module tab nào phải biết ``Settings``, ``model`` hay ``processor``.

CẢNH BÁO VỀ SỐ ẢNH VÀO: HiDream-O1 khuyến nghị ĐÚNG MỘT ảnh tham chiếu cho
editing. Ba tab truyền hai ảnh (thử đồ, ghép người, hoán đổi khuôn mặt) vẫn
chạy nhưng rơi vào đường subject-driven, KHÔNG còn ngữ nghĩa "ảnh cuối là
ảnh nền" của pipeline Qwen cũ. Chất lượng những tab đó phải được đánh giá
lại — xem ``docs/MODEL_HIDREAM_O1.md``.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import gradio as gr
import torch

from ..config import Settings, load_settings
from ..device import select_device
from ..hidream import Recipe, build_recipe, generate, load_model
from ..logging_setup import configure_logging
from ..tuning import apply_gpu_tuning, warmup
from . import prompts, tabs
from .components import RESOLUTION_PRESETS, build_demo_cache
from .context import TabContext
from .launcher import launch_with_public_link

logger = configure_logging()


def build_ui(
    model,
    processor,
    settings: Settings,
) -> gr.Blocks:
    """Build and return the Gradio Blocks interface.

    ``settings.demo_cache=True`` khiến các ví dụ mẫu trả thẳng ảnh đã dựng
    sẵn thay vì chạy model — dùng khi trình diễn. Đặt ``IMG_DEMO_CACHE=false``
    để tắt và đo hiệu năng thật.
    """
    num_steps = settings.num_steps
    default_cfg = settings.guidance_scale
    default_shift = settings.shift

    cache = build_demo_cache() if settings.demo_cache else {}
    logger.info(
        "Demo cache: %s (%d ví dụ dựng sẵn)",
        "bật" if settings.demo_cache else "tắt",
        len(cache),
    )

    def _recipe(cfg, steps, shift) -> Recipe:
        """Recipe cho một lượt chạy, lấy tham số từ widget.

        Đi qua ``build_recipe`` chứ không dựng ``Recipe`` tay: hàm đó còn
        điền ``timesteps_list`` và tham số noise của scheduler flash — bỏ
        qua là scheduler chạy sai lịch mà không báo lỗi.
        """
        return build_recipe(
            replace(
                settings,
                guidance_scale=float(cfg),
                shift=float(shift),
            ),
            num_steps=int(steps),
        )

    def _run(
        images,
        prompt_text,
        note,
        cfg,
        seed,
        steps,
        shift,
        res,
        match_input_size=False,
    ):
        """Nối ghi chú người dùng vào prompt task rồi chạy model."""
        width, height = RESOLUTION_PRESETS.get(
            res, (settings.width, settings.height)
        )
        return generate(
            model,
            processor,
            images,
            prompts.append_note(prompt_text, note),
            _recipe(cfg, steps, shift),
            seed=seed,
            width=width,
            height=height,
            # Đúng một ảnh → giữ khung ảnh vào thay vì snap về danh sách cứng.
            keep_original_aspect=len(images) == 1,
            match_input_size=match_input_size,
        )

    def _run_t2i(prompt_text, cfg, seed, steps, shift, res):
        """Sinh ảnh từ chữ — CÙNG model, chỉ khác là không có ảnh vào.

        Bản Qwen trước đây phải dựng một pipeline diffusers thứ hai cho tab
        này. HiDream-O1 là model hợp nhất nên không cần: cùng một lời gọi,
        ``images`` rỗng.
        """
        width, height = RESOLUTION_PRESETS.get(
            res, (settings.width, settings.height)
        )
        return generate(
            model,
            processor,
            [],
            prompt_text,
            _recipe(cfg, steps, shift),
            seed=seed,
            width=width,
            height=height,
        )

    ctx = TabContext(
        run=_run,
        run_t2i=_run_t2i,
        cache=cache,
        num_steps=num_steps,
        default_cfg=default_cfg,
        default_shift=default_shift,
    )

    title = "HiDream-O1-Image — Demo 7 task"
    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            "# HiDream-O1-Image (SDNQ 4-bit)\n"
            "Năm tab đầu có prompt chuyên biệt viết sẵn và ảnh mẫu "
            "bấm-là-chạy: **Virtual Try-On**, **Home Design**, **Image to "
            "Cartoon**, **Ghép 2 người ôm nhau**, **Face Swap**. Hai tab "
            "cuối để bạn tự viết prompt: **Prompt to Image** và **Image + "
            "Prompt**."
        )

        for tab in tabs.ALL:
            tab.build(ctx)

    return demo


def main() -> None:
    # Gradio in link share bằng print(). Khi stdout là pipe — Colab chạy
    # `!uv run run-app`, hoặc `docker logs` — Python block-buffer 8KB nên
    # link kẹt trong buffer, có khi tới lúc tắt app mới hiện ra. Dockerfile
    # đặt PYTHONUNBUFFERED=1 cho container; dòng này phủ nốt mọi đường chạy
    # khác. reconfigure() vắng mặt nếu stdout đã bị thay bằng stream khác.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    # Buộc stdout flush ngay lập tức cho mọi print()
    sys.stdout.flush()

    settings = load_settings()
    apply_gpu_tuning()

    device = select_device(settings.device_override)
    logger.info(
        "CUDA devices available: %d | running on %s",
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
        device,
    )

    logger.info(
        "Đang nạp HiDream-O1-Image (%s, %d bước, cfg %s) ...",
        settings.model_type,
        settings.num_steps,
        settings.guidance_scale,
    )
    model, processor = load_model(settings, device)
    logger.info("Model loaded.")

    # Warm-up TRƯỚC khi mở port: healthcheck chỉ báo xanh khi service thật
    # sự sẵn sàng phục vụ nhanh, chứ không phải lúc vừa nạp xong weight.
    # Lưu ý đây là một lượt sinh ảnh THẬT ở 2048² — tốn hàng chục giây đến
    # vài phút, không phải vài giây như bản Lightning cũ.
    if settings.warmup:
        logger.info("Đang warm-up model (lượt sinh ảnh thật)...")
        logger.info("Quá trình này có thể mất 1-3 phút tùy GPU, vui lòng đợi...")
        try:
            warmup(model, processor, settings)
            logger.info("Warm-up hoàn tất.")
        except Exception as e:
            logger.warning(f"Warm-up thất bại: {e}. Tiếp tục khởi động Gradio...")
    logger.info("Sẵn sàng khởi động Gradio app với public link...")

    demo = build_ui(model, processor, settings)
    logger.info("Đang tạo public link Gradio...")
    try:
        launch_with_public_link(demo, settings)
    except Exception as e:
        logger.error(f"Không thể tạo public link: {e}")
        logger.info("Thử khởi động Gradio ở local mode...")
        demo.launch(
            server_name=settings.server_name,
            server_port=settings.server_port,
            share=False,
            show_error=True,
            quiet=False,
        )


if __name__ == "__main__":
    main()
