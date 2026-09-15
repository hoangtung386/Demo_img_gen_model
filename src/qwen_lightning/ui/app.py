"""Gradio user interface for the demo.

Mỗi tab là một bài toán đang thử nghiệm trên cùng một pipeline
Qwen-Image-Edit-2509 Lightning: thử đồ ảo, thiết kế lại nội thất, chuyển ảnh
chụp thành hoạt hình, ghép hai người và hoán đổi khuôn mặt. Điểm khác nhau
giữa các tab nằm ở prompt (xem ``prompts.py``) và ở thứ tự ảnh đưa vào
pipeline.

THỨ TỰ ẢNH: ảnh nền (người mẫu / phòng của bạn / ảnh đầy đủ) phải là ảnh
CUỐI. Pipeline
lấy tỉ lệ khung từ ``image[-1]`` và mô hình coi ảnh cuối là ảnh chính cần
chỉnh sửa; đặt ảnh tham chiếu ở cuối khiến kết quả bê nguyên ảnh tham chiếu.
Số thứ tự trên nhãn UI khớp với "image 1"/"image 2" trong prompt.
"""
from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from pathlib import Path

import gradio as gr
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

from ..config import load_settings
from ..device import select_device
from ..inference import DEFAULT_OUTPUT_AREA, generate
from ..logging_setup import configure_logging
from ..models.loader import load_pipeline
from ..tuning import apply_gpu_tuning, warmup
from . import examples_spec as spec
from . import prompts

logger = configure_logging()

# Ảnh mẫu để tester bấm một cái là có sẵn đầu vào VÀ kết quả dựng sẵn.
# Ảnh kết quả do scripts/warm_examples.py sinh ra trước; ở đây chỉ nạp lại nên
# tester bấm ví dụ là thấy ngay, không phải đợi model chạy. Thiếu file nào thì
# case đó bị bỏ qua — demo vẫn chạy bình thường.


def _img(name: str) -> str | None:
    """Đường dẫn ảnh mẫu, hoặc ``None`` nếu chưa tải về."""
    path = spec.image_path(name)
    return str(path) if path.is_file() else None


def _image_key(image) -> bytes:
    """Vân tay nội dung một ảnh, không phụ thuộc đường dẫn file tạm."""
    if image is None:
        return b""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    return hashlib.blake2b(
        image.convert("RGB").tobytes(), digest_size=16
    ).digest()


def _demo_key(images, *texts) -> str:
    """Khoá tra ảnh dựng sẵn: nội dung ảnh đầu vào + phần chữ đi kèm."""
    digest = hashlib.blake2b(digest_size=16)
    for image in images:
        digest.update(_image_key(image))
    for text in texts:
        digest.update(str(text or "").strip().encode("utf-8"))
    return digest.hexdigest()


def _build_demo_cache() -> dict[str, str]:
    """Lập chỉ mục {khoá đầu vào -> ảnh kết quả dựng sẵn} từ examples_spec."""
    index: dict[str, str] = {}

    def add(tab, cases, key_of):
        for position, case in enumerate(cases):
            output = spec.output_path(tab, position)
            if not output.is_file():
                continue
            try:
                index[key_of(case)] = str(output)
            except (OSError, ValueError):
                continue

    add("vto", spec.VTO_CASES, lambda c: _demo_key(
        [spec.image_path(c[0]), spec.image_path(c[1])]))
    add("home", spec.HOME_CASES, lambda c: _demo_key(
        [spec.image_path(c[0])], c[1]))
    add("cartoon", spec.CARTOON_CASES, lambda c: _demo_key(
        [spec.image_path(c[0])], prompts.build_cartoon_prompt(c[1])))
    add("hug", spec.HUG_CASES, lambda c: _demo_key(
        [spec.image_path(c[0]), spec.image_path(c[1])]))
    add("faceswap", spec.FACESWAP_CASES, lambda c: _demo_key(
        [spec.image_path(c[0]), spec.image_path(c[1])],
        prompts.build_faceswap_prompt(c[2])))
    return index


# Thời gian tối thiểu một lượt "chạy" ở chế độ demo. Không trả ảnh tức thì vì
# như vậy trông như lỗi giao diện chứ không như một lượt sinh ảnh.
DEMO_MIN_SECONDS = 0.3


def _serve_cached(path: str, started: float) -> tuple[Image.Image, str]:
    """Trả ảnh dựng sẵn, giữ nhịp tối thiểu cho giống một lượt chạy thật."""
    image = Image.open(path)
    image.load()
    remaining = DEMO_MIN_SECONDS - (time.time() - started)
    if remaining > 0:
        time.sleep(remaining)
    elapsed = time.time() - started
    logger.info("Demo cache hit: %s (%.2fs)", path, elapsed)
    # TUYỆT ĐỐI không định dạng giống một lượt chạy thật. Trước đây chỗ này in
    # "Done in 0.3s" y hệt lượt chạy thật, khiến người đo tưởng model sinh ảnh
    # trong 0.3s trong khi không có bước denoise nào chạy cả.
    return image, (
        f"{spec.CACHED_STATUS}\n"
        f"{image.width}×{image.height} · ảnh đọc từ đĩa trong {elapsed:.1f}s"
    )


def _example_rows(cases, to_row) -> list[list]:
    """Các dòng cho ``gr.Examples`` — chỉ gồm cột đầu vào."""
    rows = []
    for case in cases:
        row = to_row(case)
        if any(value is None for value in row):
            continue
        rows.append(row)
    return rows


# Chi phí denoise tỉ lệ với số token latent, tức với diện tích ảnh ra. Đây là
# đòn bẩy tốc độ thật sự duy nhất còn lại: hạ resolution ẢNH VÀO không giúp gì
# vì pipeline tự chuẩn hoá mọi ảnh điều kiện về 1024² cho VAE và 384² cho
# text encoder.
OUTPUT_PRESETS: dict[str, int] = {
    "Chuẩn — 1024px (~1.0 MP)": DEFAULT_OUTPUT_AREA,
    "Nhanh — 832px (~0.7 MP)": 832 * 832,
    "Rất nhanh — 704px (~0.5 MP)": 704 * 704,
}


def _advanced_block(
    default_prompt: str, default_negative: str, default_steps: int
) -> tuple[
    gr.Textbox, gr.Textbox, gr.Slider, gr.Number, gr.Slider, gr.Dropdown,
    gr.Button,
]:
    """Khối tham số nâng cao dùng chung cho cả ba tab."""
    with gr.Accordion("Tuỳ chỉnh nâng cao", open=False):
        prompt_box = gr.Textbox(
            label="Prompt gửi cho model",
            value=default_prompt,
            lines=10,
            info=(
                "Được sinh sẵn theo task. Có thể sửa trực tiếp; bấm "
                "'Khôi phục prompt mặc định' để lấy lại bản gốc."
            ),
        )
        reset_btn = gr.Button("Khôi phục prompt mặc định", size="sm")
        negative_box = gr.Textbox(
            label="Negative prompt",
            value=default_negative,
            lines=3,
            info="Chỉ có tác dụng khi true_cfg_scale > 1.0.",
        )
        with gr.Row():
            cfg = gr.Slider(
                label="true_cfg_scale",
                minimum=1.0,
                maximum=10.0,
                value=1.0,
                step=0.5,
                info="Lightning chạy tốt nhất ở 1.0",
            )
            seed = gr.Number(
                label="Seed (-1 = ngẫu nhiên)", value=-1, precision=0
            )
            steps = gr.Slider(
                label="Số bước",
                minimum=4,
                maximum=12,
                value=default_steps,
                step=1,
                info="Nên giữ đúng số bước mà weights Lightning được luyện",
            )
        resolution = gr.Dropdown(
            label="Độ phân giải đầu ra",
            choices=list(OUTPUT_PRESETS),
            value=next(iter(OUTPUT_PRESETS)),
            info="Hạ xuống để chạy nhanh hơn, đổi lại mất chi tiết.",
        )
    return prompt_box, negative_box, cfg, seed, steps, resolution, reset_btn


def _output_block() -> tuple[gr.Image, gr.Textbox]:
    """Khối kết quả dùng chung cho cả ba tab."""
    output_image = gr.Image(label="Kết quả", type="pil", height=560)
    status = gr.Textbox(
        label="Thời gian chạy thật (GPU)",
        info=(
            "Chỉ tính thời gian model chạy trên GPU. KHÔNG gồm upload ảnh, "
            "hàng đợi Gradio, mã hoá ảnh trả về và độ trễ tunnel "
            "*.gradio.live — đó mới là phần khiến bạn thấy lâu hơn con số này."
        ),
        lines=2,
        interactive=False,
    )
    return output_image, status


def build_ui(
    pipeline: QwenImageEditPlusPipeline,
    num_steps: int,
    demo_cache: bool = True,
) -> gr.Blocks:
    """Build and return the Gradio Blocks interface.

    ``demo_cache=True`` khiến các ví dụ mẫu trả thẳng ảnh đã dựng sẵn thay vì
    chạy model — dùng khi trình diễn. Đặt ``QIE_DEMO_CACHE=false`` để tắt và đo
    hiệu năng thật.
    """
    cache = _build_demo_cache() if demo_cache else {}
    logger.info(
        "Demo cache: %s (%d ví dụ dựng sẵn)",
        "bật" if demo_cache else "tắt",
        len(cache),
    )

    def _run(
        images, prompt_text, note, negative, cfg, seed, steps, res,
        match_input_size=False,
    ):
        """Nối ghi chú người dùng vào prompt task rồi chạy pipeline.

        ``images`` phải có ảnh nền ở CUỐI — xem docstring của module.
        """
        return generate(
            pipeline,
            images,
            prompts.append_note(prompt_text, note),
            cfg,
            seed,
            int(steps),
            negative,
            OUTPUT_PRESETS.get(res, DEFAULT_OUTPUT_AREA),
            match_input_size,
        )

    title = "Qwen-Image-Edit-2509 Lightning — Demo 5 task"
    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            "# Qwen-Image-Edit-2509 Lightning (4-bit, 4-step)\n"
            "Năm tab thử nghiệm: **Virtual Try-On**, **Home Design**, "
            "**Image to Cartoon**, **Ghép 2 người ôm nhau** và "
            "**Face Swap**. Mỗi tab có prompt chuyên biệt viết sẵn và ảnh "
            "mẫu bấm-là-chạy."
        )

        # ------------------------------------------------------------------
        # Tab 1 — Virtual Try-On
        # ------------------------------------------------------------------
        with gr.Tab("1. Virtual Try-On"):
            gr.Markdown(
                "Thay trang phục cho người mẫu bằng trang phục trong ảnh "
                "tham chiếu, **giữ nguyên khuôn mặt, dáng người, tư thế, màu "
                "da và bối cảnh** của ảnh người mẫu.\n"
                "> Ảnh người mẫu là **ảnh nền**, phải đứng sau — kết quả bám "
                "theo khung của nó."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        vto_garment = gr.Image(
                            label="Ảnh 1 — Trang phục", type="pil", height=320
                        )
                        vto_person = gr.Image(
                            label="Ảnh 2 — Người mẫu (ảnh nền)",
                            type="pil",
                            height=320,
                        )
                    vto_garment_type = gr.Dropdown(
                        label="Loại trang phục cần thay",
                        choices=list(prompts.GARMENT_TYPES),
                        value=next(iter(prompts.GARMENT_TYPES)),
                    )
                    vto_note = gr.Textbox(
                        label="Ghi chú thêm (tuỳ chọn)",
                        placeholder="ví dụ: tuck the shirt into the trousers",
                        lines=2,
                    )
                    vto_run = gr.Button("Thử đồ", variant="primary")
                    (
                        vto_prompt,
                        vto_negative,
                        vto_cfg,
                        vto_seed,
                        vto_steps,
                        vto_res,
                        vto_reset,
                    ) = _advanced_block(
                        prompts.build_vto_prompt(next(iter(prompts.GARMENT_TYPES))),
                        prompts.VTO_NEGATIVE,
                        num_steps,
                    )
                with gr.Column(scale=1):
                    vto_out, vto_status = _output_block()


            # Đổi loại trang phục thì dựng lại prompt cho khớp.
            vto_garment_type.change(
                prompts.build_vto_prompt, [vto_garment_type], [vto_prompt]
            )
            vto_reset.click(
                prompts.build_vto_prompt, [vto_garment_type], [vto_prompt]
            )

            def on_vto(
                garment, person, note, prompt_text, negative, cfg, seed,
                steps, res,
            ):
                started = time.time()
                if garment is None:
                    return None, "Vui lòng upload ảnh trang phục (Ảnh 1)."
                if person is None:
                    return None, "Vui lòng upload ảnh người mẫu (Ảnh 2)."
                hit = cache.get(_demo_key([garment, person]))
                if hit:
                    return _serve_cached(hit, started)
                # Người mẫu đứng cuối: ảnh nền quyết định khung kết quả.
                return _run(
                    [garment, person], prompt_text, note, negative, cfg,
                    seed, steps, res,
                )

            def ex_vto(garment, person):
                """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả."""
                return on_vto(
                    garment, person, "", vto_prompt.value, vto_negative.value,
                    1.0, -1, num_steps, next(iter(OUTPUT_PRESETS)),
                )

            _ex_vto = _example_rows(
                spec.VTO_CASES,
                lambda case: [_img(case[0]), _img(case[1])],
            )
            if _ex_vto:
                gr.Examples(
                    examples=_ex_vto,
                    inputs=[vto_garment, vto_person],
                    outputs=[vto_out, vto_status],
                    fn=ex_vto,
                    run_on_click=True,
                    label="Ví dụ mẫu — bấm để xem ngay kết quả",
                    examples_per_page=6,
                )

            vto_run.click(
                fn=on_vto,
                inputs=[
                    vto_garment,
                    vto_person,
                    vto_note,
                    vto_prompt,
                    vto_negative,
                    vto_cfg,
                    vto_seed,
                    vto_steps,
                    vto_res,
                ],
                outputs=[vto_out, vto_status],
            )

        # ------------------------------------------------------------------
        # Tab 2 — Home Design
        # ------------------------------------------------------------------
        with gr.Tab("2. Home Design"):
            gr.Markdown(
                "Upload **một ảnh căn phòng** rồi tự mô tả phong cách nội "
                "thất mong muốn. Kết quả là chính căn phòng đó được dọn sạch "
                "đồ cũ và bài trí lại theo mô tả, **giữ nguyên tường, cửa sổ, "
                "cửa ra vào, sàn và góc chụp**."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    home_room = gr.Image(
                        label="Ảnh căn phòng của bạn",
                        type="pil",
                        height=380,
                    )
                    home_design = gr.Textbox(
                        label="Mô tả thiết kế mong muốn",
                        placeholder=prompts.HOME_DESIGN_EXAMPLE,
                        lines=4,
                        info=(
                            "Viết bằng tiếng Anh: đồ nội thất, vật liệu, "
                            "bảng màu, ánh sáng, phụ kiện trang trí."
                        ),
                    )
                    home_room_type = gr.Dropdown(
                        label="Loại phòng",
                        choices=list(prompts.ROOM_TYPES),
                        value=next(iter(prompts.ROOM_TYPES)),
                    )
                    home_run = gr.Button("Thiết kế lại", variant="primary")
                    (
                        home_prompt,
                        home_negative,
                        home_cfg,
                        home_seed,
                        home_steps,
                        home_res,
                        home_reset,
                    ) = _advanced_block(
                        prompts.build_home_prompt(
                            next(iter(prompts.ROOM_TYPES))
                        ),
                        prompts.HOME_NEGATIVE,
                        num_steps,
                    )
                with gr.Column(scale=1):
                    home_out, home_status = _output_block()


            home_room_type.change(
                prompts.build_home_prompt, [home_room_type], [home_prompt]
            )
            home_reset.click(
                prompts.build_home_prompt, [home_room_type], [home_prompt]
            )

            def on_home(
                room, design, prompt_text, negative, cfg, seed, steps, res
            ):
                started = time.time()
                if room is None:
                    return None, "Vui lòng upload ảnh căn phòng."
                if not (design or "").strip():
                    return None, "Vui lòng mô tả thiết kế mong muốn."
                hit = cache.get(_demo_key([room], design))
                if hit:
                    return _serve_cached(hit, started)
                # Khung hệ thống + mô tả người dùng; note để rỗng vì mô tả
                # thiết kế đã là phần nội dung chính.
                return _run(
                    [room],
                    prompts.compose_home_prompt(prompt_text, design),
                    "",
                    negative,
                    cfg,
                    seed,
                    steps,
                    res,
                )

            def ex_home(room, design):
                """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả."""
                return on_home(
                    room, design, home_prompt.value, home_negative.value,
                    1.0, -1, num_steps, next(iter(OUTPUT_PRESETS)),
                )

            _ex_home = _example_rows(
                spec.HOME_CASES,
                lambda case: [_img(case[0]), case[1]],
            )
            if _ex_home:
                gr.Examples(
                    examples=_ex_home,
                    inputs=[home_room, home_design],
                    outputs=[home_out, home_status],
                    fn=ex_home,
                    run_on_click=True,
                    label="Ví dụ mẫu — bấm để xem ngay kết quả",
                    examples_per_page=6,
                )

            home_run.click(
                fn=on_home,
                inputs=[
                    home_room,
                    home_design,
                    home_prompt,
                    home_negative,
                    home_cfg,
                    home_seed,
                    home_steps,
                    home_res,
                ],
                outputs=[home_out, home_status],
            )

        # ------------------------------------------------------------------
        # Tab 3 — Image to Cartoon
        # ------------------------------------------------------------------
        with gr.Tab("3. Image to Cartoon"):
            gr.Markdown(
                "Chuyển tất cả người trong ảnh thành nhân vật hoạt hình **giữ "
                "tối đa nét mặt, kiểu tóc, vóc dáng, trang phục và màu sắc**, "
                "phong cảnh xung quanh được vẽ lại cùng phong cách."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    cartoon_photo = gr.Image(
                        label="Ảnh chụp (có người và phong cảnh)",
                        type="pil",
                        height=380,
                    )
                    cartoon_style = gr.Dropdown(
                        label="Phong cách hoạt hình",
                        choices=list(prompts.CARTOON_STYLES),
                        value=next(iter(prompts.CARTOON_STYLES)),
                    )
                    cartoon_note = gr.Textbox(
                        label="Ghi chú thêm (tuỳ chọn)",
                        placeholder="ví dụ: warm sunset lighting",
                        lines=2,
                    )
                    cartoon_run = gr.Button(
                        "Chuyển sang hoạt hình", variant="primary"
                    )
                    (
                        cartoon_prompt,
                        cartoon_negative,
                        cartoon_cfg,
                        cartoon_seed,
                        cartoon_steps,
                        cartoon_res,
                        cartoon_reset,
                    ) = _advanced_block(
                        prompts.build_cartoon_prompt(
                            next(iter(prompts.CARTOON_STYLES))
                        ),
                        prompts.CARTOON_NEGATIVE,
                        num_steps,
                    )
                with gr.Column(scale=1):
                    cartoon_out, cartoon_status = _output_block()

            cartoon_style.change(
                prompts.build_cartoon_prompt, [cartoon_style], [cartoon_prompt]
            )
            cartoon_reset.click(
                prompts.build_cartoon_prompt, [cartoon_style], [cartoon_prompt]
            )

            def on_cartoon(
                photo, note, prompt_text, negative, cfg, seed, steps, res
            ):
                started = time.time()
                if photo is None:
                    return None, "Vui lòng upload ảnh cần chuyển."
                hit = cache.get(_demo_key([photo], prompt_text))
                if hit:
                    return _serve_cached(hit, started)
                return _run(
                    [photo], prompt_text, note, negative, cfg, seed, steps,
                    res,
                )

            def ex_cartoon(photo, style):
                """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả.

                Prompt dựng lại từ phong cách chứ không đưa vào bảng, để bảng
                ví dụ khỏi có một cột prompt dài loà xoà.
                """
                return on_cartoon(
                    photo, "", prompts.build_cartoon_prompt(style),
                    cartoon_negative.value, 1.0, -1, num_steps,
                    next(iter(OUTPUT_PRESETS)),
                )

            # Phong cách nằm trong dòng ví dụ để prompt hiển thị khớp với ảnh
            # dựng sẵn, thay vì giữ nguyên phong cách mặc định.
            _ex_cartoon = _example_rows(
                spec.CARTOON_CASES,
                lambda case: [_img(case[0]), case[1]],
            )
            if _ex_cartoon:
                gr.Examples(
                    examples=_ex_cartoon,
                    inputs=[cartoon_photo, cartoon_style],
                    outputs=[cartoon_out, cartoon_status],
                    fn=ex_cartoon,
                    run_on_click=True,
                    label="Ví dụ mẫu — bấm để xem ngay kết quả",
                    examples_per_page=6,
                )

            cartoon_run.click(
                fn=on_cartoon,
                inputs=[
                    cartoon_photo,
                    cartoon_note,
                    cartoon_prompt,
                    cartoon_negative,
                    cartoon_cfg,
                    cartoon_seed,
                    cartoon_steps,
                    cartoon_res,
                ],
                outputs=[cartoon_out, cartoon_status],
            )


        # ------------------------------------------------------------------
        # Tab 4 — Hai người ôm nhau
        # ------------------------------------------------------------------
        with gr.Tab("4. Ghép 2 người ôm nhau"):
            gr.Markdown(
                "Upload **ảnh hai người khác nhau**, model ghép thành một ảnh "
                "duy nhất trong đó hai người đang ôm nhau, **giữ khuôn mặt, "
                "kiểu tóc, vóc dáng và trang phục** của từng người.\n"
                "> Người ở Ảnh 2 là **ảnh nền** — kết quả bám khung ảnh đó."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        hug_a = gr.Image(
                            label="Ảnh 1 — Người thứ nhất",
                            type="pil",
                            height=320,
                        )
                        hug_b = gr.Image(
                            label="Ảnh 2 — Người thứ hai (ảnh nền)",
                            type="pil",
                            height=320,
                        )
                    hug_note = gr.Textbox(
                        label="Ghi chú thêm (tuỳ chọn)",
                        placeholder="ví dụ: outdoors in a park, warm sunlight",
                        lines=2,
                    )
                    hug_run = gr.Button("Ghép ảnh ôm nhau", variant="primary")
                    (
                        hug_prompt,
                        hug_negative,
                        hug_cfg,
                        hug_seed,
                        hug_steps,
                        hug_res,
                        hug_reset,
                    ) = _advanced_block(
                        prompts.build_hug_prompt(),
                        prompts.HUG_NEGATIVE,
                        num_steps,
                    )
                with gr.Column(scale=1):
                    hug_out, hug_status = _output_block()

            hug_reset.click(prompts.build_hug_prompt, None, [hug_prompt])

            def on_hug(
                person_a, person_b, note, prompt_text, negative, cfg, seed,
                steps, res,
            ):
                started = time.time()
                if person_a is None:
                    return None, "Vui lòng upload ảnh người thứ nhất (Ảnh 1)."
                if person_b is None:
                    return None, "Vui lòng upload ảnh người thứ hai (Ảnh 2)."
                hit = cache.get(_demo_key([person_a, person_b]))
                if hit:
                    return _serve_cached(hit, started)
                return _run(
                    [person_a, person_b], prompt_text, note, negative, cfg,
                    seed, steps, res,
                )

            def ex_hug(person_a, person_b):
                """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả."""
                return on_hug(
                    person_a, person_b, "", hug_prompt.value,
                    hug_negative.value, 1.0, -1, num_steps,
                    next(iter(OUTPUT_PRESETS)),
                )

            _ex_hug = _example_rows(
                spec.HUG_CASES,
                lambda case: [_img(case[0]), _img(case[1])],
            )
            if _ex_hug:
                gr.Examples(
                    examples=_ex_hug,
                    inputs=[hug_a, hug_b],
                    outputs=[hug_out, hug_status],
                    fn=ex_hug,
                    run_on_click=True,
                    label="Ví dụ mẫu — bấm để xem ngay kết quả",
                    examples_per_page=6,
                )

            hug_run.click(
                fn=on_hug,
                inputs=[
                    hug_a,
                    hug_b,
                    hug_note,
                    hug_prompt,
                    hug_negative,
                    hug_cfg,
                    hug_seed,
                    hug_steps,
                    hug_res,
                ],
                outputs=[hug_out, hug_status],
            )

        # ------------------------------------------------------------------
        # Tab 5 — Face Swap
        # ------------------------------------------------------------------
        with gr.Tab("5. Face Swap"):
            gr.Markdown(
                "Upload **ảnh mặt đã crop** (Ảnh 1) và **ảnh đầy đủ** có "
                "người cùng phong cảnh (Ảnh 2). Model thay khuôn mặt trong "
                "Ảnh 2 bằng khuôn mặt của Ảnh 1, **giữ nguyên phong cảnh, "
                "quần áo, dáng người và khung hình**.\n"
                "> Ảnh 2 là **ảnh nền** — ảnh kết quả trả về **đúng kích "
                "thước pixel của Ảnh 2**. Model vẫn sinh ở độ phân giải chọn "
                "trong *Tuỳ chỉnh nâng cao* rồi scale về kích thước gốc, nên "
                "ảnh vào rất lớn sẽ không vì thế mà có thêm chi tiết thật.\n"
                "> Hoạt động tốt nhất khi Ảnh 2 có **một khuôn mặt rõ**, và "
                "Ảnh 1 crop sát mặt, nhìn thẳng, không bị mờ."
            )
            _fs_scope0 = next(iter(prompts.FACE_SWAP_SCOPES))
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        fs_face = gr.Image(
                            label="Ảnh 1 — Mặt đã crop",
                            type="pil",
                            height=320,
                        )
                        fs_photo = gr.Image(
                            label="Ảnh 2 — Ảnh đầy đủ (ảnh nền)",
                            type="pil",
                            height=320,
                        )
                    fs_scope = gr.Dropdown(
                        label="Phạm vi hoán đổi",
                        choices=list(prompts.FACE_SWAP_SCOPES),
                        value=_fs_scope0,
                        info=(
                            "Mặc định chỉ đổi vùng mặt và giữ tóc của Ảnh 2 "
                            "— đổi cả tóc dễ lộ đường ghép ở chân tóc."
                        ),
                    )
                    fs_note = gr.Textbox(
                        label="Ghi chú thêm (tuỳ chọn)",
                        placeholder="ví dụ: keep the glasses from image 2",
                        lines=2,
                    )
                    fs_run = gr.Button("Hoán đổi khuôn mặt", variant="primary")
                    (
                        fs_prompt,
                        fs_negative,
                        fs_cfg,
                        fs_seed,
                        fs_steps,
                        fs_res,
                        fs_reset,
                    ) = _advanced_block(
                        prompts.build_faceswap_prompt(_fs_scope0),
                        prompts.FACESWAP_NEGATIVE,
                        num_steps,
                    )
                with gr.Column(scale=1):
                    fs_out, fs_status = _output_block()

            fs_scope.change(
                prompts.build_faceswap_prompt, [fs_scope], [fs_prompt]
            )
            fs_reset.click(
                prompts.build_faceswap_prompt, [fs_scope], [fs_prompt]
            )

            def on_faceswap(
                face, photo, note, prompt_text, negative, cfg, seed, steps,
                res,
            ):
                started = time.time()
                if face is None:
                    return None, "Vui lòng upload ảnh mặt đã crop (Ảnh 1)."
                if photo is None:
                    return None, "Vui lòng upload ảnh đầy đủ (Ảnh 2)."
                hit = cache.get(_demo_key([face, photo], prompt_text))
                if hit:
                    return _serve_cached(hit, started)
                # match_input_size=True: tab này hứa ảnh ra thay thế được ảnh
                # vào, nên kích thước phải khớp đúng pixel.
                return _run(
                    [face, photo], prompt_text, note, negative, cfg, seed,
                    steps, res, True,
                )

            def ex_faceswap(face, photo, scope):
                """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả."""
                return on_faceswap(
                    face, photo, "", prompts.build_faceswap_prompt(scope),
                    fs_negative.value, 1.0, -1, num_steps,
                    next(iter(OUTPUT_PRESETS)),
                )

            _ex_fs = _example_rows(
                spec.FACESWAP_CASES,
                lambda case: [_img(case[0]), _img(case[1]), case[2]],
            )
            if _ex_fs:
                gr.Examples(
                    examples=_ex_fs,
                    inputs=[fs_face, fs_photo, fs_scope],
                    outputs=[fs_out, fs_status],
                    fn=ex_faceswap,
                    run_on_click=True,
                    label="Ví dụ mẫu — bấm để xem ngay kết quả",
                    examples_per_page=6,
                )

            fs_run.click(
                fn=on_faceswap,
                inputs=[
                    fs_face,
                    fs_photo,
                    fs_note,
                    fs_prompt,
                    fs_negative,
                    fs_cfg,
                    fs_seed,
                    fs_steps,
                    fs_res,
                ],
                outputs=[fs_out, fs_status],
            )

    return demo


def main() -> None:
    # Gradio in link share bằng print(). Khi stdout là pipe — Colab chạy
    # `!uv run run-app`, hoặc `docker logs` — Python block-buffer 8KB nên
    # link kẹt trong buffer, có khi tới lúc tắt app mới hiện ra. Dockerfile
    # đặt PYTHONUNBUFFERED=1 cho container; dòng này phủ nốt mọi đường chạy
    # khác. reconfigure() vắng mặt nếu stdout đã bị thay bằng stream khác.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    settings = load_settings()
    apply_gpu_tuning()

    device = select_device(settings.device_override)
    logger.info(
        "CUDA devices available: %d | running on %s",
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
        device,
    )

    logger.info(
        "Loading Qwen-Image-Edit-2509 Lightning (%d-step, rank=%d) ...",
        settings.num_steps,
        settings.rank,
    )
    pipeline = load_pipeline(settings, device)

    # Thanh tiến trình của diffusers ghi ra stderr mỗi bước. Vô dụng khi chạy
    # dưới dạng service và chỉ làm nhiễu log.
    pipeline.set_progress_bar_config(disable=True)
    logger.info("Model loaded.")

    # Warm-up TRƯỚC khi mở port: healthcheck chỉ báo xanh khi service thật sự
    # sẵn sàng phục vụ nhanh, chứ không phải lúc vừa nạp xong weight.
    if settings.warmup:
        warmup(pipeline, settings.num_steps, DEFAULT_OUTPUT_AREA)
    logger.info("Ready.")

    demo = build_ui(pipeline, settings.num_steps, settings.demo_cache)
    _launch_with_public_link(demo, settings)


# Link công khai là BẮT BUỘC, không phải tuỳ chọn — không còn biến QIE_SHARE.
# Đường chạy chính của repo là Colab / máy thuê, nơi cổng 7860 không tiếp cận
# được từ ngoài: service chạy mà không có link thì coi như không chạy. Vì vậy
# thiếu link được coi là lỗi khởi động, không phải cảnh báo.
SHARE_ATTEMPTS = 3
SHARE_RETRY_WAIT = 5.0
FRPC_ATTEMPTS = 3
FRPC_RETRY_WAIT = 3.0


def _ensure_frpc_binary() -> None:
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
        logger.info("Binary tunnel frpc đã có: %s", path)
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
        logger.info("Tải binary tunnel frpc (lần %d/%d): %s",
                    attempt, FRPC_ATTEMPTS, url)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                blob = resp.read()
            digest = hashlib.sha256(blob).hexdigest()
            if digest != expected:
                raise RuntimeError(
                    f"sha256 lệch: {digest} != {expected}"
                )
        except Exception as exc:  # noqa: BLE001 - báo lại sau vòng lặp
            last_error = exc
            logger.error("Tải frpc hỏng: %s", exc)
            if attempt < FRPC_ATTEMPTS:
                time.sleep(FRPC_RETRY_WAIT)
            continue

        path.write_bytes(blob)
        path.chmod(0o755)
        logger.info("frpc sẵn sàng: %s", path)
        return

    raise RuntimeError(
        f"Không tải được binary tunnel frpc sau {FRPC_ATTEMPTS} lần thử, "
        "nên không thể tạo link công khai. Tải tay rồi đặt vào "
        f"{path}:\n    curl -L {url} -o {path}\n    chmod +x {path}"
    ) from last_error


def _launch_with_public_link(demo: gr.Blocks, settings) -> None:
    """Mở demo kèm link *.gradio.live, thử lại rồi nổ nếu không dựng được.

    ``prevent_thread_lock=True`` để lấy được share_url trả về mà kiểm tra —
    ``launch()`` mặc định block nên mọi dòng sau nó không bao giờ chạy. Gradio
    đặt share_url=None khi tunnel hỏng (không tải được frpc, bị chặn mạng,
    relay quá tải) và vẫn phục vụ tiếp ở local, đúng kiểu hỏng âm thầm cần
    chặn ở đây. Giữ thread sống lại bằng block_thread() ở cuối.
    """
    _ensure_frpc_binary()

    last_error: Exception | None = None
    for attempt in range(1, SHARE_ATTEMPTS + 1):
        try:
            _, local_url, share_url = demo.launch(
                server_name=settings.server_name,
                server_port=settings.server_port,
                share=True,
                prevent_thread_lock=True,
            )
        except Exception as exc:  # noqa: BLE001 - báo lại ở cuối vòng lặp
            last_error = exc
            share_url = None
            local_url = None
            logger.error("launch() hỏng (lần %d): %s", attempt, exc)

        if share_url:
            logger.info("Public URL: %s (sống 1 tuần)", share_url)
            logger.info("Local URL : %s", local_url)
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
            time.sleep(SHARE_RETRY_WAIT)

    raise RuntimeError(
        f"Không tạo được link công khai sau {SHARE_ATTEMPTS} lần thử — "
        "dừng hẳn thay vì phục vụ ở một cổng không ai với tới được. "
        "Kiểm tra kết nối ra Internet của máy và xem "
        "https://api.gradio.app còn sống không."
    ) from last_error
