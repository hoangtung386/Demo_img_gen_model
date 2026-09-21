"""Giao diện Gradio — bảy tab trên cùng một pipeline.

Mỗi tab là một bài toán đang thử nghiệm trên cùng một pipeline
FLUX.2-klein-9B: thử đồ ảo, thiết kế lại nội thất, chuyển ảnh chụp thành
hoạt hình, ghép hai người và hoán đổi khuôn mặt. Điểm khác nhau giữa các tab
nằm ở prompt (xem ``prompts.py``) và ở thứ tự ảnh đưa vào pipeline.

THỨ TỰ ẢNH: ảnh nền (người mẫu / phòng của bạn / ảnh đầy đủ) phải là ảnh
CUỐI. Pipeline lấy tỉ lệ khung từ ``image[-1]`` và mô hình coi ảnh cuối là
ảnh chính cần chỉnh sửa; đặt ảnh tham chiếu ở cuối khiến kết quả bê nguyên
ảnh tham chiếu. Số thứ tự trên nhãn UI khớp với "image 1"/"image 2" trong
prompt.

Chỉ dựng widget — KHÔNG nạp model. ``build_ui(None, ...)`` chạy được, và
``tests`` dựa vào điều đó để kiểm giao diện mà không cần GPU. Phần nạp model
và mở cổng nằm ở ``launch.py``.
"""

from __future__ import annotations

import logging
import time

import gradio as gr

from ..inference import DEFAULT_OUTPUT_AREA, DEFAULT_REFERENCE_AREA, generate
from . import examples_spec as spec
from . import prompts
from .demo_cache import build_index, cache_key, sample_image, serve_cached
from .widgets import (
    ASPECT_RATIOS,
    OUTPUT_PRESETS,
    advanced_block,
    example_rows,
    output_block,
    params_accordion,
)

logger = logging.getLogger("gen-image")


def build_ui(
    pipeline,
    num_steps: int,
    demo_cache: bool = True,
    reference_area: int = DEFAULT_REFERENCE_AREA,
) -> gr.Blocks:
    """Build and return the Gradio Blocks interface.

    ``demo_cache=True`` khiến các ví dụ mẫu trả thẳng ảnh đã dựng sẵn thay vì
    chạy model — dùng khi trình diễn. Đặt ``GENIMG_DEMO_CACHE=false`` để
    tắt và đo hiệu năng thật.
    """
    cache = build_index() if demo_cache else {}
    logger.info(
        "Demo cache: %s (%d ví dụ dựng sẵn)",
        "bật" if demo_cache else "tắt",
        len(cache),
    )

    def _run(
        images,
        prompt_text,
        note,
        negative,
        cfg,
        seed,
        steps,
        res,
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
            match_input_size=match_input_size,
            reference_area=reference_area,
        )

    def _run_t2i(prompt_text, negative, cfg, seed, steps, res, ratio):
        """Sinh ảnh từ chữ — CÙNG pipeline, chỉ khác là không có ảnh vào.

        Flux2KleinPipeline nhận ``image=None`` nên không cần pipeline thứ hai
        như backend Qwen trước đây.
        """
        return generate(
            pipeline,
            [],
            prompt_text,
            cfg,
            seed,
            int(steps),
            negative,
            OUTPUT_PRESETS.get(res, DEFAULT_OUTPUT_AREA),
            aspect_ratio=ASPECT_RATIOS.get(ratio, 1.0),
        )

    title = "FLUX.2-klein-9B GGUF — Demo 7 task"
    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            "# FLUX.2-klein-9B GGUF (step-distilled, 4 bước)\n"
            "Năm tab đầu có prompt chuyên biệt viết sẵn và ảnh mẫu "
            "bấm-là-chạy: **Virtual Try-On**, **Home Design**, **Image to "
            "Cartoon**, **Ghép 2 người ôm nhau**, **Face Swap**. Hai tab "
            "cuối để bạn tự viết prompt: **Prompt to Image** và **Image + "
            "Prompt**."
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
                    ) = advanced_block(
                        prompts.build_vto_prompt(next(iter(prompts.GARMENT_TYPES))),
                        prompts.VTO_NEGATIVE,
                        num_steps,
                    )
                with gr.Column(scale=1):
                    vto_out, vto_status = output_block()

            # Đổi loại trang phục thì dựng lại prompt cho khớp.
            vto_garment_type.change(
                prompts.build_vto_prompt, [vto_garment_type], [vto_prompt]
            )
            vto_reset.click(prompts.build_vto_prompt, [vto_garment_type], [vto_prompt])

            def on_vto(
                garment,
                person,
                note,
                prompt_text,
                negative,
                cfg,
                seed,
                steps,
                res,
            ):
                started = time.time()
                if garment is None:
                    return None, "Vui lòng upload ảnh trang phục (Ảnh 1)."
                if person is None:
                    return None, "Vui lòng upload ảnh người mẫu (Ảnh 2)."
                hit = cache.get(cache_key([garment, person]))
                if hit:
                    return serve_cached(hit, started)
                # Người mẫu đứng cuối: ảnh nền quyết định khung kết quả.
                return _run(
                    [garment, person],
                    prompt_text,
                    note,
                    negative,
                    cfg,
                    seed,
                    steps,
                    res,
                )

            def ex_vto(garment, person):
                """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả."""
                return on_vto(
                    garment,
                    person,
                    "",
                    vto_prompt.value,
                    vto_negative.value,
                    1.0,
                    -1,
                    num_steps,
                    next(iter(OUTPUT_PRESETS)),
                )

            _ex_vto = example_rows(
                spec.VTO_CASES,
                lambda case: [sample_image(case[0]), sample_image(case[1])],
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
                    ) = advanced_block(
                        prompts.build_home_prompt(next(iter(prompts.ROOM_TYPES))),
                        prompts.HOME_NEGATIVE,
                        num_steps,
                    )
                with gr.Column(scale=1):
                    home_out, home_status = output_block()

            home_room_type.change(
                prompts.build_home_prompt, [home_room_type], [home_prompt]
            )
            home_reset.click(prompts.build_home_prompt, [home_room_type], [home_prompt])

            def on_home(room, design, prompt_text, negative, cfg, seed, steps, res):
                started = time.time()
                if room is None:
                    return None, "Vui lòng upload ảnh căn phòng."
                if not (design or "").strip():
                    return None, "Vui lòng mô tả thiết kế mong muốn."
                hit = cache.get(cache_key([room], design))
                if hit:
                    return serve_cached(hit, started)
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
                    room,
                    design,
                    home_prompt.value,
                    home_negative.value,
                    1.0,
                    -1,
                    num_steps,
                    next(iter(OUTPUT_PRESETS)),
                )

            _ex_home = example_rows(
                spec.HOME_CASES,
                lambda case: [sample_image(case[0]), case[1]],
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
                    cartoon_run = gr.Button("Chuyển sang hoạt hình", variant="primary")
                    (
                        cartoon_prompt,
                        cartoon_negative,
                        cartoon_cfg,
                        cartoon_seed,
                        cartoon_steps,
                        cartoon_res,
                        cartoon_reset,
                    ) = advanced_block(
                        prompts.build_cartoon_prompt(
                            next(iter(prompts.CARTOON_STYLES))
                        ),
                        prompts.CARTOON_NEGATIVE,
                        num_steps,
                    )
                with gr.Column(scale=1):
                    cartoon_out, cartoon_status = output_block()

            cartoon_style.change(
                prompts.build_cartoon_prompt, [cartoon_style], [cartoon_prompt]
            )
            cartoon_reset.click(
                prompts.build_cartoon_prompt, [cartoon_style], [cartoon_prompt]
            )

            def on_cartoon(photo, note, prompt_text, negative, cfg, seed, steps, res):
                started = time.time()
                if photo is None:
                    return None, "Vui lòng upload ảnh cần chuyển."
                hit = cache.get(cache_key([photo], prompt_text))
                if hit:
                    return serve_cached(hit, started)
                return _run(
                    [photo],
                    prompt_text,
                    note,
                    negative,
                    cfg,
                    seed,
                    steps,
                    res,
                )

            def ex_cartoon(photo, style):
                """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả.

                Prompt dựng lại từ phong cách chứ không đưa vào bảng, để bảng
                ví dụ khỏi có một cột prompt dài loà xoà.
                """
                return on_cartoon(
                    photo,
                    "",
                    prompts.build_cartoon_prompt(style),
                    cartoon_negative.value,
                    1.0,
                    -1,
                    num_steps,
                    next(iter(OUTPUT_PRESETS)),
                )

            # Phong cách nằm trong dòng ví dụ để prompt hiển thị khớp với ảnh
            # dựng sẵn, thay vì giữ nguyên phong cách mặc định.
            _ex_cartoon = example_rows(
                spec.CARTOON_CASES,
                lambda case: [sample_image(case[0]), case[1]],
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
                    ) = advanced_block(
                        prompts.build_hug_prompt(),
                        prompts.HUG_NEGATIVE,
                        num_steps,
                    )
                with gr.Column(scale=1):
                    hug_out, hug_status = output_block()

            hug_reset.click(prompts.build_hug_prompt, None, [hug_prompt])

            def on_hug(
                person_a,
                person_b,
                note,
                prompt_text,
                negative,
                cfg,
                seed,
                steps,
                res,
            ):
                started = time.time()
                if person_a is None:
                    return None, "Vui lòng upload ảnh người thứ nhất (Ảnh 1)."
                if person_b is None:
                    return None, "Vui lòng upload ảnh người thứ hai (Ảnh 2)."
                hit = cache.get(cache_key([person_a, person_b]))
                if hit:
                    return serve_cached(hit, started)
                return _run(
                    [person_a, person_b],
                    prompt_text,
                    note,
                    negative,
                    cfg,
                    seed,
                    steps,
                    res,
                )

            def ex_hug(person_a, person_b):
                """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả."""
                return on_hug(
                    person_a,
                    person_b,
                    "",
                    hug_prompt.value,
                    hug_negative.value,
                    1.0,
                    -1,
                    num_steps,
                    next(iter(OUTPUT_PRESETS)),
                )

            _ex_hug = example_rows(
                spec.HUG_CASES,
                lambda case: [sample_image(case[0]), sample_image(case[1])],
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
                    ) = advanced_block(
                        prompts.build_faceswap_prompt(_fs_scope0),
                        prompts.FACESWAP_NEGATIVE,
                        num_steps,
                    )
                with gr.Column(scale=1):
                    fs_out, fs_status = output_block()

            fs_scope.change(prompts.build_faceswap_prompt, [fs_scope], [fs_prompt])
            fs_reset.click(prompts.build_faceswap_prompt, [fs_scope], [fs_prompt])

            def on_faceswap(
                face,
                photo,
                note,
                prompt_text,
                negative,
                cfg,
                seed,
                steps,
                res,
            ):
                started = time.time()
                if face is None:
                    return None, "Vui lòng upload ảnh mặt đã crop (Ảnh 1)."
                if photo is None:
                    return None, "Vui lòng upload ảnh đầy đủ (Ảnh 2)."
                hit = cache.get(cache_key([face, photo], prompt_text))
                if hit:
                    return serve_cached(hit, started)
                # match_input_size=True: tab này hứa ảnh ra thay thế được ảnh
                # vào, nên kích thước phải khớp đúng pixel.
                return _run(
                    [face, photo],
                    prompt_text,
                    note,
                    negative,
                    cfg,
                    seed,
                    steps,
                    res,
                    True,
                )

            def ex_faceswap(face, photo, scope):
                """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả."""
                return on_faceswap(
                    face,
                    photo,
                    "",
                    prompts.build_faceswap_prompt(scope),
                    fs_negative.value,
                    1.0,
                    -1,
                    num_steps,
                    next(iter(OUTPUT_PRESETS)),
                )

            _ex_fs = example_rows(
                spec.FACESWAP_CASES,
                lambda case: [
                    sample_image(case[0]),
                    sample_image(case[1]),
                    case[2],
                ],
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

        # ------------------------------------------------------------------
        # Tab 6 — Prompt to Image
        # ------------------------------------------------------------------
        with gr.Tab("6. Prompt to Image"):
            gr.Markdown(
                "Không cần ảnh vào — chỉ mô tả bằng chữ, model sinh ảnh "
                "mới. Chọn tỉ lệ khung vì ở đây không có ảnh nền để bám "
                "theo.\n"
                "> FLUX.2-klein hợp nhất sinh ảnh và sửa ảnh trong một model, "
                "nên tab này chạy đúng thứ nó được luyện — khác backend "
                "trước đây vốn là một model chuyên *sửa* ảnh. Muốn sửa một "
                "tấm ảnh có sẵn thì dùng tab 7."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    t2i_prompt = gr.Textbox(
                        label="Prompt — mô tả ảnh bạn muốn",
                        placeholder=(
                            "Tiếng Anh cho kết quả bám sát hơn. Tả càng cụ "
                            "thể càng tốt: chủ thể, bối cảnh, ánh sáng, góc "
                            "máy, phong cách."
                        ),
                        lines=6,
                    )
                    t2i_ratio = gr.Dropdown(
                        label="Tỉ lệ khung",
                        choices=list(ASPECT_RATIOS),
                        value=next(iter(ASPECT_RATIOS)),
                    )
                    t2i_run = gr.Button("Sinh ảnh", variant="primary")
                    (
                        t2i_negative,
                        t2i_cfg,
                        t2i_seed,
                        t2i_steps,
                        t2i_res,
                    ) = params_accordion(prompts.FREE_NEGATIVE, num_steps)
                    # Chỉ điền vào ô prompt, KHÔNG chạy model: tab này không
                    # có ảnh dựng sẵn nên chạy luôn sẽ mất 6-12s mỗi lần bấm.
                    gr.Examples(
                        examples=[[x] for x in prompts.T2I_EXAMPLES],
                        inputs=[t2i_prompt],
                        label="Prompt mẫu — bấm để điền vào ô trên",
                        examples_per_page=6,
                    )
                with gr.Column(scale=1):
                    t2i_out, t2i_status = output_block()

            t2i_run.click(
                fn=_run_t2i,
                inputs=[
                    t2i_prompt,
                    t2i_negative,
                    t2i_cfg,
                    t2i_seed,
                    t2i_steps,
                    t2i_res,
                    t2i_ratio,
                ],
                outputs=[t2i_out, t2i_status],
            )

        # ------------------------------------------------------------------
        # Tab 7 — Image + Prompt to Image
        # ------------------------------------------------------------------
        with gr.Tab("7. Image + Prompt"):
            gr.Markdown(
                "Upload một ảnh và tự viết yêu cầu sửa. Đây là tab tổng "
                "quát — sáu tab trên chỉ là những prompt chuyên biệt viết "
                "sẵn cho cùng pipeline này.\n"
                "> Ảnh ra bám tỉ lệ khung của ảnh vào. Nói rõ cái gì PHẢI "
                "giữ nguyên, không chỉ cái cần đổi — đó là khác biệt lớn "
                "nhất giữa một prompt sửa ảnh tốt và một prompt tệ."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    edit_image = gr.Image(
                        label="Ảnh cần sửa",
                        type="pil",
                        height=380,
                    )
                    edit_prompt = gr.Textbox(
                        label="Yêu cầu sửa",
                        placeholder=(
                            "ví dụ: Change the season to winter, cover the "
                            "ground with snow. Keep the building, the camera "
                            "angle and the people unchanged."
                        ),
                        lines=5,
                    )
                    edit_run = gr.Button("Sửa ảnh", variant="primary")
                    (
                        edit_negative,
                        edit_cfg,
                        edit_seed,
                        edit_steps,
                        edit_res,
                    ) = params_accordion(prompts.FREE_NEGATIVE, num_steps)
                    gr.Examples(
                        examples=[[x] for x in prompts.EDIT_EXAMPLES],
                        inputs=[edit_prompt],
                        label="Yêu cầu mẫu — bấm để điền vào ô trên",
                        examples_per_page=6,
                    )
                with gr.Column(scale=1):
                    edit_out, edit_status = output_block()

            def on_edit(image, prompt_text, negative, cfg, seed, steps, res):
                if image is None:
                    return None, "Vui lòng upload ảnh cần sửa."
                if not prompt_text or not prompt_text.strip():
                    return None, "Vui lòng nhập yêu cầu sửa ảnh."
                # note="" vì ô prompt ở đây CHÍNH LÀ prompt của người dùng,
                # không có prompt hệ thống nào để nối thêm vào.
                return _run(
                    [image],
                    prompt_text,
                    "",
                    negative,
                    cfg,
                    seed,
                    steps,
                    res,
                )

            edit_run.click(
                fn=on_edit,
                inputs=[
                    edit_image,
                    edit_prompt,
                    edit_negative,
                    edit_cfg,
                    edit_seed,
                    edit_steps,
                    edit_res,
                ],
                outputs=[edit_out, edit_status],
            )

    return demo
