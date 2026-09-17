"""Tab 5 — Face Swap.

Dựng bởi ``ui/app.py``; mọi thứ tab này cần đến từ :class:`TabContext`.
"""

from __future__ import annotations

import time

import gradio as gr

from .. import examples_spec as spec
from .. import prompts
from ..components import (
    DEFAULT_RESOLUTION_LABEL,
    advanced_block,
    demo_cache_key,
    example_rows,
    output_block,
    sample_image,
    serve_cached,
)
from ..context import TabContext


def build(ctx: TabContext) -> None:
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
                    fs_cfg,
                    fs_seed,
                    fs_steps,
                    fs_shift,
                    fs_res,
                    fs_reset,
                ) = advanced_block(
                    prompts.build_faceswap_prompt(_fs_scope0),
                    ctx.num_steps,
                    ctx.default_cfg,
                    ctx.default_shift,
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
            cfg,
            seed,
            steps,
            shift,
            res,
        ):
            started = time.time()
            if face is None:
                return None, "Vui lòng upload ảnh mặt đã crop (Ảnh 1)."
            if photo is None:
                return None, "Vui lòng upload ảnh đầy đủ (Ảnh 2)."
            hit = ctx.cached(demo_cache_key([face, photo], prompt_text))
            if hit:
                return serve_cached(hit, started)
            # match_input_size=True: tab này hứa ảnh ra thay thế được ảnh
            # vào, nên kích thước phải khớp đúng pixel.
            return ctx.run(
                [face, photo],
                prompt_text,
                note,
                cfg,
                seed,
                steps,
                shift,
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
                ctx.default_cfg,
                -1,
                ctx.num_steps,
                ctx.default_shift,
                DEFAULT_RESOLUTION_LABEL,
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
                fs_cfg,
                fs_seed,
                fs_steps,
                fs_shift,
                fs_res,
            ],
            outputs=[fs_out, fs_status],
        )
