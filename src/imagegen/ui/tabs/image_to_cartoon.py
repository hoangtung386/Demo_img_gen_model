"""Tab 3 — Image to Cartoon.

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
                    cartoon_cfg,
                    cartoon_seed,
                    cartoon_steps,
                    cartoon_shift,
                    cartoon_res,
                    cartoon_reset,
                ) = advanced_block(
                    prompts.build_cartoon_prompt(
                        next(iter(prompts.CARTOON_STYLES))
                    ),
                    ctx.num_steps,
                    ctx.default_cfg,
                    ctx.default_shift,
                )
            with gr.Column(scale=1):
                cartoon_out, cartoon_status = output_block()

        cartoon_style.change(
            prompts.build_cartoon_prompt, [cartoon_style], [cartoon_prompt]
        )
        cartoon_reset.click(
            prompts.build_cartoon_prompt, [cartoon_style], [cartoon_prompt]
        )

        def on_cartoon(photo, note, prompt_text, cfg, seed, steps, shift, res):
            started = time.time()
            if photo is None:
                return None, "Vui lòng upload ảnh cần chuyển."
            hit = ctx.cached(demo_cache_key([photo], prompt_text))
            if hit:
                return serve_cached(hit, started)
            return ctx.run(
                [photo],
                prompt_text,
                note,
                cfg,
                seed,
                steps,
                shift,
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
                ctx.default_cfg,
                -1,
                ctx.num_steps,
                ctx.default_shift,
                DEFAULT_RESOLUTION_LABEL,
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
                cartoon_cfg,
                cartoon_seed,
                cartoon_steps,
                cartoon_shift,
                cartoon_res,
            ],
            outputs=[cartoon_out, cartoon_status],
        )
