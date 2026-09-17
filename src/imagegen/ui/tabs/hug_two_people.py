"""Tab 4 — Ghép 2 người ôm nhau.

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
                    hug_cfg,
                    hug_seed,
                    hug_steps,
                    hug_shift,
                    hug_res,
                    hug_reset,
                ) = advanced_block(
                    prompts.build_hug_prompt(),
                    ctx.num_steps,
                    ctx.default_cfg,
                    ctx.default_shift,
                )
            with gr.Column(scale=1):
                hug_out, hug_status = output_block()

        hug_reset.click(prompts.build_hug_prompt, None, [hug_prompt])

        def on_hug(
            person_a,
            person_b,
            note,
            prompt_text,
            cfg,
            seed,
            steps,
            shift,
            res,
        ):
            started = time.time()
            if person_a is None:
                return None, "Vui lòng upload ảnh người thứ nhất (Ảnh 1)."
            if person_b is None:
                return None, "Vui lòng upload ảnh người thứ hai (Ảnh 2)."
            hit = ctx.cached(demo_cache_key([person_a, person_b]))
            if hit:
                return serve_cached(hit, started)
            return ctx.run(
                [person_a, person_b],
                prompt_text,
                note,
                cfg,
                seed,
                steps,
                shift,
                res,
            )

        def ex_hug(person_a, person_b):
            """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả."""
            return on_hug(
                person_a,
                person_b,
                "",
                hug_prompt.value,
                ctx.default_cfg,
                -1,
                ctx.num_steps,
                ctx.default_shift,
                DEFAULT_RESOLUTION_LABEL,
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
                hug_cfg,
                hug_seed,
                hug_steps,
                hug_shift,
                hug_res,
            ],
            outputs=[hug_out, hug_status],
        )
