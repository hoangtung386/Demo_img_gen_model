"""Tab 1 — Virtual Try-On.

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
                    vto_cfg,
                    vto_seed,
                    vto_steps,
                    vto_shift,
                    vto_res,
                    vto_reset,
                ) = advanced_block(
                    prompts.build_vto_prompt(
                        next(iter(prompts.GARMENT_TYPES))
                    ),
                    ctx.num_steps,
                    ctx.default_cfg,
                    ctx.default_shift,
                )
            with gr.Column(scale=1):
                vto_out, vto_status = output_block()

        # Đổi loại trang phục thì dựng lại prompt cho khớp.
        vto_garment_type.change(
            prompts.build_vto_prompt, [vto_garment_type], [vto_prompt]
        )
        vto_reset.click(
            prompts.build_vto_prompt, [vto_garment_type], [vto_prompt]
        )

        def on_vto(
            garment,
            person,
            note,
            prompt_text,
            cfg,
            seed,
            steps,
            shift,
            res,
        ):
            started = time.time()
            if garment is None:
                return None, "Vui lòng upload ảnh trang phục (Ảnh 1)."
            if person is None:
                return None, "Vui lòng upload ảnh người mẫu (Ảnh 2)."
            hit = ctx.cached(demo_cache_key([garment, person]))
            if hit:
                return serve_cached(hit, started)
            # Người mẫu đứng cuối: ảnh nền quyết định khung kết quả.
            return ctx.run(
                [garment, person],
                prompt_text,
                note,
                cfg,
                seed,
                steps,
                shift,
                res,
            )

        def ex_vto(garment, person):
            """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả."""
            return on_vto(
                garment,
                person,
                "",
                vto_prompt.value,
                ctx.default_cfg,
                -1,
                ctx.num_steps,
                ctx.default_shift,
                DEFAULT_RESOLUTION_LABEL,
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
                vto_cfg,
                vto_seed,
                vto_steps,
                vto_shift,
                vto_res,
            ],
            outputs=[vto_out, vto_status],
        )
