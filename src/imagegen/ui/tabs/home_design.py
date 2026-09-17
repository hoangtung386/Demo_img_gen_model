"""Tab 2 — Home Design.

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
                    home_cfg,
                    home_seed,
                    home_steps,
                    home_shift,
                    home_res,
                    home_reset,
                ) = advanced_block(
                    prompts.build_home_prompt(next(iter(prompts.ROOM_TYPES))),
                    ctx.num_steps,
                    ctx.default_cfg,
                    ctx.default_shift,
                )
            with gr.Column(scale=1):
                home_out, home_status = output_block()

        home_room_type.change(
            prompts.build_home_prompt, [home_room_type], [home_prompt]
        )
        home_reset.click(
            prompts.build_home_prompt, [home_room_type], [home_prompt]
        )

        def on_home(room, design, prompt_text, cfg, seed, steps, shift, res):
            started = time.time()
            if room is None:
                return None, "Vui lòng upload ảnh căn phòng."
            if not (design or "").strip():
                return None, "Vui lòng mô tả thiết kế mong muốn."
            hit = ctx.cached(demo_cache_key([room], design))
            if hit:
                return serve_cached(hit, started)
            # Khung hệ thống + mô tả người dùng; note để rỗng vì mô tả
            # thiết kế đã là phần nội dung chính.
            return ctx.run(
                [room],
                prompts.compose_home_prompt(prompt_text, design),
                "",
                cfg,
                seed,
                steps,
                shift,
                res,
            )

        def ex_home(room, design):
            """Click ví dụ: đẩy thẳng ảnh dựng sẵn ra khung kết quả."""
            return on_home(
                room,
                design,
                home_prompt.value,
                ctx.default_cfg,
                -1,
                ctx.num_steps,
                ctx.default_shift,
                DEFAULT_RESOLUTION_LABEL,
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
                home_cfg,
                home_seed,
                home_steps,
                home_shift,
                home_res,
            ],
            outputs=[home_out, home_status],
        )
