"""Tab 7 — Image + Prompt.

Dựng bởi ``ui/app.py``; mọi thứ tab này cần đến từ :class:`TabContext`.
"""

from __future__ import annotations

import gradio as gr

from .. import prompts
from ..components import (
    output_block,
    params_accordion,
)
from ..context import TabContext


def build(ctx: TabContext) -> None:
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
                    edit_cfg,
                    edit_seed,
                    edit_steps,
                    edit_shift,
                    edit_res,
                ) = params_accordion(
                    ctx.num_steps, ctx.default_cfg, ctx.default_shift
                )
                gr.Examples(
                    examples=[[x] for x in prompts.EDIT_EXAMPLES],
                    inputs=[edit_prompt],
                    label="Yêu cầu mẫu — bấm để điền vào ô trên",
                    examples_per_page=6,
                )
            with gr.Column(scale=1):
                edit_out, edit_status = output_block()

        def on_edit(image, prompt_text, cfg, seed, steps, shift, res):
            if image is None:
                return None, "Vui lòng upload ảnh cần sửa."
            if not prompt_text or not prompt_text.strip():
                return None, "Vui lòng nhập yêu cầu sửa ảnh."
            # note="" vì ô prompt ở đây CHÍNH LÀ prompt của người dùng,
            # không có prompt hệ thống nào để nối thêm vào.
            return ctx.run(
                [image],
                prompt_text,
                "",
                cfg,
                seed,
                steps,
                shift,
                res,
            )

        edit_run.click(
            fn=on_edit,
            inputs=[
                edit_image,
                edit_prompt,
                edit_cfg,
                edit_seed,
                edit_steps,
                edit_shift,
                edit_res,
            ],
            outputs=[edit_out, edit_status],
        )
