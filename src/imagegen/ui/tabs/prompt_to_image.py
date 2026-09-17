"""Tab 6 — Prompt to Image.

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
    with gr.Tab("6. Prompt to Image"):
        gr.Markdown(
            "Không cần ảnh vào — chỉ mô tả bằng chữ, model sinh ảnh "
            "mới. Chọn khung hình ở mục *Độ phân giải đầu ra* trong "
            "**Tuỳ chỉnh nâng cao**.\n"
            "> Đây là tác vụ HiDream-O1 được luyện chính, không phải "
            "đường phụ như bản Qwen-Image-**Edit** trước đây. Prompt mô "
            "tả càng dài và cụ thể càng tốt. Muốn sửa một tấm ảnh có "
            "sẵn thì dùng tab 7."
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
                t2i_run = gr.Button("Sinh ảnh", variant="primary")
                (
                    t2i_cfg,
                    t2i_seed,
                    t2i_steps,
                    t2i_shift,
                    t2i_res,
                ) = params_accordion(
                    ctx.num_steps, ctx.default_cfg, ctx.default_shift
                )
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
            fn=ctx.run_t2i,
            inputs=[
                t2i_prompt,
                t2i_cfg,
                t2i_seed,
                t2i_steps,
                t2i_shift,
                t2i_res,
            ],
            outputs=[t2i_out, t2i_status],
        )
