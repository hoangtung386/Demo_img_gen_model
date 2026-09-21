"""Widget Gradio dùng chung giữa các tab.

Tách khỏi ``app.py`` vì bảy tab đều dựng cùng một bộ điều khiển (prompt,
negative, guidance, seed, số bước, độ phân giải) và cùng một khối kết quả.
Giữ chúng ở một chỗ để sửa một lần là cả bảy tab đổi theo.
"""

from __future__ import annotations

import gradio as gr

from ..inference import DEFAULT_OUTPUT_AREA


def example_rows(cases, to_row) -> list[list]:
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
# vì Flux2KleinPipeline tự thu mọi ảnh tham chiếu xuống <=1024² trước khi
# VAE-encode, và text encoder Qwen3-8B không nhận ảnh.
OUTPUT_PRESETS: dict[str, int] = {
    "Chuẩn — 1024px (~1.0 MP)": DEFAULT_OUTPUT_AREA,
    "Nhanh — 832px (~0.7 MP)": 832 * 832,
    "Rất nhanh — 704px (~0.5 MP)": 704 * 704,
}

# Tab 6 không có ảnh nền để lấy tỉ lệ khung, nên người dùng phải chọn. Diện
# tích vẫn lấy từ OUTPUT_PRESETS ở trên; hai thứ ghép lại ra chiều rộng và
# chiều cao qua calculate_dimensions.
ASPECT_RATIOS: dict[str, float] = {
    "1:1 — vuông": 1.0,
    "16:9 — ngang": 16 / 9,
    "9:16 — dọc (điện thoại)": 9 / 16,
    "4:3 — ngang": 4 / 3,
    "3:4 — dọc": 3 / 4,
    "3:2 — ảnh chụp": 3 / 2,
    "2:3 — chân dung": 2 / 3,
}


def params_block(
    default_negative: str, default_steps: int
) -> tuple[gr.Textbox, gr.Slider, gr.Number, gr.Slider, gr.Dropdown]:
    """Negative prompt + các tham số lấy mẫu.

    Tách riêng khỏi ``_advanced_block`` vì tab 6 và 7 để người dùng tự viết
    prompt ở ô chính, không có prompt hệ thống nào để hiện trong accordion.
    """
    negative_box = gr.Textbox(
        label="Negative prompt",
        value=default_negative,
        lines=3,
        info=(
            "KHÔNG có tác dụng với bản klein distilled: guidance được nhúng "
            "vào model thay vì chạy CFG hai nhánh, nên không có nhánh "
            "negative nào để tác động. Giữ ô này cho bản 'klein-base'."
        ),
    )
    with gr.Row():
        cfg = gr.Slider(
            label="guidance_scale",
            minimum=1.0,
            maximum=10.0,
            value=1.0,
            step=0.5,
            info="Bản distilled dùng 1.0 (giá trị của model card)",
        )
        seed = gr.Number(label="Seed (-1 = ngẫu nhiên)", value=-1, precision=0)
        steps = gr.Slider(
            label="Số bước",
            minimum=4,
            maximum=12,
            value=default_steps,
            step=1,
            info="Bản distilled được chưng cất về đúng 4 bước",
        )
    resolution = gr.Dropdown(
        label="Độ phân giải đầu ra",
        choices=list(OUTPUT_PRESETS),
        value=next(iter(OUTPUT_PRESETS)),
        info="Hạ xuống để chạy nhanh hơn, đổi lại mất chi tiết.",
    )
    return negative_box, cfg, seed, steps, resolution


def params_accordion(
    default_negative: str, default_steps: int
) -> tuple[gr.Textbox, gr.Slider, gr.Number, gr.Slider, gr.Dropdown]:
    """``_params_block`` gói trong accordion — cho tab prompt tự do."""
    with gr.Accordion("Tuỳ chỉnh nâng cao", open=False):
        return params_block(default_negative, default_steps)


def advanced_block(
    default_prompt: str, default_negative: str, default_steps: int
) -> tuple[
    gr.Textbox,
    gr.Textbox,
    gr.Slider,
    gr.Number,
    gr.Slider,
    gr.Dropdown,
    gr.Button,
]:
    """Khối nâng cao cho các tab có prompt hệ thống sinh sẵn."""
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
        negative_box, cfg, seed, steps, resolution = params_block(
            default_negative, default_steps
        )
    return prompt_box, negative_box, cfg, seed, steps, resolution, reset_btn


def output_block() -> tuple[gr.Image, gr.Textbox]:
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
