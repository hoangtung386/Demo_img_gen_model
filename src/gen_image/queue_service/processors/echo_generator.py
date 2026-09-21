"""Processor giả — dùng để smoke test ĐƯỜNG QUEUE, không phải để chạy thật.

Vì sao cần: `Application.run()` nạp model TRƯỚC khi spawn consumer/publisher
(create_processor gọi ở bước 5, consumer ở bước 6). Nên trên máy không có
A100 + 26GB weight, toàn bộ chuỗi broker → worker → `queue_out` không cách
nào chạy thử được, và lỗi cấu hình queue chỉ lộ ra sau khi đã thuê GPU.

EchoImageGenerator thay chỗ model: trả về một ảnh placeholder sau một khoảng
delay giả lập. Mọi mắt xích còn lại (consume 4 tier, validate, pipeline,
audit, storage, publish sang `queue_out`) vẫn chạy nguyên bản, nên nếu smoke
test thấy reply trên `queue_out` thì phần queue đã đúng — chỉ còn model là
biến số khi lên A100.

Bật bằng config, KHÔNG có đường nào vào nhầm lúc chạy thật:

    processor:
      type: "echo"

Delay giả lập đổi qua env `GENIMG_ECHO_DELAY_SECONDS` (mặc định 0.2s).
"""

from __future__ import annotations

import math
import os
import time

from PIL import Image, ImageDraw

from ..config import ProcessorConfig
from ..observability import logger
from .base import ImageGenerator

_DEFAULT_DELAY_SECONDS = 0.2
# Trần kích thước ảnh placeholder. output_area mặc định là 1024² nhưng ảnh
# giả không cần to bằng ảnh thật — nhỏ thì upload/audit nhanh, smoke test
# không bị nghẽn ở I/O.
_MAX_SIDE = 512


class EchoImageGenerator(ImageGenerator):
    """Trả ảnh placeholder; không import torch, không chạm GPU."""

    def __init__(self, cfg: ProcessorConfig):
        self.cfg = cfg
        try:
            self._delay = float(
                os.environ.get("GENIMG_ECHO_DELAY_SECONDS", _DEFAULT_DELAY_SECONDS)
            )
        except ValueError:
            self._delay = _DEFAULT_DELAY_SECONDS
        logger.warning(
            "[processor] ECHO MODE — không nạp model, ảnh trả về là placeholder. "
            f"Chỉ dùng để smoke test đường queue (delay={self._delay}s)."
        )

    def generate(
        self,
        prompt: str,
        images: list[Image.Image],
        *,
        negative_prompt: str = "",
        seed: int = -1,
        num_steps: int = 0,
        aspect_ratio: float = 1.0,
        match_input_size: bool = False,
    ) -> tuple[Image.Image | None, str]:
        time.sleep(self._delay)

        # Bám theo tỉ lệ/kích thước mà request yêu cầu để pipeline phía sau
        # (save_output, upload) làm việc trên ảnh có hình dạng hợp lý.
        if images and match_input_size:
            width, height = images[-1].size
        else:
            side = min(_MAX_SIDE, int(math.sqrt(max(self.cfg.output_area, 1))))
            ratio = aspect_ratio if aspect_ratio > 0 else 1.0
            width = max(64, int(side * math.sqrt(ratio)))
            height = max(64, int(side / math.sqrt(ratio)))

        image = Image.new("RGB", (width, height), (32, 36, 48))
        draw = ImageDraw.Draw(image)
        draw.rectangle([8, 8, width - 8, height - 8], outline=(120, 200, 160), width=3)
        draw.text((20, 20), "ECHO MODE — placeholder", fill=(220, 220, 220))
        draw.text((20, 40), f"prompt: {prompt[:48]}", fill=(160, 170, 180))
        draw.text(
            (20, 60), f"inputs: {len(images)}  seed: {seed}", fill=(160, 170, 180)
        )

        steps = num_steps if num_steps > 0 else self.cfg.num_steps
        info = (
            f"echo placeholder {width}×{height}, steps={steps}, "
            f"seed={seed} (KHÔNG chạy model)"
        )
        return image, info
