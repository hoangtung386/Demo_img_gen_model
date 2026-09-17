"""HiDreamGenerator — cầu nối giữa queue service và model HiDream-O1-Image.

Nạp model MỘT LẦN lúc khởi động (trong ``Application.run`` trước khi flip
/readyz) rồi tái sử dụng cho mọi request. Không nạp lại giữa các request —
weight ~9.9GB, nạp mất hàng chục giây.

ProcessorConfig của queue service được map sang ``imagegen.config.Settings``
để tái dùng nguyên ``hidream.loader.load_model`` — không copy logic nạp model.

Khác bản Qwen trước đây ở ba chỗ, đều do model mới là một khối hợp nhất:
  • MỘT model cho cả edit lẫn text-to-image (không có pipeline t2i thứ hai)
  • không có cache prompt-embeds
  • ``aspect_ratio`` đi qua ``find_closest_resolution()`` nên kết quả luôn là
    một trong 11 độ phân giải cứng, không phải diện tích tuỳ ý
"""

from __future__ import annotations

from PIL import Image

from ...config import Settings as CoreSettings
from ...device import select_device
from ...hidream import Recipe, build_recipe, generate, load_model
from ...hidream.resolutions import size_for_aspect_ratio
from ...tuning import apply_gpu_tuning
from ..config import ProcessorConfig
from ..observability import logger
from .base import ImageGenerator


def _to_core_settings(cfg: ProcessorConfig) -> CoreSettings:
    """Map ProcessorConfig (queue service) → imagegen.config.Settings.

    Chỉ set các field loader/recipe thực sự đọc; phần Gradio (server_name,
    port, demo_cache…) để giá trị vô hại vì service queue không chạy UI.
    ``load_settings()`` KHÔNG được gọi ở đây — mọi giá trị đến từ base.yaml
    qua ProcessorConfig, tránh phụ thuộc biến môi trường IMG_* ở runtime
    queue.
    """
    return CoreSettings(
        model_root="models",
        base_model="",
        model_path_local=cfg.model_path or None,
        model_type=cfg.model_type,
        num_steps=cfg.num_steps,
        guidance_scale=cfg.guidance_scale,
        shift=cfg.shift,
        scheduler_name=cfg.scheduler_name,
        width=cfg.width,
        height=cfg.height,
        device_override=cfg.device or None,
        warmup=cfg.warmup,
        server_name="0.0.0.0",
        server_port=7860,
        demo_cache=False,
        attention_mode=cfg.attention_mode,
        dequantize=cfg.dequantize,
        compile_model=cfg.compile_model,
        cfg_interval_start=cfg.cfg_interval_start,
        cfg_interval_end=cfg.cfg_interval_end,
        snap_resolution=cfg.snap_resolution,
        warmup_steps=cfg.warmup_steps,
    )


class HiDreamGenerator(ImageGenerator):
    """Chạy HiDream-O1-Image cho cả text-to-image lẫn editing."""

    def __init__(self, cfg: ProcessorConfig):
        self.cfg = cfg
        self._settings = _to_core_settings(cfg)
        # TRƯỚC select_device/load_model: TF32, backend SDPA và các cờ khác
        # phải được đặt trước khi có kernel nào chạy. Đường Gradio gọi hàm
        # này trong serve.main(); queue worker trước đây thì KHÔNG — tức là
        # production chạy thiếu đúng những cờ mà benchmark có.
        apply_gpu_tuning()
        self.device = select_device(cfg.device or None)
        logger.info(f"[processor] nạp HiDream-O1 trên {self.device} ...")
        self._model, self._processor = load_model(self._settings, self.device)
        logger.info("[processor] HiDream-O1 sẵn sàng")
        if cfg.warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Trả trước chi phí lượt sinh đầu tiên.

        Chỉ chạy ``warmup_steps`` bước (mặc định 2), không phải cả
        ``num_steps``: chi phí cần trả trước — CUDA context, allocator pool,
        dequant lần đầu, autotune — là chi phí MỘT LẦN, không nhân theo số
        bước. Độ phân giải thì giữ đúng của config vì nó quyết định hình
        dạng mà allocator và kernel được autotune cho.
        """
        try:
            steps = max(1, self.cfg.warmup_steps)
            logger.info(f"[processor] warmup ({steps} bước) ...")
            generate(
                self._model,
                self._processor,
                [],
                "a warmup image, minimalist",
                self._recipe(num_steps=steps),
                seed=0,
                width=self.cfg.width,
                height=self.cfg.height,
            )
            logger.info("[processor] warmup xong")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[processor] warmup thất bại (bỏ qua): {e}")

    def _recipe(self, num_steps: int = 0) -> Recipe:
        return build_recipe(self._settings, num_steps=num_steps)

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
        # negative_prompt bị BỎ QUA: generate_image() không nhận nó — nhánh
        # uncond của CFG dùng prompt " " cố định. Giữ trong chữ ký để không
        # phá ImageGenerator ABC và message schema hiện có.
        if negative_prompt:
            logger.debug("[processor] HiDream-O1 bỏ qua negative_prompt")

        width, height = size_for_aspect_ratio(
            aspect_ratio, base=self.cfg.height
        )
        return generate(
            self._model,
            self._processor,
            images,
            prompt,
            self._recipe(num_steps),
            seed=seed,
            width=width,
            height=height,
            # keep_original_aspect chỉ có tác dụng với ĐÚNG 1 ảnh; bật khi
            # có để ảnh ra giữ khung của ảnh vào thay vì bị snap.
            keep_original_aspect=len(images) == 1,
            match_input_size=match_input_size,
        )
