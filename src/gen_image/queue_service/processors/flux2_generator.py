"""
Flux2KleinGenerator — cầu nối giữa queue service và tầng model FLUX.2.

Nạp model MỘT LẦN lúc khởi động (trong `Application.run` trước khi flip
/readyz) rồi tái sử dụng cho mọi request. Không nạp lại giữa các request —
weight ~16GB, nạp mất hàng chục giây.

Khác backend cũ ở một điểm đáng kể: chỉ có MỘT pipeline. Backend Qwen phải
dựng hai object (`QwenImageEditPlusPipeline` bắt buộc có `image=`, nên
text-to-image cần thêm một `QwenImagePipeline` dùng chung weight).
`Flux2KleinPipeline` nhận `image=None` nên cùng một object phục vụ cả hai
đường — và `inference.generate()` cũng chỉ còn một hàm thay vì hai.

ProcessorConfig của queue service được map sang gen_image.config.Settings để
tái dùng nguyên `models.loader.load_pipeline` — không copy logic nạp model.
"""

from __future__ import annotations

from PIL import Image

from ...config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_GGUF_FILE,
    DEFAULT_GGUF_REPO,
)
from ...config import Settings as ModelSettings
from ...device import select_device
from ...inference import DEFAULT_OUTPUT_AREA, configure_embed_cache, generate
from ...models.loader import load_pipeline
from ...tuning import apply_gpu_tuning, maybe_compile
from ..config import ProcessorConfig
from ..observability import logger
from .base import ImageGenerator


def _to_model_settings(cfg: ProcessorConfig) -> ModelSettings:
    """Map ProcessorConfig (queue service) → gen_image.config.Settings.

    Chỉ set các field pipeline loader thực sự đọc; phần Gradio (server_name,
    port, demo_cache…) để giá trị vô hại vì service queue không chạy UI.
    `load_settings()` KHÔNG được gọi ở đây — mọi giá trị đến từ base.yaml
    qua ProcessorConfig, tránh phụ thuộc biến môi trường GENIMG_* ở runtime
    queue.
    """
    return ModelSettings(
        num_steps=cfg.num_steps,
        guidance_scale=cfg.guidance_scale,
        quantization=cfg.quantization,
        compile_transformer=cfg.compile_transformer,
        vae_tiling=cfg.vae_tiling,
        vae_slicing=cfg.vae_slicing,
        embed_cache_size=cfg.embed_cache_size,
        model_root="models",
        base_model=DEFAULT_BASE_MODEL,
        gguf_repo=DEFAULT_GGUF_REPO,
        gguf_file=DEFAULT_GGUF_FILE,
        offload=cfg.offload,
        server_name="0.0.0.0",
        server_port=7860,
        demo_cache=False,
        device_override=cfg.device or None,
        base_model_local=cfg.base_model_local or None,
        transformer_gguf=cfg.transformer_gguf or None,
        warmup=cfg.warmup,
    )


class Flux2KleinGenerator(ImageGenerator):
    def __init__(self, cfg: ProcessorConfig):
        self.cfg = cfg
        self._settings = _to_model_settings(cfg)
        self.device = select_device(cfg.device or None)
        apply_gpu_tuning()
        logger.info(f"[processor] loading FLUX.2-klein on {self.device} ...")
        configure_embed_cache(cfg.embed_cache_size)
        self._pipeline = load_pipeline(self._settings, self.device)
        maybe_compile(self._pipeline, cfg.compile_transformer, cfg.quantization)
        logger.info("[processor] FLUX.2-klein pipeline ready")
        if cfg.warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Trả trước chi phí lượt sinh đầu (kernel, allocator, compile)."""
        from ...tuning import warmup

        try:
            logger.info("[processor] warmup ...")
            warmup(
                self._pipeline,
                self.cfg.num_steps,
                self.cfg.output_area or DEFAULT_OUTPUT_AREA,
                self.cfg.guidance_scale,
            )
            logger.info("[processor] warmup done")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[processor] warmup failed (bỏ qua): {e}")

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
        steps = num_steps if num_steps and num_steps > 0 else self.cfg.num_steps
        output_area = self.cfg.output_area or DEFAULT_OUTPUT_AREA

        # Một lời gọi duy nhất cho cả hai đường: `images` rỗng → text-to-image
        # (khung suy từ aspect_ratio), có phần tử → image-edit (khung bám ảnh
        # cuối). Không còn nhánh if/else dispatch sang hai pipeline khác nhau.
        return generate(
            self._pipeline,
            images=images,
            prompt=prompt,
            guidance_scale=self.cfg.guidance_scale,
            seed=seed,
            num_steps=steps,
            negative_prompt=negative_prompt or None,
            output_area=output_area,
            aspect_ratio=aspect_ratio,
            match_input_size=match_input_size,
        )
