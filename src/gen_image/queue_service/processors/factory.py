from ..config import ProcessorConfig
from .base import ImageGenerator


def create_processor(cfg: ProcessorConfig) -> ImageGenerator:
    """Build the image generator từ config. Nạp model NGAY (constructor)."""
    if cfg.type in ("flux2_klein", "flux2", "klein"):
        from .flux2_generator import Flux2KleinGenerator

        return Flux2KleinGenerator(cfg)
    # Smoke test đường queue trên máy không có GPU/weight — trả ảnh
    # placeholder thay vì nạp model. Xem echo_generator.py.
    if cfg.type in ("echo", "smoke", "stub"):
        from .echo_generator import EchoImageGenerator

        return EchoImageGenerator(cfg)
    raise ValueError(f"Unknown processor type: {cfg.type}")
