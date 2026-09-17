from ..config import ProcessorConfig
from .base import ImageGenerator

_HIDREAM_ALIASES = frozenset({"hidream_o1", "hidream", "hidream_o1_image"})
_ECHO_ALIASES = frozenset({"echo", "smoke", "stub"})


def create_processor(cfg: ProcessorConfig) -> ImageGenerator:
    """Build the image generator từ config. Nạp model NGAY (constructor)."""
    kind = cfg.type.strip().lower()

    if kind in _HIDREAM_ALIASES:
        from .hidream_generator import HiDreamGenerator

        return HiDreamGenerator(cfg)

    # Smoke test đường queue trên máy không có GPU/weight — trả ảnh
    # placeholder thay vì nạp model. Xem echo_generator.py.
    if kind in _ECHO_ALIASES:
        from .echo_generator import EchoImageGenerator

        return EchoImageGenerator(cfg)

    known = sorted(_HIDREAM_ALIASES | _ECHO_ALIASES)
    raise ValueError(
        f"processor.type không hợp lệ: {cfg.type!r}. Chọn một trong {known}."
    )
