from abc import ABC, abstractmethod

from PIL import Image


class ImageGenerator(ABC):
    """
    Abstract image generator. Implementation: FLUX.2-klein-9B GGUF.

    `generate()` chạy model (image-edit nếu có ảnh điều kiện, text-to-image
    nếu không) và trả (image | None, info_message). None nghĩa model không
    sinh được ảnh (→ AI_NO_RESULT).
    """

    @abstractmethod
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
        """Run the model. Raises on inference error; returns (None, info) if empty."""
