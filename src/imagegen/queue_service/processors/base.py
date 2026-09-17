from abc import ABC, abstractmethod

from PIL import Image


class ImageGenerator(ABC):
    """
    Abstract image generator. Implementation: HiDream-O1-Image (SDNQ 4-bit).

    `generate()` chạy model (editing nếu có ảnh điều kiện, text-to-image nếu
    không) và trả (image | None, info_message). None nghĩa model không sinh
    được ảnh (→ AI_NO_RESULT).

    Hai tham số được giữ trong chữ ký nhưng KHÔNG còn tác dụng với model
    hiện tại — giữ để message schema và các implementation khác (echo) không
    phải đổi:

    * `negative_prompt` — HiDream-O1 không nhận; nhánh uncond của CFG dùng
      prompt " " cố định.
    * `aspect_ratio` — chỉ chọn được TỈ LỆ, không chọn được kích thước: model
      snap về một trong 11 độ phân giải cứng, nhỏ nhất 2048x2048.
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
        """Run the model.

        Raises on inference error; returns (None, info) if empty.
        """
