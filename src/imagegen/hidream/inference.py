"""Adapter giữa giao diện của dự án và ``generate_image()`` của HiDream.

Thay cho ``inference.py`` cũ. Ba khác biệt đáng nhớ so với bản Qwen:

* **Một hàm cho cả hai đường.** HiDream-O1 là model hợp nhất: text-to-image
  và editing chỉ khác nhau ở chỗ ``images`` rỗng hay không. Không còn cặp
  ``generate`` / ``generate_t2i`` trỏ vào hai pipeline diffusers riêng.
* **Không có cache prompt-embeds.** Prompt và ảnh được tokenize chung một
  chuỗi mỗi lượt chạy; không có API nào trả embeds tái dùng được. Hai
  ``OrderedDict`` cache của bản cũ không có tương đương.
* **Không chọn được diện tích ảnh ra.** ``find_closest_resolution()`` snap
  mọi yêu cầu về một trong 11 độ phân giải cứng, nhỏ nhất 2048×2048.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from ..config import Settings
from ..logging_setup import LOGGER_NAME
from .resolutions import find_closest_resolution
from .vendor.pipeline import DEFAULT_TIMESTEPS, generate_image

logger = logging.getLogger(LOGGER_NAME)

# Scheduler "default" chạy trọn num_steps do scheduler tự sinh timestep.
# Hai scheduler còn lại dùng bảng timestep cố định 28 mốc của bản dev.
_SCHEDULERS_WITH_FIXED_TIMESTEPS = frozenset({"flow_match", "flash"})


@dataclass(frozen=True)
class Recipe:
    """Bộ tham số sinh ảnh đã giải quyết xong mọi ràng buộc chéo.

    Tồn tại vì các tham số của HiDream-O1 không độc lập với nhau: chọn
    scheduler ``flash`` mà không truyền ``noise_scale_*`` thì scheduler chạy
    với noise mặc định sai, còn chọn ``flow_match``/``flash`` mà không truyền
    ``timesteps_list`` thì bảng 28 mốc của bản dev không được dùng. Gom một
    chỗ để UI, queue worker và benchmark không ai tự suy lại — và suy sai.
    """

    num_steps: int
    guidance_scale: float
    shift: float
    scheduler_name: str
    timesteps_list: list[int] | None = None
    extra: dict[str, float] = field(default_factory=dict)

    # --- Tốc độ ---------------------------------------------------------
    #: Đường attention: "auto" | "sdpa" | "mask" | "flash".
    attn_mode: str = "auto"
    #: Khoảng bước (0..1) còn chạy nhánh uncond của CFG. Ngoài khoảng đó
    #: mỗi bước chỉ tốn MỘT forward pass thay vì hai.
    cfg_interval: tuple[float, float] = (0.0, 1.0)
    #: False → sinh đúng kích thước yêu cầu thay vì snap về bảng 11 độ
    #: phân giải. Rẻ hơn nhiều, nhưng ngoài phân bố huấn luyện.
    snap_resolution: bool = True

    @property
    def forward_passes(self) -> int:
        """Số forward pass một lượt sinh ảnh sẽ chạy — thước đo chi phí thật.

        ``num_steps`` KHÔNG phải chi phí: một bước có CFG tốn hai forward
        pass, không có CFG tốn một. Benchmark và log đều cần con số này.
        """
        if self.guidance_scale <= 1.0:
            return self.num_steps
        lo, hi = self.cfg_interval
        n = self.num_steps
        with_cfg = sum(
            1 for i in range(n) if lo <= (i / max(1, n - 1)) <= hi
        )
        return n + with_cfg


def build_recipe(
    settings: Settings, *, num_steps: int | None = None
) -> Recipe:
    """Dựng Recipe từ Settings, điền các tham số phụ thuộc.

    ``num_steps`` ghi đè giá trị trong settings (message của queue cho phép
    client chọn số bước riêng).
    """
    scheduler = settings.scheduler_name
    extra: dict[str, float] = {}

    if scheduler == "flash":
        # Giá trị của inference.py bản gốc. Scheduler flash lấy mẫu lại
        # noise ở mỗi bước nên cần cả ba, không có default hợp lý nào khác.
        extra = {
            "noise_scale_start": 7.5,
            "noise_scale_end": 7.5,
            "noise_clip_std": 2.5,
        }

    return Recipe(
        num_steps=num_steps
        if num_steps and num_steps > 0
        else settings.num_steps,
        guidance_scale=settings.guidance_scale,
        shift=settings.shift,
        scheduler_name=scheduler,
        timesteps_list=(
            DEFAULT_TIMESTEPS
            if scheduler in _SCHEDULERS_WITH_FIXED_TIMESTEPS
            else None
        ),
        extra=extra,
        attn_mode=settings.attention_mode,
        cfg_interval=(
            settings.cfg_interval_start,
            settings.cfg_interval_end,
        ),
        snap_resolution=settings.snap_resolution,
    )


def to_pil(item: Any) -> Image.Image:
    """Chuẩn hoá một ảnh đầu vào về ``PIL.Image`` RGB.

    Gradio có thể trả về ``(ảnh, caption)`` hoặc đường dẫn file tạm thay vì
    đối tượng PIL.
    """
    if isinstance(item, (tuple, list)):
        item = item[0]
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    return Image.open(item).convert("RGB")


def resolve_size(
    width: int, height: int, *, snap: bool = True
) -> tuple[int, int]:
    """Kích thước thật model sẽ sinh.

    ``snap=False`` khớp với ``Recipe.snap_resolution=False``: chỉ làm tròn
    xuống bội số 32 (kích thước patch) thay vì snap về bảng cố định.
    """
    if not snap:
        return (
            max(32, (width // 32) * 32),
            max(32, (height // 32) * 32),
        )
    return find_closest_resolution(width, height)


def _format_info(
    *,
    elapsed: float,
    step_marks: list[float],
    result: Image.Image,
    recipe: Recipe,
    seed: int,
    requested: tuple[int, int],
) -> str:
    """Dòng trạng thái hiện trên UI và ghi vào audit."""
    per_step = 0.0
    if len(step_marks) >= 2:
        per_step = (step_marks[-1] - step_marks[0]) / (len(step_marks) - 1)

    size = f"{result.width}×{result.height}"
    if (result.width, result.height) != requested:
        size += f" (yêu cầu {requested[0]}×{requested[1]})"

    # `elapsed` đo thuần thời gian tính toán trên GPU. Nó KHÔNG gồm: upload
    # ảnh lên server, hàng đợi Gradio, mã hoá PNG trả về, và độ trễ tunnel
    # *.gradio.live. Nhãn "GPU" ở đầu dòng là để không ai nhầm con số này
    # với round-trip.
    passes = recipe.forward_passes
    per_pass = elapsed / passes if passes else 0.0
    cfg_note = ""
    if recipe.cfg_interval != (0.0, 1.0) and recipe.guidance_scale > 1.0:
        cfg_note = (
            f" (CFG {recipe.cfg_interval[0]:.2f}–{recipe.cfg_interval[1]:.2f})"
        )

    return (
        f"⚡ GPU {elapsed:.1f}s ({per_step:.2f}s/bước · "
        f"{per_pass:.2f}s/forward × {passes})\n"
        f"{size} · {recipe.num_steps} bước · cfg {recipe.guidance_scale}"
        f"{cfg_note} · shift {recipe.shift} · {recipe.scheduler_name} · "
        f"seed {seed}"
    )


def generate(
    model,
    processor,
    images: list[Any],
    prompt: str,
    recipe: Recipe,
    *,
    seed: int = -1,
    width: int = 2048,
    height: int = 2048,
    keep_original_aspect: bool = False,
    match_input_size: bool = False,
) -> tuple[Image.Image | None, str]:
    """Sinh ảnh và trả ``(image, status_message)``.

    ``images`` rỗng → text-to-image. Có ảnh → editing / subject-driven.
    HiDream-O1 khuyến nghị ĐÚNG MỘT ảnh tham chiếu cho editing; nhiều ảnh
    vẫn chạy nhưng là bài toán personalization khác, chất lượng không đảm
    bảo — khác hẳn ngữ nghĩa "ảnh cuối là ảnh nền" của pipeline Qwen cũ.

    ``keep_original_aspect=True`` (chỉ có tác dụng với đúng 1 ảnh) bỏ qua
    bước snap độ phân giải và lấy khung theo ảnh tham chiếu.

    ``match_input_size=True`` thu/phóng ảnh ra về ĐÚNG kích thước pixel của
    ảnh vào. Đây là phép resize SAU khi sinh, không phải sinh ở độ phân giải
    gốc.
    """
    if not prompt or not prompt.strip():
        return None, "Vui lòng nhập prompt mô tả ảnh cần sinh."

    pil_images = [to_pil(i) for i in (images or []) if i is not None]
    base_size = pil_images[-1].size if pil_images else None

    if seed is None or seed < 0:
        seed = int(torch.randint(0, 2**32 - 1, (1,)).item())

    requested = (width, height)
    logger.info(
        "Sinh ảnh: %d ảnh vào, %dx%d, steps=%d, cfg=%s, seed=%d, prompt=%r",
        len(pil_images),
        width,
        height,
        recipe.num_steps,
        recipe.guidance_scale,
        seed,
        prompt[:80],
    )

    step_marks: list[float] = []

    def _on_step(step_idx, total, _preview):  # noqa: ANN001, ARG001
        step_marks.append(time.time())

    started = time.time()
    with torch.inference_mode():
        result = generate_image(
            model=model,
            processor=processor,
            prompt=prompt,
            # PATCH 2 của vendor cho phép truyền thẳng PIL, không cần ghi
            # ảnh ra đĩa rồi đọc lại.
            ref_image_paths=pil_images,
            height=height,
            width=width,
            num_inference_steps=recipe.num_steps,
            guidance_scale=recipe.guidance_scale,
            shift=recipe.shift,
            timesteps_list=recipe.timesteps_list,
            scheduler_name=recipe.scheduler_name,
            seed=seed,
            keep_original_aspect=keep_original_aspect,
            callback=_on_step,
            attn_mode=recipe.attn_mode,
            cfg_interval=recipe.cfg_interval,
            snap_resolution=recipe.snap_resolution,
            **recipe.extra,
        )
    elapsed = time.time() - started

    if match_input_size and base_size and result.size != base_size:
        result = result.resize(base_size, Image.LANCZOS)

    info = _format_info(
        elapsed=elapsed,
        step_marks=step_marks,
        result=result,
        recipe=recipe,
        seed=seed,
        requested=requested,
    )
    logger.info(info.replace("\n", " | "))
    return result, info
