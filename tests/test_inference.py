"""Tests cho tầng inference, chạy bằng pipeline giả (không cần GPU/weight).

Đây là chỗ duy nhất kiểm được phần plumbing giữa service và diffusers mà
không phải nạp 16GB weight: khung ảnh ra, cache prompt-embed, việc truyền
``image=None`` cho text-to-image, và chuỗi ``info`` đi thẳng vào outbound
message.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from gen_image import inference
from gen_image.inference import (
    MAX_SEQUENCE_LENGTH,
    calculate_dimensions,
    clear_embed_cache,
    generate,
)


class _Output:
    def __init__(self, image: Image.Image) -> None:
        self.images = [image]


class _Config:
    """``pipeline.config`` — chỉ có field inference.py thực sự đọc."""

    def __init__(self, is_distilled: bool) -> None:
        self.is_distilled = is_distilled


class FakePipeline:
    """Bắt chước bề mặt của ``Flux2KleinPipeline`` mà inference.py dùng tới.

    Ghi lại mọi lời gọi để test khẳng định được ĐÚNG những gì được truyền
    xuống diffusers — phần dễ lệch nhất khi đổi backend.

    ⚠️ Cố tình KHÔNG có ``do_classifier_free_guidance`` lẫn
    ``_guidance_scale``. Pipeline thật cũng vậy cho tới khi ``__call__`` chạy
    lần đầu: property đó đọc ``self._guidance_scale``, field chỉ được gán ở
    đầu ``__call__``. Một bản FakePipeline trước đây phơi
    ``do_classifier_free_guidance`` ra như thuộc tính thường, và vì thế đã
    che mất một AttributeError xảy ra ở đúng request đầu tiên trên máy thật.
    Giữ bề mặt giả HẸP HƠN bề mặt thật là cách test này còn bắt được lỗi đó.
    """

    vae_scale_factor = 8

    def __init__(self, *, distilled: bool = True) -> None:
        self._execution_device = torch.device("cpu")
        self.config = _Config(is_distilled=distilled)
        self.encode_calls: list[dict] = []
        self.call_kwargs: list[dict] = []

    def encode_prompt(self, **kwargs):
        self.encode_calls.append(kwargs)
        embeds = torch.zeros(1, 4, 8)
        text_ids = torch.zeros(4, 3)
        return embeds, text_ids

    def __call__(self, **kwargs):
        self.call_kwargs.append(kwargs)
        callback = kwargs.get("callback_on_step_end")
        if callback is not None:
            for step in range(kwargs["num_inference_steps"]):
                callback(self, step, 0, {})
        return _Output(Image.new("RGB", (kwargs["width"], kwargs["height"])))


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_embed_cache()
    yield
    clear_embed_cache()


# --------------------------------------------------------------------------
# calculate_dimensions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [1.0, 16 / 9, 9 / 16, 4 / 3, 3 / 2])
def test_dimensions_divisible_and_near_target_area(ratio):
    """Cả hai cạnh phải chia hết cho multiple_of, diện tích bám sát yêu cầu.

    ``Flux2KleinPipeline.check_inputs`` cảnh báo rồi tự resize nếu chiều
    không chia hết — im lặng đổi kích thước ảnh ra, nên phải chặn từ đây.
    """
    area = 1024 * 1024
    width, height = calculate_dimensions(area, ratio, 16)
    assert width % 16 == 0 and height % 16 == 0
    assert width / height == pytest.approx(ratio, rel=0.05)
    assert width * height == pytest.approx(area, rel=0.10)


def test_dimensions_never_collapse_to_zero():
    """Tỉ lệ cực đoan (panorama) không được làm tròn xuống 0.

    Cạnh 0 sẽ nổ sâu trong prepare_latents với một lỗi khó lần ngược.
    """
    width, height = calculate_dimensions(1024 * 1024, 200.0, 16)
    assert width >= 16 and height >= 16


# --------------------------------------------------------------------------
# text-to-image vs image-edit
# --------------------------------------------------------------------------


def test_text_to_image_passes_image_none():
    """Không có ảnh vào → phải truyền ``image=None``, KHÔNG phải list rỗng."""
    pipe = FakePipeline()
    image, info = generate(pipe, [], "a cat", 1.0, seed=0, num_steps=4)

    assert image is not None
    assert pipe.call_kwargs[0]["image"] is None
    assert "guidance 1.0" in info


def test_edit_frames_output_on_last_image():
    """Khung ảnh ra bám ảnh CUỐI (ảnh nền), không phải ảnh tham chiếu."""
    reference = Image.new("RGB", (400, 800))  # dọc
    base = Image.new("RGB", (800, 400))  # ngang
    pipe = FakePipeline()

    generate(pipe, [reference, base], "edit", 1.0, seed=0, num_steps=4)

    kwargs = pipe.call_kwargs[0]
    assert kwargs["width"] > kwargs["height"], "phải bám tỉ lệ ảnh nền"
    assert len(kwargs["image"]) == 2


def test_aspect_ratio_only_used_without_images():
    """``aspect_ratio`` bị ảnh nền ghi đè khi có ảnh vào."""
    base = Image.new("RGB", (800, 400))
    pipe = FakePipeline()

    generate(pipe, [base], "edit", 1.0, seed=0, num_steps=4, aspect_ratio=9 / 16)

    kwargs = pipe.call_kwargs[0]
    assert kwargs["width"] > kwargs["height"]


def test_match_input_size_resizes_back_to_source():
    base = Image.new("RGB", (813, 407))
    pipe = FakePipeline()

    image, _ = generate(
        pipe,
        [base],
        "edit",
        1.0,
        seed=0,
        num_steps=4,
        match_input_size=True,
    )

    assert image.size == (813, 407)


# --------------------------------------------------------------------------
# cache prompt-embed
# --------------------------------------------------------------------------


def test_embed_cache_hits_on_same_prompt_different_image():
    """Đổi ảnh mà giữ prompt PHẢI ăn cache.

    Đây là điểm khác biệt thực chất so với backend cũ: text encoder Qwen3-4B
    là text-only nên ảnh không tham gia vào khoá cache. Backend Qwen trước
    đây băm cả pixel của ảnh vào khoá và vì thế không bao giờ ăn cache ở
    kịch bản này — vốn là kịch bản lặp nhiều nhất trong production.
    """
    pipe = FakePipeline()
    first = Image.new("RGB", (512, 512), (10, 10, 10))
    second = Image.new("RGB", (512, 512), (200, 200, 200))

    _, info_a = generate(pipe, [first], "same prompt", 1.0, 0, 4)
    _, info_b = generate(pipe, [second], "same prompt", 1.0, 1, 4)

    assert len(pipe.encode_calls) == 1, "lần thứ hai phải ăn cache"
    assert "(cache)" not in info_a
    assert "(cache)" in info_b


def test_embed_cache_misses_on_different_prompt():
    pipe = FakePipeline()
    generate(pipe, [], "prompt one", 1.0, 0, 4)
    generate(pipe, [], "prompt two", 1.0, 0, 4)
    assert len(pipe.encode_calls) == 2


def test_embed_cache_is_bounded():
    """LRU phải thực sự loại bớt — cache không giới hạn là một rò VRAM."""
    pipe = FakePipeline()
    for i in range(inference._EMBED_CACHE_MAX + 3):
        generate(pipe, [], f"prompt {i}", 1.0, 0, 4)
    assert len(inference._EMBED_CACHE) == inference._EMBED_CACHE_MAX


def test_encode_uses_flux2_sequence_length():
    """512, không phải 1024 của Qwen2.5-VL — thừa sẽ bị pipeline cắt."""
    pipe = FakePipeline()
    generate(pipe, [], "x", 1.0, 0, 4)
    assert pipe.encode_calls[0]["max_sequence_length"] == MAX_SEQUENCE_LENGTH
    assert MAX_SEQUENCE_LENGTH == 512


# --------------------------------------------------------------------------
# negative prompt
# --------------------------------------------------------------------------


def test_negative_prompt_ignored_when_distilled():
    """Bản distilled không chạy CFG → không được truyền negative embeds.

    Truyền vào thì diffusers lẳng lặng bỏ qua, nhưng ta đã tốn một lượt chạy
    text encoder cho không.
    """
    pipe = FakePipeline(distilled=True)
    generate(pipe, [], "x", 1.0, 0, 4, negative_prompt="blurry, ugly")

    assert "negative_prompt_embeds" not in pipe.call_kwargs[0]
    assert len(pipe.encode_calls) == 1


def test_negative_prompt_used_when_cfg_active():
    """Bản base (không distilled) thì negative prompt phải đi xuống thật."""
    pipe = FakePipeline(distilled=False)
    generate(pipe, [], "x", 4.0, 0, 4, negative_prompt="blurry, ugly")

    assert "negative_prompt_embeds" in pipe.call_kwargs[0]
    assert len(pipe.encode_calls) == 2


# --------------------------------------------------------------------------
# hợp đồng với BE
# --------------------------------------------------------------------------


def test_empty_prompt_returns_none_not_raises():
    """Prompt rỗng → (None, message). Pipeline phải map sang AI_NO_RESULT."""
    pipe = FakePipeline()
    image, info = generate(pipe, [], "   ", 1.0, 0, 4)
    assert image is None
    assert info


def test_info_string_keeps_the_shape_be_parses():
    """`result.info` là hợp đồng với BE — giữ nguyên cấu trúc mốc giờ."""
    pipe = FakePipeline()
    _, info = generate(pipe, [], "x", 1.0, seed=7, num_steps=4)

    for token in ("GPU", "denoise", "text", "prep", "decode", "seed 7"):
        assert token in info, f"thiếu {token!r} trong: {info}"


def test_negative_seed_is_randomised_and_reported():
    pipe = FakePipeline()
    _, info = generate(pipe, [], "x", 1.0, seed=-1, num_steps=4)
    assert "seed -1" not in info


def test_does_not_touch_pipeline_cfg_property_before_first_call():
    """Không được đọc ``do_classifier_free_guidance`` trước ``__call__``.

    Property đó của diffusers đọc ``self._guidance_scale``, field chỉ tồn tại
    sau khi ``__call__`` chạy. Đọc sớm → AttributeError ở đúng request đầu
    tiên, tức là service chết ngay sau khi deploy chứ không phải lúc test.
    """
    pipe = FakePipeline(distilled=True)

    def _boom(self):
        raise AssertionError("đọc do_classifier_free_guidance quá sớm")

    type(pipe).do_classifier_free_guidance = property(_boom)
    try:
        image, _ = generate(pipe, [], "x", 1.0, 0, 4, negative_prompt="blurry")
        assert image is not None
    finally:
        del type(pipe).do_classifier_free_guidance


# --------------------------------------------------------------------------
# sức chứa cache prompt-embed
# --------------------------------------------------------------------------


def test_configure_embed_cache_changes_capacity():
    from gen_image.inference import configure_embed_cache

    pipe = FakePipeline()
    configure_embed_cache(3)
    try:
        for i in range(6):
            generate(pipe, [], f"p{i}", 1.0, 0, 4)
        assert len(inference._EMBED_CACHE) == 3
    finally:
        configure_embed_cache(8)


def test_shrinking_capacity_evicts_immediately():
    """Hạ sức chứa phải loại bớt NGAY, không đợi request tiếp theo.

    Nếu không, một cache 500 entry vẫn ngốn VRAM sau khi ai đó hạ config
    xuống 10 — và chỉ co lại vào lúc không ai ngờ.
    """
    from gen_image.inference import configure_embed_cache

    pipe = FakePipeline()
    configure_embed_cache(10)
    try:
        for i in range(10):
            generate(pipe, [], f"q{i}", 1.0, 0, 4)
        assert len(inference._EMBED_CACHE) == 10
        configure_embed_cache(2)
        assert len(inference._EMBED_CACHE) == 2
    finally:
        configure_embed_cache(8)
