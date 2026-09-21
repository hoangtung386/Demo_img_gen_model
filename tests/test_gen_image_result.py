"""Tests for GenImagePipeline._build_result / _build_openai_payload."""

from __future__ import annotations

import base64

from PIL import Image

from gen_image.queue_service.messaging.schemas import InboundMessage
from gen_image.queue_service.pipelines.gen_image import GenImagePipeline

_BASE = {
    "_id": "req-1",
    "os": "android",
    "firebase_token": "tok",
    "appid": "app",
    "country": "vn",
    "device_id": "dev",
    "prompt": "a cat",
}


def _write_test_png(path) -> None:
    Image.new("RGB", (16, 24), color="red").save(path)


def test_build_result_default_provider_keeps_old_shape(tmp_path):
    msg = InboundMessage.from_dict(_BASE)
    img_path = tmp_path / "out.png"
    _write_test_png(img_path)

    result = GenImagePipeline._build_result(
        msg, url="https://example.com/x.png", info="", image_path=str(img_path)
    )

    assert result == {"url": "https://example.com/x.png", "info": ""}
    assert "openai" not in result


def test_resize_if_oversized_keeps_image_within_limit_unchanged():
    im = Image.new("RGB", (1024, 1024))
    out = GenImagePipeline._resize_if_oversized(im, max_dimension=1280)
    assert out.size == (1024, 1024)


def test_resize_if_oversized_shrinks_wide_image_keeping_aspect_ratio():
    im = Image.new("RGB", (2560, 1280))  # 2:1
    out = GenImagePipeline._resize_if_oversized(im, max_dimension=1280)
    assert out.width <= 1280 and out.height <= 1280
    assert out.size == (1280, 640)  # tỉ lệ 2:1 giữ nguyên


def test_resize_if_oversized_shrinks_tall_image_keeping_aspect_ratio():
    im = Image.new("RGB", (500, 2000))  # 1:4
    out = GenImagePipeline._resize_if_oversized(im, max_dimension=1280)
    assert out.width <= 1280 and out.height <= 1280
    assert out.size == (320, 1280)


def test_resize_if_oversized_never_upscales_small_image():
    im = Image.new("RGB", (256, 256))
    out = GenImagePipeline._resize_if_oversized(im, max_dimension=1280)
    assert out.size == (256, 256)


def test_build_result_openai_provider_adds_nested_block(tmp_path):
    msg = InboundMessage.from_dict({**_BASE, "provider": "openai"})
    img_path = tmp_path / "out.png"
    _write_test_png(img_path)

    result = GenImagePipeline._build_result(
        msg, url="https://example.com/x.png", info="", image_path=str(img_path)
    )

    # Old fields untouched — consumers reading result.url/info unaffected.
    assert result["url"] == "https://example.com/x.png"
    assert result["info"] == ""

    openai_block = result["openai"]
    assert openai_block["background"] == "opaque"
    assert openai_block["output_format"] == "png"
    assert openai_block["quality"] == "high"
    assert openai_block["size"] == "16x24"
    assert isinstance(openai_block["created"], int)
    assert len(openai_block["data"]) == 1

    decoded = base64.b64decode(openai_block["data"][0]["b64_json"])
    with img_path.open("rb") as f:
        assert decoded == f.read()
