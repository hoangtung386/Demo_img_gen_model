"""Resolve đường dẫn model — cây file phẳng, không còn model_index.json."""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from imagegen.hidream.artifacts import (
    REQUIRED_FILES,
    SPECIAL_TOKENS,
)


def _stub_heavy_imports(monkeypatch):
    """Cho phép import hidream.loader mà không cần torch/transformers/sdnq.

    ``resolve_model_path`` là hàm thuần trên hệ thống file, nhưng nó nằm
    cùng module với lời gọi ``from_pretrained``. Không stub thì test này chỉ
    chạy được trên máy đã cài đủ ~3GB wheel — tức là không bao giờ chạy
    trong CI.
    """
    monkeypatch.setitem(sys.modules, "sdnq", types.ModuleType("sdnq"))

    torch = types.ModuleType("torch")
    torch.bfloat16 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)

    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = object
    transformers.PreTrainedTokenizerBase = type(
        "PreTrainedTokenizerBase", (), {}
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    vendor = "imagegen.hidream.vendor.qwen3_vl_transformers"
    module = types.ModuleType(vendor)
    module.Qwen3VLForConditionalGeneration = object
    monkeypatch.setitem(sys.modules, vendor, module)

    # Module có thể đã được import ở test khác với torch thật/vắng mặt.
    monkeypatch.delitem(sys.modules, "imagegen.hidream.loader", raising=False)


@pytest.fixture
def loader(monkeypatch):
    _stub_heavy_imports(monkeypatch)
    from imagegen.hidream import loader as mod

    yield mod

    # DỌN SẠCH, không chỉ khôi phục env.
    #
    # ``monkeypatch.setitem(sys.modules, "torch", stub)`` chỉ hoàn lại mục
    # "torch"; nó không biết gì về ``imagegen.hidream.loader`` mà lệnh import
    # ở trên vừa tạo ra. Module đó giữ cứng tham chiếu tới module torch GIẢ
    # trong biến toàn cục của nó, và nằm lại trong sys.modules cho mọi test
    # sau — test nào import loader sau đây sẽ nhận một `torch` chỉ có đúng
    # thuộc tính `bfloat16`, rồi hỏng ở một dòng không liên quan gì tới nó.
    sys.modules.pop("imagegen.hidream.loader", None)


def _make_repo(tmp_path, required):
    for name in required:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    return tmp_path


def test_resolve_local(loader, base_settings, tmp_path):
    repo = _make_repo(tmp_path, REQUIRED_FILES)
    settings = dataclasses.replace(base_settings, model_path_local=str(repo))
    assert loader.resolve_model_path(settings) == str(repo)


@pytest.mark.parametrize(
    "missing",
    ["config.json", "model.safetensors.index.json", "tokenizer.json"],
)
def test_missing_required_file_raises(
    loader, base_settings, tmp_path, missing
):
    """Snapshot tải dở phải dừng NGAY.

    Đây chính là chế độ hỏng mà preflight sinh ra để bắt: config.json có mà
    shard hoặc tokenizer thì thiếu, và lỗi chỉ lộ ra sau hàng chục giây nạp
    model.
    """
    repo = _make_repo(tmp_path, REQUIRED_FILES)
    (repo / missing).unlink()
    settings = dataclasses.replace(base_settings, model_path_local=str(repo))
    with pytest.raises(FileNotFoundError):
        loader.resolve_model_path(settings)


def test_falls_back_to_hub(loader, base_settings):
    assert loader.resolve_model_path(base_settings) == base_settings.base_model


def test_special_tokens_are_attached(loader):
    """5 token này không nằm trong tokenizer_config.json.

    vendor/pipeline.py đọc thẳng chúng trên tokenizer; quên gắn thì lỗi là
    AttributeError giữa lúc build sample, SAU khi đã nạp xong 10GB weight.
    """

    class FakeTokenizer:
        pass

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

    processor = FakeProcessor()
    loader.attach_special_tokens(processor)
    for attr, value in SPECIAL_TOKENS.items():
        assert getattr(processor.tokenizer, attr) == value
