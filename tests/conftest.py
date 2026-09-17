"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from imagegen.config import DEFAULT_MODEL_REPO, Settings


@pytest.fixture
def base_settings() -> Settings:
    """A fully populated default Settings instance for unit tests.

    Settings là frozen dataclass không có giá trị mặc định, nên mọi field
    phải được liệt kê ở đây. Thêm field mới vào Settings mà quên cập nhật
    fixture này thì toàn bộ test dùng nó sẽ đổ TypeError.
    """
    return Settings(
        model_root="models",
        base_model=DEFAULT_MODEL_REPO,
        model_path_local=None,
        model_type="full",
        num_steps=50,
        guidance_scale=5.0,
        shift=3.0,
        scheduler_name="default",
        width=2048,
        height=2048,
        device_override=None,
        warmup=False,
        server_name="0.0.0.0",
        server_port=7860,
        demo_cache=True,
        # Field tốc độ: giữ ĐÚNG default của dataclass để test recipe đo
        # được hành vi mặc định, không phải một cấu hình "nhanh" nào đó.
        attention_mode="auto",
        dequantize="auto",
        compile_model=False,
        cfg_interval_start=0.0,
        cfg_interval_end=1.0,
        snap_resolution=True,
        warmup_steps=2,
    )


@pytest.fixture
def recipe_mod(monkeypatch):
    """Import ``hidream.inference`` với torch/PIL được stub.

    Module đó import torch chỉ để dùng trong ``generate()``; ``build_recipe``
    thì thuần Python. Stub để test chạy trên CI không GPU.
    """
    import sys
    import types

    torch = types.ModuleType("torch")
    torch.inference_mode = lambda: types.SimpleNamespace(
        __enter__=lambda s: None, __exit__=lambda s, *a: False
    )
    torch.randint = lambda *a, **k: types.SimpleNamespace(item=lambda: 123)
    monkeypatch.setitem(sys.modules, "torch", torch)

    for name in ("einops", "numpy", "tqdm", "torchvision", "diffusers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    vendor_pipeline = types.ModuleType("imagegen.hidream.vendor.pipeline")
    vendor_pipeline.DEFAULT_TIMESTEPS = [999, 500, 8]
    vendor_pipeline.generate_image = lambda **kw: None
    monkeypatch.setitem(
        sys.modules, "imagegen.hidream.vendor.pipeline", vendor_pipeline
    )
    monkeypatch.delitem(
        sys.modules, "imagegen.hidream.inference", raising=False
    )

    from imagegen.hidream import inference

    return inference
