"""Tầng chạy HiDream-O1-Image (SDNQ 4-bit).

Thay cho ``models/loader.py`` + ``models/offload.py`` + ``inference.py`` cũ
(đường diffusers + Nunchaku). Xem ``docs/MODEL_HIDREAM_O1.md``.

    model, processor = load_model(settings, device)
    recipe = build_recipe(settings)
    image, info = generate(model, processor, [], "a red apple", recipe)

Các symbol nặng (``generate``, ``load_model``, …) được nạp LƯỜI qua PEP 562.
Lý do: ``loader``/``inference`` import torch + transformers + sdnq, cộng lại
vài giây và ~3GB wheel. UI, queue config, preflight và test chỉ cần bảng độ
phân giải trong ``resolutions`` (thuần stdlib) — bắt chúng kéo cả CUDA chỉ
để đọc một tuple là sai, và làm test không chạy nổi trên máy không GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .resolutions import (
    MIN_SHORT_SIDE,
    PREDEFINED_RESOLUTIONS,
    find_closest_resolution,
    size_for_aspect_ratio,
)

if TYPE_CHECKING:  # pragma: no cover - chỉ để type checker nhìn thấy
    from .inference import (
        Recipe,
        build_recipe,
        generate,
        resolve_size,
        to_pil,
    )
    from .loader import load_model, resolve_model_path

# Tên symbol → module con chứa nó.
_LAZY = {
    "Recipe": "inference",
    "build_recipe": "inference",
    "generate": "inference",
    "resolve_size": "inference",
    "to_pil": "inference",
    "load_model": "loader",
    "resolve_model_path": "loader",
}

__all__ = [
    "MIN_SHORT_SIDE",
    "PREDEFINED_RESOLUTIONS",
    "Recipe",
    "build_recipe",
    "find_closest_resolution",
    "generate",
    "load_model",
    "resolve_model_path",
    "resolve_size",
    "size_for_aspect_ratio",
    "to_pil",
]


def __getattr__(name: str):
    """Nạp lười submodule nặng khi symbol được dùng lần đầu."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # cache: lần sau không qua __getattr__ nữa
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
