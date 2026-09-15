"""Bảng quyết định của chiến lược offload.

``choose_strategy`` là hàm thuần nên kiểm được toàn bộ tổ hợp phần cứng mà
không cần GPU thật.
"""
from __future__ import annotations

import pytest

from qwen_lightning.models.offload import choose_strategy


@pytest.mark.parametrize(
    "gpu_mem,has_second_gpu,expected",
    [
        # A100-40GB một card: get_gpu_memory() trả 39 (làm tròn xuống GiB).
        # Weight resident 26.4 GiB nên phải giữ nguyên trên GPU, KHÔNG offload.
        (39, False, "none"),
        (79, False, "none"),
        # Đúng ngưỡng.
        (32, False, "none"),
        (31, False, "model"),
        # 24GB một card: không đủ chỗ cho text_encoder + transformer.
        (23, False, "model"),
        # Hai card: split luôn thắng, kể cả khi mỗi card nhỏ.
        (23, True, "split"),
        (39, True, "split"),
        # Card quá nhỏ.
        (11, False, "sequential"),
        # Không dò được VRAM mà chỉ có một card: chọn phương án an toàn nhất.
        (None, False, "sequential"),
        # Không dò được VRAM nhưng có hai card: split vẫn hợp lệ.
        (None, True, "split"),
    ],
)
def test_auto_resolution(gpu_mem, has_second_gpu, expected):
    assert choose_strategy("auto", gpu_mem, has_second_gpu) == expected


@pytest.mark.parametrize(
    "strategy", ["none", "split", "model", "sequential"]
)
def test_explicit_strategy_is_never_overridden(strategy):
    """Chọn tay thì phải được tôn trọng, bất kể phần cứng dò ra là gì."""
    assert choose_strategy(strategy, 39, False) == strategy
    assert choose_strategy(strategy, 11, True) == strategy


def test_strategy_is_case_insensitive():
    assert choose_strategy("AUTO", 39, False) == "none"
    assert choose_strategy("Model", 39, False) == "model"


def test_a100_40gb_does_not_fall_back_to_cpu_offload():
    """Chống hồi quy cho lỗi tốn ~35s/ảnh.

    Logic cũ gắn "đủ VRAM" với "có 2 GPU", nên A100-40GB một card rơi xuống
    enable_model_cpu_offload và phải nạp ~27GB qua PCIe mỗi lần sinh ảnh.
    """
    assert choose_strategy("auto", 39, False) != "model"
