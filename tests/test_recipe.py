"""``build_recipe`` — quyết định tham số phụ thuộc của một lượt sinh ảnh.

Đây là logic dễ sai nhất trong tầng lõi vì các tham số của HiDream-O1 KHÔNG
độc lập: chọn scheduler mà quên `timesteps_list`, hoặc chọn `flash` mà quên
`noise_scale_*`, thì scheduler chạy sai lịch **mà không báo lỗi** — ảnh ra
vẫn là một tấm ảnh, chỉ là sai. Không có test nào bắt được chuyện đó ngoài
những test ở đây.
"""

from __future__ import annotations

import dataclasses

import pytest

# Fixture ``recipe_mod`` (stub torch rồi import hidream.inference) nằm ở
# tests/conftest.py — tests/test_speed_path.py dùng chung.


def test_default_scheduler_has_no_fixed_timesteps(recipe_mod, base_settings):
    """Bản `full` để scheduler tự sinh timestep theo `num_steps`.

    Truyền nhầm bảng 28 mốc của bản dev vào đây sẽ ép 50 bước chạy trên lịch
    28 mốc.
    """
    r = recipe_mod.build_recipe(base_settings)
    assert r.scheduler_name == "default"
    assert r.timesteps_list is None
    assert r.extra == {}
    assert r.num_steps == 50


@pytest.mark.parametrize("scheduler", ["flow_match", "flash"])
def test_dev_schedulers_get_fixed_timesteps(
    recipe_mod, base_settings, scheduler
):
    """Hai scheduler của bản dev BẮT BUỘC dùng bảng 28 mốc cố định."""
    settings = dataclasses.replace(
        base_settings, scheduler_name=scheduler, num_steps=28
    )
    r = recipe_mod.build_recipe(settings)
    assert r.timesteps_list == recipe_mod.DEFAULT_TIMESTEPS


def test_flash_scheduler_gets_noise_params(recipe_mod, base_settings):
    """`flash` lấy mẫu lại noise mỗi bước — thiếu 3 tham số này là chạy sai."""
    settings = dataclasses.replace(base_settings, scheduler_name="flash")
    r = recipe_mod.build_recipe(settings)
    assert set(r.extra) == {
        "noise_scale_start",
        "noise_scale_end",
        "noise_clip_std",
    }


def test_non_flash_schedulers_get_no_noise_params(recipe_mod, base_settings):
    """``generate_image`` không nhận noise_* khi scheduler khác flash.

    Truyền thừa vào là TypeError ngay lúc chạy, sau khi đã nạp xong model.
    """
    for scheduler in ("default", "flow_match"):
        settings = dataclasses.replace(base_settings, scheduler_name=scheduler)
        assert recipe_mod.build_recipe(settings).extra == {}


@pytest.mark.parametrize(
    "override,expected",
    [(0, 50), (-1, 50), (None, 50), (28, 28), (4, 4)],
)
def test_num_steps_override(recipe_mod, base_settings, override, expected):
    """0/âm/None nghĩa là "dùng mặc định" — đó là quy ước của message queue."""
    r = recipe_mod.build_recipe(base_settings, num_steps=override)
    assert r.num_steps == expected


def test_recipe_is_immutable(recipe_mod, base_settings):
    """Recipe frozen: hai request chạy song song không giẫm lên nhau."""
    r = recipe_mod.build_recipe(base_settings)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.num_steps = 1


def test_recipe_carries_cfg_and_shift(recipe_mod, base_settings):
    settings = dataclasses.replace(
        base_settings, guidance_scale=0.0, shift=1.0
    )
    r = recipe_mod.build_recipe(settings)
    assert r.guidance_scale == 0.0
    assert r.shift == 1.0
