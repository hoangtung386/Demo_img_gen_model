"""Các đường tối ưu tốc độ — cái nào "nhanh hơn" phải vẫn là cái cũ.

Hai nhóm test, hai kiểu hỏng khác nhau:

* ``forward_passes`` / ``cfg_interval``: số học thuần. Hỏng ở đây nghĩa là
  benchmark báo sai đơn giá và ``avg_inference_seconds`` trong base.yaml đi
  theo — backlog queue phình mà không ai biết vì sao.
* ``_block_prefix_sdpa``: ĐÂY mới là chỗ đáng sợ. Nó thay một mask 4D dày
  đặc bằng hai lời gọi SDPA không mask. Sai thì model VẪN CHẠY và VẪN trả
  về một tấm ảnh — chỉ là ảnh sai, không có exception nào. Test so trực
  tiếp với phép attention dùng đúng mask cũ.
"""

from __future__ import annotations

import dataclasses
import types

import pytest


# ---------------------------------------------------------------------------
# Chi phí một lượt sinh ảnh
# ---------------------------------------------------------------------------
def test_forward_passes_counts_cfg(recipe_mod, base_settings):
    """Một bước CÓ CFG tốn HAI forward pass, không phải một.

    ``num_steps`` là thứ hiện trên UI; ``forward_passes`` là thứ GPU phải
    trả. Nhầm hai con số này là nhầm gấp đôi.
    """
    recipe = recipe_mod.build_recipe(base_settings)
    assert recipe.num_steps == 50
    assert recipe.forward_passes == 100


def test_forward_passes_without_cfg(recipe_mod, base_settings):
    """guidance_scale <= 1.0 → không có nhánh uncond, một pass mỗi bước."""
    settings = dataclasses.replace(base_settings, guidance_scale=0.0)
    recipe = recipe_mod.build_recipe(settings)
    assert recipe.forward_passes == recipe.num_steps


@pytest.mark.parametrize(
    "lo,hi,expected",
    [
        (0.0, 1.0, 100),   # mặc định: không bỏ bước nào
        (0.0, 0.5, 75),    # bỏ CFG ở nửa cuối → 25/50 bước còn uncond
        (0.0, 0.0, 51),    # chỉ bước đầu có CFG
        (0.5, 1.0, 75),    # bỏ CFG ở nửa đầu
    ],
)
def test_cfg_interval_reduces_forward_passes(
    recipe_mod, base_settings, lo, hi, expected
):
    settings = dataclasses.replace(
        base_settings, cfg_interval_start=lo, cfg_interval_end=hi
    )
    assert recipe_mod.build_recipe(settings).forward_passes == expected


def test_default_settings_do_not_change_behaviour(recipe_mod, base_settings):
    """Default của mọi knob tốc độ phải là "y như trước khi có knob".

    Chốt chặn cho cách hỏng khó chịu nhất của một đợt tối ưu: có người đặt
    default sang chế độ nhanh-nhưng-xấu rồi cả tháng sau mới có ai nhận ra
    ảnh PRD đã khác.
    """
    recipe = recipe_mod.build_recipe(base_settings)
    assert recipe.cfg_interval == (0.0, 1.0)
    assert recipe.snap_resolution is True
    assert recipe.forward_passes == 2 * recipe.num_steps


# ---------------------------------------------------------------------------
# Tương đương của đường attention nhanh
# ---------------------------------------------------------------------------
def _reference_attention(torch, q, k, v, n_ar):
    """Attention theo ĐÚNG mask 4D mà upstream dựng trong _forward_generation.

    Chép lại có chủ ý thay vì gọi vào vendor: nếu ai đó sửa cả hai bên cùng
    lúc thì test này phải fail, không phải trôi theo.
    """
    import math

    seq = q.shape[2]
    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    min_val = torch.finfo(q.dtype).min
    mask = torch.full((seq, seq), min_val, dtype=q.dtype)
    mask = torch.triu(mask, diagonal=1)
    mask[n_ar:, :] = 0  # token gen nhìn thấy tất cả

    scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    scores = scores + mask
    return torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype) @ v


@pytest.mark.parametrize("n_ar", [0, 1, 7, 23])
def test_block_prefix_sdpa_matches_dense_mask(n_ar):
    """Đường nhanh phải cho ra CÙNG tensor với mask 4D của upstream.

    Chạy fp32 trên CPU: ở đây kiểm tra tính ĐÚNG của phép tách hai lượt,
    không phải tốc độ. Sai lệch duy nhất được phép là sai số dấu phẩy động.
    """
    torch = pytest.importorskip("torch")
    # Module vendor kéo theo transformers (nó là bản sửa của modeling code
    # Qwen3-VL). Bỏ qua khi thiếu — trên máy đã `uv sync` thì luôn có.
    pytest.importorskip("transformers")
    from imagegen.hidream.vendor.qwen3_vl_transformers import (
        _block_prefix_sdpa,
    )

    torch.manual_seed(0)
    batch, heads, kv_heads, seq, dim = 1, 4, 2, 40, 16
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, kv_heads, seq, dim)
    v = torch.randn(batch, kv_heads, seq, dim)

    fast = _block_prefix_sdpa(q, k, v, n_ar)
    reference = _reference_attention(torch, q, k, v, n_ar)

    assert fast.shape == reference.shape
    torch.testing.assert_close(fast, reference, rtol=1e-4, atol=1e-5)


def test_ar_prefix_detection_rejects_interleaved_layout():
    """Token AR xen giữa token gen → PHẢI rơi về đường mask.

    Phép tách hai lượt chỉ đúng khi token AR là tiền tố liên tục. Đường
    sinh ảnh hiện tại luôn thoả, nhưng một thay đổi ở ``build_*_sample``
    có thể phá điều đó trong im lặng — nên việc nhận biết phải được kiểm.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from imagegen.hidream.vendor.qwen3_vl_transformers import Qwen3VLModel

    check = Qwen3VLModel._ar_tokens_are_a_prefix

    # 3 token AR rồi 5 token gen — layout thật của pipeline.
    ok, n_ar = check(torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]]))
    assert (ok, n_ar) == (True, 3)

    # Token AR nằm xen giữa → không áp dụng được.
    ok, _ = check(torch.tensor([[0, 1, 0, 1, 1]]))
    assert ok is False

    # Toàn bộ là token gen (không có phần text) vẫn hợp lệ.
    ok, n_ar = check(torch.tensor([[1, 1, 1, 1]]))
    assert (ok, n_ar) == (True, 0)


def _tiny_generation_model(torch):
    """Một Qwen3VL cỡ đồ chơi đủ để chạy đường sinh ảnh thật.

    538K tham số, 2 layer, hidden 64 — chạy trên CPU trong chớp mắt. Mục
    đích không phải sinh ảnh đẹp mà là cho HAI đường attention cùng đi qua
    đúng những module của production: q_norm/k_norm, RoPE 3D, o_proj,
    deepstack, final norm.
    """
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    from imagegen.hidream.vendor.qwen3_vl_transformers import (
        Qwen3VLForConditionalGeneration,
    )

    text = Qwen3VLTextConfig(
        vocab_size=200,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA thật, như model thật (32/8)
        head_dim=16,
        rope_scaling={
            "rope_type": "default",
            # mrope_section cộng lại phải bằng head_dim/2.
            "mrope_section": [4, 2, 2],
            "mrope_interleaved": True,
        },
    )
    vision = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=64,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=64,
    )
    config = Qwen3VLConfig(
        text_config=text.to_dict(), vision_config=vision.to_dict()
    )
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).eval()


def test_sdpa_and_mask_paths_give_the_same_prediction():
    """Đường nhanh và đường gốc phải cho ra CÙNG ``x_pred``.

    Test trên chạy hết cả `_forward_generation`, không chỉ phép attention:
    nó bắt được cả lỗi lắp ráp (nhầm thứ tự transpose, quên q_norm, RoPE sai
    trục, bỏ sót o_proj) — những lỗi mà test đơn vị của
    ``_block_prefix_sdpa`` không thấy vì nó nhận q/k/v đã dựng sẵn.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    model = _tiny_generation_model(torch)

    txt_len, img_len = 5, 9
    total = txt_len + img_len
    inputs = {
        "input_ids": torch.randint(0, 200, (1, txt_len)),
        "position_ids": torch.arange(total)
        .view(1, 1, -1)
        .expand(3, 1, -1)
        .contiguous(),
        # 3 kênh × patch 32×32
        "vinputs": torch.randn(1, img_len, 3 * 32 * 32),
        "timestep": torch.tensor([0.4]),
        "token_types": torch.zeros(1, total, dtype=torch.long),
    }
    # Layout của pipeline: tms token + toàn bộ token ảnh là "gen".
    inputs["token_types"][0, txt_len - 1:] = 1

    with torch.no_grad():
        with_mask = model(**inputs, attn_mode="mask").x_pred
        with_sdpa = model(**inputs, attn_mode="sdpa").x_pred

    assert with_sdpa.shape == with_mask.shape
    torch.testing.assert_close(with_sdpa, with_mask, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Hạ scale của SDNQ về bf16
# ---------------------------------------------------------------------------
def test_shrink_scales_keeps_parameters_as_parameters():
    """``scale`` của SDNQ là ``nn.Parameter`` — gán Tensor trần vào là nổ.

    ``nn.Module.__setattr__`` ném ``TypeError: cannot assign ... as
    parameter`` cho một tên đang nằm trong ``_parameters``. Nó ném lúc NẠP
    MODEL, tức sau khi đã đọc xong ~10GB weight — kiểu lỗi đắt nhất để phát
    hiện bằng tay.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("sdnq")  # loader import nó ở module level
    from imagegen.hidream.loader import _shrink_scales

    class FakeSDNQLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sdnq_dequantizer = types.SimpleNamespace(
                result_dtype=torch.bfloat16
            )
            self.scale = torch.nn.Parameter(
                torch.ones(4, 1, dtype=torch.float32), requires_grad=False
            )
            self.zero_point = torch.nn.Parameter(
                torch.zeros(4, 1, dtype=torch.float32), requires_grad=False
            )

    layer = FakeSDNQLinear()
    assert _shrink_scales([layer]) == 1

    for name in ("scale", "zero_point"):
        tensor = getattr(layer, name)
        assert tensor.dtype is torch.bfloat16
        # Vẫn phải là Parameter, nếu không state_dict và .to(device) sẽ bỏ sót.
        assert isinstance(tensor, torch.nn.Parameter), name
        assert name in layer._parameters
        assert tensor.requires_grad is False


def test_shrink_scales_can_be_disabled(monkeypatch):
    """``IMG_SDNQ_FP32_SCALES=true`` là đường lui để đối chiếu numerics."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("sdnq")
    from imagegen.hidream.loader import _shrink_scales

    monkeypatch.setenv("IMG_SDNQ_FP32_SCALES", "true")
    assert _shrink_scales([object()]) == 0
