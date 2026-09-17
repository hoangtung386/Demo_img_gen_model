# Nguồn vendor

| | |
|---|---|
| Repo | https://github.com/HiDream-ai/HiDream-O1-Image |
| Commit | `2c2d29ff729e48f33e41f49edfdbd81d5ac103b4` |
| Ngày copy | 2026-09-17 |
| License | MIT (Copyright (c) 2026 HiDream.ai) |

File copy từ `models/` của repo gốc:

```
pipeline.py  qwen3_vl_transformers.py  utils.py
fm_solvers_unipc.py  flash_scheduler.py
```

`__init__.py` là của dự án này, không phải bản upstream.

## Patch đã áp — KHÔNG được mất khi cập nhật vendor

Tất cả nằm trong `pipeline.py`, đánh dấu bằng comment `# PATCH <n>`.

| # | Chỗ | Sửa | Vì sao |
|---|---|---|---|
| 1 | `forward_once()` | `"use_flash_attn": True` → `_USE_FLASH_ATTN` (đọc env `IMG_USE_FLASH_ATTN`, mặc định tắt) | `flash-attn` không có trong image runtime. Hard-code `True` làm inference nổ ở forward đầu tiên với `AssertionError: Flash attention is not available` — sau khi đã nạp xong 10GB weight |
| 2 | `generate_image()` ×2 | `Image.open(p)` → `_as_pil(p)` | Tầng Gradio đưa `PIL.Image` vào. Không patch thì mọi request phải ghi ảnh ra đĩa rồi đọc lại chỉ để thoả chữ ký hàm |
| 3 | `generate_image()` ×4 | `print()` → `logger` (`logging.getLogger("hidream-o1")`) | `print()` không đi qua `logging_setup.py`; log của queue service mất các dòng này |

Từ patch 4 trở đi có cả trong `qwen3_vl_transformers.py`. Đây là các patch
**tốc độ** — xem [docs/PERFORMANCE.md](../../../../docs/PERFORMANCE.md) cho số
đo và cách tắt từng cái.

| # | File / chỗ | Sửa | Vì sao |
|---|---|---|---|
| 4 | `qwen3_vl_transformers.py`: `_block_prefix_sdpa()`, `Qwen3VLModel._run_decoder_sdpa()`, `_ar_tokens_are_a_prefix()`, nhánh chọn đường trong `_forward_generation()`, tham số `attn_mode` trên hai `forward()` | Thêm đường attention thứ ba: tách hai lượt SDPA thay cho mask 4D dày đặc | Mask `[B, 1, S, S]` tường minh **cấm** PyTorch dùng backend flash của SDPA. Nó rơi về mem-efficient, nơi mask bị expand ra `[B, heads, S, S]` (~1.1GB ở 2048²) và copy contiguous lại ở **mỗi layer, mỗi bước** — 36 × 100 lần một ảnh. Mask đó có cấu trúc "tiền tố AR causal + phần còn lại bidirectional" nên tách được thành hai lời gọi SDPA **không mask**, kết quả y hệt. Khác `_run_decoder_flash` ở chỗ không cần wheel `flash-attn` |
| 5 | `pipeline.py`: `generate_image()` nhận `attn_mode`, `cfg_interval`, `snap_resolution`; `ref_max_size` | Ba knob tốc độ cho tầng gọi | `snap_resolution=False` bỏ chặn sàn 2048² (rẻ hơn nhiều, nhưng ngoài phân bố huấn luyện); `ref_max_size` là knob DUY NHẤT hạ được chi phí đường editing |
| 6 | `pipeline.py`: `_cfg_on[]` + hai nhánh dùng nó | Bỏ forward pass uncond ở các bước ngoài `cfg_interval` | CFG là đúng 50% chi phí. "Guidance interval" (arXiv:2404.07724) cho thấy guidance gần như vô dụng ở hai đầu quỹ đạo |

`qwen3_vl_transformers.py` còn 4 lời gọi `print()` nhưng đều nằm sau cờ
`DEBUG_MEM=1` nên không đụng tới.

## Patch 4 đúng tới mức nào

`tests/test_speed_path.py::test_block_prefix_sdpa_matches_dense_mask` so
`_block_prefix_sdpa()` với một bản attention viết tay dùng **đúng** mask 4D
của upstream, ở fp32 trên CPU. Đây không phải test cho có: đường nhanh sai
thì model vẫn chạy và vẫn trả về một tấm ảnh — chỉ là ảnh sai, không
exception nào.

Phép tách chỉ đúng khi token AR là **tiền tố liên tục** của chuỗi. Cả hai
đường của `generate_image` đều thoả (mọi token gen — tms, ảnh đích, ref
patch — đều nằm sau phần text). `_ar_tokens_are_a_prefix()` kiểm tra điều đó
ở mỗi forward và rơi về đường mask nếu không thoả, nên một thay đổi ở
`build_t2i_text_sample()` không thể lặng lẽ làm hỏng kết quả.

## Cách cập nhật

```bash
scripts/vendor_hidream.sh <commit-sha-moi>
# rồi áp lại 6 patch trên, cập nhật dòng Commit + Ngày copy ở đầu file này
pytest tests/test_speed_path.py   # patch 4 còn đúng không
```
