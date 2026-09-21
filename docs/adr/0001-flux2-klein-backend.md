# ADR 0001 — Backend sinh ảnh: FLUX.2-klein-4B

- **Trạng thái:** Đã thực thi ở tầng code · **Phase 0 CHƯA nghiệm thu**
- **Ngày:** 2026-09-21
- **Thay thế:** Qwen-Image-Edit-2509 Lightning (INT4 / Nunchaku)
- **Kế hoạch:** [REFACTOR_FLUX2_KLEIN_4B.md](../REFACTOR_FLUX2_KLEIN_4B.md)

---

## 1. Quyết định

Chuyển toàn bộ backend sinh ảnh sang `black-forest-labs/FLUX.2-klein-4B`
(bản **step-distilled**, 4 bước, `guidance_scale=1.0`), chạy `bf16` thường
trú trên một GPU L4 24GB. Nhánh GGUF (`unsloth/FLUX.2-klein-4B-GGUF`) được
hỗ trợ nhưng **không phải mặc định**.

## 2. Lý do

Backend cũ cần **26.4 GiB** weight thường trú (text_encoder Qwen2.5-VL 15.4 +
transformer INT4 10.7 + vae 0.24) — không vừa 24 GiB của L4. Toàn bộ tầng
`models/offload.py` (4 chiến lược, ~200 dòng) tồn tại chỉ để né giới hạn đó,
với giá là ~15.4 GiB đi qua PCIe mỗi request.

FLUX.2-klein-4B dùng text encoder **Qwen3-4B text-only** (~8 GiB) thay vì bản
VL, nên tổng chỉ ~16 GiB — vừa L4 với chỗ dư cho activation. Đó là toàn bộ
luận điểm: **model mới vừa GPU, nên mọi cơ chế offload trở thành mã chết.**

## 3. Vì sao bf16 là mặc định chứ không phải GGUF

Dù người đặt hàng chỉ đích danh repo GGUF, bf16 được chọn làm mặc định:

| | bf16 | GGUF Q8_0 |
|---|---|---|
| Transformer trên đĩa | 7.75 GB | 4.3 GB |
| Tổng VRAM | ~16 GB | ~12.6 GB |
| Vừa L4 24GB | ✅ | ✅ |
| Chi phí giải nén trong forward | không | **có** |
| `torch.compile` | ✅ | ❌ ([#10795][10795]) |

GGUF trong diffusers **giải nén on-the-fly về `compute_dtype`** mỗi lần
forward — nó đổi *dung lượng* lấy *băng thông*, mà băng thông (300 GB/s trên
L4) mới là nút cổ chai. Vì bf16 *đã vừa*, đổi lấy dung lượng là đổi lấy thứ
ta không thiếu.

GGUF vẫn được giữ cho đúng một mục đích: ép đủ chỗ cho **2 worker process
trên cùng một L4** (2 × 16 GB không vừa, 2 × Q4_K_M ~9 GB thì vừa). Đây là
đánh đổi latency lấy throughput và **phải quyết bằng số đo**, không phải
bằng suy đoán.

## 4. Rủi ro đã né được ở tầng code

`Flux2Transformer2DModel.from_single_file()` với GGUF klein từng nổ shape
mismatch vì state-dict rỗng bị khởi tạo theo chiều của FLUX.2 **dev**
([#13001][13001]). Đọc mã `diffusers/loaders/single_file_model.py` (0.40.0)
cho thấy đường né: truyền `config=<repo klein>` + `subfolder="transformer"`
sẽ **bỏ qua hẳn** `fetch_diffusers_config()` — tức bỏ qua đúng bước đoán sai.
`models/loader.py::_load_gguf_transformer` làm đúng vậy.

**Đây là suy luận từ mã nguồn, chưa phải quan sát trên GPU.** Tiêu chí C của
Phase 0 tồn tại để xác nhận nó.

## 5. Kết quả Phase 0 — ⚠️ CHƯA CHẠY

Máy phát triển dùng cho refactor này là **RTX 3080 10GB**, không đủ VRAM cho
bf16 (~16 GB). Không có số đo nào ở đây được lấy từ phần cứng thật.

> **Cập nhật 2026-09-21:** ADR này viết cho bản **4B**. Dự án đã chuyển sang
> **9B**, nơi text encoder là Qwen3-8B (16.4 GiB bf16) chứ không phải
> Qwen3-4B (8 GiB). Mọi con số VRAM trong §2/§3 ở trên thuộc về 4B và đã bị
> thay thế — xem [PERFORMANCE.md §0](../PERFORMANCE.md). Tiêu chí A và B
> dưới đây (đặt quanh giả định bf16 là mặc định) không còn áp dụng: ở 9B,
> bf16 không vừa L4 dù text encoder đã nén.

Chạy trên L4 rồi điền bảng dưới:

```bash
python scripts/spike_flux2.py --modes gguf fp8 --runs 3
python scripts/spike_flux2.py --modes fp8 --compile --runs 3
```

| # | Tiêu chí | Kết quả | Số đo |
|---|---|---|---|
| A | bf16 chạy được, peak VRAM < 20 GiB | ❌ **không áp dụng ở 9B** | bf16 + NF4 = ~23.3 GiB > 22.5 |
| B | Latency bf16 ≤ 8s @1024²/4 bước | ❌ **không áp dụng ở 9B** | |
| C | GGUF load được (không dính #13001) | ✅ **ĐẠT** | Colab L4, Q4_K_M, load 81.9s |
| D | GGUF nhanh hơn hoặc bằng bf16 | ⬜ chưa đo | GGUF t2i: **10.36s** median (3 lượt) |
| E | `torch.compile` không recompile mỗi shape | ⬜ chưa đo | chỉ kiểm được ở nhánh fp8/bf16 |
| F | Ảnh edit hợp lý bằng mắt | ⬜ chưa đo | ảnh t2i đã sinh, chưa ai xem |
| G | FP8 nhanh hơn GGUF | ⬜ chưa đo | lần chạy đầu trượt vì thiếu `ninja` |
| H | **Vì sao service chậm hơn spike 75%?** | 🔴 **ĐANG MỞ** | spike 10.36s vs Gradio 18.1s, cùng 1024²/4 bước/GGUF |

**Peak VRAM đo được:** 13.81 GiB (GGUF Q4_K_M + text encoder NF4) trên L4
22.5 GiB — khớp ước tính ~11.2 GiB weight cộng ~2.6 GiB activation.

**Tiêu chí H là thứ đáng đuổi trước FP8.** Spike gọi thẳng
`Flux2KleinPipeline` với prompt ngắn; service gọi qua `inference.generate`
với `prompt_embeds` đã cache, `max_sequence_length=512` và prompt demo dài
~1840 ký tự. Một trong những khác biệt đó đang tốn 7.7s — nhiều hơn phần
FP8 có thể lấy lại. Đường tách: `python scripts/benchmark.py --mode t2i`
chạy ĐÚNG đường service nhưng với prompt ngắn.

**Nếu C trượt:** đặt `quantization: "bf16"` (đã là mặc định) và ghi rõ ở đây
rằng nhánh GGUF không dùng được với phiên bản diffusers đang ghim. Không cần
đổi code — nhánh đó chỉ được chạm tới khi config yêu cầu.

**Nếu A trượt:** đây là tình huống duy nhất không có đường lùi tốt; phải
tính lại lựa chọn model.

## 6. Hằng số phụ thuộc bảng trên

Các giá trị dưới đây đang là **giả định**, phải trích nguồn từ §5 sau khi đo:

| Hằng số | Nơi đặt | Giá trị hiện tại | Cơ sở |
|---|---|---|---|
| `quantization` | `base.example.yaml` | `bf16` | suy luận §3 |
| `offload` | `base.example.yaml` | `resident` | phép tính VRAM §2 |
| `compile_transformer` | `base.example.yaml` | `false` | thận trọng (E chưa đo) |
| `num_steps` | `base.example.yaml` | `4` | model card |
| `guidance_scale` | `base.example.yaml` | `1.0` | model card |
| `_EXPECTED_MAX_GB` | `scripts/preflight.py` | `24.0` | phép tính VRAM §2 |

## 7. Hệ quả

**Được:**
- Xoá `models/offload.py` (−200 dòng) và `tests/test_offload_strategy.py`.
- Xoá `nunchaku` cùng khối `[tool.uv.sources]` ghim 4 URL wheel theo
  `python × torch × CUDA` → `diffusers` được nâng tự do (0.36 → 0.40).
- Một pipeline thay vì hai (`Flux2KleinPipeline` nhận `image=None`), nên
  `build_t2i_pipeline()` và `generate_t2i()` biến mất.
- Cache prompt-embed khoá theo **text**, không còn băm pixel ảnh → đổi ảnh
  mà giữ prompt giờ ăn cache.
- Hết monkey-patch `VAE_IMAGE_SIZE` và hết import module *internal* của
  diffusers (`pipelines.qwenimage`) — chính thứ từng ghim repo vào 0.36.

**Mất:**
- `negative_prompt` **không còn tác dụng** ở cấu hình mặc định: bản distilled
  nhúng guidance vào model nên diffusers tắt hẳn CFG. Hợp đồng inbound không
  đổi (BE vẫn gửi field đó), service ghi log khi bỏ qua. Xem
  `inference._negative_embeds`.
- Prompt trong `ui/prompts.py` được tinh chỉnh cho Qwen, **chưa đánh giá lại**
  trên FLUX.2 — xem §2.4 của kế hoạch.
- Tiền tố env `QIE_*` bị bỏ hẳn, không có lớp tương thích. Cố ý: env thắng
  YAML, nên một `QIE_TRANSFORMER_PATH` còn sót sẽ âm thầm ghi đè cấu hình
  đúng. `tests/test_config.py::test_no_stale_qie_prefix_is_honoured` giữ
  điều này.

**Không đổi:**
- Hợp đồng RabbitMQ inbound/outbound. `tests/test_messaging_schemas.py` pass
  nguyên văn, không sửa một dòng — đó là bằng chứng.

[13001]: https://github.com/huggingface/diffusers/issues/13001
[10795]: https://github.com/huggingface/diffusers/issues/10795
