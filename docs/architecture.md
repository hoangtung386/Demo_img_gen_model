# Kiến Trúc Dự Án (Architecture)

Tài liệu dành cho team tiếp nhận, mô tả trách nhiệm từng module và luồng dữ
liệu của service sinh ảnh `FLUX.2-klein-9B`.

## Nguyên tắc

- **Mỗi module một trách nhiệm** (config / device / loader / placement /
  inference / ui).
- **Một nguồn cấu hình duy nhất** (`src/gen_image/config.py`), dùng chung cho
  cả tải trọng số (`download`) và chạy service (`serve`, `main_queue`).
- **Logic thuần tách khỏi GPU/UI** để dễ test (config, path resolution,
  `calculate_dimensions`, cache prompt-embed).
- **Ranh giới `ImageGenerator.generate()` là bất biến.** Toàn bộ tầng queue
  (messaging, retry/DLQ, audit, storage, worker pool) không biết model nào
  đang chạy phía dưới — đổi backend không được chạm vào nó.

## Sơ đồ module

```
scripts/serve.py ──► gen_image.serve.main
                        │
                        ▼
                   ui.launch.main
                   ├─ config.load_settings()      # base.yaml + env GENIMG_*
                   ├─ device.select_device()      # chọn cuda
                   ├─ models.loader.load_pipeline # Flux2KleinPipeline
                   │     └─ models.placement.place    # resident | model_offload
                   ├─ tuning.maybe_compile()      # torch.compile (bf16 only)
                   ├─ tuning.warmup()             # t2i + edit
                   ├─ ui.app.build_ui()           # Gradio Blocks (KHÔNG cần model)
                   └─ ui.launch._launch_with_public_link()  # frpc + *.gradio.live

main_queue.py ──► gen_image.queue_service.Application
                   ├─ config.load_config()        # base.yaml (KHÔNG đọc env GENIMG_*)
                   ├─ processors.create_processor # Flux2KleinGenerator
                   │     └─ (dùng lại models.loader.load_pipeline)
                   └─ workers/ messaging/ audit/ storage/

scripts/download_model.py ──► gen_image.download.main
                                ├─ config.load_settings()
                                ├─ download.download()   # snapshot + (tuỳ chọn) gguf
                                └─ ghi .model_paths.env  # path local cho serve
```

## Trách nhiệm module

| Module | Trách nhiệm |
| :----- | :---------- |
| `config.py` | `Settings` (frozen dataclass) + `load_settings()`. Env `GENIMG_*` thắng `base.yaml` thắng default. |
| `device.py` | `select_device()` — tự động `cuda:1` nếu ≥2 GPU, ngược lại `cuda:0`, hoặc override. |
| `logging_setup.py` | `configure_logging()` — logger `gen-image`, format có timestamp. |
| `models/loader.py` | `resolve_base_model()`, `resolve_gguf_path()`, `load_pipeline()`. |
| `models/placement.py` | `place()` — `resident` (mặc định) hoặc `model_offload`. |
| `tuning.py` | `apply_gpu_tuning()`, `maybe_compile()`, `warmup()`. |
| `inference.py` | `generate()` — MỘT hàm cho cả t2i lẫn edit; cache prompt-embed; `calculate_dimensions()`. |
| `ui/app.py` | `build_ui()` — dựng 7 tab. KHÔNG import torch, KHÔNG nạp model. |
| `ui/launch.py` | `main()` — nạp model, warm-up, mở cổng, dựng tunnel công khai. |
| `ui/widgets.py` | Widget dùng chung giữa các tab (params, output, examples). |
| `ui/demo_cache.py` | Ánh xạ {đầu vào ví dụ → ảnh kết quả dựng sẵn trên đĩa}. |
| `ui/prompts.py` | Prompt hệ thống theo từng task. ⚠️ chưa tinh chỉnh lại cho FLUX.2. |
| `download.py` | `download()`, `main()` tải trọng số. |
| `serve.py` | Entry point chạy demo (re-export `ui.launch.main`). |
| `queue_service/` | RabbitMQ consumer. Không phụ thuộc model cụ thể. |

## Vì sao không còn `models/offload.py`

Backend trước (Qwen-Image-Edit-2509) cần 26.4 GiB weight thường trú — không
vừa L4 24GB — nên phải có 4 chiến lược offload và một hàm tự dò GPU để chọn.
FLUX.2-klein-9B cần ~11 GiB ở cấu hình mặc định (transformer GGUF Q4_K_M +
text encoder Qwen3-8B NF4 + VAE), vừa thoải mái, nên cả khối đó trở thành mã
chết. Lưu ý đó là con số **sau khi nén cả hai** component lớn: để text
encoder ở bf16 thì tổng lên 22.6 GiB và không vừa. Chi tiết:
[PERFORMANCE.md §0](PERFORMANCE.md) và
[adr/0001-flux2-klein-backend.md](adr/0001-flux2-klein-backend.md) (ADR viết
cho bản 4B, các con số trong đó đã bị bản 9B thay thế).

## Ranh giới `ui/app.py` ↔ `ui/launch.py`

`build_ui(pipeline, ...)` chỉ đóng gói `pipeline` vào closure của các nút
bấm — nó **không gọi tới model lúc dựng**. Nhờ vậy `build_ui(None, ...)`
chạy được, và giao diện kiểm được trên máy không có GPU.
`tests/test_ui_structure.py` khoá bất biến đó lại (kể cả việc `app.py` không
được import `torch`), vì nếu hai mối quan tâm trộn lại thì không còn cách nào
test UI mà không thuê GPU.

## Cấu hình quan trọng

Nguồn config duy nhất: `config_setup/base.yaml`.

- Khối `app:` → `serve.py` + `download.py`, override được bằng env `GENIMG_*`.
- Khối `processor:` → queue service. **KHÔNG đọc env `GENIMG_*`** — mọi giá
  trị phải nằm trong file.

`base_model` phải là bản **distilled** (`FLUX.2-klein-9B`), không phải
`FLUX.2-klein-base-9B`. `scripts/preflight.py` cảnh báo nếu `model_index.json`
thiếu `is_distilled`.

Tiền tố env cũ `QIE_*` **không còn tác dụng** — xem ADR §7.

## Quy trình chuẩn khi phát triển

```bash
uv sync --extra dev            # cài dependency theo uv.lock
make check                     # lint + test — cổng trước khi push
make fmt                       # tự sửa format + thứ tự import
python scripts/preflight.py    # kiểm cây model trên đĩa (KHÔNG cần GPU)
pre-commit install             # chạy lint tự động trước commit
```

Trên GPU thật:

```bash
python scripts/spike_flux2.py  # cổng chặn Phase 0 — chạy TRƯỚC khi deploy
python scripts/benchmark.py    # ma trận đo latency/VRAM
```

## Hiệu năng

Cái gì đã tối ưu sẵn, cái gì đo rồi mới làm, và cái gì **đừng làm** (đổi
sampler trên model distilled): [PERFORMANCE.md](PERFORMANCE.md).

## Lưu ý bảo mật

- `.env` chứa `HF_TOKEN` — **bị gitignore**, không commit.
- `config_setup/base.yaml` chứa secret RabbitMQ/GCS — **bị gitignore**.
- `.model_paths.env` sinh tự động bởi `download` — cũng bị gitignore.
- `models/` (trọng số) rất lớn — bị gitignore.
