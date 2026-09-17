# Kiến Trúc Dự Án (Architecture)

Tài liệu dành cho team tiếp nhận, mô tả trách nhiệm từng module và luồng dữ
liệu của service `HiDream-O1-Image (SDNQ 4-bit)`.

> Lịch sử: dự án trước đây chạy `Qwen-Image-Edit-2509 Lightning` qua
> diffusers + Nunchaku. Xem [MODEL_HIDREAM_O1.md](MODEL_HIDREAM_O1.md)
> để biết cái gì đã đổi và vì sao.

## Nguyên tắc

- **Mỗi module một trách nhiệm** (config / device / loader / inference / ui).
- **Một nguồn cấu hình duy nhất** (`src/imagegen/config.py`), dùng chung cho
  cả tải trọng số (`download`) và chạy service (`serve` / `main_queue`).
- **Logic thuần tách khỏi GPU/UI** để test được không cần card: `config`,
  `hidream/resolutions`, `hidream/artifacts`, `queue_service/config`.
- **Code bên thứ ba nằm riêng** trong `hidream/vendor/`, không sửa ngoài các
  patch ghi ở `VENDOR.md`, và bị loại khỏi ruff.
- **Tầng UI không biết model.** Hai closure trong `ui/app.py` là chỗ duy
  nhất chạm tới `model`/`processor`; chúng đi xuống các tab qua
  `TabContext`.

## Sơ đồ module

```
scripts/serve.py ──► imagegen.serve.main
                        │
                        ▼
                   ui.app.main
                   ├─ config.load_settings()      # env + base.yaml + .model_paths.env
                   ├─ device.select_device()      # chọn cuda
                   ├─ hidream.load_model()        # AutoProcessor + Qwen3VL (SDNQ)
                   ├─ tuning.warmup()             # 1 lượt sinh ảnh thật
                   ├─ ui.app.build_ui()           # dựng TabContext + Blocks
                   │    ├─ ui.components          # widget + preset dùng chung
                   │    └─ ui.tabs.ALL            # 7 module tab, mỗi cái build(ctx)
                   └─ ui.launcher.launch_with_public_link()

main_queue.py ──► imagegen.queue_service.Application.run()
                   ├─ config.load_config()             # config_setup/base.yaml
                   ├─ processors.create_processor()    # ← model load ở đây
                   │    └─ HiDreamGenerator → hidream.load_model/generate
                   ├─ pipelines.GenImagePipeline       # download → gen → upload
                   └─ messaging.*                      # consumer / publisher / DLQ

scripts/download_model.py ──► imagegen.download.main
                                ├─ config.load_settings()
                                ├─ snapshot_download()    # MỘT lời gọi, ~9.9GB
                                └─ ghi .model_paths.env   # IMG_MODEL_PATH
```

## Trách nhiệm module

| Module | Trách nhiệm |
| :----- | :---------- |
| `config.py` | `Settings` (frozen dataclass) + `load_settings()`. Env `IMG_*` > `.model_paths.env` > `base.yaml` > default. |
| `device.py` | `select_device()` — `cuda:1` nếu ≥2 GPU, ngược lại `cuda:0`, hoặc override. |
| `logging_setup.py` | `LOGGER_NAME` + `configure_logging()`. Mọi module log vào cùng một logger. |
| `download.py` | `download()` — một `snapshot_download`, ghi `.model_paths.env`. |
| `tuning.py` | `apply_gpu_tuning()` (TF32 on, cudnn.benchmark off) + `warmup()`. |
| `hidream/artifacts.py` | **Thuần stdlib.** Hợp đồng file của snapshot: `MODEL_DIR_NAME`, `REQUIRED_FILES`, `SPECIAL_TOKENS`, quant method mong đợi. |
| `hidream/resolutions.py` | **Thuần stdlib.** 11 độ phân giải cố định + `find_closest_resolution()`. |
| `hidream/loader.py` | `load_model()` — `import sdnq` → `AutoProcessor` → `Qwen3VLForConditionalGeneration`, gắn 5 special token. |
| `hidream/inference.py` | `Recipe` + `build_recipe()` + `generate()`. Một hàm cho cả t2i lẫn editing. |
| `hidream/vendor/` | Code inference của HiDream, copy từ GitHub. Xem `VENDOR.md`. |
| `ui/app.py` | Điểm ráp: nạp model, dựng `TabContext` + Blocks, `main()`. Chỉ ~210 dòng. |
| `ui/tabs/` | Bảy tab, mỗi tab một module export `build(ctx)`. Thêm tab = thêm module + thêm vào `ALL`. |
| `ui/context.py` | `TabContext` — dependency mà một tab cần, đã đóng sẵn model + settings. |
| `ui/components.py` | Widget dùng chung, preset độ phân giải, chỉ mục ảnh demo dựng sẵn. |
| `ui/launcher.py` | Dựng tunnel `*.gradio.live`, thử lại, fail-fast nếu không có link. |
| `ui/prompts.py` | Prompt hệ thống theo task. |
| `queue_service/` | RabbitMQ consumer/publisher, pipeline, audit, storage, health. Không đổi khi thay lõi. |

### Vì sao `hidream/__init__.py` nạp lười

`loader` và `inference` kéo theo torch + transformers + sdnq (~3GB wheel, vài
giây khởi động). UI, queue config, `preflight` và test chỉ cần bảng độ phân
giải và hợp đồng file. `__getattr__` (PEP 562) cho `from imagegen.hidream
import generate` vẫn chạy, nhưng `import imagegen.hidream.resolutions` không
chạm tới CUDA — đó là điều kiện để test chạy trên CI không GPU.

## Cấu hình quan trọng

Nguồn config duy nhất: `config_setup/base.yaml`.

* Khối `app:` → `imagegen.config.Settings` (Gradio + download).
* Khối `processor:` → `queue_service.config.ProcessorConfig` (queue worker).

Mọi key override được bằng biến môi trường `IMG_*` (env THẮNG YAML THẮNG
default).

**Ba giá trị phải để ý:**

| Giá trị | Vì sao |
| :-- | :-- |
| `rabbitmq.avg_inference_seconds` | Quyết định `effective_prefetch()`. Đặt sai → consumer ôm quá nhiều message → backlog phình, timeout hàng loạt. Chạy `scripts/benchmark.py`, nó in ra con số cần điền. |
| `processor.width/height` | Chỉ chọn được TỈ LỆ. Model snap về 1 trong 11 độ phân giải cứng, nhỏ nhất 2048×2048 — không có cách sinh ảnh nhỏ hơn. |
| `processor.model_type` | `full` = 50 bước + CFG 5.0; `dev` = 28 bước + CFG 0.0 (nhanh ~3.5×). Đổi `model_type` thì `num_steps`/`guidance_scale`/`shift` phải đổi theo. |

### Thêm một tab mới

```bash
cp src/imagegen/ui/tabs/image_and_prompt.py src/imagegen/ui/tabs/my_tab.py
# sửa nội dung, rồi thêm vào ui/tabs/__init__.py: import + ALL
```

Tab chỉ được dùng `ctx.run(...)` / `ctx.run_t2i(...)`; đừng import `generate`
hay `Settings` vào module tab — đó là cách tầng UI giữ được độc lập với model.

## Quy trình chuẩn khi phát triển

```bash
uv sync --frozen                    # cài đúng uv.lock
ruff check src tests scripts        # lint (0 lỗi là bắt buộc)
ruff format src tests scripts       # format
pytest                              # unit tests (không cần GPU)
python scripts/preflight.py         # kiểm tra cây model trên đĩa
pre-commit install                  # lint/format tự động trước commit
```

Cập nhật code vendor:

```bash
scripts/vendor_hidream.sh <commit-sha>
# rồi áp lại 3 patch trong src/imagegen/hidream/vendor/VENDOR.md
pytest tests/test_resolution.py     # bắt trôi bảng độ phân giải
```

## Test suite

Chạy được **không cần GPU, không cần torch**. Đây là ràng buộc thiết kế,
không phải may mắn — xem mục lazy import ở trên.

| File | Kiểm gì |
| :-- | :-- |
| `test_config.py` | Thứ tự ưu tiên env/yaml/default; `model_type` kéo theo `num_steps`; không còn tiền tố env cũ |
| `test_recipe.py` | `build_recipe()` — ràng buộc chéo giữa scheduler / timesteps / noise params |
| `test_resolution.py` | Snap độ phân giải; `aspect_ratio` hỏng không làm chết request; không trôi khỏi bảng vendor |
| `test_model_paths.py` | Resolve thư mục model; phát hiện snapshot tải dở; gắn đủ 5 special token |
| `test_cache_location.py` | `HF_HOME` nằm trong project; `download` không gọi `login()` |
| `test_config_contract.py` | `base.example.yaml` khớp dataclass; `config_env.py` map đủ key |

Thứ **chưa** có test: `generate()` (cần model thật), tầng Gradio (cần
browser), tầng queue (cần broker — dùng `make smoke` thay thế).

## Lưu ý bảo mật

- `.env` chứa `HF_TOKEN` — **bị gitignore**, không commit. (Model hiện tại là
  MIT và không gated nên token là tuỳ chọn.)
- `.model_paths.env` sinh tự động bởi `download` — cũng bị gitignore.
- `models/` (trọng số) rất lớn — bị gitignore.
- `config_setup/base.yaml` + `config_setup/credentials/*.json` chứa secret —
  bị gitignore và bị `.dockerignore` chặn khỏi build context.
