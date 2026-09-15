# Kiến Trúc Dự Án (Architecture)

Tài liệu dành cho team tiếp nhận, mô tả trách nhiệm từng module và luồng dữ liệu
của demo `Qwen-Image-Edit-2509 Lightning`.

## Nguyên tắc

- **Mỗi module một trách nhiệm** (config / device / loader / offload / inference / ui).
- **Một nguồn cấu hình duy nhất** (`src/qwen_lightning/config.py`), dùng chung cho
  cả tải trọng số (`download`) và chạy demo (`serve`).
- **Logic thuần tách khỏi GPU/UI** để dễ test (config, path resolution, scheduler).

## Sơ đồ module

```
scripts/serve.py ──► qwen_lightning.serve.main
                        │
                        ▼
                   ui.app.main
                   ├─ config.load_settings()      # đọc env + .model_paths.env
                   ├─ device.select_device()      # chọn cuda
                   ├─ models.loader.load_pipeline # build_scheduler
                   │                              # resolve_transformer_path
                   │                              # + models.offload.apply_offload
                   └─ ui.app.build_ui()           # Gradio Blocks + bind generate

scripts/download_model.py ──► qwen_lightning.download.main
                                ├─ config.load_settings()
                                ├─ download.detect_precision()
                                ├─ download.download()         # snapshot + hf_hub_download
                                └─ ghi .model_paths.env        # path local cho serve
```

## Trách nhiệm module

| Module | Trách nhiệm |
| :----- | :---------- |
| `config.py` | `Settings` (dataclass frozen) + `load_settings()`. Đọc `.env` rồi `.model_paths.env`. |
| `device.py` | `select_device()` — tự động `cuda:1` nếu ≥2 GPU, ngược lại `cuda:0`, hoặc override. |
| `logging_setup.py` | `configure_logging()` — logger chuẩn, format có timestamp. |
| `models/loader.py` | `build_scheduler()`, `resolve_transformer_path()`, `load_pipeline()`. |
| `models/offload.py` | `apply_offload()` — chiến lược `auto/none/model/sequential`. |
| `inference.py` | `generate()` — validate input, seed, chạy pipeline, trả `(image, status)`. |
| `ui/app.py` | `build_ui()` dựng Gradio, `main()` khởi tạo và `launch()`. |
| `download.py` | `detect_precision()`, `download()`, `main()` tải trọng số. |
| `serve.py` | Entry point chạy demo (re-export `ui.app.main`). |

## Biến môi trường quan trọng

Xem `config/example.env`. Các giá trị `QIE_NUM_STEPS`, `QIE_RANK`,
`QIE_PRECISION`, `QIE_TRANSFORMER_SUBDIR` phải **khớp nhau** giữa lúc tải
(`download`) và lúc chạy (`serve`), nếu không sẽ tải/pick sai weights.

## Quy trình chuẩn khi phát triển

```bash
uv pip install -e ".[dev]"     # hoặc: uv pip install -e .
ruff check src tests scripts   # lint
ruff format src tests scripts  # format
pytest                         # unit tests (không cần GPU)
pre-commit install             # chạy lint/format tự động trước commit
```

## Lưu ý bảo mật

- `.env` chứa `HF_TOKEN` — **bị gitignore**, không commit.
- `.model_paths.env` sinh tự động bởi `download` — cũng bị gitignore.
- `models/` (trọng số) rất lớn — bị gitignore.
