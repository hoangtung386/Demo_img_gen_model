# HiDream-O1-Image (SDNQ 4-bit) — Gradio Demo + Queue Worker

Service sinh và chỉnh sửa ảnh chạy
[**HiDream-O1-Image**](https://huggingface.co/HiDream-ai/HiDream-O1-Image) bản
đã lượng tử hoá 4-bit bằng
[SDNQ](https://github.com/Disty0/sdnq):
[`WaveCut/HiDream-O1-Image-SDNQ-4bit-dynamic-uint4-th1e-2`](https://huggingface.co/WaveCut/HiDream-O1-Image-SDNQ-4bit-dynamic-uint4-th1e-2).

HiDream-O1 là một **Pixel-level Unified Transformer**: một model duy nhất, không
VAE, không text encoder rời — text-to-image và chỉnh sửa ảnh đi qua cùng một
lời gọi.

> **Đã thay lõi.** Dự án trước đây chạy `Qwen-Image-Edit-2509 Lightning` qua
> Nunchaku + diffusers. Xem
> [docs/MODEL_HIDREAM_O1.md](docs/MODEL_HIDREAM_O1.md) để biết cái gì
> đổi, vì sao, và những gì còn phải kiểm chứng trên GPU thật.

## Bắt đầu từ đâu

README này dài — dưới đây là đường ngắn nhất cho từng vai trò.

| Bạn muốn | Đọc |
| :-- | :-- |
| Chạy thử demo trên máy có GPU | [Cài đặt](#cài-đặt) → [Tải trọng số](#tải-trọng-số) → [Chạy trực tiếp](#chạy-trực-tiếp-không-docker) |
| Deploy lên server | [Triển khai trên Google Cloud](#triển-khai-trên-google-cloud-trọng-số-từ-gcs) → [Chạy bằng Docker](#chạy-bằng-docker) |
| Nối queue với backend | [docs/QUEUE_SERVICE.md](docs/QUEUE_SERVICE.md) |
| Hiểu code để sửa | [docs/architecture.md](docs/architecture.md) |
| Ảnh sinh quá chậm | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| Hiểu model & những gì chưa kiểm chứng | [docs/MODEL_HIDREAM_O1.md](docs/MODEL_HIDREAM_O1.md) |

> **Nếu bạn vừa nhận bàn giao repo này:** đọc
> [docs/MODEL_HIDREAM_O1.md §5 — Việc CÒN LẠI](docs/MODEL_HIDREAM_O1.md) trước
> tiên. Model lõi vừa được thay và **chưa có gì chạy qua một forward pass
> thật**; có 7 việc cần GPU để đóng lại.

## Yêu cầu

- Python >= 3.10, < 3.14 (khuyên dùng `uv`)
- GPU NVIDIA có bf16 (Ampere trở lên: A100, RTX 30/40/50-series).
  SDNQ thuần PyTorch nên **không phải chọn int4/fp4 theo đời card** như
  Nunchaku trước đây.
- **~9.9 GB đĩa** cho trọng số, **~11 GiB VRAM** lúc chạy. Card 24GB (RTX 4090,
  A10G) giờ đủ — bản Qwen cũ cần 26.4 GiB.
- **Không cần HuggingFace token.** Cả model gốc lẫn bản lượng tử hoá đều là MIT
  và không gated.

⚠️ **Đọc mục [Tốc độ](#tốc-độ) trước khi lên kế hoạch tải.** Model này chạy 50
bước có CFG ở 2048×2048 — chậm hơn bản Lightning 4-bước một bậc độ lớn.

Thuê A100 trên Google Cloud: xem [mục riêng bên dưới](#thuê-a100-trên-google-cloud).

## Cài đặt

Cách nhanh nhất (uv quản lý venv + lock, cài package ở chế độ editable):

```bash
# Cài đặt cơ bản
uv sync

# Kèm công cụ dev (ruff, pytest)
uv sync --extra dev
```

Sau đó chạy thông qua `uv run` (không cần kích hoạt venv thủ công):

```bash
uv run download-model     # tải trọng số (~9.9GB)
uv run run-app            # chạy demo
```

Hoặc cách truyền thống (pip-style):

```bash
uv venv
uv pip install -e .
# uv pip install -e ".[dev]"   # kèm dev tools
```

### Về ghim phiên bản

Ba ghim trong `pyproject.toml` **không phải tuỳ tiện** — đổi một cái là phải
kiểm tra cả ba:

| Package | Ghim | Vì sao |
| :-- | :-- | :-- |
| `transformers` | `==4.57.1` | Bản HiDream ghim. `hidream/vendor/qwen3_vl_transformers.py` bám vào internals 4.57 (`masking_utils`, `modeling_layers`, `models.qwen3_vl.*`). |
| `torch` | `>=2.10` | README của HiDream nói rõ **không** dùng 2.9.x — [Qwen3-VL#1811](https://github.com/QwenLM/Qwen3-VL/issues/1811). |
| `gradio` | `==6.17.3` | Bản mới nhất còn cho phép `huggingface-hub < 1.0`. Từ 6.18 gradio đòi hub >= 1.2, xung đột cứng với transformers 4.57.1. |

## Cấu hình (config_setup/base.yaml)

Toàn bộ config nằm trong **một file** `config_setup/base.yaml`. `download`,
`serve`, `main_queue` và các script model đều đọc từ đây. Sao chép template rồi
điền giá trị:

```bash
cp config_setup/base.example.yaml config_setup/base.yaml
```

> Mọi key override được bằng biến môi trường `IMG_* / HF_TOKEN / MODEL_ROOT`
> (env THẮNG YAML). `.env` cũ vẫn được nạp nếu tồn tại (backward-compat).

> ⚠️ **Tiền tố biến môi trường đã đổi `QIE_*` → `IMG_*`.** `QIE` viết tắt của
> "Qwen Image Edit" nên mất nghĩa. Biến `QIE_*` cũ **không còn được đọc** — có
> một test (`test_no_qie_prefix_left_in_source`) canh cho nó không quay lại.
> Bảng chuyển đổi ở [mục dưới](#bảng-chuyển-đổi-biến-môi-trường).

Các key quan trọng trong `app:` (đọc bởi cả `download` và `serve`):

| Biến | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `IMG_MODEL_TYPE` | `full` | `full` (50 bước) hoặc `dev` (28 bước, nhanh ~3.5×) |
| `IMG_NUM_STEPS` | `50` | Số bước. Bỏ trống thì tự theo `model_type`. |
| `IMG_GUIDANCE_SCALE` | `5.0` | CFG. **> 1.0 nhân đôi số forward pass.** Đặt `0.0` cho bản dev. |
| `IMG_SHIFT` | `3.0` | Độ lệch lịch nhiễu. `1.0` cho bản dev. |
| `IMG_SCHEDULER` | `default` | `default` / `flow_match` / `flash` |
| `IMG_WIDTH` / `IMG_HEIGHT` | `2048` | Chỉ chọn **tỉ lệ** — xem cảnh báo dưới |
| `IMG_DEVICE` | (auto) | `cuda:1` nếu ≥2 GPU, ngược lại `cuda:0` |
| `IMG_WARMUP` | `true` | Sinh một ảnh thật lúc khởi động để người dùng đầu tiên không phải chờ |
| `IMG_PORT` | `7860` | Cổng Gradio |
| `IMG_SERVER_NAME` | `0.0.0.0` | Bind address |
| `IMG_DEMO_CACHE` | `true` | Ví dụ mẫu trả ảnh dựng sẵn thay vì chạy model |
| `IMG_MODEL_PATH` | (từ `.model_paths.env`) | Thư mục model local |
| `IMG_USE_FLASH_ATTN` | `false` | Bật nếu image có `flash-attn` khớp torch+CUDA |

### ⚠️ Không chọn được kích thước ảnh ra

`IMG_WIDTH`/`IMG_HEIGHT` chỉ chọn **tỉ lệ khung**. Model snap mọi yêu cầu về
một trong **11 độ phân giải cố định**:

```
2048×2048  2304×1728  1728×2304  2560×1440  1440×2560  2496×1664
1664×2496  3104×1312  1312×3104  2304×1792  1792×2304
```

Đặt `1024×1024` **không** cho ảnh 1024 — nó vẫn ra 2048×2048. Không có đòn bẩy
nào để sinh ảnh nhỏ hơn cho nhanh, khác hẳn `output_area` của bản Qwen cũ (đã
bị bỏ khỏi config).

### Không còn chiến lược offload

Bản Qwen cũ có `QIE_OFFLOAD` với 5 chế độ vì phải xếp text_encoder (15.4 GiB) +
transformer (10.7 GiB) + vae qua một hay nhiều card. Model mới là **một khối
~11 GiB** — `accelerate device_map` đặt trọn lên card đã chọn, không còn gì để
chia. Biến đó đã bị bỏ.

### Bảng chuyển đổi biến môi trường

| Cũ (`QIE_*`) | Mới | Ghi chú |
| :-- | :-- | :-- |
| `QIE_BASE_MODEL_LOCAL` + `QIE_TRANSFORMER_PATH` | `IMG_MODEL_PATH` | Gộp thành một thư mục |
| `QIE_OFFLOAD` | — | Bỏ hẳn |
| `QIE_PRECISION`, `QIE_RANK` | — | Bỏ hẳn (đặc thù Nunchaku) |
| `QIE_TRANSFORMER_REPO`, `QIE_TRANSFORMER_SUBDIR` | — | Bỏ hẳn |
| `QIE_DOWNLOAD_BASE_TRANSFORMER` | — | Bỏ hẳn (không còn shard thừa) |
| `QIE_TEXT_ENCODER_DEVICE` | — | Bỏ hẳn (không có text encoder rời) |
| `QIE_NUM_STEPS`, `QIE_DEVICE`, `QIE_WARMUP`, `QIE_PORT`, `QIE_SERVER_NAME`, `QIE_DEMO_CACHE` | đổi tiền tố thành `IMG_` | Ngữ nghĩa giữ nguyên |
| `QIE_RMQ_*`, `QIE_QUEUE_CONFIG`, `QIE_MODELS_*`, `QIE_GCS_KEY_FILE` | đổi tiền tố thành `IMG_` | Ngữ nghĩa giữ nguyên |
| — | `IMG_MODEL_TYPE`, `IMG_GUIDANCE_SCALE`, `IMG_SHIFT`, `IMG_SCHEDULER`, `IMG_WIDTH`, `IMG_HEIGHT`, `IMG_USE_FLASH_ATTN` | Mới |

## Tải trọng số

```bash
python scripts/download_model.py
# hoặc: uv run download-model
```

Script tải **một** snapshot (~9.9 GB) về `models/HiDream-O1-Image-SDNQ-uint4/`,
rồi ghi đường dẫn vào `.model_paths.env` để `serve` dùng trực tiếp.

Khác bản Qwen cũ: model mới là một repo `transformers` **phẳng** (config + 3
shard + tokenizer ở cấp gốc), không còn `model_index.json` và các thư mục con
`vae/`, `text_encoder/`, `transformer/`. Vì thế cũng chỉ còn một lời gọi tải,
không còn bước bỏ qua 39GB shard BF16 thừa.

**Không có gì rơi vào `~/.cache/huggingface`.** Cache của HuggingFace đã được
ghim vào `models/.hf` ngay trong `imagegen/__init__.py`, nên toàn bộ dữ liệu
tải về nằm trong project. Đặt `HF_HOME` tường minh nếu muốn chỗ khác.

Kiểm tra cây model bất cứ lúc nào (vài giây, không cần GPU, không cần torch):

```bash
make preflight
```

Nó kiểm tra đủ file, đủ shard, đúng `quant_method: sdnq` và đúng kiến trúc —
bắt được cả trường hợp tải nhầm bản BF16 chưa lượng tử hoá (chạy được nhưng
tốn 17 GiB VRAM thay vì 11).

## Triển khai trên Google Cloud (trọng số từ GCS)

Đây là đường deploy chính: clone repo, thả file key, chạy một lệnh. **Không
cần `.env`, không cần `HF_TOKEN`, không chạm tới HuggingFace.**

```bash
git clone -b develop https://github.com/hoangtung386/Demo_img_gen_model.git
cd Demo_img_gen_model
cp /duong/dan/toi/<key-file>.json .      # key service account GCS
docker compose up -d --build
```

Thế thôi. `docker compose up` chạy hai service theo thứ tự:

1. **`model-fetcher`** — xác thực bằng `<key-file>.json`, tải
   `...zip` (21.5 GB) về rồi giải nén vào
   `./models/`, tự kiểm tra cây thư mục, xong thì thoát.
2. **`qwen-lightning`** — chỉ khởi động khi bước trên trả về 0
   (`depends_on: service_completed_successfully`).

Lần chạy sau, fetcher thấy `models/.fetched-from` khớp URI và còn đủ file thì
bỏ qua, không tải lại. Ép tải lại: `IMG_MODELS_FORCE=true docker compose up`.

> **Đến đây là xong — không phải chạy thêm lệnh nào.** `CMD` của image chính
> là `python scripts/serve.py`, và compose không ghi đè nó. Mục
> [Chạy trực tiếp](#chạy-trực-tiếp-không-docker) bên dưới là đường **thay
> thế** cho ai chạy không qua Docker (Colab, máy thuê), không phải bước tiếp
> theo của mục này.

### Vòng đời container

```bash
docker compose logs -f              # theo dõi tải model + warm-up (~2-3 phút)
docker compose ps                   # xem trạng thái
docker compose restart qwen-lightning   # khởi động lại app, không tải lại model
docker compose down                 # dừng và xoá container
```

Dùng `docker compose down` chứ không phải `docker stop`: `docker stop` chỉ
dừng container app và để nó nằm lại ở trạng thái exited, lần sau `up` sẽ báo
xung đột tên. `down` dọn cả app lẫn container fetcher đã thoát.

Thư mục `./models` **không** bị `down` đụng tới — nó là bind mount trên host,
nên lần `up` sau không phải tải lại 21.5 GB.

Không có `docker run` một dòng cho service này, vì nó là **hai** container
chạy có thứ tự: fetcher phải xong trước, app mới được khởi động
(`depends_on: service_completed_successfully`). Muốn viết tay bằng `docker
run` thì phải tự chạy hai lệnh đúng thứ tự và tự truyền lại `--gpus all`,
ánh xạ cổng, ba volume mount và khoảng mười biến môi trường mà
`docker-compose.yml` đang giữ. Compose tồn tại chính là để khỏi phải làm việc
đó.

### Build cài những gì

`Dockerfile` dùng **Python 3.13 + uv**, tương đương một lệnh:

```bash
uv sync --frozen --no-dev
```

Không còn bước cài wheel riêng như thời Nunchaku — mọi dependency đều đến từ
PyPI qua `uv.lock`, kể cả `sdnq`. Cũng không còn `[tool.uv.sources]`.

Build vẫn có bước chốt phiên bản đọc metadata (không `import`, vì builder
không có GPU), nên ghim lệch là fail ngay lúc build chứ không phải sau vài
chục giây nạp model:

```
OK — torch 2.14.0 | transformers 4.57.1 | sdnq 0.2.6 | diffusers 0.36.0
```

**`flash-attn` KHÔNG có trong image.** Build nó mất khoảng một giờ và phải
khớp cả torch lẫn CUDA. Code vendor của HiDream hard-code `use_flash_attn=True`
— dự án đã patch thành cờ `IMG_USE_FLASH_ATTN` (mặc định `false`, xem
`src/imagegen/hidream/vendor/VENDOR.md`). Không có patch này thì inference nổ
ngay forward đầu tiên, sau khi đã nạp xong 10GB weight.

### Xác thực GCS

Hai key nằm chung trong `config_setup/credentials/` (mỗi key một vai trò):

- `model-download-key.json` — tải trọng số model nội bộ (`fetch_models.sh`).
- `user-upload-key.json` — upload ảnh kết quả lên bucket cho user (queue service).

`scripts/fetch_models.sh` dò key download theo thứ tự: `$IMG_GCS_KEY_FILE` →
`/credentials/model-download-key.json` → `/project/config_setup/credentials/
model-download-key.json` → **`<key-file>.json` ở gốc repo** → bất kỳ
`<key-file>*.json` nào. `config_setup/credentials` được mount `:ro` vào
`/credentials` trong container fetcher; key **không** đi vào image và không nằm
trong build context (`.dockerignore` chặn `*-key.json`, `<key-file>*.json`).

Không có file key nào thì script rơi về ADC — **cách nên dùng trên GCE**: gắn
service account vào VM, không có file key nào để rò rỉ, thu hồi bằng một lệnh
IAM. Quyền tối thiểu: `roles/storage.objectViewer` trên bucket.

### Đổi archive

```yaml
IMG_MODELS_URI: "${IMG_MODELS_URI:-gs://<your-bucket>/hidream-o1-sdnq-uint4.zip}"
```

> ⚠️ **Archive cũ không dùng lại được.** Nó chứa trọng số Qwen; fetcher sẽ
> chạy thành công rồi app báo thiếu file. Phải đóng gói lại:
> `make download && make pack DEST=gs://<bucket>/`, rồi cập nhật
> `IMG_MODELS_URI` trong `docker-compose.yml`.

`fetch_models.sh` nhận `.zip`, `.tar.zst`, `.tar.gz`, `.tar`, hoặc một prefix
thư mục. Archive bọc trong một cấp `models/` — như `zip -r models.zip models/`
tạo ra — được nhận diện và bỏ cấp đó tự động, nên không ra `models/models/...`.

Nó cũng kiểm tra `IMG_MODELS_REQUIRE` sau khi giải nén; mặc định là
`HiDream-O1-Image-SDNQ-uint4/config.json` +
`HiDream-O1-Image-SDNQ-uint4/model.safetensors.index.json`. Giải nén ra sai
cấu trúc thì fetcher fail ở đó chứ không để app phát hiện sau.

**Nên dùng `.tar.zst` thay vì `.zip`** — không phải vì tỉ lệ nén (chênh ~2%)
mà vì **tar stream được, zip thì không**: central directory của zip nằm ở cuối
file nên VM buộc phải tải trọn archive xuống đĩa rồi mới giải nén, tức cần gấp
đôi dung lượng. Với `.tar.zst`, `gcloud storage cat | zstd -d | tar -x` chỉ
cần đủ chỗ cho bản giải nén, và một lần tải dở dang cũng không để lại archive
cụt. zstd còn nhanh hơn gzip hàng chục lần.

Ở quy mô ~9.9 GB của model này thì khoản chênh đó nhỏ hơn nhiều so với thời
Qwen (21.5 GB archive + 26.4 GB giải nén), nhưng lý do kỹ thuật thì không đổi.

```bash
make pack DEST=gs://<your-bucket>/      # -> models.tar.zst rồi upload
```

Xem riêng log tải mà không lẫn log nạp model:

```bash
make fetch
```

## Chạy trực tiếp (không Docker)

Đường này dành cho Colab, máy thuê, hoặc khi đang sửa code — **bỏ qua nếu bạn
dùng `docker compose`**, vì compose đã chạy sẵn đúng lệnh dưới đây bên trong
container.

```bash
python scripts/serve.py
# hoặc: uv run run-app
```

Cần `uv sync` và trọng số nằm sẵn trong `./models` trước (xem
[Cài đặt](#cài-đặt) và [Tải trọng số](#tải-trọng-số)).

## Giao diện (7 tab)

Mở trình duyệt tại `http://<server-ip>:7860`. Năm tab đầu đã
có prompt chuyên biệt viết sẵn (xem `src/imagegen/ui/prompts.py`):

| Tab | Đầu vào | Kết quả |
| :--- | :--- | :--- |
| **1. Virtual Try-On** | Ảnh 1: trang phục — Ảnh 2: người mẫu *(ảnh nền)* | Người mẫu mặc trang phục mới, giữ nguyên khuôn mặt, dáng, tư thế, màu da và bối cảnh |
| **2. Home Design** | **Một** ảnh căn phòng + mô tả thiết kế tự viết | Chính căn phòng đó được dọn sạch đồ cũ và bài trí lại theo mô tả, giữ nguyên tường, cửa sổ, cửa, sàn và góc chụp |
| **3. Image to Cartoon** | 1 ảnh chụp có người và phong cảnh | Toàn bộ người trong ảnh thành nhân vật hoạt hình giữ tối đa nét mặt, kiểu tóc, vóc dáng, trang phục; phong cảnh vẽ lại cùng phong cách |
| **4. Ghép 2 người ôm nhau** | Ảnh 1: người thứ nhất — Ảnh 2: người thứ hai *(ảnh nền)* | Một ảnh duy nhất hai người đang ôm nhau, giữ khuôn mặt, kiểu tóc, vóc dáng và trang phục của từng người |
| **5. Face Swap** | Ảnh 1: mặt đã crop — Ảnh 2: ảnh đầy đủ có người và phong cảnh *(ảnh nền)* | Chính ảnh 2 nhưng khuôn mặt đã đổi sang người ở ảnh 1, giữ nguyên phong cảnh, quần áo, dáng người, khung hình — và trả về **đúng kích thước pixel của ảnh 2** |
| **6. Prompt to Image** | Chỉ prompt, **không cần ảnh** | Ảnh mới sinh hoàn toàn từ mô tả — đây là tác vụ HiDream-O1 mạnh nhất |
| **7. Image + Prompt** | 1 ảnh + yêu cầu sửa tự viết | Ảnh đã sửa theo yêu cầu — tab tổng quát, sáu tab trên chỉ là prompt chuyên biệt viết sẵn cho cùng model này |

Năm tab đầu có khối **Ví dụ mẫu** với 3 test case, bảng chỉ hiện các cột đầu
vào. Tab 6 và 7 thì ví dụ chỉ **điền vào ô prompt** chứ không chạy model -
chúng không có ảnh dựng sẵn để trả về, chạy luôn mỗi lần bấm sẽ rất tốn.

### Tab 6 — Prompt to Image

Không còn pipeline riêng. HiDream-O1 là **model hợp nhất**: text-to-image và
editing đi qua đúng một lời gọi, chỉ khác ở chỗ có truyền ảnh vào hay không.
Bản Qwen trước đây phải dựng thêm một pipeline diffusers thứ hai cho tab
này, kèm một cái bẫy dtype phải né. Toàn bộ đoạn đó đã biến mất.

**Về chất lượng:** khác hẳn bản cũ, đây là tác vụ HiDream-O1 được luyện chính,
không phải đường phụ. Prompt mô tả càng dài và cụ thể càng tốt — model gốc còn
có một *Reasoning-Driven Prompt Agent* riêng để viết lại prompt, chưa tích hợp
vào dự án này.

Khung hình chọn ở mục *Độ phân giải đầu ra* trong **Tuỳ chỉnh nâng cao** (11
giá trị cố định). Không còn dropdown "tỉ lệ khung" riêng: độ phân giải đã bao
hàm tỉ lệ.

### Chế độ demo (`IMG_DEMO_CACHE`, mặc định bật)

Khi bật, mọi đầu vào **trùng với một test case mẫu** sẽ trả ảnh đã dựng sẵn
trong `examples/outputs/` thay vì chạy model:

- Bấm vào một ví dụ → ảnh kết quả hiện thẳng ra khung kết quả.
- Bấm nút chạy với đúng ảnh mẫu đó → cũng trả ảnh dựng sẵn, mất ~0.3s.
- Đầu vào khác (ảnh tester tự upload) → **chạy model thật** như bình thường.

Mục đích là trình diễn không phải chờ cả phút mỗi lần. Ô trạng thái nói thẳng đây
là ảnh dựng sẵn:

```
Kết quả dựng sẵn cho ví dụ này (không phải vừa chạy). Bấm nút chạy để model sinh lại ảnh mới.
2048×2048 · ảnh đọc từ đĩa trong 0.3s
```

Log phía server cũng ghi `Demo cache hit: ...` mỗi lần trả ảnh dựng sẵn, và số
ví dụ nạp được in lúc khởi động (`Demo cache: bật (12 ví dụ dựng sẵn)`).

Đo hiệu năng thật thì tắt đi:

```bash
# .env
IMG_DEMO_CACHE=false
```

Ảnh dựng sẵn sinh bằng:

```bash
python scripts/warm_examples.py       # demo phải đang chạy
```

> ⚠️ **Ảnh trong `examples/outputs/` hiện là của model CŨ.** Chưa chạy lại
> `warm_examples.py` thì demo đang trưng ra kết quả không phải do model
> đang chạy sinh ra. Mỗi case giờ tốn hàng chục giây đến vài phút, cả bộ
> có thể mất 30+ phút.

Phải chạy lại script này mỗi khi đổi ảnh mẫu, đổi danh sách case trong
`src/imagegen/ui/examples_spec.py`, hoặc sửa prompt — nếu không, ảnh dựng
sẵn sẽ lệch với những gì model thực sự sinh ra. Nguồn và giấy phép ảnh xem
[`examples/README.md`](examples/README.md).

### ⚠️ Số ảnh đầu vào — cần đánh giá lại sau khi thay lõi

Bản Qwen cũ dùng `QwenImageEditPlusPipeline` với ngữ nghĩa rõ ràng: nhận N ảnh,
ảnh **cuối** là ảnh nền (quyết định khung hình), ảnh trước là tham chiếu. Bốn
tab được xây quanh ngữ nghĩa đó: Virtual Try-On, Ghép 2 người, Face Swap và
Home Design bản cũ.

**HiDream-O1 không có ngữ nghĩa đó.** Tài liệu của model khuyến nghị **đúng
một** ảnh tham chiếu cho editing — khi đó ảnh ra giữ khung của ảnh vào
(`keep_original_aspect`). Truyền nhiều ảnh vẫn chạy, nhưng rơi vào đường
subject-driven / personalization, chất lượng chưa được kiểm chứng ở đây.

Tình trạng hiện tại của từng tab:

| Tab | Số ảnh | Trạng thái |
| :-- | :-- | :-- |
| 2. Home Design | 1 | ✅ Hợp với model mới |
| 3. Image to Cartoon | 1 | ✅ Hợp với model mới |
| 6. Prompt to Image | 0 | ✅ Tác vụ mạnh nhất của model |
| 7. Image + Prompt | 1 | ✅ Hợp với model mới |
| 1. Virtual Try-On | 2 | ⚠️ Phải A/B lại trên GPU thật |
| 4. Ghép 2 người ôm nhau | 2 | ⚠️ Phải A/B lại trên GPU thật |
| 5. Face Swap | 2 | ⚠️ Phải A/B lại trên GPU thật |

Ba tab hai-ảnh vẫn giữ nguyên trong code — chưa có cơ sở để bỏ chúng trước khi
chạy thử. Xem [docs/MODEL_HIDREAM_O1.md](docs/MODEL_HIDREAM_O1.md)
§2.3.

### Tab 5 — Face Swap

Ảnh 1 là mặt đã crop, Ảnh 2 là ảnh đầy đủ và cũng là **ảnh nền**.

- **Kích thước ảnh ra bằng đúng ảnh vào.** Tab này bật `match_input_size`, tức
  ảnh kết quả được scale về đúng kích thước pixel của Ảnh 2. Model vẫn sinh ở
  độ phân giải chọn trong *Tuỳ chỉnh nâng cao* (mặc định ~1 MP) rồi mới scale,
  nên ảnh vào 12 MP sẽ **không** vì thế mà có thêm chi tiết thật.
  `calculate_dimensions` làm tròn hai cạnh về bội của 32 nên tỉ lệ khung lệch
  tối đa ~0.9% so với ảnh gốc; phép scale kéo lại đúng phần đó.
- **Dropdown *Phạm vi hoán đổi***: mặc định chỉ đổi vùng mặt và giữ tóc của
  Ảnh 2. Đổi cả tóc dễ lộ đường ghép ở chân tóc, nhất là khi Ảnh 2 có tóc che
  trán hoặc đội mũ.
- Prompt của tab này là prompt dài nhất trong demo (~1840 ký tự). Ba đoạn giữa
  là phần quyết định: *không dán phẳng ảnh crop*, *relight theo ánh sáng Ảnh
  2*, và *khớp tông da với cổ/tai/tay*. Nếu phải cắt ngắn cho model bớt lạc, hãy giữ ba đoạn đó lại.
- Hoạt động tốt nhất khi Ảnh 2 có **một khuôn mặt rõ**, và Ảnh 1 crop sát mặt,
  nhìn thẳng, không mờ.

Lưu ý khác:

- Tab 1, 2, 3, 5 có dropdown thu hẹp bài toán (loại trang phục / loại phòng /
  phong cách hoạt hình / phạm vi hoán đổi) — đổi dropdown sẽ dựng lại prompt
  tương ứng.
- Tab 2: ô **Mô tả thiết kế mong muốn** là bắt buộc, viết bằng tiếng Anh.
- Ô **Ghi chú thêm** ở các tab khác được nối vào cuối prompt.
- Mục **Tuỳ chỉnh nâng cao** cho phép sửa trực tiếp prompt, `guidance_scale`,
  `shift`, seed, số bước và **độ phân giải đầu ra**.
- **Không còn ô negative prompt.** HiDream-O1 không nhận nó — nhánh uncond của
  CFG dùng prompt `" "` cố định. Ô đó đã bị gỡ khỏi UI thay vì để lại một thứ
  không có tác dụng; sáu hằng `*_NEGATIVE` trong `prompts.py` cũng đã xoá.
- `guidance_scale` mặc định `5.0`. **Mỗi giá trị > 1.0 làm mỗi bước chạy hai
  forward pass** (cond + uncond), tức đắt gấp đôi. Đặt `0.0` nếu chạy bản `dev`
  đã chưng cất.
- **Prompt dài là điểm mạnh, không phải điểm yếu** — ngược hẳn model 4-bước cũ.
  HiDream-O1 ăn mô tả dài, cụ thể. Các prompt hệ thống hiện tại được soạn cho
  model cũ và **chưa được A/B lại**.
- **Cùng seed không đảm bảo cùng một ảnh.** Kernel lượng tử hoá không tất định
  hoàn toàn. Seed thu hẹp biến thiên, không tái lập tuyệt đối.

### Tốc độ

> ⚠️ **Chưa có số đo trên model mới.** Bảng benchmark của bản Qwen cũ đã bị gỡ
> vì nó không còn đúng một chút nào. Chạy `make benchmark` trên GPU của bạn để
> có số thật — script in ra đúng giá trị cần điền vào
> `rabbitmq.avg_inference_seconds`.

#### Vì sao phải đo lại: phép tính thô

| | Forward pass / ảnh | Pixel | Tương đối |
| :-- | --: | --: | --: |
| Qwen Lightning (4 bước, no CFG, 1024²) | 4 | 1.05M | 1× |
| HiDream-O1 **full** (50 bước, CFG 5.0, 2048²) | **100** | 4.19M | **~100×** |
| HiDream-O1 **dev** (28 bước, CFG 0.0, 2048²) | 28 | 4.19M | ~28× |

Tác giả bản lượng tử hoá đo **30.6s/ảnh** trên RTX PRO 6000 Blackwell. A100 là
Ampere (sm_80): không có tensor core FP4/FP8, không chạy được flash-attn 3.
**Dự kiến 60–150s/ảnh trên A100** — nhưng đó là dự kiến, không phải số đo.

#### Đòn bẩy tốc độ

Toàn bộ nằm ở **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** — chi phí một
tấm ảnh nằm ở đâu, knob nào đổi gì lấy gì, và cách đo.

Tóm tắt: hai tối ưu **không đổi ảnh ra** đã bật sẵn. Số trong cột cuối đo
trên RTX 3080 — **không phải L4**, và là tỉ lệ trên *phần* đó chứ không phải
trên tổng (xem PERFORMANCE.md §1 cho phân bổ thật).

| | Sửa gì | Knob | Đo được |
| :-- | :-- | :-- | :-- |
| Giải nén trọng số | SDNQ giải nén **toàn bộ** trọng số ở **mỗi** forward pass — 100 lần một ảnh. Giải nén một lần lúc nạp thay vì vậy (VRAM ~11 → ~18 GiB, nên `auto` tự đo card) | `dequantize: auto` | 1.33× trên phần Linear (~90% thời gian) |
| Đường attention | Mask 4D dày đặc cấm PyTorch dùng backend flash của SDPA, và bắt expand `[B, heads, S, S]` (~1.1GB ở 2048²) lại ở **mỗi layer, mỗi bước**. Thay bằng hai lời gọi SDPA không mask, kết quả y hệt | `attention_mode: auto` | 1.9–2.0× trên phần attention (~10–17% thời gian) |

Ba đòn bẩy nữa **đổi chất lượng lấy tốc độ**, mặc định tắt: hạ `num_steps`
(scheduler UniPC là solver bậc cao, 50 bước là giá trị an toàn chứ không phải
ngưỡng cần thiết), thu hẹp `cfg_interval_*` (CFG là đúng 50% chi phí), và
`snap_resolution: false` (bỏ chặn sàn 2048²).

Đo trước/sau trên chính GPU của bạn — và nếu tổng thời gian không khớp với
những gì PERFORMANCE.md §1 giải thích được thì hỏi thẳng GPU thay vì đoán:

```bash
python scripts/benchmark.py --compare   # đường cũ vs đường mới
python scripts/benchmark.py --profile   # thời gian đi vào kernel nào
```

> Đường **editing** nối ref patch vào chuỗi nên chuỗi dài **gấp đôi** so với
> t2i cùng kích thước, mà attention thì O(S²). Một request editing đắt hơn
> nhiều so với vẻ ngoài của nó.

Hai cách giảm nữa, ở tầng model chứ không phải tầng code:

1. **Dùng bản `dev`** (`model_type: dev`, `num_steps: 28`, `guidance_scale: 0.0`,
   `shift: 1.0`) — nhanh khoảng 3.5×. Vướng mắc: WaveCut chỉ lượng tử hoá bản
   `full`. Muốn `dev` + SDNQ thì phải tự quant, hoặc chạy `dev` ở BF16 (~17 GiB,
   vẫn vừa A100-40GB).
2. **Chọn variant lượng tử hoá khác** — WaveCut có 5 bản. Bản đang dùng là bản
   *nhanh nhất* trong nhóm 4-bit; các bản còn lại chậm hơn 6–15% đổi lấy VRAM
   thấp hơn (tới 8.26 GiB).

#### Điều chỉnh SLA của queue theo số đo

Đây là hệ quả dễ bỏ sót nhất. `config_setup/base.yaml`:

```yaml
rabbitmq:
  avg_inference_seconds: 90.0   # ← ĐIỀN SỐ ĐO THẬT
  target_sla_seconds: 600.0
  prefetch_count: 1
  prefetch_max: 4
```

Để nguyên giá trị của model cũ (`8.0`) thì `effective_prefetch()` cho mỗi
consumer ôm 8 message × ~2 phút = 16 phút backlog, và message timeout hàng
loạt. Xem [docs/QUEUE_SERVICE.md](docs/QUEUE_SERVICE.md).

#### Con số trên UI là thời gian GPU, không phải round-trip

Ô **Thời gian chạy thật (GPU)** hiển thị:

```
⚡ GPU 92.4s (1.84s/bước · 0.92s/forward × 100)
2048×2048 · 50 bước · cfg 5.0 · shift 3.0 · default · seed 12345
```

`s/forward` là đơn giá thật: một bước **có CFG** tốn hai forward pass, nên
`s/bước` luôn gấp đôi `s/forward` khi `guidance_scale > 1.0`. Thu hẹp
`cfg_interval_*` sẽ làm số forward pass tụt xuống dưới `2 × num_steps`.

Khoảng thời gian này đo thuần tính toán trên GPU. Nó **không** bao gồm:

| Không được tính | Vì sao đáng kể |
| :--- | :--- |
| Upload ảnh từ trình duyệt lên server | Vài MB mỗi ảnh |
| Hàng đợi Gradio | ~0.3s round-trip ngay cả khi rảnh |
| Mã hoá PNG kết quả rồi trả về | Ảnh 2048×2048 không nhỏ |
| Tunnel `*.gradio.live` | Traffic đi vòng qua server relay của HuggingFace |

Muốn loại hẳn yếu tố mạng khi đo: truy cập qua LAN / SSH port-forward, hoặc
dùng `make benchmark` (chạy thẳng trong container, không qua HTTP).

#### VRAM

Model SDNQ uint4 đỉnh **~10.9 GiB** (bản BF16 gốc là 17.4 GiB). Không còn
chiến lược offload nào để chỉnh: `accelerate device_map` đặt trọn khối lên card
đã chọn. Đây là một thắng lợi thật của lần thay lõi — card 24GB giờ chạy được,
trong khi bản Qwen cũ cần 26.4 GiB.

Đo trên máy của bạn:

```bash
make benchmark                                  # trong container đang chạy
python scripts/benchmark.py --runs 3            # trên host
python scripts/benchmark.py --width 2560 --height 1440
```

## Triển khai trên server

```bash
uv sync --extra dev
uv run download-model
uv run run-app   # phục vụ tại 0.0.0.0:7860
```

### Lấy link public để chia sẻ

Không cần cấu hình gì: app **luôn** dựng tunnel và in ra hai dòng khi khởi
động. Không dựng được thì nó dừng hẳn kèm `RuntimeError` (sau 3 lần thử)
thay vì phục vụ im lặng ở một cổng không ai với tới được.

```
* Running on local URL:  http://0.0.0.0:7860
* Running on public URL: https://xxxxxxxxxxxx.gradio.live
```

Link `*.gradio.live` **sống tối đa 1 tuần** và chết ngay khi tiến trình demo dừng.

### Chạy nền và đọc log

```bash
nohup .venv/bin/python -u scripts/serve.py > serve.log 2>&1 &
grep -a gradio.live serve.log
```

Cờ `-u` là **bắt buộc** khi chạy nền: Gradio in link bằng `print()`, mà stdout bị
redirect vào file sẽ chuyển sang block-buffer 8KB — không có `-u` thì link nằm kẹt
trong buffer, log chỉ hiện các dòng logger (ghi qua stderr) và tưởng như thiếu link.

## Chạy bằng Docker

Yêu cầu: Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Kiểm tra trước khi build:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Lệnh trên phải in ra bảng GPU. Nếu không, toolkit chưa được đăng ký với Docker
và `make run` sẽ thất bại ở bước cấp GPU.

Trọng số **không** nằm trong image — chúng được mount từ `./models` trên host.
Có hai cách đưa chúng vào đó:

- **Từ GCS (mặc định, khuyên dùng).** Không cần làm gì cả: service
  `model-fetcher` tự kéo về ở lần `up` đầu tiên. Xem
  [Triển khai trên Google Cloud](#triển-khai-trên-google-cloud-trọng-số-từ-gcs).
- **Từ HuggingFace.** Chạy `make download` ở host trước ít nhất một lần. Cách
  này cần `HF_TOKEN` trong `.env` và đã accept license của model.

```bash
make build           # dựng image (~12.7GB, lần đầu mất ~8 phút)
make run             # kéo trọng số (lần đầu) rồi khởi động container
make logs            # theo dõi load model + warm-up
make preflight       # kiểm tra cây model trên đĩa, không cần GPU
make benchmark       # đo tốc độ sinh ảnh thật
make stop            # dừng và xoá container
```

Cổng 7860 đang bận thì đổi: `make run HOST_PORT=7861`.

Không dùng `make` thì thay bằng `docker compose build` / `up -d` / `logs -f` / `down`.

### Những điều compose đã xử lý sẵn

| Vấn đề | Cách xử lý trong `docker-compose.yml` |
| :--- | :--- |
| Trọng số ~9.9GB | Mount `./models:/app/models:ro`, và `.dockerignore` loại `models/` khỏi build context |
| `.model_paths.env` ghi đường dẫn tuyệt đối của host | Bị loại khỏi image; compose set `IMG_MODEL_PATH` trỏ vào `/app/models` thay thế |
| Vô tình tải lại ~10GB giữa production | `HF_HUB_OFFLINE=1` biến mọi đường rò xuống Hub thành lỗi dừng hẳn |
| Phân mảnh VRAM khi shape ảnh thay đổi liên tục | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Người dùng đầu tiên phải gánh chi phí khởi tạo | `IMG_WARMUP=true` — warm-up chạy TRƯỚC khi mở cổng. Warm-up là một lượt sinh ảnh THẬT nên `HEALTHCHECK start-period` để 900s |
| `HF_TOKEN` | Không cần trong container (đã có weights local); compose ghi đè thành rỗng |
| Link share không hiện trong `docker logs` | `PYTHONUNBUFFERED=1` đã set sẵn trong Dockerfile |
| Restart policy khác nhau giữa bàn dev và server | Makefile dò GPU rồi set `IMG_RESTART_POLICY` — xem `make gpu-info` |

> `IMG_MODEL_PATH` trỏ vào MỘT thư mục duy nhất, không còn phụ thuộc
> `num_steps`/`rank`/`precision` như thời Nunchaku (tên file weight mã hoá
> các tham số đó). Đổi số bước giờ không phải sửa đường dẫn.

### Restart policy tự dò

`make run` gọi `nvidia-smi` và tự quyết định:

| Phần cứng | `restart` | Lý do |
| :--- | :--- | :--- |
| A100 (bất kể số lượng) | `unless-stopped` | Server production — container tự dậy lại sau reboot / OOM-kill |
| Đúng 1 GPU | `unless-stopped` | Cùng lý do |
| Nhiều GPU không phải A100 | `no` | Bàn dev — crash thì dừng hẳn cho dễ đọc traceback |
| Không dò được GPU | `no` | Mặc định an toàn |

Ghi đè tay: `make run IMG_RESTART_POLICY=no`.

Cấu hình hiện tại phục vụ **một tải tại một thời điểm** (một container, hàng đợi
Gradio mặc định xử lý tuần tự).

## Thuê A100 trên Google Cloud

Toàn bộ phần này dành cho việc dựng service từ một project GCP trống.

### 0. Chuẩn bị

Cần quota GPU trước — project mới luôn có quota A100 bằng 0:

```bash
gcloud auth login
gcloud config set project "$PROJECT_ID"

# Xem quota A100 hiện có ở vùng định thuê
gcloud compute regions describe us-central1 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -i a100
```

Nếu limit = 0 thì xin tăng ở **IAM & Admin → Quotas**, chọn metric
`NVIDIA_A100_GPUS`. Duyệt thường mất vài giờ đến một ngày làm việc.

A100 40GB có ở `us-central1-a/b/c/f`, `us-west1-b`, `europe-west4-a`,
`asia-southeast1-b/c` và một số zone khác — danh sách đổi theo thời điểm:

```bash
gcloud compute accelerator-types list --filter="name=nvidia-tesla-a100"
```

### 1. Tạo VM

Máy `a2-highgpu-1g` = **1× A100 40GB**, 12 vCPU, 85GB RAM. Với họ máy `a2-*`
thì GPU đã gắn sẵn trong machine type, **không cần** cờ `--accelerator`.

Dùng ảnh Deep Learning VM để có sẵn driver NVIDIA + Docker + nvidia-container-toolkit:

```bash
# Xem các image family đang có
gcloud compute images list --project deeplearning-platform-release \
  --filter="family~'common-cu'" --format="value(family)" | sort -u

gcloud compute instances create qwen-lightning \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --image-family=common-cu124-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=250GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --tags=qwen-lightning
```

Vài lựa chọn đã cân nhắc:

- **`--maintenance-policy=TERMINATE` là bắt buộc.** VM có GPU không live-migrate được.
- **Ổ 250GB.** Cần chỗ cho ~10GB weights + ~10GB image + OS — dư nhiều so với
  trước. Vẫn nên giữ ổ lớn: trên Persistent Disk, băng thông đọc **tăng theo
  dung lượng ổ**, ổ quá nhỏ làm bước nạp weights lên VRAM chậm hẳn.
- **A100 80GB** thì đổi sang `a2-ultragpu-1g`. Hoàn toàn không cần — model chỉ
  chiếm ~11 GiB. Thực ra A100 40GB cũng đã dư; giới hạn giờ là **tốc độ tính
  toán**, không phải VRAM.
- **Spot VM** (`--provisioning-model=SPOT`) rẻ hơn đáng kể nhưng bị thu hồi bất kỳ
  lúc nào. Hợp để test, không hợp cho service đang phục vụ thật.

> Tính tiền theo thời gian VM **chạy**, không theo mức dùng GPU. Không dùng thì
> `gcloud compute instances stop qwen-lightning --zone=us-central1-a`.
> Ổ đĩa vẫn tính tiền khi VM đã stop.

### 2. Kiểm tra máy

```bash
gcloud compute ssh qwen-lightning --zone=us-central1-a
```

Lần SSH đầu, ảnh DLVM sẽ hỏi cài driver NVIDIA — trả lời `y` (hoặc đã tự cài nếu
truyền `install-nvidia-driver=True`). Sau đó xác nhận:

```bash
nvidia-smi                    # phải thấy A100-SXM4-40GB và driver >= 525
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Driver phải từ **525 trở lên**: image dùng torch cu128, chạy được trên driver cũ
hơn CUDA 12.8 nhờ CUDA minor version compatibility, nhưng dưới 525 thì không.

### 3. Đưa code và weights lên

```bash
# Trên VM
git clone <repo-url> Demo_Qwen_Lightning && cd Demo_Qwen_Lightning
cp config_setup/base.example.yaml config_setup/base.yaml
nano config_setup/base.yaml    # sửa khối app: (hf_token, models_uri…)
```

Tải weights **thẳng trên VM** — nhanh hơn nhiều so với copy 27GB từ máy bạn lên,
và ingress vào GCP không mất phí:

```bash
make download                  # ~27GB
make preflight                 # xác nhận đủ file trước khi build
```

Nếu đã có sẵn weights trong một GCS bucket thì dùng cách này thay thế:

```bash
gcloud storage cp -r "gs://$BUCKET/qwen-models/*" ./models/
make preflight
```

### 4. Chạy

```bash
make build
make gpu-info                  # xác nhận dò ra A100 -> unless-stopped
make run
make logs                      # chờ "Warm-up xong ..." rồi "Ready."
```

Lần khởi động đầu mất vài phút: nạp 26.4GB weights từ đĩa lên VRAM, rồi warm-up
một lượt sinh ảnh. Healthcheck có `start-period=600s` nên container không bị báo
unhealthy trong lúc đó.

Đo tốc độ thật:

```bash
make benchmark
```

### 5. Truy cập

**Cách an toàn (khuyên dùng) — SSH port-forward, không mở cổng nào ra Internet:**

```bash
# Chạy trên MÁY BẠN
gcloud compute ssh qwen-lightning --zone=us-central1-a -- -L 7860:localhost:7860
```

Rồi mở `http://localhost:7860`.

**Cách mở cổng công khai** — chỉ làm khi thật sự cần, và luôn giới hạn IP nguồn:

```bash
MY_IP=$(curl -s ifconfig.me)
gcloud compute firewall-rules create allow-qwen-7860 \
  --allow=tcp:7860 \
  --target-tags=qwen-lightning \
  --source-ranges="$MY_IP/32"
```

Đừng dùng `--source-ranges=0.0.0.0/0`: Gradio không có xác thực, ai biết IP là vào
được và chạy model bằng tiền GPU của bạn.

Lưu ý link `*.gradio.live` mà app luôn tạo **cũng** là một đường công khai
không xác thực, và không tắt được bằng cấu hình. Trên server trả tiền theo
giờ, đừng để demo chạy khi không dùng — xem [Dọn dẹp](#dọn-dẹp-sau-khi-test).

### 6. Xong việc

```bash
gcloud compute instances stop qwen-lightning --zone=us-central1-a     # giữ ổ đĩa
gcloud compute instances delete qwen-lightning --zone=us-central1-a   # xoá hẳn
```

### Khắc phục sự cố

| Triệu chứng | Nguyên nhân thường gặp |
| :--- | :--- |
| `could not select device driver "" with capabilities: [[gpu]]` | nvidia-container-toolkit chưa đăng ký với Docker. Chạy lại lệnh kiểm tra ở mục 2 |
| Log dừng ở `Loading ...` rất lâu | Đang nạp 26.4GB từ Persistent Disk. Ổ nhỏ = đọc chậm; xem mục 1 |
| `FileNotFoundError: IMG_MODEL_PATH trỏ tới ...` | Mount `./models` sai hoặc chưa tải weights. Chạy `make preflight` |
| `OfflineModeIsEnabled` / `LocalEntryNotFound` | Thiếu file trong `models/`, code định tải từ Hub nhưng đã bị khoá offline. `make download` lại |
| `CUDA out of memory` | Có tiến trình khác đang giữ VRAM. `nvidia-smi` xem PID |
| Container `unhealthy` nhưng vẫn chạy | Warm-up lâu hơn `start-period=600s`. Xem `make logs` |

## Cho máy khác trong mạng LAN truy cập

Máy chủ demo: **`<demo-server-ip>`** (interface `enp6s0`, subnet `192.168.5.0/24`).
Container đã publish `0.0.0.0:7860` nên đã lắng nghe trên mọi interface — địa chỉ
để các máy khác mở là:

```
http://<demo-server-ip>:7860
```

### Docker và ufw: điều cần biết trước

Máy này có `ufw` đang **active**. Nhưng cổng do **Docker publish không đi qua
ufw**: Docker chèn rule riêng vào chain `DOCKER` của iptables, được xét trước
luật ufw. Nghĩa là:

- **Chạy bằng Docker** (`make run`): LAN thường **đã truy cập được ngay**, không
  cần `ufw allow`. Cứ thử từ máy khác trước khi đụng vào firewall.
- **Chạy trực tiếp trên host** (`uv run run-app`): ufw **có** chặn, phải mở cổng.

### Mở cổng cho subnet công ty

```bash
sudo ufw allow from 192.168.5.0/24 to any port 7860 proto tcp comment 'Qwen demo LAN'
sudo ufw status numbered          # kiểm tra luật đã vào
```

Giới hạn theo subnet thay vì `sudo ufw allow 7860` để demo không mở ra ngoài
phạm vi mạng nội bộ.

### Kiểm tra từ máy khác

```bash
curl -I --max-time 5 http://<demo-server-ip>:7860     # mong đợi: HTTP/1.1 200 OK
```

Không thông thì lần lượt loại trừ:

```bash
# trên máy chủ: container còn sống và đang map cổng?
docker compose ps
ss -ltnp | grep 7860              # phải thấy 0.0.0.0:7860

# hai máy có cùng subnet không?
ip -4 addr show                   # trên máy client
ping <demo-server-ip>
```

Nếu `ss` cho thấy `127.0.0.1:7860` thay vì `0.0.0.0:7860` thì `IMG_SERVER_NAME`
đang sai — `docker-compose.yml` đã ép `0.0.0.0`, chỉ xảy ra khi chạy ngoài Docker.

### Link public luôn bật — không tắt được

Mỗi lần khởi động app tạo một link `*.gradio.live` mở ra Internet, và không
có biến môi trường nào tắt được. Đây là lựa chọn có chủ ý: đường chạy chính
của repo là Colab / máy thuê, nơi cổng 7860 không tiếp cận được từ ngoài nên
thiếu link đồng nghĩa demo không dùng được.

Đổi lại, **mọi lần chạy đều mở một endpoint không xác thực ra Internet**, kể
cả khi bạn chỉ định test trong LAN. Test nội bộ xong thì dừng demo ngay
(xem [Dọn dẹp](#dọn-dẹp-sau-khi-test)) chứ đừng để chạy nền. Cần chạy dài
ngày mà không muốn link thì sửa thẳng `launch_with_public_link()` trong
`src/imagegen/ui/launcher.py`.

### Lưu ý khi test tải

Demo phục vụ **tuần tự một request tại một thời điểm** (hàng đợi Gradio mặc
định, một container, một pipeline trên GPU). Nhiều máy bấm cùng lúc sẽ **xếp
hàng** chứ không chạy song song: 5 máy đồng thời thì máy cuối đợi khoảng
5 × 11s ≈ 55s. Đó là hành vi đúng chứ không phải nghẽn — chạy song song trên
cùng GPU chỉ làm chậm tất cả và dễ hết VRAM.

Theo dõi hàng đợi khi test:

```bash
docker compose logs -f | grep -a "⚡ GPU"
```

## Dọn dẹp (giải phóng tài nguyên)

Sau khi dùng xong, demo vẫn chiếm **VRAM** và **cổng 7860** ngay cả khi bạn đã
đóng tab trình duyệt. Chạy script sau để dừng demo — script xử lý **cả hai kiểu
chạy**: container Docker của dự án và tiến trình chạy trực tiếp trên host:

```bash
# Cách 1: thực thi trực tiếp (file đã có quyền thực thi)
./cleanup.sh

# Cách 2: qua bash
bash cleanup.sh
```

`cleanup.sh` sẽ:
- Đọc cổng từ env `IMG_PORT` / `.env` (ưu tiên) rồi mới tới `config_setup/base.yaml` (`app.server_port`).
- **Dừng container Docker của dự án trước tiên** (`docker compose down`). Thứ tự
  này bắt buộc: khi container đang map cổng, `fuser` trả về PID của
  `docker-proxy` — kill nó không dừng container (VRAM vẫn bị giữ) mà chỉ phá
  port mapping của Docker. Nếu cổng do một container **ngoài** compose của dự án
  giữ, script chỉ cảnh báo chứ không kill.
- Tìm PID đang chiếm cổng (dùng `fuser` hoặc `lsof`) và gửi `SIGTERM`, nếu không
  phản hồi sau 2s thì `SIGKILL`.
- Dọn tiến trình tunnel `frpc` (app luôn tạo) **chỉ của đúng cổng
  đó** — máy có thể đang chạy demo Gradio khác, không đụng tới tunnel của họ.
- In trạng thái VRAM hiện tại (`nvidia-smi`) nếu có.

> Lưu ý: trọng số model nằm trong thư mục `models/` (đã được `.gitignore`) **không**
> bị xoá — chúng chỉ chiếm dung lượng ổ đĩa, không chiếm RAM/VRAM khi demo đã dừng.
> Nếu muốn xoá luôn trọng số để thu hồi ổ đĩa:
>

```bash
rm -rf models .model_paths.env
```

## Queue service trên Docker — vì sao `queue_out` im lặng

Ba cái bẫy, theo thứ tự hay gặp:

1. **`docker compose up -d` KHÔNG chạy queue worker.** Service `qwen-queue`
   nằm sau `profiles: ["queue"]`, nên lệnh deploy mặc định chỉ dựng
   `model-fetcher` + `qwen-lightning` (Gradio). Container chạy xanh, GPU có
   tải, mà không ai consume inbound queue → `gen-image-queue-out` mãi mãi
   rỗng. Chạy đúng:

   ```bash
   make queue                  # = docker compose --profile queue up -d qwen-queue
   make queue-logs
   curl -s localhost:8395/readyz
   ```

2. **Thiếu `config_setup/base.yaml`.** File này bị `.gitignore` chặn nên
   KHÔNG có sau khi clone; compose mount `./config_setup` từ host chứ không
   bake vào image. Thiếu nó thì container chết ngay lúc khởi động.
   `make queue` kiểm tra trước và báo lỗi thay vì để container chết im.

3. **Thiếu key upload GCS** (`config_setup/credentials/user-upload-key.json`).
   `create_storage()` chạy trước khi consumer start, nên key sai/thiếu cũng
   giết service trước khi có message nào được đọc.

Cả 3 đều có chung một triệu chứng — inbound queue đầy dần, `queue_out` = 0,
`consumers` = 0 trên RabbitMQ UI. Nhìn cột `consumers` trước khi nghi ngờ
model.

### Smoke test đường queue — chạy được khi CHƯA có A100

`Application.run()` nạp model TRƯỚC khi spawn consumer/publisher, nên bình
thường không có cách nào thử chuỗi broker → worker → `queue_out` nếu máy
không có GPU + 26GB weight. Service `qwen-queue-smoke` (profile `smoke`) chạy
cùng image nhưng với `config_setup/smoke.yaml`: `processor.type: echo` (trả
ảnh placeholder, không nạp model) + `storage.backend: local` (không cần key
GCS). Mọi mắt xích còn lại giữ nguyên bản.

```bash
export IMG_RMQ_HOST=<broker-host> IMG_RMQ_VHOST=gen-image \
       IMG_RMQ_USER=<username> IMG_RMQ_PASSWORD='...'

make smoke                                              # dựng container
python scripts/loadtest_queue.py --rate 5 --duration 10  # bắn tải
make smoke-logs                                          # xem [OUTBOUND]
make smoke-stop
```

Reply trên `queue_out` có đúng hình dạng production:

```json
{
  "_id": "...", "os": "ios", "firebase_token": "...",
  "status_code": 200, "message": "success",
  "result": { "url": "file:///app/data/smoke_out/...png", "info": "echo placeholder 512×512" }
}
```

Chỉ khác `result.url`: smoke trả `file://`, production trả URL GCS/CDN.
Smoke pass mà bản thật im lặng → lỗi nằm ở model/GPU, không phải ở queue.

Thông số broker của cả hai service đều nhận env `IMG_RMQ_HOST` / `PORT` /
`USER` / `PASSWORD` / `VHOST` / `EXCHANGE` ghi đè lên file yaml, nên password
không bắt buộc phải nằm trong `base.yaml`. Đổi hẳn file config bằng
`IMG_QUEUE_CONFIG=/app/config_setup/<file>.yaml`.

## Test tải queue (RabbitMQ)

`scripts/loadtest_queue.py` bắn message liên tục vào 4 inbound queue để đo
broker + đường vào. Nó **chỉ publish** (và tuỳ chọn đọc `queue_out`), không
nạp model — chạy được từ máy dev không có GPU. Phụ thuộc duy nhất là
`pika` + `pyyaml`, nên không cần cài cả môi trường torch:

```bash
uv venv /tmp/ltvenv && uv pip install --python /tmp/ltvenv/bin/python pika pyyaml
```

```bash
# Smoke 30s, 5 msg/s, tự tạo topology nếu vhost còn trống
export IMG_RMQ_PASSWORD='...'        # secret: để trong env, đừng gõ ra dòng lệnh
/tmp/ltvenv/bin/python scripts/loadtest_queue.py \
    --host <broker-host> --vhost gen-image --user <username> \
    --declare --rate 5 --duration 30

# Đẩy tải: 200 msg/s trong 2 phút, 4 connection, tắt publisher confirm
... --rate 200 --duration 120 --workers 4 --no-confirm

# Đo end-to-end (cần queue worker đang chạy): đọc luôn queue_out
... --rate 2 --duration 300 --drain-out
```

Thông số broker lấy theo thứ tự **CLI > env `IMG_RMQ_*` > `config_setup/base.yaml`**,
nên trên server đã có `base.yaml` thì chỉ cần `--rate` / `--duration`; không
phải gõ mật khẩu ra dòng lệnh.

Vài điểm cần biết trước khi chạy:

- **`--mix premium:basic_1:basic_2:basic_3`** (mặc định `1:5:3:2`) — phần
  basic khớp `worker.basic_tier_weights`, tức đúng tỉ lệ weighted round-robin
  mà `PriorityRouter` dùng để *lấy* message ra. Script trải weights bằng
  smooth WRR (thuật toán của nginx) nên không bắn 5 message basic_1 liền
  nhau — bắn theo cụm sẽ tạo burst giả và làm sai cả số liệu backlog lẫn
  hành vi ưu tiên của consumer.
- **`--no-confirm`** tắt publisher confirm. Với confirm bật, `basic_publish`
  block tới khi broker ack → trần ~550 msg/s trên một connection; tắt đi thì
  vượt 10k msg/s nhưng con số đó không còn là "broker đã nhận thật".
- **`--drain-out`** consume `queue_out` để tính e2e latency + phổ
  `status_code`. Nó **lấy mất** reply khỏi queue — đừng bật khi BE thật đang
  đọc queue đó.
- **`--purge`** xoá sạch 4 inbound queue trước khi bắn, để số liệu không lẫn
  backlog cũ.
- **`--declare`** tạo exchange + queue đúng như `topology_mode:
  declare_minimal` (không DLX args), nên service start sau đó không dính
  `PRECONDITION_FAILED 406` vì lệch argument.

Payload dựng theo đúng hợp đồng trong
`queue_service/messaging/schemas.py`. Mặc định là text-to-image; thêm
`--image-url` (lặp nhiều lần, ảnh CUỐI là ảnh nền) để test nhánh image-edit.
Sửa schema bên đó thì phải sửa `build_payload()` trong script, nếu không
worker sẽ reply `BAD_REQUEST` cho toàn bộ tải test.

## Tests

```bash
uv run pytest                 # toàn bộ, không cần GPU
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

Test suite cố tình **không cần torch**: `imagegen.hidream` nạp lười (PEP 562),
và bảng độ phân giải + hợp đồng file model nằm ở hai module thuần stdlib
(`hidream/resolutions.py`, `hidream/artifacts.py`). Một test
(`test_table_matches_vendor`) đối chiếu bảng chép tay với bản upstream trong
`vendor/` và tự bỏ qua khi thiếu torch — nó là chốt chặn sau mỗi lần chạy
`scripts/vendor_hidream.sh`.

| File | Kiểm gì |
| :-- | :-- |
| `test_config.py` | Thứ tự ưu tiên env/yaml/default, `model_type` kéo theo `num_steps`, không còn tiền tố `IMG_` |
| `test_model_paths.py` | Resolve thư mục model, phát hiện snapshot tải dở, gắn đủ 5 special token |
| `test_resolution.py` | Snap độ phân giải, không trôi khỏi bảng upstream |
| `test_cache_location.py` | `HF_HOME` nằm trong project, `download` không gọi `login()` |
| `test_recipe.py` | `build_recipe()` — ràng buộc chéo scheduler / timesteps / noise params |
| `test_config_contract.py` | `base.example.yaml` khớp dataclass; `config_env.py` map đủ key |

## Cấu trúc dự án

```
src/imagegen/
  __init__.py              # ghim HF_HOME + allocator CUDA vào project
  config.py                # Settings + load_settings() — nguồn config duy nhất
  device.py  tuning.py     # chọn GPU; cờ torch + warm-up
  download.py              # tải snapshot model, ghi .model_paths.env
  logging_setup.py         # LOGGER_NAME + configure_logging()
  hidream/                 # TẦNG LÕI
    artifacts.py           #   hợp đồng file snapshot (thuần stdlib)
    resolutions.py         #   11 độ phân giải cố định (thuần stdlib)
    loader.py              #   sdnq + AutoProcessor + Qwen3VL
    inference.py           #   Recipe + generate() cho cả t2i lẫn editing
    vendor/                #   code HiDream copy từ GitHub — xem VENDOR.md
  ui/
    app.py                 #   7 tab Gradio + main()
    components.py          #   widget + preset dùng chung
    launcher.py            #   tunnel *.gradio.live
    prompts.py             #   prompt hệ thống theo task
    examples_spec.py       #   danh sách test case mẫu
  queue_service/           # RabbitMQ consumer/publisher, pipeline, audit, storage
scripts/                   # download_model, serve, preflight, benchmark,
                           #   warm_examples, loadtest_queue, vendor_hidream.sh,
                           #   fetch_models.sh, pack_models.sh, config_env.py
tests/                     # pytest — chạy được không cần GPU
config_setup/base.yaml           # NGUỒN CONFIG DUY NHẤT (app + queue service)
config_setup/base.example.yaml   #   template (commit); base.yaml bị gitignore
config_setup/smoke.yaml          #   smoke test đường queue (processor: echo)
docs/                      # architecture.md, QUEUE_SERVICE.md,
                           #   MODEL_HIDREAM_O1.md
main_queue.py              # entry point queue worker
cleanup.sh                 # dọn tiến trình / giải phóng VRAM, port
Dockerfile                 # image 2 stage (builder venv -> runtime)
Dockerfile.fetch           # image riêng kéo weights từ GCS
docker-compose.yml         # imagegen / imagegen-queue / imagegen-queue-smoke
Makefile                   # download / preflight / build / run / queue / smoke
```
