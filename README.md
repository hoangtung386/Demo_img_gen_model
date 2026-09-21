# FLUX.2-klein-9B GGUF — Service sinh ảnh

Sinh ảnh từ chữ và sửa ảnh theo tham chiếu bằng
[**FLUX.2-klein-9B**](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
với transformer [**GGUF Q4_K_M của Unsloth**](https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF)
(bản step-distilled, 4 bước) qua `diffusers`. Ship hai entrypoint dùng chung
một tầng model:

- **RabbitMQ consumer** (`main_queue.py`) — đường production.
- **Gradio demo 7 tab** (`scripts/serve.py`) — để thử và trình diễn.

> Backend trước đây là Qwen-Image-Edit-2509 Lightning (INT4 / Nunchaku). Lý
> do đổi, số liệu VRAM và những gì mất đi: xem
> [docs/adr/0001-flux2-klein-backend.md](docs/adr/0001-flux2-klein-backend.md).
>
> 🚀 **Trước khi áp dụng bất kỳ hướng dẫn "tối ưu image gen" nào**, đọc
> [docs/PERFORMANCE.md](docs/PERFORMANCE.md). Phần lớn lời khuyên trên mạng
> viết cho SDXL 30 bước; ở đây có cái đã có sẵn, và có cái **làm hỏng ảnh**
> (đổi sampler trên model đã chưng cất).

## Yêu cầu

- Python >= 3.10 (khuyên dùng môi trường ảo `uv`)
- GPU NVIDIA >= 16GB VRAM. Mục tiêu tối ưu là **L4 24GB**: pipeline ở cấu
  hình mặc định (transformer GGUF + text encoder NF4) chiếm ~11 GiB nên
  thường trú được, không cần offload.
- ~20GB đĩa cho trọng số
- Đã accept license của
  [`black-forest-labs/FLUX.2-klein-9B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
  trên Hugging Face (model Apache 2.0, nhưng repo có thể vẫn yêu cầu đồng ý)

> ⚠️ **Phase 0 chưa nghiệm thu.** Mọi số VRAM/latency trong README này là
> tính toán hoặc suy luận từ mã nguồn diffusers, **chưa đo trên L4 thật**.
> Chạy `python scripts/spike_flux2.py` rồi điền vào ADR trước khi deploy.

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
uv run download-model     # tải trọng số
uv run run-app            # chạy demo
```

Hoặc cách truyền thống (pip-style):

```bash
uv venv
uv pip install -e .
# uv pip install -e ".[dev]"   # kèm dev tools
```

## Cấu hình (config_setup/base.yaml)

Toàn bộ config nằm trong **một file** `config_setup/base.yaml` (khối `app:`) —
gộp từ `config/example.env` cũ. `download`, `serve` và các script model đều đọc
từ đây. Sao chép template rồi điền giá trị:

```bash
cp config_setup/base.example.yaml config_setup/base.yaml
# mở config_setup/base.yaml, sửa khối app: (đặt hf_token nếu tải model gated,
# hoặc để trống rồi export HF_TOKEN qua môi trường cho an toàn)
```

> Mọi key vẫn override được bằng biến môi trường `GENIMG_* / HF_TOKEN / MODEL_ROOT`
> (env THẮNG YAML) — docker-compose set env nên không phá deploy hiện tại.
> `.env` cũ vẫn được nạp nếu tồn tại (backward-compat).

### Token HuggingFace — ba nguồn, theo thứ tự

`black-forest-labs/FLUX.2-klein-9B` là repo **gated**. Repo GGUF của Unsloth
chỉ chứa transformer, nên nó **không** thay thế được repo gated: text
encoder, VAE, scheduler và tokenizer vẫn phải lấy từ đó.

1. env `HF_TOKEN` — chắc ăn nhất, dùng cho CI/Colab
2. `app.hf_token` trong `base.yaml`
3. token do `huggingface_hub.login()` ghi ra đĩa

Nguồn (3) được đọc ở **cả hai** vị trí: `$HF_HOME/token` (đã ghim vào
`models/.hf`) lẫn `~/.cache/huggingface/token` (mặc định của hub). Không có
nhánh thứ hai đó thì một `login()` chạy trong notebook cell sẽ báo thành
công nhưng `download-model` vẫn đi ẩn danh và nhận 401 — hai tiến trình nhìn
vào hai file khác nhau.

Các key quan trọng trong `app:` (đọc bởi cả `download` và `serve`):

| Biến | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `GENIMG_NUM_STEPS` | `4` | Bản distilled được chưng cất về đúng 4 bước |
| `GENIMG_GUIDANCE_SCALE` | `1.0` | Giá trị model card. Bản distilled KHÔNG chạy CFG |
| `GENIMG_QUANTIZATION` | `gguf` | Transformer: `gguf` / `bf16` / `fp8` — 9B mặc định GGUF |
| `GENIMG_TEXT_ENCODER_QUANTIZATION` | `nf4` | **Text encoder** Qwen3-8B: `nf4` / `int8` / `bf16`. Knob riêng — `GENIMG_QUANTIZATION` không chạm tới nó |
| `GENIMG_COMPILE` | `false` | `torch.compile` transformer; chỉ có tác dụng ở nhánh bf16 |
| `GENIMG_OFFLOAD` | `resident` | `resident` / `model_offload` |
| `GENIMG_BASE_MODEL` | `black-forest-labs/FLUX.2-klein-9B` | PHẢI là bản distilled |
| `GENIMG_BASE_MODEL_LOCAL` | (rỗng) | Path weight đã có trên đĩa; rỗng → tải từ Hub |
| `GENIMG_TRANSFORMER_GGUF` | (rỗng) | Path file `.gguf`; chỉ đọc khi `quantization: gguf` |
| `GENIMG_WARMUP` | `true` | Chạy lượt sinh ảnh giả lúc khởi động để người dùng đầu tiên không phải chờ |
| `GENIMG_PORT` | `7860` | Cổng Gradio |
| `GENIMG_SERVER_NAME` | `0.0.0.0` | Bind address (để chạy trên server) |
| `GENIMG_DEMO_CACHE` | `true` | Ví dụ mẫu trả ảnh dựng sẵn thay vì chạy model (xem bên dưới) |

Chỉ còn hai chiến lược đặt model, không còn `auto` tự dò phần cứng:

| Giá trị | Khi nào | Ghi chú |
| :--- | :--- | :--- |
| `resident` | GPU >= 20 GiB (L4, A100, H100…) | Mọi component ở lại GPU. **Mặc định.** |
| `model_offload` | Card nhỏ hơn, hoặc nhiều process/GPU | Weight ở CPU, kéo lên theo module. Chậm hơn đáng kể. |

> `auto` đã bị bỏ có chủ đích: tự dò nghe tiện nhưng nó biến một lỗi cấu hình
> thành một quyết định âm thầm — service vẫn khởi động, chỉ là chậm gấp đôi
> vì đã lặng lẽ rơi về offload. Giá trị sai giờ là lỗi dừng hẳn.

> ⚠️ Tiền tố cũ `QIE_*` **không còn tác dụng**. Đây là cú cắt sạch có chủ
> đích: vì env thắng YAML, một `QIE_TRANSFORMER_PATH` còn sót trong môi
> trường deploy sẽ âm thầm ghi đè cấu hình đúng và trỏ pipeline về trọng số
> Qwen không còn tồn tại. Quét sạch môi trường trước khi deploy.

### Deploy trên L4 24GB

Cấu hình đọc từ `config_setup/base.yaml`, mount read-only qua
`- ./config_setup:/app/config_setup:ro` — **cùng file** cho cả hai service,
chỉ khác khối được đọc: `gen-image` (Gradio) đọc khối `app:`,
`gen-image-queue` đọc khối `processor:`.

```bash
cp config_setup/base.example.yaml config_setup/base.yaml
# base.example.yaml đã điền sẵn giá trị khuyến nghị cho L4:
#   quantization: "gguf"   text_encoder_quantization: "nf4"
#   offload: "resident"    num_steps: 4
docker compose up -d --build
```

#### Vì sao không cần offload nữa

| | Backend cũ (Qwen) | FLUX.2-klein-9B (mặc định) |
| :--- | ---: | ---: |
| `text_encoder` | Qwen2.5-VL **15.4 GB** | Qwen3-8B **NF4 ~5.0 GB** (bf16 là 16.4) |
| `transformer` | INT4 10.7 GB | GGUF Q4_K_M 5.9 GB |
| `vae` | 0.24 GB | ~0.3 GB |
| **Tổng thường trú** | **26.4 GB — không vừa L4** | **~11.2 GB — vừa** |

> 🔴 Con số 11.2 GB chỉ đúng khi nén **cả hai**. `quantization: "gguf"` một
> mình KHÔNG chạm tới text encoder: để nó ở `bf16` thì tổng là 22.6 GB, sát
> trần L4 tới mức OOM ngay lượt denoise đầu. Xem
> [docs/PERFORMANCE.md §0](docs/PERFORMANCE.md).

Backend cũ phải đẩy text encoder ra CPU và chuyển qua lại PCIe mỗi request.
FLUX.2-klein-9B đã nén thì vừa GPU, nên `offload: "resident"` giữ mọi
component tại chỗ và **không có lần chuyển CPU↔GPU nào lúc suy luận**. Cả tầng `models/offload.py`
(4 chiến lược, ~200 dòng) đã bị xoá.

Còn đúng một đường lùi, `offload: "model_offload"`, cho card nhỏ hơn L4 —
chậm hơn đáng kể, đừng dùng nếu không bắt buộc.

#### Bắt buộc kiểm tra sau khi đổi máy/VM mới

`base.yaml` bị `.gitignore` chặn (chứa secret) nên không tự có sau
`git clone`. Nếu thiếu file hoặc mount sai đường dẫn, `gen_image.config`
**không dừng hẳn** — nó chỉ in WARNING rồi rơi về default hard-code. Luôn
xác nhận trước khi tin tưởng:

```bash
python scripts/preflight.py                      # KHÔNG cần GPU, chạy vài giây
docker compose exec gen-image-queue cat /app/config_setup/base.yaml | grep -E "quantization|offload"
docker compose logs gen-image-queue | grep -iE "CẢNH BÁO|Đặt model|Quantization"
```

`scripts/preflight.py` cũng cảnh báo nếu `model_index.json` thiếu
`is_distilled` — dấu hiệu đã tải nhầm `FLUX.2-klein-base-4B` (không
distilled, chậm hơn ~4 lần ở cùng cấu hình).

#### bf16 hay GGUF?

Mặc định là **bf16**, dù người đặt hàng chỉ đích danh repo GGUF. Lý do: GGUF
trong diffusers giải nén về `compute_dtype` **ngay trong forward** — đổi dung
lượng lấy băng thông, mà băng thông (300 GB/s trên L4) mới là nút cổ chai.
Vì bf16 *đã vừa* 24 GB, đổi lấy dung lượng là đổi lấy thứ ta không thiếu.
GGUF cũng không `torch.compile` được
([diffusers#10795](https://github.com/huggingface/diffusers/issues/10795)).

Chọn `quantization: "gguf"` khi — và chỉ khi — cần nhồi **2 worker process
lên cùng một L4**; xem [docs/MULTI_PROCESS_WORKERS.md](docs/MULTI_PROCESS_WORKERS.md).

#### Đo lại trên phần cứng của bạn

Đừng tin bảng nào trong README này: chúng là tính toán, chưa phải số đo.

```bash
python scripts/spike_flux2.py       # cổng chặn Phase 0 (xem ADR 0001)
make benchmark                      # ma trận t2i / edit / edit2
```

## Tải trọng số

```bash
python scripts/download_model.py
# hoặc: uv run download-model
```

Script tải:
1. Các component pipeline cần cho GGUF (`black-forest-labs/FLUX.2-klein-9B`) →
   `models/FLUX.2-klein-9B` — gồm text encoder, VAE, scheduler, tokenizer và
   `transformer/config.json`; transformer BF16 không được tải.
   vae, scheduler, tokenizer.
2. Đúng **một** file `.gguf` (`flux-2-klein-9b-Q4_K_M.gguf`) →
   `models/gguf/`. Repo GGUF có ~15 bản lượng tử hoá; script cố tình không
   `snapshot_download` cả repo.

Sau đó ghi đường dẫn local vào `.model_paths.env` để `serve` dùng trực tiếp
(không tải lại lúc chạy).

Hai điều đáng biết:

- **Không có gì rơi vào `~/.cache/huggingface`.** Cache của HuggingFace đã được
  ghim vào `models/.hf` ngay trong `gen_image/__init__.py`, nên toàn bộ dữ
  liệu tải về nằm trong project. Đặt `HF_HOME` tường minh nếu muốn chỗ khác.
- **Tải đúng bản distilled.** `FLUX.2-klein-base-4B` là repo KHÁC: không
  distilled, cần vài chục bước và guidance thật — chậm hơn ~4 lần ở cùng
  cấu hình. `make preflight` cảnh báo nếu tải nhầm.

Kiểm tra cây model bất cứ lúc nào (không cần GPU):

```bash
make preflight
```

## Triển khai trên Google Cloud (trọng số từ GCS)

Đây là đường deploy chính: clone repo, thả file key, chạy một lệnh. **Không
cần `.env`, không cần `HF_TOKEN`, không chạm tới HuggingFace.**

```bash
git clone -b demo/FLUX_2_klein_9B https://github.com/hoangtung386/Demo_img_gen_model.git
cd Demo_img_gen_model
cp /directory-path/ai-service-account.json .      # key service account GCS
docker compose up -d --build
```

Thế thôi. `docker compose up` chạy hai service theo thứ tự:

1. **`model-fetcher`** — xác thực bằng `ai-service-account.json`, tải
   `gs://ai-service-account/models.tar.zst` về rồi giải nén vào
   `./models/`, tự kiểm tra cây thư mục, xong thì thoát. Stream thẳng qua
   `gcloud storage cat | zstd -d | tar -x` — không có archive trung gian
   chạm đĩa, chỉ cần ~27 GB đĩa trống (xem mục "Đổi archive" bên dưới).
2. **`gen-image`** — chỉ khởi động khi bước trên trả về 0
   (`depends_on: service_completed_successfully`).

Lần chạy sau, fetcher thấy `models/.fetched-from` khớp URI và còn đủ file thì
bỏ qua, không tải lại. Ép tải lại: `GENIMG_MODELS_FORCE=true docker compose up`.

> **Đến đây là xong — không phải chạy thêm lệnh nào.** `CMD` của image chính
> là `python scripts/serve.py`, và compose không ghi đè nó. Mục
> [Chạy trực tiếp](#chạy-trực-tiếp-không-docker) bên dưới là đường **thay
> thế** cho ai chạy không qua Docker (Colab, máy thuê), không phải bước tiếp
> theo của mục này.

### Vòng đời container

```bash
docker compose logs -f              # theo dõi tải model + warm-up (~2-3 phút)
docker compose ps                   # xem trạng thái
docker compose restart gen-image   # khởi động lại app, không tải lại model
docker compose down                 # dừng và xoá container
```

Dùng `docker compose down` chứ không phải `docker stop`: `docker stop` chỉ
dừng container app và để nó nằm lại ở trạng thái exited, lần sau `up` sẽ báo
xung đột tên. `down` dọn cả app lẫn container fetcher đã thoát.

Thư mục `./models` **không** bị `down` đụng tới — nó là bind mount trên host,
nên lần `up` sau không phải tải lại.

Không có `docker run` một dòng cho service này, vì nó là **hai** container
chạy có thứ tự: fetcher phải xong trước, app mới được khởi động
(`depends_on: service_completed_successfully`). Muốn viết tay bằng `docker
run` thì phải tự chạy hai lệnh đúng thứ tự và tự truyền lại `--gpus all`,
ánh xạ cổng, ba volume mount và khoảng mười biến môi trường mà
`docker-compose.yml` đang giữ. Compose tồn tại chính là để khỏi phải làm việc
đó.

### Build cài những gì

`Dockerfile` dùng **Python 3.13 + uv**, tương đương hai lệnh chạy tay:

```bash
uv sync --frozen --no-dev --no-install-project
```

Một lệnh là đủ — không còn wheel ngoài PyPI nào phải cài tay. Trước đây
`nunchaku` chỉ phát hành wheel trên GitHub releases, dựng riêng cho từng tổ
hợp `python × torch × CUDA`, nên `pyproject.toml` phải ghim 4 URL trong
`[tool.uv.sources]` và Dockerfile phải cài lại tường minh **sau** `uv sync`
để diffusers không bị kéo ngược phiên bản. Cả khối đó đã biến mất.

Build có bước chốt phiên bản (`scripts/verify_build.py`) đọc metadata —
không import torch/diffusers, vì builder không có GPU. Nó cũng khẳng định
`nunchaku` **vắng mặt**: package đó ghim `diffusers==0.36` và sẽ kéo tụt
phiên bản xuống dưới ngưỡng `Flux2KleinPipeline` cần nếu lỡ quay lại qua một
dependency bắc cầu.

```
✅ OK — diffusers 0.40.0, torch 2.9.0, transformers 5.15.0
```

### Xác thực GCS

Hai key nằm chung trong `config_setup/credentials/` (mỗi key một vai trò):

- `model-download-key.json` — tải trọng số model nội bộ (`fetch_models.sh`).
- `user-upload-key.json` — upload ảnh kết quả lên bucket cho user (queue service).

`scripts/fetch_models.sh` dò key download theo thứ tự: `$GENIMG_GCS_KEY_FILE` →
`/credentials/model-download-key.json` → `/project/config_setup/credentials/
model-download-key.json` → **`ai-asset-amb.json` ở gốc repo** → bất kỳ
`ai-asset*.json` nào. `config_setup/credentials` được mount `:ro` vào
`/credentials` trong container fetcher; key **không** đi vào image và không nằm
trong build context (`.dockerignore` chặn `*-key.json`, `ai-asset*.json`).

Không có file key nào thì script rơi về ADC — **cách nên dùng trên GCE**: gắn
service account vào VM, không có file key nào để rò rỉ, thu hồi bằng một lệnh
IAM. Quyền tối thiểu: `roles/storage.objectViewer` trên bucket.

### Đổi archive

Mặc định đã ghim trong `docker-compose.yml`, không cần đặt biến nào:

```yaml
GENIMG_MODELS_URI: "${GENIMG_MODELS_URI:-gs://<your-bucket>/models.tar.zst}"
```

`fetch_models.sh` nhận `.zip`, `.tar.zst`, `.tar.gz`, `.tar`, hoặc một prefix
thư mục. Archive bọc trong một cấp `models/` — như `zip -r models.zip models/`
tạo ra — được nhận diện và bỏ cấp đó tự động, nên không ra `models/models/...`.

**Dùng `.tar.zst` chứ không phải `.zip`** — từng dùng `.zip` và đã hết đĩa
thật giữa production. Số đo trên chính trọng số của repo này, mẫu 500MB lấy
từ giữa file transformer INT4:

| | Thời gian | Tiết kiệm |
| :--- | ---: | ---: |
| `gzip -6` (tức `zip`) | 42.9s | 32.8% |
| `zstd -3 -T0` | **0.7s** | **34.9%** |
| `zstd -10 -T0` | 7.0s | 35.9% |

zstd nhanh hơn 60 lần **và** nén tốt hơn. Nhưng lý do quan trọng hơn là
**tar stream được, zip thì không**: central directory của zip nằm ở cuối
file, nên VM buộc phải tải trọn 21.5 GB xuống đĩa rồi mới giải nén thêm
26.4 GB nữa — tức **~48 GB đĩa trống** phải có. Với `.tar.zst`,
`gcloud storage cat | zstd -d | tar -x` chỉ cần 27 GB, và một lần tải dở dang
cũng không để lại archive cụt.

Đóng gói lại khi đổi trọng số (rank/steps/precision khác, hoặc cập nhật
model):

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
có prompt chuyên biệt viết sẵn (xem `src/gen_image/ui/prompts.py`):

| Tab | Đầu vào | Kết quả |
| :--- | :--- | :--- |
| **1. Virtual Try-On** | Ảnh 1: trang phục — Ảnh 2: người mẫu *(ảnh nền)* | Người mẫu mặc trang phục mới, giữ nguyên khuôn mặt, dáng, tư thế, màu da và bối cảnh |
| **2. Home Design** | **Một** ảnh căn phòng + mô tả thiết kế tự viết | Chính căn phòng đó được dọn sạch đồ cũ và bài trí lại theo mô tả, giữ nguyên tường, cửa sổ, cửa, sàn và góc chụp |
| **3. Image to Cartoon** | 1 ảnh chụp có người và phong cảnh | Toàn bộ người trong ảnh thành nhân vật hoạt hình giữ tối đa nét mặt, kiểu tóc, vóc dáng, trang phục; phong cảnh vẽ lại cùng phong cách |
| **4. Ghép 2 người ôm nhau** | Ảnh 1: người thứ nhất — Ảnh 2: người thứ hai *(ảnh nền)* | Một ảnh duy nhất hai người đang ôm nhau, giữ khuôn mặt, kiểu tóc, vóc dáng và trang phục của từng người |
| **5. Face Swap** | Ảnh 1: mặt đã crop — Ảnh 2: ảnh đầy đủ có người và phong cảnh *(ảnh nền)* | Chính ảnh 2 nhưng khuôn mặt đã đổi sang người ở ảnh 1, giữ nguyên phong cảnh, quần áo, dáng người, khung hình — và trả về **đúng kích thước pixel của ảnh 2** |
| **6. Prompt to Image** | Chỉ prompt + tỉ lệ khung, **không cần ảnh** | Ảnh mới sinh hoàn toàn từ mô tả |
| **7. Image + Prompt** | 1 ảnh + yêu cầu sửa tự viết | Ảnh đã sửa theo yêu cầu — tab tổng quát, sáu tab trên chỉ là prompt chuyên biệt viết sẵn cho cùng pipeline này |

Năm tab đầu có khối **Ví dụ mẫu** với 3 test case, bảng chỉ hiện các cột đầu
vào. Tab 6 và 7 thì ví dụ chỉ **điền vào ô prompt** chứ không chạy model —
chúng không có ảnh dựng sẵn để trả về, chạy luôn mỗi lần bấm sẽ mất 6–12s.

### Tab 6 — Prompt to Image

FLUX.2-klein **hợp nhất sinh ảnh và sửa ảnh trong một model**:
`Flux2KleinPipeline.__call__` nhận `image=None`, nên text-to-image và
image-edit đi qua đúng một pipeline object và đúng một hàm
`inference.generate()`.

Đây là thay đổi kiến trúc đáng kể so với backend cũ, nơi
`QwenImageEditPlusPipeline` **bắt buộc** phải có `image=` và vì thế phải dựng
thêm một `QwenImagePipeline` thứ hai trỏ vào cùng các module đã nạp. Hàm
`build_t2i_pipeline()` cùng mọi cái bẫy quanh nó (đừng dùng
`from_pipe()` — nó cast weight sang fp32 và hỏng luôn pipeline gốc) đã biến
mất.

**Về chất lượng:** khác backend cũ (một model chuyên *sửa* ảnh, sinh ảnh từ
chữ chỉ là tác dụng phụ), tab này giờ chạy đúng thứ model được luyện.

Tab 6 không có ảnh nền nên tỉ lệ khung phải chọn tay (1:1, 16:9, 9:16, 4:3,
3:4, 3:2, 2:3); diện tích vẫn lấy từ *Độ phân giải đầu ra*.

### Chế độ demo (`GENIMG_DEMO_CACHE`, mặc định bật)

Khi bật, mọi đầu vào **trùng với một test case mẫu** sẽ trả ảnh đã dựng sẵn
trong `examples/outputs/` thay vì chạy model:

- Bấm vào một ví dụ → ảnh kết quả hiện thẳng ra khung kết quả.
- Bấm nút chạy với đúng ảnh mẫu đó → cũng trả ảnh dựng sẵn, mất ~0.3s.
- Đầu vào khác (ảnh tester tự upload) → **chạy model thật** như bình thường.

Mục đích là trình diễn không phải chờ 6–12s mỗi lần. Ô trạng thái nói thẳng đây
là ảnh dựng sẵn:

```
Kết quả dựng sẵn cho ví dụ này (không phải vừa chạy). Bấm nút chạy để model sinh lại ảnh mới.
1024×1024 · ảnh đọc từ đĩa trong 0.3s
```

Log phía server cũng ghi `Demo cache hit: ...` mỗi lần trả ảnh dựng sẵn, và số
ví dụ nạp được in lúc khởi động (`Demo cache: bật (12 ví dụ dựng sẵn)`).

Đo hiệu năng thật thì tắt đi:

```bash
# .env
GENIMG_DEMO_CACHE=false
```

Ảnh dựng sẵn sinh bằng:

```bash
python scripts/warm_examples.py       # demo phải đang chạy; 15 case
```

Phải chạy lại script này mỗi khi đổi ảnh mẫu, đổi danh sách case trong
`src/gen_image/ui/examples_spec.py`, hoặc sửa prompt — nếu không, ảnh dựng
sẵn sẽ lệch với những gì model thực sự sinh ra.

> ⚠️ **Ảnh mẫu KHÔNG nằm trong repo** (`examples/` chỉ có `outputs/` rỗng và
> `queue_payloads/`). Team tiếp nhận phải tự chuẩn bị bộ ảnh theo đúng tên
> file khai trong `examples_spec.py`, và **tự chịu trách nhiệm về nguồn gốc
> cùng giấy phép** của chúng — ảnh có người thật trong một demo public là
> chuyện pháp lý, không phải chuyện kỹ thuật. Thiếu ảnh thì UI tự bỏ qua
> case đó, demo vẫn chạy bình thường.

### Thứ tự ảnh — quan trọng với tab 1 và tab 4

**Ảnh nền phải là ảnh CUỐI** (nhãn UI đánh nó là "Ảnh 2"). Pipeline lấy tỉ lệ
khung từ `image[-1]` và mô hình coi ảnh cuối là ảnh chính cần chỉnh sửa. Số trên
nhãn UI khớp với `image 1` / `image 2` trong prompt, nên khi tự sửa prompt hãy
giữ đúng cách đánh số đó.

> **Vì sao tab 2 chỉ còn một ảnh.** Bản trước nhận thêm ảnh phòng tham chiếu và
> kết quả gần như luôn bê nguyên ảnh tham chiếu đó — sửa thứ tự ảnh chỉ chữa
> được tỉ lệ khung, không chữa được nội dung. Rút prompt ngắn lại thì đúng, và
> bỏ hẳn ảnh tham chiếu (mô tả phong cách bằng chữ) thì không còn ảnh nào để
> model copy nhầm. Đó là cách tab 2 hoạt động hiện nay.

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
  2*, và *khớp tông da với cổ/tai/tay*. Nếu phải cắt ngắn cho model 4 bước bớt
  lạc, hãy giữ ba đoạn đó lại.
- Hoạt động tốt nhất khi Ảnh 2 có **một khuôn mặt rõ**, và Ảnh 1 crop sát mặt,
  nhìn thẳng, không mờ.

Lưu ý khác:

- Tab 1, 2, 3, 5 có dropdown thu hẹp bài toán (loại trang phục / loại phòng /
  phong cách hoạt hình / phạm vi hoán đổi) — đổi dropdown sẽ dựng lại prompt
  tương ứng.
- Tab 2: ô **Mô tả thiết kế mong muốn** là bắt buộc, viết bằng tiếng Anh.
- Ô **Ghi chú thêm** ở các tab khác được nối vào cuối prompt.
- Mục **Tuỳ chỉnh nâng cao** cho phép sửa trực tiếp prompt, negative prompt,
  `guidance_scale`, seed, số bước và **độ phân giải đầu ra**.
- **Negative prompt không có tác dụng** với bản klein distilled, ở bất kỳ
  `guidance_scale` nào: guidance được nhúng thẳng vào model thay vì chạy CFG
  hai nhánh, nên diffusers tắt hẳn nhánh negative. Ô này chỉ sống lại nếu
  đổi `base_model` sang bản `klein-base` (không distilled). Service ghi log
  mỗi lần bỏ qua — xem `inference._negative_embeds`.
- **Prompt càng dài càng dễ hỏng.** Model 4 bước bám prompt kém; prompt quá dài
  khiến nó bỏ chỉ dẫn và rơi về hành vi mặc định. Sửa prompt thì nên ngắn, đặt
  câu mệnh lệnh lên trước.
- **Cùng seed không cho lại đúng cùng một ảnh.** Kernel INT4 không tất định giữa
  các lần chạy; đo thực tế cho chênh lệch pixel tối đa ~240/255 dù giữ nguyên mọi
  đầu vào. Seed chỉ giúp thu hẹp biến thiên, không tái lập tuyệt đối.

### Tốc độ

> ⚠️ **Đã có số đo đầu tiên trên L4, và nó CHƯA ĐẠT mục tiêu.** Chi tiết +
> phân tích tỉ trọng: [docs/PERFORMANCE.md §0b](docs/PERFORMANCE.md).

| Kịch bản @1024², 4 bước | Backend cũ trên L4 (đã đo) | klein-9B GGUF (đã đo) | Mục tiêu |
| :--- | ---: | ---: | ---: |
| Text-to-image | 12.9s | **10.8s** | ≤ 6s |
| Text-to-image, prompt dài ~1800 ký tự | — | **10.8s** | — |
| Image-edit, tab demo @1024² | 16.8s | **18.1s** | ≤ 8s |
| VRAM đỉnh | ~23 GB | **13.7 GB** | ≤ 18 GB |

VRAM đạt mục tiêu, tốc độ t2i đã vượt backend cũ. Phát hiện quan trọng nhất:
**độ dài prompt gần như miễn phí** (hai dòng đầu bằng nhau), còn **ảnh tham
chiếu thì rất đắt**. Ảnh tham chiếu 1024² thêm 4096 latent token vào chuỗi
denoise ở MỌI bước — đúng bằng toàn bộ ảnh ra ở 1024². Knob mới
`reference_area` cắt thẳng vào đó. Chi tiết + số đo:
[PERFORMANCE.md §0c](docs/PERFORMANCE.md).

Cơ sở của mục tiêu: ComfyUI báo ~1.2s @1024² trên RTX 5090; L4 chậm hơn
khoảng 4–5× về compute và ~3× về băng thông (300 GB/s). Đây là phép ngoại
suy, không phải cam kết.

**Denoise chiếm phần lớn thời gian.** Hệ quả: tối ưu bất cứ phần nào ngoài
denoise đều gần như vô nghĩa; hai đòn bẩy còn lại là **số ảnh tham chiếu** và
**độ phân giải đầu ra** (`OUTPUT_PRESETS` trong UI, `output_area` trong config).

Lần chạy đầu với mỗi prompt mất thêm thời gian cho text encoder; các lần sau
ăn cache và hiện `text 0.0s (cache)`. Khác backend cũ, **đổi ảnh mà giữ
nguyên prompt vẫn ăn cache** — text encoder Qwen3-8B là text-only nên ảnh
không tham gia vào khoá cache.

#### Con số trên UI là thời gian GPU, không phải round-trip

Ô **Thời gian chạy thật (GPU)** hiển thị:

```
⚡ GPU 6.3s  =  denoise 5.6s (1.39s/bước) + text 0.0s (cache) + prep 0.3s + decode 0.4s
1024×1024 · 4 bước · guidance 1.0 · seed 12345
```

Khoảng thời gian này đo từ **trước** `encode_prompt` đến **sau** khi pipeline
trả kết quả — thuần tính toán trên GPU. Nó **không** bao gồm:

| Không được tính | Vì sao đáng kể |
| :--- | :--- |
| Upload ảnh từ trình duyệt lên server | Vài MB mỗi ảnh |
| Hàng đợi Gradio | ~0.3s round-trip ngay cả khi rảnh |
| Mã hoá PNG kết quả rồi trả về | Ảnh 1024×1024 |
| Tunnel `*.gradio.live` | **Thường là phần lớn nhất** — traffic đi vòng qua server relay của HuggingFace |

Nên nếu bấm đồng hồ thấy 15s mà ô trạng thái ghi `GPU 6.3s` thì model vẫn
chạy đúng 6.3s; phần còn lại là mạng. Muốn loại hẳn yếu tố này khi đo: truy
cập trực tiếp qua LAN hoặc SSH port-forward, hoặc đo bằng `make benchmark`
(chạy thẳng trong container, không qua HTTP).

> Chuỗi này đi **thẳng vào `result.info`** của outbound message — đổi format
> là đổi hợp đồng với BE.

#### Đo trên máy của bạn

```bash
make benchmark                                   # trong container đang chạy
python scripts/benchmark.py --runs 10 --mode t2i  # hoặc trực tiếp
python scripts/benchmark.py --area 768            # hạ độ phân giải
```

`benchmark.py` xoá cache prompt-embed trước mỗi lượt — giữ nguyên prompt qua
các lượt sẽ ăn cache từ lượt thứ hai và ta chỉ còn đang đo tốc độ tra dict.

> Cách đọc dòng trạng thái: `callback_on_step_end` của diffusers chỉ chạy sau
> khi một bước kết thúc, nên nếu đo ngây thơ thì phần "prep" sẽ nuốt trọn một
> bước denoise. Số ở trên đã trừ ra, và `x.xxs/bước` là thời gian một bước thật.

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
| Trọng số ~16GB | Mount `./models:/app/models:ro`, và `.dockerignore` loại `models/` khỏi build context |
| `.model_paths.env` ghi đường dẫn tuyệt đối của host | Bị loại khỏi image; compose set `GENIMG_BASE_MODEL_LOCAL` trỏ vào `/app/models` thay thế |
| Vô tình tải lại ~16GB giữa production | `HF_HUB_OFFLINE=1` biến mọi đường rò xuống Hub thành lỗi dừng hẳn |
| Phân mảnh VRAM khi shape ảnh thay đổi liên tục | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Người dùng đầu tiên phải chờ nạp kernel | `GENIMG_WARMUP=true` — warm-up (t2i + edit) chạy TRƯỚC khi mở cổng |
| `HF_TOKEN` | Không cần trong container (đã có weights local); compose ghi đè thành rỗng |
| Link share không hiện trong `docker logs` | `PYTHONUNBUFFERED=1` đã set sẵn trong Dockerfile |
| Restart policy khác nhau giữa bàn dev và server | Makefile dò GPU rồi set `GENIMG_RESTART_POLICY` — xem `make gpu-info` |

> Đổi `quantization` sang `gguf` thì phải set thêm `transformer_gguf` trỏ tới
> đúng file `.gguf` đã tải — và chạy lại `make download` sau khi đổi
> `gguf_file` trong khối `app:`.

### Restart policy tự dò

`make run` gọi `nvidia-smi` và tự quyết định:

| Phần cứng | `restart` | Lý do |
| :--- | :--- | :--- |
| A100 (bất kể số lượng) | `unless-stopped` | Server production — container tự dậy lại sau reboot / OOM-kill |
| Đúng 1 GPU | `unless-stopped` | Cùng lý do |
| Nhiều GPU không phải A100 | `no` | Bàn dev — crash thì dừng hẳn cho dễ đọc traceback |
| Không dò được GPU | `no` | Mặc định an toàn |

Ghi đè tay: `make run GENIMG_RESTART_POLICY=no`.

Cấu hình hiện tại phục vụ **một tải tại một thời điểm** (một container, hàng đợi
Gradio mặc định xử lý tuần tự).

## Thuê GPU L4 trên Google Cloud

Toàn bộ phần này dành cho việc dựng service từ một project GCP trống. Phần
cứng đích là **L4 24GB** (`g2-*`) — FLUX.2-klein-9B đã nén chỉ cần ~11 GiB nên
không còn lý do trả tiền cho A100 40GB như backend cũ (26.4 GiB).

### 0. Chuẩn bị

Cần quota GPU trước — project mới luôn có quota bằng 0:

```bash
gcloud auth login
gcloud config set project "$PROJECT_ID"

# Xem quota L4 hiện có ở vùng định thuê
gcloud compute regions describe us-central1 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -i l4
```

Nếu limit = 0 thì xin tăng ở **IAM & Admin → Quotas**, chọn metric
`NVIDIA_L4_GPUS`. Quota L4 thường được duyệt nhanh hơn A100 đáng kể.

L4 có ở rất nhiều zone — danh sách đổi theo thời điểm:

```bash
gcloud compute accelerator-types list --filter="name=nvidia-l4"
```

### 1. Tạo VM

Máy `g2-standard-8` = **1× L4 24GB**, 8 vCPU, 32GB RAM. Với họ máy `g2-*`
thì GPU đã gắn sẵn trong machine type, **không cần** cờ `--accelerator`.

Dùng ảnh Deep Learning VM để có sẵn driver NVIDIA + Docker + nvidia-container-toolkit:

```bash
# Xem các image family đang có
gcloud compute images list --project deeplearning-platform-release \
  --filter="family~'common-cu'" --format="value(family)" | sort -u

gcloud compute instances create gen-image \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --image-family=common-cu124-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --tags=gen-image
```

Vài lựa chọn đã cân nhắc:

- **`--maintenance-policy=TERMINATE` là bắt buộc.** VM có GPU không
  live-migrate được.
- **Ổ 200GB.** Cần chỗ cho ~17GB weights + ~10GB image + OS. Quan trọng hơn:
  trên Persistent Disk, băng thông đọc **tăng theo dung lượng ổ**. Ổ quá nhỏ
  làm bước nạp weights lên VRAM chậm hẳn. Cần khởi động nhanh hơn nữa thì gắn
  Local SSD.
- **RAM 32GB là đủ** với `offload: "resident"` — weight nằm trên GPU, host RAM
  chỉ dùng lúc nạp. Nếu phải rơi về `model_offload` thì cân nhắc
  `g2-standard-12` trở lên.
- **Spot VM** (`--provisioning-model=SPOT`) rẻ hơn đáng kể nhưng bị thu hồi bất
  kỳ lúc nào. Hợp để test, không hợp cho service đang phục vụ thật.

> Tính tiền theo thời gian VM **chạy**, không theo mức dùng GPU. Không dùng thì
> `gcloud compute instances stop gen-image --zone=us-central1-a`.
> Ổ đĩa vẫn tính tiền khi VM đã stop.

### 2. Kiểm tra máy

```bash
gcloud compute ssh gen-image --zone=us-central1-a
```

Lần SSH đầu, ảnh DLVM sẽ hỏi cài driver NVIDIA — trả lời `y` (hoặc đã tự cài nếu
truyền `install-nvidia-driver=True`). Sau đó xác nhận:

```bash
nvidia-smi                    # phải thấy NVIDIA L4 (23034MiB) và driver >= 525
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Driver phải từ **525 trở lên**: image dùng torch cu128, chạy được trên driver cũ
hơn CUDA 12.8 nhờ CUDA minor version compatibility, nhưng dưới 525 thì không.

### 3. Đưa code và weights lên

```bash
# Trên VM
git clone <repo-url> gen-image && cd gen-image
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
gcloud storage cp -r "gs://$BUCKET/gen-image-models/*" ./models/
make preflight
```

### 4. Chạy

```bash
make build
make gpu-info                  # xác nhận dò ra 1 GPU -> unless-stopped
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
gcloud compute ssh gen-image --zone=us-central1-a -- -L 7860:localhost:7860
```

Rồi mở `http://localhost:7860`.

**Cách mở cổng công khai** — chỉ làm khi thật sự cần, và luôn giới hạn IP nguồn:

```bash
MY_IP=$(curl -s ifconfig.me)
gcloud compute firewall-rules create allow-gen-image-7860 \
  --allow=tcp:7860 \
  --target-tags=gen-image \
  --source-ranges="$MY_IP/32"
```

Đừng dùng `--source-ranges=0.0.0.0/0`: Gradio không có xác thực, ai biết IP là vào
được và chạy model bằng tiền GPU của bạn.

Lưu ý link `*.gradio.live` mà app luôn tạo **cũng** là một đường công khai
không xác thực, và không tắt được bằng cấu hình. Trên server trả tiền theo
giờ, đừng để demo chạy khi không dùng — xem [Dọn dẹp](#dọn-dẹp-sau-khi-test).

### 6. Xong việc

```bash
gcloud compute instances stop gen-image --zone=us-central1-a     # giữ ổ đĩa
gcloud compute instances delete gen-image --zone=us-central1-a   # xoá hẳn
```

### Khắc phục sự cố

| Triệu chứng | Nguyên nhân thường gặp |
| :--- | :--- |
| `could not select device driver "" with capabilities: [[gpu]]` | nvidia-container-toolkit chưa đăng ký với Docker. Chạy lại lệnh kiểm tra ở mục 2 |
| Log dừng ở `Loading ...` rất lâu | Đang nạp 26.4GB từ Persistent Disk. Ổ nhỏ = đọc chậm; xem mục 1 |
| `FileNotFoundError: GENIMG_BASE_MODEL_LOCAL trỏ tới ...` | Mount `./models` sai hoặc chưa tải weights. Chạy `make preflight` |
| `OfflineModeIsEnabled` / `LocalEntryNotFound` | Thiếu file trong `models/`, code định tải từ Hub nhưng đã bị khoá offline. `make download` lại |
| `CUDA out of memory` | Có tiến trình khác đang giữ VRAM. `nvidia-smi` xem PID |
| Container `unhealthy` nhưng vẫn chạy | Warm-up lâu hơn `start-period=600s`. Xem `make logs` |

## Cho máy khác trong mạng LAN truy cập

Máy chủ demo: **`<demo-server-ip>`** (interface `enp6s0`, subnet `<demo-subnet>`).
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
sudo ufw allow from <demo-subnet> to any port 7860 proto tcp comment 'gen-image demo LAN'
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

Nếu `ss` cho thấy `127.0.0.1:7860` thay vì `0.0.0.0:7860` thì `GENIMG_SERVER_NAME`
đang sai — `docker-compose.yml` đã ép `0.0.0.0`, chỉ xảy ra khi chạy ngoài Docker.

### Link public luôn bật — không tắt được

Mỗi lần khởi động app tạo một link `*.gradio.live` mở ra Internet, và không
có biến môi trường nào tắt được. Đây là lựa chọn có chủ ý: đường chạy chính
của repo là Colab / máy thuê, nơi cổng 7860 không tiếp cận được từ ngoài nên
thiếu link đồng nghĩa demo không dùng được.

Đổi lại, **mọi lần chạy đều mở một endpoint không xác thực ra Internet**, kể
cả khi bạn chỉ định test trong LAN. Test nội bộ xong thì dừng demo ngay
(xem [Dọn dẹp](#dọn-dẹp-sau-khi-test)) chứ đừng để chạy nền. Cần chạy dài
ngày mà không muốn link thì sửa thẳng `_launch_with_public_link()` trong
`src/gen_image/ui/launch.py`.

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
- Đọc cổng từ env `GENIMG_PORT` / `.env` (ưu tiên) rồi mới tới `config_setup/base.yaml` (`app.server_port`).
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

1. **`docker compose up -d` KHÔNG chạy queue worker.** Service `gen-image-queue`
   nằm sau `profiles: ["queue"]`, nên lệnh deploy mặc định chỉ dựng
   `model-fetcher` + `gen-image` (Gradio). Container chạy xanh, GPU có
   tải, mà không ai consume inbound queue → `gen-image-queue-out` mãi mãi
   rỗng. Chạy đúng:

   ```bash
   make queue                  # = docker compose up -d gen-image-queue
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

### Smoke test đường queue — chạy được khi CHƯA có GPU

`Application.run()` nạp model TRƯỚC khi spawn consumer/publisher, nên bình
thường không có cách nào thử chuỗi broker → worker → `queue_out` nếu máy
không có GPU + 26GB weight. Service `gen-image-queue-smoke` (profile `smoke`) chạy
cùng image nhưng với `config_setup/smoke.yaml`: `processor.type: echo` (trả
ảnh placeholder, không nạp model) + `storage.backend: local` (không cần key
GCS). Mọi mắt xích còn lại giữ nguyên bản.

```bash
export GENIMG_RMQ_HOST=<broker-host> GENIMG_RMQ_VHOST=gen-image \
       GENIMG_RMQ_USER=<user> GENIMG_RMQ_PASSWORD='...'

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

Thông số broker của cả hai service đều nhận env `GENIMG_RMQ_HOST` / `PORT` /
`USER` / `PASSWORD` / `VHOST` / `EXCHANGE` ghi đè lên file yaml, nên password
không bắt buộc phải nằm trong `base.yaml`. Đổi hẳn file config bằng
`GENIMG_QUEUE_CONFIG=/app/config_setup/<file>.yaml`.

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
export GENIMG_RMQ_PASSWORD='...'        # secret: để trong env, đừng gõ ra dòng lệnh
/tmp/ltvenv/bin/python scripts/loadtest_queue.py \
    --host <broker-host> --vhost gen-image --user <user> \
    --declare --rate 5 --duration 30

# Đẩy tải: 200 msg/s trong 2 phút, 4 connection, tắt publisher confirm
... --rate 200 --duration 120 --workers 4 --no-confirm

# Đo end-to-end (cần queue worker đang chạy): đọc luôn queue_out
... --rate 2 --duration 300 --drain-out
```

Thông số broker lấy theo thứ tự **CLI > env `GENIMG_RMQ_*` > `config_setup/base.yaml`**,
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
uv run pytest
```

## Cấu trúc dự án

```
src/gen_image/                  # mã nguồn chính
  __init__.py                   #   ghim HF_HOME + allocator CUDA vào project
  config.py                     #   NGUỒN CONFIG: base.yaml + env GENIMG_*
  device.py                     #   chọn cuda:N
  inference.py                  #   generate() — MỘT hàm cho t2i lẫn edit
  tuning.py                     #   cờ torch, torch.compile, warm-up
  download.py                   #   tải weight + ghi .model_paths.env
  models/loader.py              #   dựng Flux2KleinPipeline (bf16 | gguf)
  models/placement.py           #   resident | model_offload
  ui/app.py                     #   build_ui() — 7 tab, KHÔNG nạp model
  ui/launch.py                  #   main() — nạp model, mở cổng, tunnel
  ui/widgets.py                 #   widget dùng chung giữa các tab
  ui/demo_cache.py              #   ảnh kết quả dựng sẵn cho khối ví dụ
  ui/prompts.py                 #   prompt hệ thống ⚠️ chưa chỉnh cho FLUX.2
  queue_service/                #   RabbitMQ consumer — KHÔNG phụ thuộc model
scripts/                        # CLI: download_model.py, serve.py,
                                #   preflight.py   (kiểm model trên đĩa, không cần GPU)
                                #   spike_flux2.py (cổng chặn Phase 0, cần GPU)
                                #   benchmark.py   (ma trận latency/VRAM)
                                #   verify_build.py(chốt phiên bản lúc build image)
                                #   loadtest_queue.py (bắn tải vào RabbitMQ)
tests/                          # pytest, KHÔNG cần GPU: config, path resolution,
                                #   inference (pipeline giả), cấu trúc UI,
                                #   vị trí cache, schema (hợp đồng BE)
config_setup/base.yaml          # NGUỒN CONFIG DUY NHẤT (app + queue service)
config_setup/base.example.yaml  #   template (commit); base.yaml bị gitignore
docs/                           # kiến trúc, hiệu năng, queue service
docs/PERFORMANCE.md             #   đã tối ưu gì / đừng làm gì
docs/adr/                       #   quyết định kiến trúc + kết quả nghiệm thu
docs/REFACTOR_FLUX2_KLEIN_4B.md #   kế hoạch chuyển backend Qwen -> FLUX.2
cleanup.sh                      # dọn tiến trình / giải phóng VRAM, port
Dockerfile                      # image 2 stage (builder venv -> runtime)
docker-compose.yml              # gen-image (dev UI) + gen-image-queue (production)
Makefile                        # download / preflight / build / run / benchmark
```
