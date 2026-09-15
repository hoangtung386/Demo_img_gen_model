# Qwen-Image-Edit-2509 Lightning — Gradio Demo

Demo chạy thử mô hình **Qwen-Image-Edit-2509 Lightning** (4-bit, 4-step) thông qua
thư viện [Nunchaku](https://github.com/nunchaku-tech/nunchaku) và `diffusers`.
Hỗ trợ chỉnh sửa một hoặc nhiều ảnh, tối ưu cho suy luận tương tác (tăng tốc 12–25 lần so với base).

## Yêu cầu

- Python >= 3.10 (khuyên dùng môi trường ảo `uv`)
- GPU NVIDIA có 4-bit tensorcore: Turing / Ampere / Ada dùng **INT4**
  (A100, RTX 30-, 40-series), Blackwell (RTX 50-series) dùng **FP4**. Chọn sai
  precision thì Nunchaku báo lỗi ngay lúc load.
- ~27GB đĩa cho trọng số, và **26.4GB VRAM** nếu muốn giữ toàn bộ pipeline
  thường trú trên GPU (xem [Tốc độ](#tốc-độ))
- Đã accept license của [`Qwen/Qwen-Image-Edit-2509`](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) trên Hugging Face

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
uv run download-model     # tải trọng số
uv run run-app            # chạy demo
```

Hoặc cách truyền thống (pip-style):

```bash
uv venv
uv pip install -e .
# uv pip install -e ".[dev]"   # kèm dev tools
```

## Cấu hình (.env)

Sao chép template và điền `HF_TOKEN` (bắt buộc với model gated):

```bash
cp config/example.env .env
# sau đó mở .env và đặt HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

Các biến quan trọng trong `.env` (đọc bởi cả `download` và `serve`):

| Biến | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `QIE_NUM_STEPS` | `4` | Số bước suy luận (4 hoặc 8) — phải khớp với weights |
| `QIE_RANK` | `32` | Rank lượng tử hóa (32 hoặc 128, 128 = chất lượng tốt hơn) |
| `QIE_PRECISION` | (auto) | `int4` / `fp4` hoặc để trống để tự phát hiện |
| `QIE_OFFLOAD` | `auto` | `auto` / `none` / `split` / `model` / `sequential` — xem bảng bên dưới |
| `QIE_WARMUP` | `true` | Chạy một lượt sinh ảnh giả lúc khởi động để người dùng đầu tiên không phải chờ |
| `QIE_PORT` | `7860` | Cổng Gradio |
| `QIE_SERVER_NAME` | `0.0.0.0` | Bind address (để chạy trên server) |
| `QIE_DEMO_CACHE` | `true` | Ví dụ mẫu trả ảnh dựng sẵn thay vì chạy model (xem bên dưới) |
| `QIE_DOWNLOAD_BASE_TRANSFORMER` | `false` | Tải cả 5 shard transformer BF16 (~39GB) mà pipeline không dùng tới |

`QIE_OFFLOAD=auto` dò phần cứng rồi tự chọn:

| Phần cứng | Chọn | Ghi chú |
| :--- | :--- | :--- |
| >= 2 GPU | `split` | text_encoder ở card phụ, transformer+vae ở card chính |
| 1 GPU >= 32 GiB (A100, H100) | `none` | Giữ hết trên GPU — nhanh ngang `split` |
| 1 GPU 18–32 GiB (RTX 3090/4090) | `model` | Nạp ~27GB qua PCIe mỗi lần sinh ảnh |
| Nhỏ hơn | `sequential` | Offload từng block |

Trên server production nên đặt tường minh thay vì để `auto` — `docker-compose.yml`
đã set sẵn `QIE_OFFLOAD=none` cho A100 một card.

## Tải trọng số

```bash
python scripts/download_model.py
# hoặc: uv run download-model
```

Script tải:
1. Pipeline gốc (`Qwen/Qwen-Image-Edit-2509`) → `models/Qwen-Image-Edit-2509`
2. Transformer Lightning 4-bit → `models/lightning-251115/`

Sau đó ghi đường dẫn local vào `.model_paths.env` để `serve` dùng trực tiếp (không tải lại lúc chạy).

Hai điều đáng biết:

- **Không có gì rơi vào `~/.cache/huggingface`.** Cache của HuggingFace đã được
  ghim vào `models/.hf` ngay trong `qwen_lightning/__init__.py`, nên toàn bộ dữ
  liệu tải về nằm trong project. Đặt `HF_HOME` tường minh nếu muốn chỗ khác.
- **Bỏ qua 39GB không dùng tới.** Repo gốc có 5 shard transformer BF16, nhưng
  pipeline luôn thay chúng bằng bản Nunchaku INT4 nên thư mục đó không bao giờ
  được đọc. Mặc định script bỏ qua; tổng dung lượng còn ~27GB thay vì ~65GB.

Kiểm tra cây model bất cứ lúc nào (không cần GPU):

```bash
make preflight
```

## Chạy demo

```bash
python scripts/serve.py
# hoặc: uv run run-app
```

Mở trình duyệt tại `http://<server-ip>:7860`. Giao diện có **3 tab**, mỗi tab đã
có prompt chuyên biệt viết sẵn (xem `src/qwen_lightning/ui/prompts.py`):

| Tab | Đầu vào | Kết quả |
| :--- | :--- | :--- |
| **1. Virtual Try-On** | Ảnh 1: trang phục — Ảnh 2: người mẫu *(ảnh nền)* | Người mẫu mặc trang phục mới, giữ nguyên khuôn mặt, dáng, tư thế, màu da và bối cảnh |
| **2. Home Design** | **Một** ảnh căn phòng + mô tả thiết kế tự viết | Chính căn phòng đó được dọn sạch đồ cũ và bài trí lại theo mô tả, giữ nguyên tường, cửa sổ, cửa, sàn và góc chụp |
| **3. Image to Cartoon** | 1 ảnh chụp có người và phong cảnh | Toàn bộ người trong ảnh thành nhân vật hoạt hình giữ tối đa nét mặt, kiểu tóc, vóc dáng, trang phục; phong cảnh vẽ lại cùng phong cách |
| **4. Ghép 2 người ôm nhau** | Ảnh 1: người thứ nhất — Ảnh 2: người thứ hai *(ảnh nền)* | Một ảnh duy nhất hai người đang ôm nhau, giữ khuôn mặt, kiểu tóc, vóc dáng và trang phục của từng người |

Mỗi tab có khối **Ví dụ mẫu** với 3 test case, bảng chỉ hiện các cột đầu vào.

### Chế độ demo (`QIE_DEMO_CACHE`, mặc định bật)

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
QIE_DEMO_CACHE=false
```

Ảnh dựng sẵn sinh bằng:

```bash
python scripts/warm_examples.py       # demo phải đang chạy; ~2 phút cho 12 case
```

Phải chạy lại script này mỗi khi đổi ảnh mẫu, đổi danh sách case trong
`src/qwen_lightning/ui/examples_spec.py`, hoặc sửa prompt — nếu không, ảnh dựng
sẵn sẽ lệch với những gì model thực sự sinh ra. Nguồn và giấy phép ảnh xem
[`examples/README.md`](examples/README.md).

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

Lưu ý khác:

- Tab 1, 2, 3 có dropdown thu hẹp bài toán (loại trang phục / loại phòng /
  phong cách hoạt hình) — đổi dropdown sẽ dựng lại prompt tương ứng.
- Tab 2: ô **Mô tả thiết kế mong muốn** là bắt buộc, viết bằng tiếng Anh.
- Ô **Ghi chú thêm** ở các tab khác được nối vào cuối prompt.
- Mục **Tuỳ chỉnh nâng cao** cho phép sửa trực tiếp prompt, negative prompt,
  `true_cfg_scale`, seed, số bước và **độ phân giải đầu ra**.
- `true_cfg_scale` mặc định `1.0` (Lightning chạy tốt nhất ở 1.0); ở mức này CFG
  bị tắt nên **negative prompt không có tác dụng** — muốn dùng phải tăng > 1.0
  (đổi lại mỗi bước denoise đắt gấp đôi).
- **Prompt càng dài càng dễ hỏng.** Model 4 bước bám prompt kém; prompt quá dài
  khiến nó bỏ chỉ dẫn và rơi về hành vi mặc định. Sửa prompt thì nên ngắn, đặt
  câu mệnh lệnh lên trước.
- **Cùng seed không cho lại đúng cùng một ảnh.** Kernel INT4 không tất định giữa
  các lần chạy; đo thực tế cho chênh lệch pixel tối đa ~240/255 dù giữ nguyên mọi
  đầu vào. Seed chỉ giúp thu hẹp biến thiên, không tái lập tuyệt đối.

### Tốc độ

Đo trên 2×RTX 3090 Ti, 4 bước, lần chạy lặp lại (prompt embeddings đã cache):

| Kịch bản | Tổng | text | prep | denoise | decode |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2 ảnh vào, 1024px | 11.1s | 0.0s | 0.5s | **10.2s** (2.54s/bước) | 0.4s |
| 2 ảnh vào, 832px | ~9.5s | 0.0s | 0.5s | ~8.6s | 0.3s |
| 2 ảnh vào, 704px | 8.6s | 0.0s | 0.5s | 7.8s (1.96s/bước) | 0.2s |
| 1 ảnh vào, 1024px | 6.3s | 0.0s | 0.3s | 5.6s (1.39s/bước) | 0.4s |

**Denoise chiếm ~90% thời gian.** Mọi thứ còn lại — chuẩn bị, VAE encode/decode,
hàng đợi Gradio (~0.3s round-trip) — đều không đáng kể. Hệ quả: tối ưu bất cứ
phần nào ngoài denoise đều gần như vô nghĩa; chỉ còn hai đòn bẩy là **số ảnh đầu
vào** (ảnh điều kiện thứ hai làm mỗi bước đắt thêm ~83%) và **độ phân giải đầu ra**.

Lần chạy đầu với mỗi cặp (ảnh, prompt) mất thêm 0.4–2.0s cho text encoder; các
lần sau ăn cache và hiện `text 0.0s (cache)`.

#### Con số trên UI là thời gian GPU, không phải round-trip

Ô **Thời gian chạy thật (GPU)** hiển thị:

```
⚡ GPU 6.3s  =  denoise 5.6s (1.39s/bước) + text 0.0s (cache) + prep 0.3s + decode 0.4s
1024×1024 · 4 bước · cfg 1.0 · seed 12345
```

Khoảng thời gian này đo từ **trước** `encode_prompt` đến **sau** khi pipeline
trả kết quả — thuần tính toán trên GPU. Nó **không** bao gồm:

| Không được tính | Vì sao đáng kể |
| :--- | :--- |
| Upload ảnh từ trình duyệt lên server | Vài MB mỗi ảnh |
| Hàng đợi Gradio | ~0.3s round-trip ngay cả khi rảnh |
| Mã hoá PNG kết quả rồi trả về | Ảnh 1024×1024 |
| Tunnel `*.gradio.live` | **Thường là phần lớn nhất** — traffic đi vòng qua server relay của HuggingFace |

Nên nếu bấm đồng hồ thấy 15s mà ô trạng thái ghi `GPU 6.3s` thì model vẫn chạy
đúng 6.3s; ~9s còn lại là mạng. Muốn loại hẳn yếu tố này khi đo: đặt
truy cập trực tiếp qua LAN hoặc SSH port-forward thay vì qua link
`*.gradio.live`, hoặc đo bằng `make benchmark` (chạy thẳng trong container,
không qua HTTP).

#### Trên một card A100 40GB

Toàn bộ pipeline chiếm **26.4 GiB** (text_encoder 15.4 + transformer INT4 10.7 +
vae 0.24), vừa trong 39 GiB của A100 với khoảng 13 GiB dư cho activation. Nghĩa là
`QIE_OFFLOAD=none` giữ được mọi component thường trú — không có lần chuyển
CPU↔GPU nào lúc sinh ảnh, tương đương chế độ `split` hai card.

Đây là điểm dễ mất tốc độ nhất: nếu để chiến lược rơi về `model`, mỗi lần sinh ảnh
phải nạp ~27GB qua PCIe. `docker-compose.yml` đã chốt `none` sẵn.

Đo trên máy của bạn:

```bash
make benchmark            # trong container đang chạy
# hoặc: python scripts/benchmark.py --runs 10 --area 1024
```

> Cách đọc dòng trạng thái: `callback_on_step_end` của diffusers chỉ chạy sau khi
> một bước kết thúc, nên nếu đo ngây thơ thì phần "prep" sẽ nuốt trọn một bước
> denoise. Số ở trên đã trừ ra, và `@x.xxs/step` là thời gian một bước thật.

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

Trọng số **không** nằm trong image — tải ở host rồi mount vào, nên phải chạy
`make download` trước ít nhất một lần.

```bash
make download        # tải trọng số về ./models (~27GB)
make preflight       # kiểm tra cây model trên đĩa, không cần GPU
make build           # dựng image (~10GB, lần đầu mất vài phút)
make run             # khởi động container
make logs            # theo dõi load model + warm-up
make benchmark       # đo tốc độ sinh ảnh thật
make stop            # dừng
```

Cổng 7860 đang bận thì đổi: `make run HOST_PORT=7861`.

Không dùng `make` thì thay bằng `docker compose build` / `up -d` / `logs -f` / `down`.

### Những điều compose đã xử lý sẵn

| Vấn đề | Cách xử lý trong `docker-compose.yml` |
| :--- | :--- |
| Trọng số 27GB | Mount `./models:/app/models:ro`, và `.dockerignore` loại `models/` khỏi build context |
| `.model_paths.env` ghi đường dẫn tuyệt đối của host | Bị loại khỏi image; compose set `QIE_BASE_MODEL_LOCAL` / `QIE_TRANSFORMER_PATH` trỏ vào `/app/models` thay thế |
| Vô tình tải lại 27GB giữa production | `HF_HUB_OFFLINE=1` biến mọi đường rò xuống Hub thành lỗi dừng hẳn |
| Phân mảnh VRAM khi shape ảnh thay đổi liên tục | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Người dùng đầu tiên phải chờ nạp cubin | `QIE_WARMUP=true` — warm-up chạy TRƯỚC khi mở cổng |
| `HF_TOKEN` | Không cần trong container (đã có weights local); compose ghi đè thành rỗng |
| Link share không hiện trong `docker logs` | `PYTHONUNBUFFERED=1` đã set sẵn trong Dockerfile |
| Restart policy khác nhau giữa bàn dev và server | Makefile dò GPU rồi set `QIE_RESTART_POLICY` — xem `make gpu-info` |

> Đổi `QIE_RANK`, `QIE_NUM_STEPS` hoặc `QIE_PRECISION` thì phải sửa luôn tên file
> trong `QIE_TRANSFORMER_PATH` ở `docker-compose.yml` cho khớp weights tương ứng.

### Restart policy tự dò

`make run` gọi `nvidia-smi` và tự quyết định:

| Phần cứng | `restart` | Lý do |
| :--- | :--- | :--- |
| A100 (bất kể số lượng) | `unless-stopped` | Server production — container tự dậy lại sau reboot / OOM-kill |
| Đúng 1 GPU | `unless-stopped` | Cùng lý do |
| Nhiều GPU không phải A100 | `no` | Bàn dev — crash thì dừng hẳn cho dễ đọc traceback |
| Không dò được GPU | `no` | Mặc định an toàn |

Ghi đè tay: `make run QIE_RESTART_POLICY=no`.

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
- **Ổ 250GB.** Cần chỗ cho 27GB weights + ~10GB image + OS. Quan trọng hơn: trên
  Persistent Disk, băng thông đọc **tăng theo dung lượng ổ**. Ổ quá nhỏ làm bước nạp
  26.4GB weights lên VRAM chậm hẳn. Cần khởi động nhanh hơn nữa thì gắn Local SSD.
- **A100 80GB** thì đổi sang `a2-ultragpu-1g`. Không cần cho dự án này — chỉ dùng
  hết 26.4GB.
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
cp config/example.env .env
nano .env                      # điền HF_TOKEN
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
| `FileNotFoundError: QIE_BASE_MODEL_LOCAL trỏ tới ...` | Mount `./models` sai hoặc chưa tải weights. Chạy `make preflight` |
| `OfflineModeIsEnabled` / `LocalEntryNotFound` | Thiếu file trong `models/`, code định tải từ Hub nhưng đã bị khoá offline. `make download` lại |
| `CUDA out of memory` | Có tiến trình khác đang giữ VRAM. `nvidia-smi` xem PID |
| Container `unhealthy` nhưng vẫn chạy | Warm-up lâu hơn `start-period=600s`. Xem `make logs` |

## Cho máy khác trong mạng LAN truy cập

Máy chủ demo: **`192.168.5.233`** (interface `enp6s0`, subnet `192.168.5.0/24`).
Container đã publish `0.0.0.0:7860` nên đã lắng nghe trên mọi interface — địa chỉ
để các máy khác mở là:

```
http://192.168.5.233:7860
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
curl -I --max-time 5 http://192.168.5.233:7860     # mong đợi: HTTP/1.1 200 OK
```

Không thông thì lần lượt loại trừ:

```bash
# trên máy chủ: container còn sống và đang map cổng?
docker compose ps
ss -ltnp | grep 7860              # phải thấy 0.0.0.0:7860

# hai máy có cùng subnet không?
ip -4 addr show                   # trên máy client
ping 192.168.5.233
```

Nếu `ss` cho thấy `127.0.0.1:7860` thay vì `0.0.0.0:7860` thì `QIE_SERVER_NAME`
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
`src/qwen_lightning/ui/app.py`.

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
- Đọc cổng từ `.env` (ưu tiên) rồi mới tới `config/example.env`.
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

## Tests

```bash
uv run pytest
```

## Cấu trúc dự án

```
src/qwen_lightning/   # mã nguồn chính (config, device, tuning, models, inference, ui)
  __init__.py         #   ghim HF_HOME + allocator CUDA vào project
  tuning.py           #   cờ torch + warm-up lúc khởi động
  models/offload.py   #   chọn chiến lược đặt component lên GPU
scripts/              # wrapper CLI: download_model.py, serve.py,
                      #   preflight.py (kiểm tra model), benchmark.py (đo tốc độ)
tests/                # pytest (config, path resolution, scheduler,
                      #   vị trí cache, bảng quyết định offload)
config/example.env    # template biến môi trường
docs/                 # tài liệu kiến trúc & phân tích kỹ thuật
cleanup.sh            # dọn tiến trình / giải phóng VRAM, port
Dockerfile            # image 2 stage (builder venv -> runtime)
docker-compose.yml    # 1 container, mount models read-only, offline mode
Makefile              # download / preflight / build / run / benchmark / gpu-info
```
