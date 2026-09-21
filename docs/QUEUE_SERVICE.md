# Queue worker service (RabbitMQ)

Service consumer chạy nền: nhận request sinh/sửa ảnh từ RabbitMQ, chạy pipeline
FLUX.2-klein-4B, upload kết quả lên GCS và publish URL trả về.
Kiến trúc mirror service `edit_any_image`.

Package: `src/gen_image/queue_service/`. Entry point: `main_queue.py`.
Config: **một file duy nhất** `config_setup/base.yaml` (giống `edit_any_image`).

## Chạy

```bash
# 1. Cài dependency (pika/loguru/pyyaml/psutil/google-cloud-storage đã nằm
#    trong nhóm mặc định — dùng chung với bản Gradio demo)
uv sync

# 2. Tạo config thật từ file mẫu rồi điền broker host/user/password/vhost +
#    bucket. base.yaml bị .gitignore chặn (chứa secret) — chỉ commit example.
cp config_setup/base.example.yaml config_setup/base.yaml

# 3. Đặt key upload (bucket cho user) vào
#    config_setup/credentials/user-upload-key.json — storage.backends[].
#    credentials_path trong base.yaml đã trỏ tới đó. (Key tải model nội bộ là
#    file khác: config_setup/credentials/model-download-key.json.)

# 4. Chạy consumer
python main_queue.py
# hoặc console script:  run-queue

# Docker (dùng chung image với Gradio, entrypoint riêng). Compose mount
# ./config_setup từ host, nên base.yaml + credentials phải có sẵn trên host:
docker compose up -d gen-image-queue
```

> **Lưu ý bảo mật:** `config_setup/base.yaml` và `config_setup/credentials/*.json`
> KHÔNG được commit (đã có trong `.gitignore` + `.dockerignore`). Image chỉ bake
> `base.example.yaml`; config thật được inject qua volume mount lúc chạy.

Health: `http://localhost:8395/healthz` (liveness), `/readyz` (readiness —
503 khi model chưa nạp xong hoặc broker unreachable).

## Cấu trúc config (`config_setup/base.yaml`)

File chia 2 phần giống `edit_any_image`:

- **PHẦN 1 (BE/DevOps chỉnh)** — `rabbitmq` + `storage`. Tên queue/exchange,
  broker host, GCS bucket, CDN domain. Đồng nhất pattern với mọi AI service
  trong org.
- **PHẦN 2 (AI internal, để mặc định)** — `processor` + `worker` +
  `observability` + `audit`. Đặc trưng service; chỉ AI team chỉnh khi
  benchmark/debug.

### `rabbitmq`

4 tier queue vào (`premium` + `basic_1/2/3`) + 1 queue ra. `topology_mode`:

| mode | mô tả |
|------|-------|
| `declare` | tạo full topology: exchange main/`.retry`/`.dlq` + DLX args + retry/DLQ sibling mỗi tier. |
| `declare_minimal` | **mặc định QAS/PRD**: exchange + 4 queue vào + 1 queue ra + bind. Không DLX. Tương thích queue BE pre-create. Failure → reply qua `queue_out` + ack + audit trên disk. |
| `passive` | chỉ verify queue đã tồn tại, không tạo. DevOps kiểm soát toàn quyền. |

`prefetch_count: 1` cho service GPU 1 worker (4 tier × 1 = 4 unacked/container,
fair distribution multi-container). `prefetch_count: 0` → auto-tune theo
`target_sla_seconds / avg_inference_seconds`, cap `prefetch_max`.

`retry_max_attempts: 0` (canonical): mọi lỗi DLQ ngay, không loop. AI reply vẫn
đi về BE qua `queue_out`; BE re-publish thủ công nếu cần.

## ⚠️ Retry: đi dây đầy đủ nhưng ĐANG TẮT

Đọc mục này trước khi bạn mất một buổi truy vì sao vài hàm không ai gọi.

Topology retry **có được dựng thật** (`topology.py`: queue `.retry` với
`x-message-ttl = retry_initial_delay_ms`, DLX trỏ về exchange chính). Nhưng
quyết định *định tuyến* thì hard-code: `AIWorker._route_failure` đẩy mọi
failure thẳng vào DLQ, không bao giờ đi qua đường retry.

Hệ quả — bốn thứ dưới đây **tồn tại nhưng không được gọi**, và đó là có chủ
đích, không phải bỏ sót:

| Thứ | Ở đâu | Sống lại khi nào |
|---|---|---|
| `should_retry()` | `messaging/retry.py` | `_route_failure` gọi nó |
| `compute_backoff_ms()` | `messaging/retry.py` | dùng backoff luỹ thừa |
| `Envelope.retry()` | `messaging/envelope.py` | `_route_failure` chọn nhánh retry |
| `BrokerHealthMonitor.is_alive_now()` | `messaging/broker_health.py` | ai đó cần đọc trạng thái thay vì nhận callback |

`attempts_so_far()` thì **đang được dùng** (`ai_worker.py` đọc header
`x-death` để ghi số lần thử vào audit) — đừng xoá nhầm nó cùng nhóm trên.

**Muốn bật retry:** đặt `retry_max_attempts > 0` là CHƯA đủ. Phải sửa
`_route_failure` để gọi `should_retry()` rồi `envelope.retry()`, **và** rà lại
classifier trong `messaging/status.py` cho chắc mọi lỗi PERMANENT (4xx từ GCS,
input hỏng) không lọt vào vòng lặp retry vô ích.

### `storage`

Multi-backend list. Inbound `storage_index` (int) chọn entry, default
`default_index`. `cdn_base_url` rỗng → signed URL của bucket; có giá trị → URL
CDN public `{cdn_base_url}/{object_key}`.

### `processor` (FLUX.2 inference)

Map thẳng vào `gen_image.config.Settings` khi nạp pipeline — dùng chung
`models.loader.load_pipeline`, không copy logic:

| field | mặc định | ý nghĩa |
|-------|----------|---------|
| `type` | `flux2_klein` | `flux2_klein` / `echo` (smoke test, không nạp model) |
| `num_steps` | 4 | bản distilled được chưng cất về đúng 4 bước |
| `quantization` | `bf16` | `bf16` (mặc định) / `gguf` — xem ADR 0001 §3 |
| `compile_transformer` | false | `torch.compile`; chỉ có tác dụng ở nhánh bf16 |
| `offload` | `resident` | `resident` / `model_offload` (đường lùi khi thiếu VRAM) |
| `guidance_scale` | 1.0 | giá trị model card; bản distilled KHÔNG chạy CFG |
| `device` | "" | rỗng → auto (cuda:1 nếu ≥2 GPU) |
| `output_area` | 1048576 | diện tích ảnh ra (1024²) |
| `max_input_dimension` | 1280 | trần mỗi chiều của ảnh tham chiếu (chỉ thu, không phóng) |
| `base_model_local` | "" | rỗng → tải từ Hub; đặt path để đọc weight đã có trên đĩa |
| `transformer_gguf` | "" | path file `.gguf`; chỉ đọc khi `quantization: gguf` |
| `warmup` | true | trả trước chi phí lượt đầu lúc startup |

⚠️ `negative_prompt` trong inbound message vẫn được nhận (hợp đồng không
đổi) nhưng **không có tác dụng** với bản distilled: guidance được nhúng vào
model thay vì chạy CFG hai nhánh. Service ghi log khi bỏ qua.

## Message schema

### Inbound (queue vào)

```jsonc
{
  "_id": "req-123",              // bắt buộc, request id (idempotency key)
  "os": "android",              // bắt buộc field (type str), giá trị tự do — android|ios|"" đều hợp lệ, giá trị lạ fallback "unknown" khi đặt tên file GCS
  "firebase_token": "…",        // bắt buộc field (type str), "" hợp lệ — chỉ echo lại, BE dùng để push kết quả nên rỗng = user không nhận được thông báo (không phải lỗi ở service này)
  "appid": "com.example.app",   // bắt buộc
  "country": "VN",              // optional
  "device_id": "abc",           // bắt buộc field (type str), giá trị tự do — "" hợp lệ, fallback "unknown-device" khi đặt tên file GCS
  "prompt": "a cat astronaut",  // bắt buộc field (type str), "" hợp lệ — đi thẳng vào text encoder, không crash nhưng ra ảnh không theo hướng dẫn nào

  // Ảnh điều kiện — OPTIONAL. Có → image-edit; không → text-to-image.
  // Ảnh CUỐI là ảnh nền (lấy tỉ lệ + là ảnh chính); ảnh trước là reference.
  "images": ["https://…/ref.png", "https://…/base.png"],
  // "image": "https://…/one.png"  // alias 1 ảnh cũng chấp nhận

  "negative_prompt": "",        // optional; KHÔNG có tác dụng với bản distilled (không chạy CFG)
  "seed": 42,                   // optional; <0 → random
  "num_steps": 0,               // optional; 0 → dùng default processor
  "aspect_ratio": 1.0,          // optional; chỉ cho text-to-image
  "match_input_size": 0,        // optional; 1 → resize ảnh ra = ảnh nền
  "storage_index": 0            // optional; chọn backend upload
}
```

### Outbound (queue ra)

```jsonc
{
  "_id": "req-123",
  "os": "android",
  "firebase_token": "…",
  "status_code": 200,
  "message": "success",
  "result": { "url": "https://cdn…/…png", "info": "⚡ GPU 5.2s …" }
}
```

Status code registry: `messaging/status.py`. 2xx success, 4xx input fault
(PERMANENT → DLQ), 5xx/6xx service/upstream (TRANSIENT).

## Audit trên disk

`data_predict/model_gen_img/<Y>/<M>/<D>/{success,fail,error}/<request_id>/attempt_N/`
với `input.json` (payload + checksum), `input.png` (ảnh đầu vào đầu tiên nếu
có), `output.json` (reply + status), `output.png` (ảnh sinh ra). Tự dọn sau
`audit.retention_days` (mặc định 7 ngày). Lỗi cấp RabbitMQ (mất kết nối…) lưu
ở `rmq_errors/`.
