# Queue worker service (RabbitMQ)

Service consumer chạy nền: nhận request sinh/sửa ảnh từ RabbitMQ, chạy pipeline
HiDream-O1-Image (SDNQ 4-bit), upload kết quả lên GCS và publish URL trả về.
Kiến trúc mirror service `edit_any_image`.

Package: `src/imagegen/queue_service/`. Entry point: `main_queue.py`.
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
docker compose --profile queue up imagegen-queue
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

### `storage`

Multi-backend list. Inbound `storage_index` (int) chọn entry, default
`default_index`. `cdn_base_url` rỗng → signed URL của bucket; có giá trị → URL
CDN public `{cdn_base_url}/{object_key}`.

### `processor` (HiDream-O1-Image inference)

Map thẳng vào `imagegen.config.Settings` khi nạp model — dùng chung
`imagegen.hidream.load_model`, không copy logic:

| field | mặc định | ý nghĩa |
|-------|----------|---------|
| `type` | `hidream_o1` | `hidream_o1` \| `echo` (smoke test, không nạp model) |
| `model_type` | `full` | `full` (50 bước) / `dev` (28 bước, nhanh ~3.5×) |
| `num_steps` | 50 | 50 cho full, 28 cho dev |
| `guidance_scale` | 5.0 | **> 1.0 nhân đôi số forward pass.** 0.0 cho dev |
| `shift` | 3.0 | độ lệch lịch nhiễu; 1.0 cho dev |
| `scheduler_name` | `default` | `default` \| `flow_match` \| `flash` |
| `width` / `height` | 2048 | chỉ chọn **tỉ lệ** — xem cảnh báo dưới |
| `model_path` | "" | rỗng → theo `IMG_MODEL_PATH` / `.model_paths.env` |
| `device` | "" | rỗng → auto (cuda:1 nếu ≥2 GPU) |
| `warmup` | true | trả trước chi phí lượt đầu lúc startup |

> **`width`/`height` không chọn được kích thước.** Model snap mọi yêu cầu về
> một trong 11 độ phân giải cố định, nhỏ nhất 2048×2048. Đặt 1024 không cho
> ảnh 1024 — nó vẫn ra 2048, chỉ đổi tỉ lệ khung. Không có đòn bẩy nào để
> sinh ảnh nhỏ hơn cho nhanh.

> **`avg_inference_seconds` của khối `rabbitmq` phải khớp số đo thật.**
> 50 bước × 2 forward pass (CFG) ở 2048² chậm hơn model cũ một bậc độ lớn.
> Chạy `scripts/benchmark.py` — nó in ra đúng giá trị cần điền. Để nguyên
> giá trị của model cũ thì `effective_prefetch()` cho consumer ôm quá nhiều
> message và backlog phình ra.

## Message schema

### Inbound (queue vào)

```jsonc
{
  "_id": "req-123",              // bắt buộc, request id (idempotency key)
  "os": "android",              // bắt buộc, android|ios
  "firebase_token": "…",        // bắt buộc (để BE push kết quả)
  "appid": "com.example.app",   // bắt buộc
  "country": "VN",              // optional
  "device_id": "abc",           // bắt buộc
  "prompt": "a cat astronaut",  // bắt buộc

  // Ảnh điều kiện — OPTIONAL. Có → editing; không → text-to-image.
  // HiDream-O1 khuyến nghị ĐÚNG MỘT ảnh cho editing (khi đó ảnh ra giữ khung
  // của ảnh vào). Nhiều ảnh vẫn chạy nhưng là đường subject-driven khác.
  "images": ["https://…/base.png"],
  // "image": "https://…/one.png"  // alias 1 ảnh cũng chấp nhận

  "negative_prompt": "",        // ⚠️ KHÔNG CÒN TÁC DỤNG — xem ghi chú
  "seed": 42,                   // optional; <0 → random
  "num_steps": 0,               // optional; 0 → dùng default processor
  "aspect_ratio": 1.0,          // optional; chỉ chọn TỈ LỆ — xem ghi chú
  "match_input_size": 0,        // optional; 1 → resize ảnh ra = ảnh vào
  "storage_index": 0            // optional; chọn backend upload
}
```

**Hai field BE vẫn gửi được nhưng không còn tác dụng như trước:**

| field | trạng thái |
| :-- | :-- |
| `negative_prompt` | **Bị bỏ qua.** HiDream-O1 không nhận negative prompt — nhánh uncond của CFG dùng prompt `" "` cố định. Giữ trong schema để BE/App không phải đổi payload; processor log ở mức debug rồi bỏ. |
| `aspect_ratio` | Chỉ chọn **tỉ lệ khung**, không chọn kích thước. Giá trị được ánh xạ sang độ phân giải gần nhất trong 11 giá trị cố định (nhỏ nhất 2048×2048). |

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
