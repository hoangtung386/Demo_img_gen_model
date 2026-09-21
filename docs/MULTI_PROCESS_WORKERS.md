# Chạy nhiều process worker song song trên 1 GPU

> **Trạng thái hiện tại (L4-24GB): TẮT.** `docker-compose.yml` chỉ có một
> service `gen-image-queue`. FLUX.2-klein-9B ở cấu hình mặc định chiếm
> ~11 GiB trong 24 GiB — 2 process **về lý thuyết** vừa, nhưng chưa ai đo
> phần activation chồng lên nhau. Tài liệu này mô tả điều kiện để bật
> lại, và **cảnh báo rằng tiền đề ban đầu của nó không còn đúng** — đọc
> mục "Batching đã khả thi trở lại" trước khi làm theo.

## Vấn đề

`gen-image-queue` mặc định chỉ có `worker.num_ai_workers: 1` — 1 thread duy
nhất gọi GPU tuần tự. GPU chỉ bận một phần thời gian; phần còn lại là I/O
(consume RabbitMQ, download/upload ảnh, publish reply).

## Tại sao KHÔNG tăng `num_ai_workers`

Đã thử `num_ai_workers: 2` (nhiều thread trong CÙNG 1 process, share 1
pipeline instance) — kết quả:

- Throughput **giảm** (0.017 req/s, tệ hơn 12× so với 1 worker)
- Lỗi thật: `IndexError: index 5 is out of bounds for dimension 0 with
  size 5` trong `FlowMatchEulerDiscreteScheduler.step()` — race condition
  vì `pipeline.scheduler` là state DÙNG CHUNG (`self.scheduler.sigmas`,
  `self._current_timestep`...), 2 thread cùng ghi/đọc đồng thời làm hỏng
  state của nhau.

Giới hạn này **vẫn đúng với FLUX.2**: nó thuộc về `scheduler` của diffusers,
không phải về model. Một pipeline instance = xử lý tuần tự.

## ⚠️ Batching đã khả thi trở lại — kiểm trước khi chọn multi-process

Lý do **ban đầu** khiến multi-process là lựa chọn duy nhất: kernel CUDA INT4
của Nunchaku không hỗ trợ `batch_size > 1`
([#148](https://github.com/nunchaku-tech/nunchaku/issues/148),
[#289](https://github.com/nunchaku-tech/nunchaku/issues/289)). Nunchaku đã bị
gỡ khỏi dự án. `Flux2KleinPipeline` là pipeline diffusers tiêu chuẩn với
`num_images_per_prompt` và `prompt: list[str]`.

**Nếu batching thật sự chạy, nó thắng multi-process về mọi mặt**: một model
instance, một bản weight trong VRAM, không nhân đôi RAM host, không nhân đôi
thời gian khởi động. Hãy đo điều đó **trước** khi dựng process thứ hai.

## Nếu vẫn cần multi-process: hạ quantization

VRAM một instance trên GPU:

VRAM một instance trên GPU (bản **9B**, text encoder Qwen3-8B):

| `quantization` | `text_encoder_quantization` | Tổng | 2 process trên L4-24GB? |
|---|---|---:|---|
| `bf16` | `bf16` | ~34.7 GB | ❌ (1 process cũng không vừa) |
| `gguf` Q4_K_M | `bf16` | ~22.6 GB | ❌ (1 process đã sát trần → OOM) |
| `gguf` Q4_K_M | `int8` | ~14.7 GB | ❌ (2 × 14.7 = 29.4) |
| `gguf` Q4_K_M | `nf4` | **~11.2 GB** | ⚠️ 2 × 11.2 = 22.4 — vừa weight, chưa tính activation |

Điểm khác biệt lớn nhất so với bản 4B: text encoder **Qwen3-8B** (16.4 GB
bf16) chứ không phải Qwen3-4B (8 GB), và nó **không** giảm theo
`quantization` của transformer. Ở 9B, nó là component to nhất, nên
`text_encoder_quantization` mới là knob quyết định có nhét vừa hay không.

Cấu hình (`config_setup/base.yaml`, mục `processor:`):

```yaml
processor:
  quantization: "gguf"
  text_encoder_quantization: "nf4"
  transformer_gguf: "/app/models/gguf/flux-2-klein-4b-Q4_K_M.gguf"
  offload: "resident"
```

Đổi `gguf_file` trong khối `app:` rồi chạy lại `make download` để kéo đúng
file Q4_K_M.

**Đánh đổi:** GGUF giải nén về bf16 ngay trong forward, nên latency mỗi ảnh
CAO HƠN bf16, và nhánh này **không** `torch.compile` được
([diffusers#10795](https://github.com/huggingface/diffusers/issues/10795)).
Bạn đang đổi latency lấy throughput. Quyết bằng `scripts/benchmark.py`, không
bằng suy đoán.

## Triển khai: 2 container Docker

Thêm một service thứ hai vào `docker-compose.yml` theo đúng pattern của
`gen-image-queue` (đổi `container_name` + port healthcheck). Mỗi container
tự load model riêng, cùng consume 4 queue inbound
(`gen-image-queue-in-{premium,basic-tier-1,2,3}`). **RabbitMQ tự chia message
cho các consumer cùng queue theo round-robin ack** — không cần code gì thêm
để "phân việc" giữa 2 process.

Healthcheck 2 container qua 2 port khác nhau trên host:
- process 1: `${QUEUE_HEALTH_PORT:-8395}` → container port 8395
- process 2: `${QUEUE_HEALTH_PORT_2:-8397}` → container port 8395

## Đánh đổi cần biết

- **RAM host tăng**: mỗi process giữ bản weight riêng (2 process OS riêng
  biệt, không share như 2 thread).
- **Thời gian khởi động**: mỗi container tự load model riêng, 2 container
  khởi động cùng lúc sẽ cạnh tranh CPU/disk I/O.
- **Scale thêm**: kiểm VRAM còn đủ bằng `nvidia-smi` sau khi 2 process đã
  chạy ổn định, đừng tính nhẩm — phân mảnh allocator là thật.

## Đo throughput

Đếm dòng `PROCESS_OK` trong log qua khoảng thời gian cố định (không phụ
thuộc rate publish của test client):

```bash
A=$(docker logs gen-image-queue 2>&1 | grep -c 'PROCESS_OK')
B=$(docker logs gen-image-queue-2 2>&1 | grep -c 'PROCESS_OK')
sleep 60
A2=$(docker logs gen-image-queue 2>&1 | grep -c 'PROCESS_OK')
B2=$(docker logs gen-image-queue-2 2>&1 | grep -c 'PROCESS_OK')
echo "throughput = $(( (A2 - A) + (B2 - B) )) req / 60s"
```
