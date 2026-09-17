.PHONY: build run logs link stop shell ps clean download preflight \
        preflight-docker benchmark gpu-info pack fetch \
        queue queue-logs queue-stop smoke smoke-logs smoke-stop

# Cổng host; đổi khi 7860 đang bận: make run HOST_PORT=7861
HOST_PORT ?= 7860
export HOST_PORT

# --- Restart policy tự dò ------------------------------------------------
# Một GPU, hoặc GPU là A100 => coi như server production => 'unless-stopped'
# để container tự dậy lại sau reboot / OOM-kill. Bàn dev nhiều GPU giữ 'no'
# cho app crash thì dừng hẳn, dễ đọc traceback.
# Ghi đè tay:  make run IMG_RESTART_POLICY=no
# Đếm bằng `wc -l` chứ KHÔNG dùng $(words ...): $(shell) nuốt newline thành
# khoảng trắng, nên $(words) sẽ đếm số TỪ trong tên card
# ("NVIDIA GeForce RTX 3080" = 4) thay vì số GPU.
GPU_NAMES  := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd'|' -)
GPU_COUNT  := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
IS_A100    := $(findstring A100,$(GPU_NAMES))

ifeq ($(GPU_COUNT),1)
IMG_RESTART_POLICY ?= unless-stopped
else ifneq ($(IS_A100),)
IMG_RESTART_POLICY ?= unless-stopped
else
IMG_RESTART_POLICY ?= no
endif
export IMG_RESTART_POLICY

# In ra phần cứng dò được và quyết định đi kèm.
gpu-info:
	@echo "GPU            : $(GPU_NAMES)"
	@echo "Số GPU         : $(GPU_COUNT)"
	@echo "Restart policy : $(IMG_RESTART_POLICY)"

# Tải weight về ./models (~9.9GB). Cache HF đã bị ghim vào project nên
# không có gì rơi ra ~/.cache.
download:
	python scripts/download_model.py

# Đóng gói ./models thành models.tar.zst để đẩy lên GCS. Thêm đích để
# upload luôn:  make pack DEST=gs://my-bucket/
# Nhớ cập nhật IMG_MODELS_URI trong docker-compose.yml cho khớp.
pack:
	scripts/pack_models.sh models.tar.zst $(DEST)

# Chạy riêng bước kéo trọng số từ GCS (bình thường `make run` tự chạy trước
# app). Dùng khi muốn tải trước, hoặc để đọc log tải mà không lẫn log model.
fetch:
	docker compose run --rm model-fetcher

# Kiểm tra cây model trên đĩa trước khi tốn hàng chục giây load. Không
# cần GPU, không cần torch.
preflight:
	python scripts/preflight.py

# Chạy đúng bài kiểm tra đó BÊN TRONG container, để bắt lỗi mount sai.
preflight-docker:
	docker compose run --rm --entrypoint python imagegen scripts/preflight.py

build:
	docker compose build

run:
	@echo ">>> restart policy = $(IMG_RESTART_POLICY) (GPU: $(GPU_NAMES))"
	docker compose up -d
	@echo ">>> Đang load model + warm-up (~3-5 phút: warm-up là một lượt"
	@echo ">>> sinh ảnh thật ở 2048²). Theo dõi: make logs"

# Đo tốc độ sinh ảnh thật bên trong container đang chạy. In ra giá trị
# cần điền vào rabbitmq.avg_inference_seconds của base.yaml.
benchmark:
	docker compose exec imagegen python scripts/benchmark.py

logs:
	docker compose logs -f

# --- Queue worker (RabbitMQ consumer) ------------------------------------
# `make run` / `docker compose up` KHÔNG kéo service này lên: nó nằm sau
# profile "queue". Đây là lý do số một khiến `gen-image-queue-out` im lặng
# sau khi deploy — container Gradio chạy ngon lành nhưng không ai consume
# inbound queue. Cần config_setup/base.yaml + credentials/user-upload-key.json
# trên HOST trước khi chạy (compose mount vào, không bake trong image).
queue:
	@test -f config_setup/base.yaml || { \
		echo "THIẾU config_setup/base.yaml — cp config_setup/base.example.yaml config_setup/base.yaml rồi điền rabbitmq + storage"; \
		exit 1; }
	docker compose --profile queue up -d imagegen-queue
	@echo ">>> Đang nạp model + warm-up (~3-5 phút). Sẵn sàng khi /readyz trả 200:"
	@echo "    curl -s localhost:$${QUEUE_HEALTH_PORT:-8395}/readyz"
	@echo ">>> Log: make queue-logs"

queue-logs:
	docker compose --profile queue logs -f imagegen-queue

queue-stop:
	docker compose --profile queue stop imagegen-queue

# --- Smoke test đường queue (KHÔNG cần GPU, KHÔNG nạp model) -------------
# Chạy cùng image nhưng với config_setup/smoke.yaml (processor: echo +
# storage: local) để chứng minh chuỗi consume → pipeline → publish sang
# queue_out hoạt động. Dùng TRƯỚC khi thuê GPU.
#   export IMG_RMQ_HOST=... IMG_RMQ_USER=... IMG_RMQ_PASSWORD=... IMG_RMQ_VHOST=...
#   make smoke && make smoke-logs
smoke:
	docker compose --profile smoke up -d --no-deps imagegen-queue-smoke
	@echo ">>> Bắn tải: python scripts/loadtest_queue.py --rate 5 --duration 10"
	@echo ">>> Log: make smoke-logs"

smoke-logs:
	docker compose --profile smoke logs -f imagegen-queue-smoke

smoke-stop:
	docker compose --profile smoke down --remove-orphans

# In link share công khai từ log container.
link:
	@docker compose logs 2>/dev/null | grep -aoE 'https://[a-z0-9]+\.gradio\.live' | tail -1 \
		|| echo "Chưa có link — model có thể còn đang load, xem: make logs"

stop:
	docker compose down

shell:
	docker compose exec imagegen bash

ps:
	docker compose ps

# Xoá cả image và volume cache
clean:
	docker compose down -v --rmi local
