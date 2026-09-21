.PHONY: build run logs link stop shell ps clean download preflight \
        preflight-docker benchmark gpu-info pack fetch \
        queue queue-logs queue-stop smoke smoke-logs smoke-stop \
        test lint fmt check help

# Cổng host; đổi khi 7860 đang bận: make run HOST_PORT=7861
HOST_PORT ?= 7860
export HOST_PORT

# --- Restart policy tự dò ------------------------------------------------
# Một GPU, hoặc GPU là A100 => coi như server production => 'unless-stopped'
# để container tự dậy lại sau reboot / OOM-kill. Bàn dev nhiều GPU giữ 'no'
# cho app crash thì dừng hẳn, dễ đọc traceback.
# Ghi đè tay:  make run GENIMG_RESTART_POLICY=no
# Đếm bằng `wc -l` chứ KHÔNG dùng $(words ...): $(shell) nuốt newline thành
# khoảng trắng, nên $(words) sẽ đếm số TỪ trong tên card
# ("NVIDIA GeForce RTX 3080" = 4) thay vì số GPU.
GPU_NAMES  := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd'|' -)
GPU_COUNT  := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
IS_A100    := $(findstring A100,$(GPU_NAMES))

ifeq ($(GPU_COUNT),1)
GENIMG_RESTART_POLICY ?= unless-stopped
else ifneq ($(IS_A100),)
GENIMG_RESTART_POLICY ?= unless-stopped
else
GENIMG_RESTART_POLICY ?= no
endif
export GENIMG_RESTART_POLICY

# --- Vòng lặp phát triển ----------------------------------------------
# Ba target này là thứ CI và pre-commit chạy; giữ chúng xanh.

# Unit test. KHÔNG cần GPU, không chạm mạng, không nạp model.
test:
	uv run pytest -q

# Lint. `ruff format --check` chạy TRƯỚC `ruff check`: sai format thì sửa
# bằng `make fmt`, còn lỗi của `check` thì phải sửa tay — tách ra để biết
# mình đang gặp loại nào.
lint:
	uv run ruff format --check src scripts tests main_queue.py
	uv run ruff check src scripts tests main_queue.py

# Tự sửa những gì sửa được (format + import order).
fmt:
	uv run ruff format src scripts tests main_queue.py
	uv run ruff check --fix src scripts tests main_queue.py

# Cổng trước khi push: đúng những gì pre-commit + CI sẽ chạy.
check: lint test
	@echo ">>> lint + test xanh."

help:
	@echo "Phát triển:  make check | test | lint | fmt"
	@echo "Model:       make download | preflight | benchmark"
	@echo "Docker:      make build | run | logs | queue | smoke | stop"
	@echo "Chẩn đoán:   make gpu-info | preflight-docker | shell | ps"

# In ra phần cứng dò được và quyết định đi kèm.
gpu-info:
	@echo "GPU            : $(GPU_NAMES)"
	@echo "Số GPU         : $(GPU_COUNT)"
	@echo "Restart policy : $(GENIMG_RESTART_POLICY)"

# Tải weight về ./models. Cache HF đã bị ghim vào project nên không có gì
# rơi ra ~/.cache. Kéo component pipeline FLUX.2-klein-9B cần thiết và
# transformer GGUF Q4_K_M; không tải transformer BF16 không được dùng.
download:
	python scripts/download_model.py

# Đóng gói ./models thành models.tar.zst để đẩy lên GCS. Thêm đích để
# upload luôn:  make pack DEST=gs://my-bucket/gen-image/
pack:
	scripts/pack_models.sh models.tar.zst $(DEST)

# Chạy riêng bước kéo trọng số từ GCS (bình thường `make run` tự chạy trước
# app). Dùng khi muốn tải trước, hoặc để đọc log tải mà không lẫn log model.
fetch:
	docker compose run --rm model-fetcher

# Kiểm tra cây model trên đĩa trước khi tốn 2 phút load. Không cần GPU.
preflight:
	python scripts/preflight.py

# Chạy đúng bài kiểm tra đó BÊN TRONG container, để bắt lỗi mount sai.
# Dùng gen-image-queue: đó là service chạy mặc định (gen-image là UI dev,
# nằm sau profile "dev" nên `docker compose run gen-image` sẽ báo no such
# service nếu không kèm --profile dev).
preflight-docker:
	docker compose run --rm --entrypoint python gen-image-queue scripts/preflight.py

build:
	docker compose build

run:
	@echo ">>> restart policy = $(GENIMG_RESTART_POLICY) (GPU: $(GPU_NAMES))"
	docker compose up -d
	@echo ">>> Đang load model + warm-up (~1-2 phút). Theo dõi: make logs"

# Đo tốc độ sinh ảnh thật bên trong container đang chạy.
# Nhắm vào gen-image-queue vì đó là container `make run` dựng lên. Muốn đo
# trong container UI thì: docker compose --profile dev exec gen-image ...
benchmark:
	docker compose exec gen-image-queue python scripts/benchmark.py

logs:
	docker compose logs -f

# --- Queue worker (RabbitMQ consumer) ------------------------------------
# gen-image-queue chạy MẶC ĐỊNH (`make run` / `docker compose up` đã kéo nó
# lên) — service UI Gradio mới là thứ nằm sau profile "dev". Target này giữ
# lại vì nó kiểm tra base.yaml tồn tại trước khi khởi động: thiếu file đó là
# lý do số một khiến `gen-image-queue-out` im lặng sau khi deploy.
# Cần config_setup/base.yaml + credentials/user-upload-key.json trên HOST
# (compose mount vào, không bake trong image).
queue:
	@test -f config_setup/base.yaml || { \
		echo "THIẾU config_setup/base.yaml — cp config_setup/base.example.yaml config_setup/base.yaml rồi điền rabbitmq + storage"; \
		exit 1; }
	docker compose up -d gen-image-queue
	@echo ">>> Đang nạp model (~1-2 phút). Sẵn sàng khi /readyz trả 200:"
	@echo "    curl -s localhost:$${QUEUE_HEALTH_PORT:-8395}/readyz"
	@echo ">>> Log: make queue-logs"

queue-logs:
	docker compose logs -f gen-image-queue

queue-stop:
	docker compose stop gen-image-queue

# --- Smoke test đường queue (KHÔNG cần GPU, KHÔNG nạp model) -------------
# Chạy cùng image nhưng với config_setup/smoke.yaml (processor: echo +
# storage: local) để chứng minh chuỗi consume → pipeline → publish sang
# queue_out hoạt động. Dùng TRƯỚC khi thuê GPU.
#   export GENIMG_RMQ_HOST=... GENIMG_RMQ_USER=... GENIMG_RMQ_PASSWORD=... GENIMG_RMQ_VHOST=...
#   make smoke && make smoke-logs
smoke:
	docker compose --profile smoke up -d --no-deps gen-image-queue-smoke
	@echo ">>> Bắn tải: python scripts/loadtest_queue.py --rate 5 --duration 10"
	@echo ">>> Log: make smoke-logs"

smoke-logs:
	docker compose --profile smoke logs -f gen-image-queue-smoke

smoke-stop:
	docker compose --profile smoke down --remove-orphans

# In link share công khai khi GENIMG_SHARE=true
link:
	@docker compose logs 2>/dev/null | grep -aoE 'https://[a-z0-9]+\.gradio\.live' | tail -1 \
		|| echo "Chưa có link — model có thể còn đang load, xem: make logs"

stop:
	docker compose down

shell:
	docker compose exec gen-image-queue bash

ps:
	docker compose ps

# Xoá cả image và volume cache
clean:
	docker compose down -v --rmi local
