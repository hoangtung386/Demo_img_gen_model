.PHONY: build run logs link stop shell ps clean download preflight \
        preflight-docker benchmark gpu-info

# Cổng host; đổi khi 7860 đang bận: make run HOST_PORT=7861
HOST_PORT ?= 7860
export HOST_PORT

# --- Restart policy tự dò ------------------------------------------------
# Một GPU, hoặc GPU là A100 => coi như server production => 'unless-stopped'
# để container tự dậy lại sau reboot / OOM-kill. Bàn dev nhiều GPU giữ 'no'
# cho app crash thì dừng hẳn, dễ đọc traceback.
# Ghi đè tay:  make run QIE_RESTART_POLICY=no
# Đếm bằng `wc -l` chứ KHÔNG dùng $(words ...): $(shell) nuốt newline thành
# khoảng trắng, nên $(words) sẽ đếm số TỪ trong tên card
# ("NVIDIA GeForce RTX 3080" = 4) thay vì số GPU.
GPU_NAMES  := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd'|' -)
GPU_COUNT  := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
IS_A100    := $(findstring A100,$(GPU_NAMES))

ifeq ($(GPU_COUNT),1)
QIE_RESTART_POLICY ?= unless-stopped
else ifneq ($(IS_A100),)
QIE_RESTART_POLICY ?= unless-stopped
else
QIE_RESTART_POLICY ?= no
endif
export QIE_RESTART_POLICY

# In ra phần cứng dò được và quyết định đi kèm.
gpu-info:
	@echo "GPU            : $(GPU_NAMES)"
	@echo "Số GPU         : $(GPU_COUNT)"
	@echo "Restart policy : $(QIE_RESTART_POLICY)"

# Tải weight về ./models. Cache HF đã bị ghim vào project nên không có gì
# rơi ra ~/.cache. Bỏ qua 5 shard transformer BF16 (~39GB) mà pipeline không
# dùng tới; đặt QIE_DOWNLOAD_BASE_TRANSFORMER=true nếu muốn bản repo đầy đủ.
download:
	python scripts/download_model.py

# Kiểm tra cây model trên đĩa trước khi tốn 2 phút load. Không cần GPU.
preflight:
	python scripts/preflight.py

# Chạy đúng bài kiểm tra đó BÊN TRONG container, để bắt lỗi mount sai.
preflight-docker:
	docker compose run --rm --entrypoint python qwen-lightning scripts/preflight.py

build:
	docker compose build

run:
	@echo ">>> restart policy = $(QIE_RESTART_POLICY) (GPU: $(GPU_NAMES))"
	docker compose up -d
	@echo ">>> Đang load model + warm-up (~2-3 phút). Theo dõi: make logs"

# Đo tốc độ sinh ảnh thật bên trong container đang chạy.
benchmark:
	docker compose exec qwen-lightning python scripts/benchmark.py

logs:
	docker compose logs -f

# In link share công khai khi QIE_SHARE=true
link:
	@docker compose logs 2>/dev/null | grep -aoE 'https://[a-z0-9]+\.gradio\.live' | tail -1 \
		|| echo "Chưa có link — model có thể còn đang load, xem: make logs"

stop:
	docker compose down

shell:
	docker compose exec qwen-lightning bash

ps:
	docker compose ps

# Xoá cả image và volume cache
clean:
	docker compose down -v --rmi local
