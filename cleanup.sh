#!/usr/bin/env bash
# cleanup.sh — giải phóng VRAM và cổng mà demo FLUX.2-klein đang chiếm giữ.
# Dọn cả hai kiểu chạy: container Docker của dự án, và tiến trình chạy trực
# tiếp trên host (nhận diện qua cổng GENIMG_PORT). Không ảnh hưởng tiến trình khác.
set -u

# Chạy được từ bất kỳ đâu: `docker compose` cần đúng thư mục có
# docker-compose.yml, và .env cũng được đọc theo đường dẫn tương đối.
cd "$(dirname "$0")" || exit 1

PORT="${GENIMG_PORT:-7860}"
# Đọc cổng: env GENIMG_PORT thắng, rồi .env (GENIMG_PORT=...), cuối cùng
# config_setup/base.yaml (app.server_port) — nguồn config duy nhất.
if [ -z "${GENIMG_PORT:-}" ] && [ -f .env ]; then
    _env_port="$(grep -E '^GENIMG_PORT=' .env | head -n1 | cut -d= -f2- \
        | cut -d'#' -f1 | tr -d ' \r')"
    [ -n "$_env_port" ] && PORT="$_env_port"
fi
if [ -z "${GENIMG_PORT:-}" ] && [ "$PORT" = "7860" ] && [ -f config_setup/base.yaml ]; then
    # Lấy server_port trong khối app: (dòng thụt lề dưới `app:`).
    _yaml_port="$(awk '/^app:/{a=1;next} /^[a-zA-Z]/{a=0} a&&/^[[:space:]]+server_port:/{gsub(/[^0-9]/,"");print;exit}' config_setup/base.yaml)"
    [ -n "$_yaml_port" ] && PORT="$_yaml_port"
fi

echo ">>> Dọn dẹp demo trên cổng: $PORT"

# --- 1. Container Docker của dự án -----------------------------------------
# Phải làm TRƯỚC phần xử lý cổng: khi container đang map cổng, `fuser` trả về
# PID của `docker-proxy`. Kill docker-proxy không dừng container (container vẫn
# giữ VRAM) mà chỉ phá port mapping của Docker.
if command -v docker >/dev/null 2>&1 && [ -f docker-compose.yml ]; then
    if [ -n "$(docker compose ps -q 2>/dev/null)" ]; then
        echo ">>> Tìm thấy container của dự án. Đang dừng..."
        docker compose down
        echo ">>> Đã dừng container. VRAM và cổng $PORT đã được giải phóng."
    else
        echo ">>> Không có container nào của dự án đang chạy."
    fi
fi

# --- 2. Tiến trình chạy trực tiếp trên host --------------------------------
PID=""
if command -v fuser >/dev/null 2>&1; then
    PID="$(fuser "${PORT}/tcp" 2>/dev/null | tr -s ' ')"
elif command -v lsof >/dev/null 2>&1; then
    PID="$(lsof -ti "tcp:${PORT}" 2>/dev/null)"
fi

if [ -n "${PID// }" ]; then
    # Cổng vẫn bị docker-proxy giữ => còn container khác (không thuộc compose
    # của dự án) đang map cổng này. Cảnh báo, tuyệt đối không kill.
    for _p in $PID; do
        _comm="$(cat "/proc/${_p}/comm" 2>/dev/null || true)"
        if [ "$_comm" = "docker-proxy" ]; then
            echo ">>> CẢNH BÁO: cổng $PORT do docker-proxy (PID $_p) giữ — một"
            echo ">>>   container ngoài compose của dự án. Tự kiểm tra:"
            echo ">>>   docker ps --filter publish=$PORT"
            PID=""
            break
        fi
    done
fi

if [ -n "${PID// }" ]; then
    echo ">>> Tìm thấy tiến trình host (PID: $PID). Đang dừng..."
    # SIGTERM trước, nếu không phản hồi mới SIGKILL
    kill -TERM $PID 2>/dev/null || true
    sleep 2
    if kill -0 $PID 2>/dev/null; then
        kill -9 $PID 2>/dev/null || true
    fi
    echo ">>> Đã dừng tiến trình. VRAM và cổng $PORT đã được giải phóng."
    # GENIMG_SHARE=true khiến Gradio chạy kèm tiến trình tunnel `frpc`. Bị SIGKILL
    # theo tiến trình cha thì frpc thành mồ côi và vẫn giữ link share cũ.
    # Lọc theo `-l $PORT`: máy có thể đang chạy demo Gradio khác, kill hết
    # `frpc` sẽ ngắt luôn tunnel của họ.
    FRPC="$(pgrep -f "frpc_linux.* -l ${PORT}( |\$)" 2>/dev/null | tr '\n' ' ')"
    if [ -n "${FRPC// }" ]; then
        echo ">>> Dọn tunnel Gradio còn sót (frpc, PID: $FRPC)..."
        kill -9 $FRPC 2>/dev/null || true
    fi
else
    echo ">>> Không có tiến trình host nào chiếm cổng $PORT."
    # Demo có thể đã chết phần Gradio nhưng tiến trình Python vẫn sống và
    # tiếp tục giữ VRAM trên cả hai GPU. Chỉ cảnh báo, không tự kill.
    ORPHAN="$(pgrep -f "scripts/serve.py|run-app|frpc_linux.* -l ${PORT}( |\$)" \
        2>/dev/null | tr '\n' ' ')"
    if [ -n "${ORPHAN// }" ]; then
        echo ">>> CẢNH BÁO: còn tiến trình demo/tunnel không giữ cổng (PID: $ORPHAN)."
        echo ">>>   VRAM hoặc link share có thể vẫn sống. Kiểm tra rồi dừng thủ công:"
        echo ">>>   kill -9 $ORPHAN"
    fi
fi

# Hiển thị trạng thái VRAM hiện tại (nếu có nvidia-smi)
if command -v nvidia-smi >/dev/null 2>&1; then
    echo ">>> Trạng thái VRAM hiện tại:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
fi
