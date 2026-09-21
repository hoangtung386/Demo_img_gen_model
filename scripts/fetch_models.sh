#!/usr/bin/env sh
#
# Kéo trọng số từ Google Cloud Storage vào thư mục model, chạy một lần trước
# khi app khởi động. Thay hẳn cho việc tải từ HuggingFace Hub: không cần
# HF_TOKEN, không cần accept license lúc deploy, không cần .env.
#
# Biến môi trường:
#   GENIMG_MODELS_URI      gs://bucket/path — BẮT BUỘC. Xem phần "Dạng URI".
#   GENIMG_MODELS_DIR      thư mục đích (mặc định /models)
#   GENIMG_GCS_KEY_FILE    key download model (config_setup/credentials/
#                       model-download-key.json). Bỏ trống thì tự dò (_find_key)
#   GENIMG_MODELS_REQUIRE  các đường dẫn tương đối phải tồn tại sau khi giải nén
#   GENIMG_MODELS_FORCE    "true" => tải lại kể cả khi đã có sẵn
#
# Dạng URI — quyết định cách tải:
#   *.tar.zst  stream thẳng: cat | zstd -d | tar -x. KHÔNG có file archive
#              nào chạm đĩa, nên chỉ cần 27GB trống thay vì 54GB. Nhanh nhất.
#   *.tar.gz   như trên nhưng giải nén chậm hơn nhiều (gzip một luồng).
#   *.tar      như trên, không nén.
#   *.zip      PHẢI tải trọn file xuống đĩa rồi mới giải nén được: central
#              directory của zip nằm ở CUỐI file nên không stream được. Cần
#              gấp đôi dung lượng đĩa. Script kiểm tra chỗ trống trước.
#   còn lại    coi là prefix thư mục -> gcloud storage rsync -r. Tải song
#              song, dở dang thì chạy lại là tiếp tục chứ không làm lại từ đầu.
set -eu

# Nạp config từ config_setup/base.yaml (nguồn config duy nhất) NẾU có sẵn.
# Trong container model-fetcher config đến từ docker-compose env (base.yaml
# không mount vào đó) → guard này no-op, env thắng. Trên host thì đọc base.yaml.
_SELF_DIR="$(cd "$(dirname "$0")" && pwd 2>/dev/null || echo .)"
if command -v python3 >/dev/null 2>&1 && [ -f "$_SELF_DIR/config_env.py" ]; then
    eval "$(python3 "$_SELF_DIR/config_env.py" 2>/dev/null || true)"
fi

DIR="${GENIMG_MODELS_DIR:-/models}"
REQUIRE="${GENIMG_MODELS_REQUIRE:-FLUX.2-klein-9B/model_index.json}"
MARKER="$DIR/.fetched-from"
STAGE="$DIR/.extract-tmp"

log() { echo "[fetch-models] $*"; }
die() { echo "[fetch-models] LỖI: $*" >&2; exit 1; }

# Đường dẫn tương đối đầu tiên trong REQUIRE, dùng để nhận ra gốc archive.
_first_require() { for r in $REQUIRE; do echo "$r"; return; done; }

_find_key() {
    # Thứ tự: biến môi trường -> chỗ mount chuyên dụng -> gốc repo. Gốc repo
    # là chỗ người dùng thả file key sau khi clone, nên phải dò tới.
    # Key DOWNLOAD model nội bộ = config_setup/credentials/model-download-key.json
    # (mount vào /credentials trong container). KHÁC key upload bucket user.
    for c in "${GENIMG_GCS_KEY_FILE:-}" \
             /credentials/model-download-key.json \
             /project/config_setup/credentials/model-download-key.json \
             /project/ai-asset-amb.json; do
        if [ -n "$c" ] && [ -f "$c" ]; then echo "$c"; return; fi
    done
    for c in /project/ai-asset*.json /project/*service-account*.json; do
        if [ -f "$c" ]; then echo "$c"; return; fi
    done
}

_missing() {
    m=""
    for path in $REQUIRE; do
        [ -e "$DIR/$path" ] || m="$m $path"
    done
    echo "$m"
}

# Dời nội dung từ thư mục tạm sang thư mục đích, bỏ cấp bọc ngoài nếu có.
#
# models.zip thường được đóng bằng `zip -r models.zip models/` nên bên trong
# có đúng một cấp "models/". Giải thẳng vào /models sẽ ra /models/models/... và
# app không tìm thấy gì. Nhận diện bằng chính REQUIRE chứ không đoán theo tên:
# chỉ bỏ cấp bọc khi file bắt buộc nằm BÊN TRONG nó chứ không nằm ở gốc.
_flatten_into() {
    src="$1"
    first="$(_first_require)"
    inner="$src"

    if [ ! -e "$src/$first" ]; then
        for candidate in "$src"/*; do
            if [ -d "$candidate" ] && [ -e "$candidate/$first" ]; then
                inner="$candidate"
                log "Archive bọc trong $(basename "$candidate")/ — bỏ cấp này."
                break
            fi
        done
    fi

    # mv cùng filesystem nên tức thì, không copy lại 27GB.
    for entry in "$inner"/* "$inner"/.[!.]*; do
        [ -e "$entry" ] || continue
        name=$(basename "$entry")
        rm -rf "$DIR/$name"
        mv "$entry" "$DIR/$name"
    done
    rm -rf "$src"
}

[ -n "${GENIMG_MODELS_URI:-}" ] || die "chưa đặt GENIMG_MODELS_URI (gs://bucket/...)"
URI="$GENIMG_MODELS_URI"

mkdir -p "$DIR"

# --- Đã có sẵn thì thôi -----------------------------------------------------
# Container này chạy lại mỗi lần `docker compose up`. Không có bước này thì
# mỗi lần restart service là một lần tải hàng chục GB.
if [ "${GENIMG_MODELS_FORCE:-false}" != "true" ] &&
   [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$URI" ]; then
    missing=$(_missing)
    if [ -z "$missing" ]; then
        log "Model đã có sẵn từ $URI — bỏ qua. (GENIMG_MODELS_FORCE=true để tải lại)"
        exit 0
    fi
    log "Marker khớp nhưng thiếu:$missing — tải lại."
fi

# --- Xác thực ---------------------------------------------------------------
KEY=$(_find_key)
if [ -n "$KEY" ]; then
    log "Xác thực bằng key service account: $KEY"
    gcloud auth activate-service-account --key-file="$KEY" --quiet
else
    # Trên GCE, service account gắn với VM là cách đúng: không có file key nào
    # để rò rỉ, xoay vòng credential do Google lo, thu hồi bằng một lệnh IAM.
    log "Không thấy file key — dùng ADC (service account gắn với VM)."
fi

log "Nguồn : $URI"
log "Đích  : $DIR"

rm -rf "$STAGE"
mkdir -p "$STAGE"

# --- Tải --------------------------------------------------------------------
case "$URI" in
    *.tar.zst|*.tzst)
        log "Stream tar.zst (không cần chỗ chứa archive)"
        gcloud storage cat "$URI" | zstd -d -T0 | tar -x -C "$STAGE"
        ;;
    *.tar.gz|*.tgz)
        log "Stream tar.gz"
        gcloud storage cat "$URI" | tar -xz -C "$STAGE"
        ;;
    *.tar)
        log "Stream tar"
        gcloud storage cat "$URI" | tar -x -C "$STAGE"
        ;;
    *.zip)
        # Zip không stream được, nên phải kiểm tra đĩa TRƯỚC: hết chỗ giữa
        # chừng sẽ để lại một archive cụt, đúng kiểu hỏng khó đọc nhất.
        size_b=$(gcloud storage ls -l "$URI" | awk 'NR==1{print $1}')
        case "$size_b" in
            ''|*[!0-9]*) size_b=0 ;;
        esac
        # printf "%.0f" chứ không phải print/‘%d’ trần: awk in số > 999999 ở
        # dạng khoa học (7.1e+10) mà shell không hiểu trong $(( )), và %d trên
        # awk tràn ở số nguyên 32-bit (2147483647) — cả hai đều gặp thật với
        # ổ đĩa cỡ chục/trăm GB. %.0f là dạng duy nhất an toàn cho cả hai.
        free_b=$(df -P "$DIR" | awk 'NR==2{printf "%.0f", $4 * 1024}')
        need_b=$(( size_b * 22 / 10 ))
        log "Zip: $((size_b / 1073741824))GiB nén, cần ~$((need_b / 1073741824))GiB trống (archive + bản giải nén), đang có $((free_b / 1073741824))GiB"
        [ "$size_b" -eq 0 ] || [ "$free_b" -gt "$need_b" ] ||
            die "không đủ đĩa. Dùng .tar.zst để stream thẳng, chỉ cần một nửa."
        tmp="$DIR/.download.zip"
        rm -f "$tmp"
        gcloud storage cp "$URI" "$tmp"
        log "Giải nén..."
        unzip -q -o "$tmp" -d "$STAGE"
        rm -f "$tmp"
        ;;
    *)
        log "Đồng bộ thư mục (rsync, tải song song và tiếp tục được)"
        rm -rf "$STAGE"
        gcloud storage rsync -r "$URI" "$DIR"
        ;;
esac

[ -d "$STAGE" ] && _flatten_into "$STAGE"

# --- Kiểm tra ---------------------------------------------------------------
# Tải xong không có nghĩa là đúng cây thư mục: archive đóng sai gốc sẽ cho
# /models/models/FLUX.2-... và app chỉ báo lỗi sau 2 phút nạp model.
missing=$(_missing)
if [ -n "$missing" ]; then
    log "Cây thư mục thực tế ở $DIR:"
    ls -la "$DIR" >&2 || true
    die "thiếu:$missing — kiểm tra archive có chứa FLUX.2-klein-9B/ không."
fi

printf '%s' "$URI" > "$MARKER"
log "Xong. Tổng dung lượng: $(du -sh "$DIR" | cut -f1)"
