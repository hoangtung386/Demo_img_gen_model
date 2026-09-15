#!/usr/bin/env sh
#
# Kéo trọng số từ Google Cloud Storage vào thư mục model, chạy một lần trước
# khi app khởi động. Thay hẳn cho việc tải từ HuggingFace Hub: không cần
# HF_TOKEN, không cần accept license lúc deploy, không cần .env.
#
# Biến môi trường:
#   QIE_MODELS_URI      gs://bucket/path — BẮT BUỘC. Xem phần "Dạng URI".
#   QIE_MODELS_DIR      thư mục đích (mặc định /models)
#   QIE_GCS_KEY_FILE    key service account. Bỏ trống thì tự dò (xem _find_key)
#   QIE_MODELS_REQUIRE  các đường dẫn tương đối phải tồn tại sau khi giải nén
#   QIE_MODELS_FORCE    "true" => tải lại kể cả khi đã có sẵn
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

DIR="${QIE_MODELS_DIR:-/models}"
REQUIRE="${QIE_MODELS_REQUIRE:-Qwen-Image-Edit-2509/model_index.json lightning-251115}"
MARKER="$DIR/.fetched-from"
STAGE="$DIR/.extract-tmp"

log() { echo "[fetch-models] $*"; }
die() { echo "[fetch-models] LỖI: $*" >&2; exit 1; }

# Đường dẫn tương đối đầu tiên trong REQUIRE, dùng để nhận ra gốc archive.
_first_require() { for r in $REQUIRE; do echo "$r"; return; done; }

_find_key() {
    # Thứ tự: biến môi trường -> chỗ mount chuyên dụng -> gốc repo. Gốc repo
    # là chỗ người dùng thả file key sau khi clone, nên phải dò tới.
    for c in "${QIE_GCS_KEY_FILE:-}" \
             /secrets/gcs-key.json \
             /secrets/ai-asset-amb.json \
             /project/ai-asset-amb.json \
             /project/secrets/gcs-key.json; do
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

[ -n "${QIE_MODELS_URI:-}" ] || die "chưa đặt QIE_MODELS_URI (gs://bucket/...)"
URI="$QIE_MODELS_URI"

mkdir -p "$DIR"

# --- Đã có sẵn thì thôi -----------------------------------------------------
# Container này chạy lại mỗi lần `docker compose up`. Không có bước này thì
# mỗi lần restart service là một lần tải hàng chục GB.
if [ "${QIE_MODELS_FORCE:-false}" != "true" ] &&
   [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$URI" ]; then
    missing=$(_missing)
    if [ -z "$missing" ]; then
        log "Model đã có sẵn từ $URI — bỏ qua. (QIE_MODELS_FORCE=true để tải lại)"
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
        free_b=$(df -P "$DIR" | awk 'NR==2{print $4 * 1024}')
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
# /models/models/Qwen-... và app chỉ báo lỗi sau 2 phút nạp model.
missing=$(_missing)
if [ -n "$missing" ]; then
    log "Cây thư mục thực tế ở $DIR:"
    ls -la "$DIR" >&2 || true
    die "thiếu:$missing — kiểm tra archive có chứa Qwen-Image-Edit-2509/ và lightning-251115/ không."
fi

printf '%s' "$URI" > "$MARKER"
log "Xong. Tổng dung lượng: $(du -sh "$DIR" | cut -f1)"
