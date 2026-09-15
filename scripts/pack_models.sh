#!/usr/bin/env bash
#
# Đóng gói ./models thành một archive để đẩy lên Google Cloud Storage.
#
#   scripts/pack_models.sh                                  # -> models.tar.zst
#   scripts/pack_models.sh models.tar.zst gs://bucket/path/ # đóng gói rồi upload
#
# Vì sao .tar.zst chứ không phải .zip — đo trên chính trọng số của repo này,
# mẫu 500MB lấy từ giữa file transformer INT4:
#
#   gzip -6      42.9s   tiết kiệm 32.8%
#   zstd -3 -T0   0.7s   tiết kiệm 34.9%     <- nhanh hơn 60 lần VÀ nhỏ hơn
#   zstd -10 -T0  7.0s   tiết kiệm 35.9%
#
# Quan trọng hơn cả tốc độ: tar stream được, zip thì không. Central directory
# của zip nằm ở cuối file nên phía nhận phải tải trọn archive xuống đĩa rồi
# mới giải nén — 27GB archive + 27GB bản giải nén = 54GB đĩa trống phải có.
# Với tar.zst thì `gcloud storage cat | zstd -d | tar -x` chỉ cần 27GB.
set -euo pipefail

SRC="${QIE_MODELS_SRC:-models}"
OUT="${1:-models.tar.zst}"
DEST="${2:-}"
LEVEL="${QIE_ZSTD_LEVEL:-3}"

[ -d "$SRC" ] || { echo "Không thấy thư mục $SRC" >&2; exit 1; }
command -v zstd >/dev/null || {
    echo "Thiếu zstd. Cài: sudo apt-get install -y zstd" >&2; exit 1; }

# Đóng gói từ BÊN TRONG models/ (tar -C "$SRC" .) nên archive không chứa cấp
# "models/" thừa. Phía nhận giải vào /models là ra đúng cây, không lồng thêm.
#
# Bỏ .hf và .cache: đó là cache lúc tải từ HuggingFace, không phải trọng số.
# Container chạy với HF_HOME riêng và HF_HUB_OFFLINE=1 nên không đọc tới.
echo ">>> Đóng gói $SRC -> $OUT (zstd -$LEVEL, đa luồng)"
du -sh "$SRC"
tar -c -C "$SRC" --exclude=./.hf --exclude=./.cache . \
    | zstd "-$LEVEL" -T0 -f -o "$OUT"

ls -lh "$OUT" | awk '{print ">>> Xong: " $9 " — " $5}'

if [ -n "$DEST" ]; then
    echo ">>> Upload lên $DEST"
    gcloud storage cp "$OUT" "$DEST"
    echo ">>> Đặt biến này trong .env của server:"
    echo "    QIE_MODELS_URI=${DEST%/}/$(basename "$OUT")"
else
    echo ">>> Upload bằng:"
    echo "    gcloud storage cp $OUT gs://<bucket>/<path>/"
fi
