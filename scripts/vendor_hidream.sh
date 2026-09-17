#!/usr/bin/env bash
# Kéo models/ từ repo HiDream-O1-Image vào src/imagegen/hidream/vendor/.
#
# Script này CHỈ copy. Sau khi chạy PHẢI áp lại các patch ghi trong
# vendor/VENDOR.md — thiếu chúng thì inference nổ ngay forward đầu tiên
# (flash-attn) và tầng Gradio không truyền được PIL vào pipeline.
#
#   scripts/vendor_hidream.sh <commit-sha-hoặc-tag>
set -euo pipefail

REPO="https://github.com/HiDream-ai/HiDream-O1-Image.git"
REF="${1:?usage: $0 <commit-sha-hoac-tag>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/src/imagegen/hidream/vendor"
FILES=(pipeline qwen3_vl_transformers utils fm_solvers_unipc flash_scheduler)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --filter=blob:none --no-checkout "$REPO" "$TMP/src"
git -C "$TMP/src" checkout "$REF" -- models

for f in "${FILES[@]}"; do
    cp "$TMP/src/models/$f.py" "$DEST/$f.py"
    # Upstream có file dính BOM; Python import được nhưng công cụ phân tích
    # tĩnh thì không.
    sed -i '1s/^\xEF\xBB\xBF//' "$DEST/$f.py"
    echo "  copied $f.py"
done

cat <<MSG

Đã copy từ $REF vào $DEST

BƯỚC BẮT BUỘC TIẾP THEO — áp lại patch trong $DEST/VENDOR.md:
  1. pipeline.py  use_flash_attn -> cờ env IMG_USE_FLASH_ATTN
  2. pipeline.py  _as_pil() nhận PIL.Image ngoài đường dẫn
  3. pipeline.py  print() -> logger
Rồi cập nhật dòng Commit + Ngày copy trong VENDOR.md.
MSG
