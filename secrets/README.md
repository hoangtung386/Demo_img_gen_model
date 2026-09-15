Đặt key service account của GCS vào đây, tên `gcs-key.json`:

    cp ~/Downloads/ai-asset-amb.json secrets/gcs-key.json

Thư mục này bị `.gitignore` bỏ qua (trừ chính file README này).

**Không cần key nếu VM trên GCE đã được gắn service account** — đó mới là
cách nên dùng trên production: không có file key nào để rò rỉ, Google lo
xoay vòng credential, thu hồi bằng một lệnh IAM. Khi đó cứ để thư mục này
rỗng, `scripts/fetch_models.sh` tự rơi về ADC.

Quyền IAM tối thiểu cho service account: `roles/storage.objectViewer` trên
đúng bucket chứa trọng số.
