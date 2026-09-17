# Credentials (service-account key của GCS)

Thư mục secret chung, chứa **2 file key** cho 2 mục đích khác nhau:

| File | Mục đích | Ai đọc |
|------|----------|--------|
| `model-download-key.json` | Tải trọng số model nội bộ từ GCS bucket | `scripts/fetch_models.sh` (container `model-fetcher`) |
| `user-upload-key.json`    | Upload ảnh kết quả lên bucket cho user | queue service (`storage.backends[].credentials_path`) |

## Đặt key

```bash
# Key tải model nội bộ
cp ~/Downloads/<key-tai-model>.json config_setup/credentials/model-download-key.json
# Key upload bucket user
cp ~/Downloads/<key-upload>.json     config_setup/credentials/user-upload-key.json
```

Trong `config_setup/base.yaml`:
- `app.gcs_key_file` trỏ key download (đường dẫn TRONG container: `/credentials/model-download-key.json`).
- `storage.backends[].credentials_path` trỏ key upload (`config_setup/credentials/user-upload-key.json`).

Nếu 2 mục đích dùng CHUNG một service account, cứ copy cùng file vào cả 2 tên.

## Không commit key thật

`.gitignore` chặn `config_setup/credentials/*.json` — chỉ `README.md` +
`.gitkeep` được commit. `.dockerignore` cũng chặn để key không lọt vào image;
key thật inject qua volume mount lúc chạy (xem `docker-compose.yml`).

## Dùng ADC thay vì file key (khuyến nghị trên production)

**Không cần file key nếu VM trên GCE đã được gắn service account** — đó mới là
cách nên dùng trên production: không có file key nào để rò rỉ, Google lo xoay
vòng credential, thu hồi bằng một lệnh IAM. Khi đó cứ để thư mục này rỗng:
`fetch_models.sh` tự rơi về ADC, và storage backend cũng vậy.

Quyền IAM tối thiểu:
- Key download: `roles/storage.objectViewer` trên bucket chứa trọng số.
- Key upload: `roles/storage.objectAdmin` (hoặc `objectCreator` + `objectViewer`)
  trên bucket phục vụ ảnh cho user.
