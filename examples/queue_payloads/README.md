# Payload mẫu — RabbitMQ inbound/outbound

Body JSON thật của message gửi vào/nhận từ queue của `gen-image-queue`
(production worker, xem `docker-compose.yml`). Schema/validate:
`src/gen_image/queue_service/messaging/schemas.py`.

Hợp đồng này KHÔNG đổi khi backend chuyển từ Qwen-Image-Edit-2509 sang
FLUX.2-klein-4B — cùng field, cùng status code.

- `text_to_image.json` — request chỉ có prompt, không ảnh → pipeline chạy
  với `image=None`.
- `image_edit.json` — request có `images` → pipeline nhận danh sách ảnh
  tham chiếu (tối đa 10). Ảnh **cuối cùng** trong `images` là ảnh nền (lấy
  tỉ lệ khung + là ảnh chính cần sửa); ảnh trước đó là ảnh tham chiếu
  (trang phục, mẫu phòng...).
- `image_edit_batch.json` — 5 request image-edit, mỗi cái 1 ảnh nguồn khác
  nhau + 1 prompt khác nhau. Ảnh lấy từ `tests/images_test/` đã upload lên
  `<your-bucket>/images_test/` (public URL, cùng tên file với bản
  local trong `tests/images_test/` để dễ đối chiếu).
- `outbound_reply.json` — reply mà worker publish lại lên `queue_out` sau
  khi xử lý xong (hoặc lỗi — xem `status_code`, đăng ký đầy đủ tại
  `src/gen_image/queue_service/messaging/status.py`).

Field bắt buộc trong inbound: `_id`, `os`, `firebase_token`, `appid`,
`device_id`, `prompt` — chỉ `_id`/`appid` không được rỗng; `os`/`device_id`/
`firebase_token`/`prompt` rỗng vẫn hợp lệ (xem `messaging/schemas.py`), và
`country` được phép rỗng như từ trước. LƯU Ý: `prompt` rỗng không crash
nhưng tốn 1 lượt GPU cho ảnh không theo hướng dẫn nào — client vẫn nên
tránh gửi rỗng. Thiếu/sai kiểu bất kỳ field nào trong nhóm này → worker
reply lỗi ngay, không tốn GPU.

Dùng để:
- Publish tay qua RabbitMQ Management UI (paste nguyên nội dung file vào ô
  Payload khi publish message).
- Làm mẫu khi viết script test khác ngoài `scripts/loadtest_queue.py` (script
  đó đã tự build payload qua `build_payload()`, không cần đọc file này).

## Test tải nhanh

`scripts/run_loadtest.py` — wrapper quanh `scripts/loadtest_queue.py`, fix
cứng sẵn broker (host/port/user/password/vhost) + ảnh test thật (bucket
`<your-bucket>/images_test/`), chỉ cần truyền 2 tham số:

```bash
python scripts/run_loadtest.py --rate 5 --duration 60
```

- `--rate`: số message/giây tổng (mặc định `2.0` nếu bỏ qua)
- `--duration`: chạy bao nhiêu giây (mặc định `60.0` nếu bỏ qua)

Tự động `--drain-out` (đọc `queue_out` để đo latency end-to-end thật). Đổi
vhost/ảnh test thì sửa 5 hằng số ở đầu file `run_loadtest.py`, không cần
nhớ lại cú pháp CLI đầy đủ của `loadtest_queue.py`.
