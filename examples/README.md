# Ảnh mẫu cho khối `gr.Examples`

Thư mục này là **asset chức năng, được commit** (xem mục "GIỮ LẠI có chủ đích"
ở cuối `.gitignore`). UI đọc nó để hiện ví dụ bấm-là-chạy; thiếu thì demo vẫn
chạy bình thường, chỉ là khối ví dụ trống —
`ui/components.py::sample_image` trả `None` và `example_rows()` bỏ qua case đó.

Docker **không** bake thư mục này vào image; `docker-compose.yml` mount
`./examples:ro` lúc chạy. Thả ảnh mới rồi restart container là ví dụ hiện ra,
không cần build lại.

## Bố cục

```
examples/
├── README.md              ← file này
├── <tên-ảnh>.jpg          ← ảnh ĐẦU VÀO, tên khớp ui/examples_spec.py
└── outputs/
    └── <tab>_<n>.png      ← ảnh KẾT QUẢ dựng sẵn, do warm_examples.py sinh
```

Danh sách tên file đầu vào do `src/imagegen/ui/examples_spec.py` quy định —
đó là nguồn sự thật duy nhất. Xem `VTO_CASES`, `HOME_CASES`, `CARTOON_CASES`,
`HUG_CASES`, `FACESWAP_CASES` trong đó.

## Thêm hoặc đổi ảnh mẫu

1. Thả ảnh đầu vào vào `examples/` với đúng tên trong `examples_spec.py`.
2. Khởi động demo (`uv run run-app`).
3. Sinh lại ảnh kết quả dựng sẵn:

   ```bash
   python scripts/warm_examples.py            # localhost:7860
   python scripts/warm_examples.py http://<ip>:7860
   ```

> ⚠️ Mỗi case tốn hàng chục giây đến vài phút trên HiDream-O1 (50 bước, CFG,
> 2048²). Cả bộ có thể mất 30+ phút.

## ⚠️ Trạng thái sau khi thay lõi model

Repo hiện **chưa có bộ ảnh nào được commit** (`outputs/` chỉ có `.gitkeep`).

Nếu bạn khôi phục một bộ ảnh cũ từ lịch sử git: ảnh trong `outputs/` là kết
quả của **model cũ** (Qwen-Image-Edit-2509 Lightning). Chế độ demo
(`IMG_DEMO_CACHE=true`, mặc định bật) sẽ trưng ra chúng như thể model đang
chạy vừa sinh ra. Phải chạy lại `warm_examples.py` trước khi demo cho khách,
hoặc đặt `IMG_DEMO_CACHE=false`.

## Giấy phép

Ảnh đặt vào đây phải có quyền sử dụng hợp lệ cho mục đích demo. Khi bổ sung
một bộ ảnh mới, ghi nguồn và giấy phép vào bảng dưới — đây là lý do file này
tồn tại.

| Bộ ảnh | Nguồn | Giấy phép |
| :-- | :-- | :-- |
| _(chưa có bộ nào được commit)_ | — | — |
