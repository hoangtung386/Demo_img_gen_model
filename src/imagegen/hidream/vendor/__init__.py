"""Code inference của HiDream-O1-Image, copy nguyên từ repo gốc.

KHÔNG SỬA TAY CÁC FILE TRONG THƯ MỤC NÀY ngoài các patch đã ghi ở VENDOR.md.
Cập nhật: chạy ``scripts/vendor_hidream.sh <sha>`` rồi áp lại patch.

Vì sao phải vendor: HiDream không phát hành code inference lên PyPI. Model
card bảo ``git clone`` rồi chạy từ trong repo, tức ``from models.pipeline
import ...`` phụ thuộc cwd — không dùng được trong container.

Thư mục này bị loại khỏi ruff/pytest (xem ``[tool.ruff] extend-exclude``
trong pyproject.toml): đây là code của bên thứ ba, giữ nguyên dạng để lần
cập nhật sau còn diff được với upstream.
"""
