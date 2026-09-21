"""Chốt tổ hợp phiên bản trong venv NGAY LÚC BUILD image.

Chạy ở builder stage của Dockerfile, nơi KHÔNG có GPU. Vì vậy chỉ đọc
metadata (``importlib.metadata.version``) chứ không ``import torch`` /
``import diffusers`` — import ở đó vừa chậm vừa có thể nổ vì thiếu driver.

Bước này tồn tại vì một phiên bản dependency sai KHÔNG làm build đỏ; nó chỉ
lộ ra sau vài phút nạp model bằng một ImportError khó lần.

    python scripts/verify_build.py
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

# Ghim tuyệt đối: hai package này được pin cứng trong pyproject.toml.
EXACT: dict[str, str] = {
    "torch": "2.9.0",
    "transformers": "5.15.0",
}

# ``Flux2KleinPipeline`` xuất hiện từ diffusers 0.37; 0.40 là dải đã kiểm.
MIN_DIFFUSERS: tuple[int, int] = (0, 40)

# Phải VẮNG MẶT. nunchaku khai báo ``diffusers`` không ghim phiên bản và
# thực tế chỉ chạy với 0.36 — nếu nó quay lại qua một dependency bắc cầu,
# resolver có thể kéo diffusers tụt xuống dưới ngưỡng trên, và triệu chứng
# sẽ là một ImportError lúc nạp model chứ không phải lỗi lúc build.
FORBIDDEN: tuple[str, ...] = ("nunchaku",)


def _installed(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> int:
    errors: list[str] = []

    for name, want in EXACT.items():
        got = _installed(name)
        if got != want:
            errors.append(f"{name}: cần {want}, đang là {got or '<thiếu>'}")

    diffusers = _installed("diffusers")
    if diffusers is None:
        errors.append("diffusers: thiếu")
    else:
        parts = tuple(int(x) for x in diffusers.split(".")[:2])
        if parts < MIN_DIFFUSERS:
            wanted = ".".join(str(x) for x in MIN_DIFFUSERS)
            errors.append(
                f"diffusers {diffusers} quá cũ — Flux2KleinPipeline cần >= {wanted}"
            )

    for name in FORBIDDEN:
        got = _installed(name)
        if got is not None:
            errors.append(
                f"{name} {got} vẫn còn trong venv — package này phải được "
                "gỡ hẳn (nó ghim diffusers==0.36)"
            )

    if errors:
        print("❌ VERIFY BUILD HỎNG:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        f"✅ OK — diffusers {diffusers}, "
        + ", ".join(f"{n} {v}" for n, v in EXACT.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
