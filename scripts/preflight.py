"""Kiểm tra model trên đĩa TRƯỚC khi nạp.

Chỉ dùng stdlib — không cần torch, không cần GPU, không chạm mạng. Mục đích
là bắt lỗi thiếu file / sai mount trong vài giây thay vì sau hàng chục giây
nạp model rồi đổ traceback giữa chừng.

    python scripts/preflight.py

Trong container:

    docker compose run --rm --entrypoint python imagegen scripts/preflight.py

Cây model đã đổi hẳn so với bản Qwen: giờ là một repo transformers PHẲNG
(config.json + 3 shard + tokenizer ở cấp gốc), không còn model_index.json và
các thư mục con vae/ text_encoder/ transformer/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from imagegen.config import load_settings  # noqa: E402

# artifacts là module THUẦN STDLIB — import nó không kéo theo torch,
# transformers hay sdnq. Đó là điều kiện để script này giữ đúng lời hứa
# ở docstring: chạy được trong vài giây trên máy không GPU.
from imagegen.hidream.artifacts import (  # noqa: E402
    ARCHITECTURE,
    EXPECTED_WEIGHT_GB,
    QUANT_METHOD,
    REQUIRED_FILES,
    WEIGHT_SIZE_TOLERANCE,
)

_GB = 1024**3


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def check_repo(root: Path) -> tuple[bool, int]:
    """Kiểm tra một snapshot repo transformers. Trả ``(đạt, tổng byte)``."""
    if not root.is_dir():
        _fail(f"{root} — không phải thư mục")
        return False, 0

    missing = [f for f in REQUIRED_FILES if not (root / f).is_file()]
    if missing:
        _fail(f"thiếu file bắt buộc: {missing}")
        return False, 0
    _ok(f"{len(REQUIRED_FILES)} file config/tokenizer đầy đủ")

    index_path = root / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = set(index["weight_map"].values())
    except (OSError, ValueError, KeyError) as exc:
        _fail(f"{index_path.name} hỏng hoặc thiếu weight_map: {exc}")
        return False, 0

    absent = sorted(s for s in shards if not (root / s).is_file())
    if absent:
        _fail(f"thiếu {len(absent)}/{len(shards)} shard: {absent[:3]}")
        return False, 0

    total = sum((root / s).stat().st_size for s in shards)
    _ok(f"{len(shards)} shard, {total / _GB:.2f} GB")
    return True, total


def check_quantization(root: Path) -> bool:
    """Xác nhận config khai báo quantizer SDNQ.

    Nạp nhầm một repo BF16 chưa lượng tử hoá vẫn chạy được nhưng tốn 17 GiB
    VRAM thay vì 11 — im lặng cho tới lúc OOM trên card nhỏ.
    """
    try:
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail(f"config.json không đọc được: {exc}")
        return False

    method = (cfg.get("quantization_config") or {}).get("quant_method")
    if method != QUANT_METHOD:
        _fail(
            f"quant_method={method!r}, mong đợi {QUANT_METHOD!r}. "
            "Đây có phải bản đã "
            "lượng tử hoá không?"
        )
        return False
    _ok(f"quantization_config.quant_method = {QUANT_METHOD}")

    arch = cfg.get("architectures") or []
    if ARCHITECTURE not in arch:
        _fail(f"architectures={arch}, mong đợi {ARCHITECTURE}")
        return False
    _ok(f"architectures = {arch[0]}")
    return True


def main() -> int:
    settings = load_settings()
    print("=" * 66)
    print("PREFLIGHT — kiểm tra model trên đĩa")
    print("=" * 66)

    print(f"\nHF_HOME        = {os.environ.get('HF_HOME', '<default>')}")
    print(f"HF_HUB_OFFLINE = {os.environ.get('HF_HUB_OFFLINE', '0')}")
    home = os.environ.get("HF_HOME", "")
    if home and (Path.home() / ".cache") in Path(home).parents:
        print(
            "  ! HF_HOME vẫn nằm trong ~/.cache — model có thể bị "
            "tải ra ngoài project"
        )

    print(f"\n[1] Model: {settings.model_path_local}")
    if not settings.model_path_local:
        _fail("IMG_MODEL_PATH chưa được set — sẽ tải từ Hub")
        return 1

    root = Path(settings.model_path_local)
    files_ok, total = check_repo(root)
    quant_ok = check_quantization(root) if files_ok else False
    ok = files_ok and quant_ok

    if files_ok:
        gb = total / _GB
        if (
            abs(gb - EXPECTED_WEIGHT_GB) / EXPECTED_WEIGHT_GB
            > WEIGHT_SIZE_TOLERANCE
        ):
            print(
                f"  ! {gb:.2f} GB lệch nhiều so với mức mong đợi "
                f"~{EXPECTED_WEIGHT_GB} GB — kiểm tra lại nguồn tải"
            )

    print("\n" + "=" * 66)
    print(f"Tổng weight sẽ nạp vào VRAM: {total / _GB:.2f} GB")
    print(f"Biến thể      : {settings.model_type}")
    print(f"Số bước       : {settings.num_steps}")
    print(f"guidance_scale: {settings.guidance_scale}")
    print(f"Kích thước    : {settings.width}x{settings.height}")
    print("=" * 66)
    print("\n✅ PREFLIGHT ĐẠT" if ok else "\n❌ PREFLIGHT HỎNG")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
