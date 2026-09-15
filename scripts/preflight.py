"""Kiểm tra cây model trên đĩa TRƯỚC khi load pipeline.

Chỉ dùng stdlib — không cần torch, không cần GPU, không chạm mạng. Mục đích
là bắt lỗi thiếu file / sai mount trong vài giây thay vì sau 2 phút load model
rồi đổ traceback giữa chừng.

    python scripts/preflight.py

Trong container:

    docker compose run --rm --entrypoint python qwen-lightning \
        scripts/preflight.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qwen_lightning.config import load_settings  # noqa: E402

# Component được thay bằng bản Nunchaku INT4 nên thư mục BF16 gốc (~39GB)
# không bao giờ được diffusers đọc. Xem ghi chú trong download.py.
SKIPPED = {"transformer"}

_GB = 1024**3


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _shard_names(component: Path) -> set[str] | None:
    """Tên các shard khai báo trong *.index.json, nếu component bị chia nhỏ."""
    for index in component.glob("*.index.json"):
        weight_map = json.loads(index.read_text(encoding="utf-8")).get(
            "weight_map", {}
        )
        return set(weight_map.values())
    return None


def check_component(base: Path, name: str) -> tuple[bool, int]:
    """Trả về (đạt, tổng byte) cho một component của pipeline."""
    component = base / name
    if not component.is_dir():
        _fail(f"{name}/ — thiếu thư mục")
        return False, 0

    shards = _shard_names(component)
    if shards is not None:
        missing = [s for s in sorted(shards) if not (component / s).is_file()]
        if missing:
            _fail(f"{name}/ — thiếu {len(missing)} shard: {missing[:3]}")
            return False, 0
        total = sum((component / s).stat().st_size for s in shards)
        _ok(f"{name}/ — {len(shards)} shard, {total / _GB:.1f} GB")
        return True, total

    weights = list(component.glob("*.safetensors"))
    configs = list(component.glob("*.json")) + list(component.glob("*.jinja"))
    if not weights and not configs:
        _fail(f"{name}/ — rỗng")
        return False, 0
    total = sum(f.stat().st_size for f in weights)
    if weights:
        _ok(f"{name}/ — {len(weights)} file weight, {total / _GB:.2f} GB")
    else:
        _ok(f"{name}/ — {len(configs)} file config")
    return True, total


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

    ok = True
    total = 0

    print(f"\n[1] Base pipeline: {settings.base_model_local}")
    if not settings.base_model_local:
        _fail("QIE_BASE_MODEL_LOCAL chưa được set — sẽ tải từ Hub")
        ok = False
    else:
        base = Path(settings.base_model_local)
        index = base / "model_index.json"
        if not index.is_file():
            _fail(f"thiếu {index}")
            ok = False
        else:
            spec = json.loads(index.read_text(encoding="utf-8"))
            components = [
                k for k in spec if not k.startswith("_")
            ]
            for name in sorted(components):
                if name in SKIPPED:
                    print(f"  – {name}/ — bỏ qua (dùng bản Nunchaku INT4)")
                    continue
                passed, size = check_component(base, name)
                ok = ok and passed
                total += size

    print(f"\n[2] Nunchaku transformer: {settings.transformer_path}")
    if not settings.transformer_path:
        _fail("QIE_TRANSFORMER_PATH chưa được set — sẽ tải từ Hub")
        ok = False
    else:
        weights = Path(settings.transformer_path)
        if not weights.is_file():
            _fail(f"thiếu {weights}")
            ok = False
        else:
            size = weights.stat().st_size
            total += size
            _ok(f"{weights.name} — {size / _GB:.1f} GB")
            if settings.precision and settings.precision not in weights.name:
                _fail(
                    f"QIE_PRECISION={settings.precision} nhưng tên file là "
                    f"{weights.name}"
                )
                ok = False

    print("\n" + "=" * 66)
    print(f"Tổng weight sẽ nạp vào VRAM: {total / _GB:.1f} GB")
    print(f"Chiến lược offload: {settings.offload}")
    print("=" * 66)
    print("\n✅ PREFLIGHT ĐẠT" if ok else "\n❌ PREFLIGHT HỎNG")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
