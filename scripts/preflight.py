"""Kiểm tra cây model trên đĩa TRƯỚC khi load pipeline.

Chỉ dùng stdlib — không cần torch, không cần GPU, không chạm mạng. Mục đích
là bắt lỗi thiếu file / sai mount trong vài giây thay vì sau 2 phút load model
rồi đổ traceback giữa chừng.

    python scripts/preflight.py

Trong container:

    docker compose run --rm --entrypoint python gen-image scripts/preflight.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gen_image.config import load_settings  # noqa: E402

_GB = 1024**3

# Ngưỡng cảnh báo tổng weight. FLUX.2-klein-4B bf16 đo được ~16 GB
# (transformer 7.75 + text_encoder Qwen3-4B ~8 + vae ~0.3). Vượt xa ngưỡng
# này nghĩa là đang trỏ nhầm sang repo khác — nhiều khả năng là FLUX.2-dev
# (32B, ~64GB) hoặc bản klein 9B.
_EXPECTED_MAX_GB = 24.0


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _warn(msg: str) -> None:
    print(f"  ! {msg}")


def _shard_names(component: Path) -> set[str] | None:
    """Tên các shard khai báo trong *.index.json, nếu component bị chia nhỏ."""
    for index in component.glob("*.index.json"):
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
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


def _check_distilled(spec: dict) -> None:
    """Kêu nếu model_index.json không đánh dấu bản distilled.

    Nạp nhầm ``FLUX.2-klein-base-4B`` vẫn chạy được, chỉ là mỗi ảnh chậm
    gấp bội ở cùng num_steps và ảnh ra ở 4 bước thì nhiễu. Bắt ở đây rẻ hơn
    nhiều so với việc đổ lỗi cho GPU sau khi deploy.
    """
    if spec.get("is_distilled") is True:
        _ok("is_distilled: true — đúng bản step-distilled 4 bước")
    else:
        _warn(
            "model_index.json KHÔNG có is_distilled=true — nhiều khả năng "
            "đây là bản 'klein-base' chứ không phải bản distilled. Ảnh ra ở "
            "num_steps=4 sẽ nhiễu và CFG chạy hai nhánh (chậm gấp đôi)."
        )


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
            "  ! HF_HOME vẫn nằm trong ~/.cache — model có thể bị tải ra ngoài project"
        )

    ok = True
    total = 0

    print(f"\n[1] Base pipeline: {settings.base_model_local}")
    if not settings.base_model_local:
        _fail("GENIMG_BASE_MODEL_LOCAL chưa được set — sẽ tải từ Hub")
        ok = False
    else:
        base = Path(settings.base_model_local)
        index = base / "model_index.json"
        if not index.is_file():
            _fail(f"thiếu {index}")
            ok = False
        else:
            spec = json.loads(index.read_text(encoding="utf-8"))
            _check_distilled(spec)
            components = [k for k in spec if not k.startswith("_")]
            # Nhánh GGUF thay transformer bf16 bằng file .gguf, nên thư mục
            # transformer/ không cần weight — nhưng config.json của nó thì
            # VẪN cần: from_single_file đọc đúng file đó qua config=+subfolder=
            # để khỏi đoán nhầm kiến trúc (xem models/loader.py).
            skipped = set()
            if settings.quantization == "gguf":
                skipped = {"transformer"}
            for name in sorted(components):
                if not isinstance(spec.get(name), (list, tuple)):
                    continue
                if name in skipped:
                    cfg = base / name / "config.json"
                    if cfg.is_file():
                        _ok(f"{name}/ — bỏ qua weight (dùng GGUF), có config.json")
                    else:
                        _fail(
                            f"{name}/config.json thiếu — from_single_file cần "
                            "file này để đọc đúng kiến trúc klein"
                        )
                        ok = False
                    continue
                passed, size = check_component(base, name)
                ok = ok and passed
                total += size

    print(f"\n[2] Transformer GGUF (quantization={settings.quantization})")
    if settings.quantization != "gguf":
        print("  – bỏ qua: đang chạy bf16, transformer nằm trong base pipeline")
    elif not settings.transformer_gguf:
        _fail("GENIMG_TRANSFORMER_GGUF chưa được set — sẽ tải từ Hub")
        ok = False
    else:
        weights = Path(settings.transformer_gguf)
        if not weights.is_file():
            _fail(f"thiếu {weights}")
            ok = False
        else:
            size = weights.stat().st_size
            total += size
            _ok(f"{weights.name} — {size / _GB:.1f} GB")

    print("\n" + "=" * 66)
    print(f"Tổng weight sẽ nạp vào VRAM: {total / _GB:.1f} GB")
    if total / _GB > _EXPECTED_MAX_GB:
        _warn(
            f"vượt {_EXPECTED_MAX_GB:.0f} GB — FLUX.2-klein-4B bf16 chỉ "
            "khoảng 16 GB. Kiểm tra xem có đang trỏ nhầm sang FLUX.2-dev "
            "hoặc bản klein 9B không."
        )
    print(f"Chiến lược đặt model: {settings.offload}")
    print("=" * 66)
    print("\n✅ PREFLIGHT ĐẠT" if ok else "\n❌ PREFLIGHT HỎNG")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
