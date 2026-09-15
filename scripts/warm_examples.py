"""Chạy trước mọi test case mẫu và lưu ảnh kết quả vào examples/outputs/.

Demo phải đang chạy (host hoặc container) trước khi gọi script này, vì nó
gửi request qua Gradio API chứ không tự nạp model:

    python scripts/warm_examples.py              # localhost:7860
    python scripts/warm_examples.py http://192.168.5.233:7860

Chạy lại mỗi khi đổi danh sách trong ``ui/examples_spec.py``, đổi ảnh mẫu,
hoặc sửa prompt — ảnh dựng sẵn cũ sẽ không còn khớp với những gì model sinh ra.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gradio_client import Client, handle_file  # noqa: E402

from qwen_lightning.ui import examples_spec as spec  # noqa: E402
from qwen_lightning.ui import prompts  # noqa: E402

DEFAULT_URL = "http://localhost:7860/"


def _run(client: Client, tab: str, index: int, **kwargs) -> float:
    """Gọi một endpoint, lưu ảnh kết quả, trả về số giây đã chạy."""
    started = time.time()
    result, status = client.predict(**kwargs)
    elapsed = time.time() - started

    target = spec.output_path(tab, index)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(result, target)
    print(f"  [{tab} {index+1}] {elapsed:5.1f}s -> {target.name} | {status}")
    return elapsed


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    print(f">>> Kết nối {url}")
    client = Client(url, verbose=False)
    common = {"cfg": 1.0, "steps": 4}
    total = 0.0

    print(">>> Tab 1 — Virtual Try-On")
    for i, (garment, person) in enumerate(spec.VTO_CASES):
        total += _run(
            client, "vto", i,
            garment=handle_file(str(spec.image_path(garment))),
            person=handle_file(str(spec.image_path(person))),
            note="", seed=1000 + i, api_name="/on_vto", **common,
        )

    print(">>> Tab 2 — Home Design")
    for i, (room, design) in enumerate(spec.HOME_CASES):
        total += _run(
            client, "home", i,
            room=handle_file(str(spec.image_path(room))),
            design=design, seed=2000 + i, api_name="/on_home", **common,
        )

    print(">>> Tab 3 — Image to Cartoon")
    # Dropdown phong cách không phải tham số của endpoint, nó chỉ dựng lại
    # prompt trong UI — nên ở đây phải truyền thẳng prompt của phong cách đó.
    for i, (photo, style) in enumerate(spec.CARTOON_CASES):
        total += _run(
            client, "cartoon", i,
            photo=handle_file(str(spec.image_path(photo))),
            prompt_text=prompts.build_cartoon_prompt(style),
            note="", seed=3000 + i, api_name="/on_cartoon", **common,
        )

    print(">>> Tab 4 — Ghép 2 người ôm nhau")
    for i, (person_a, person_b) in enumerate(spec.HUG_CASES):
        total += _run(
            client, "hug", i,
            person_a=handle_file(str(spec.image_path(person_a))),
            person_b=handle_file(str(spec.image_path(person_b))),
            note="", seed=4000 + i, api_name="/on_hug", **common,
        )

    print(">>> Tab 5 — Face Swap")
    # Giống tab 3: dropdown phạm vi chỉ dựng lại prompt trong UI chứ không
    # phải tham số endpoint, nên truyền thẳng prompt của phạm vi đó.
    for i, (face, photo, scope) in enumerate(spec.FACESWAP_CASES):
        total += _run(
            client, "faceswap", i,
            face=handle_file(str(spec.image_path(face))),
            photo=handle_file(str(spec.image_path(photo))),
            prompt_text=prompts.build_faceswap_prompt(scope),
            note="", seed=5000 + i, api_name="/on_faceswap", **common,
        )

    print(f">>> Xong. Tổng {total:.1f}s, ảnh nằm ở {spec.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
