"""Quyết định các component của pipeline nằm ở ĐÂU.

File này thay cho ``models/offload.py`` cũ (~200 dòng, 4 chiến lược, hàm
``choose_strategy()`` tự dò GPU). Toàn bộ khối đó tồn tại vì một lý do duy
nhất: backend cũ (Qwen-Image-Edit-2509) cần 26.4 GiB weight thường trú —
text_encoder Qwen2.5-VL 15.4 + transformer INT4 10.7 + vae 0.24 — trong khi
L4 chỉ có 24 GiB. Không cách nào nhét vừa, nên phải xoay: tách text_encoder
sang GPU thứ hai, hoặc đẩy nó ra CPU rồi chuyển qua lại PCIe mỗi request.

FLUX.2-klein-9B cần nén CẢ HAI component lớn thì mới vừa L4. Với cấu hình
mặc định — transformer GGUF Q4_K_M ~5.9 GiB + text_encoder Qwen3-8B NF4
~5.0 GiB + vae ~0.3 GiB ≈ 11 GiB — còn dư hơn 10 GiB cho activation, nên
chiến lược đúng lại chỉ còn một: nhét hết lên GPU và không chuyển đi đâu cả.

⚠️ Con số trên phụ thuộc ``text_encoder_quantization``. Để nó ở ``bf16`` thì
riêng text encoder đã là 16.4 GiB, tổng ~22.6 GiB — sát trần tới mức OOM ở
activation, và ``model_offload`` trở thành bắt buộc. Đây chính là chỗ bản 9B
khác bản 4B: ở 4B, text encoder chỉ 8 GiB nên GGUF một mình là đủ.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("gen-image")

# Chiến lược mặc định: mọi component thường trú trên GPU. Không một lần
# chuyển CPU<->GPU nào trong suốt vòng đời request.
RESIDENT = "resident"

# Đường lùi: diffusers giữ weight ở CPU và chỉ kéo từng module lên GPU ngay
# trước forward của module đó. Chậm hơn đáng kể (mỗi request trả toàn bộ weight qua
# PCIe) — chỉ dùng khi card nhỏ hơn L4, hoặc khi phải nhồi nhiều process
# worker lên cùng một card.
MODEL_OFFLOAD = "model_offload"

STRATEGIES = (RESIDENT, MODEL_OFFLOAD)


def place(pipeline, strategy: str, device: str) -> None:
    """Đặt pipeline lên ``device`` theo ``strategy``.

    Không tự dò VRAM rồi tự chọn như ``choose_strategy()`` cũ. Tự dò nghe
    tiện nhưng nó biến một lỗi cấu hình thành một quyết định âm thầm: service
    vẫn khởi động, chỉ là chậm gấp đôi vì đã lặng lẽ rơi về offload. Ở đây
    giá trị sai là lỗi dừng hẳn, và giá trị đúng phải do người deploy viết ra.
    """
    normalized = (strategy or "").strip().lower()
    if normalized not in STRATEGIES:
        raise ValueError(
            f"Chiến lược đặt model không hợp lệ: {strategy!r}. "
            f"Chọn một trong: {' | '.join(STRATEGIES)}."
        )

    if normalized == MODEL_OFFLOAD:
        # enable_model_cpu_offload tự gắn hook và tự quản device — KHÔNG
        # được gọi pipeline.to(device) trước hay sau nó, làm vậy là kéo hết
        # weight lên GPU và vô hiệu hoá chính cái offload vừa bật.
        pipeline.enable_model_cpu_offload(device=device)
        logger.info(
            "Đặt model: '%s' trên %s — weight ở CPU, kéo lên GPU theo từng "
            "module lúc forward. Chậm hơn 'resident'.",
            MODEL_OFFLOAD,
            device,
        )
        return

    pipeline.to(device)
    logger.info(
        "Đặt model: '%s' — mọi component thường trú trên %s, không có lần "
        "chuyển CPU<->GPU nào lúc suy luận.",
        RESIDENT,
        device,
    )
