# Tối ưu hiệu năng — cái gì đã có, cái gì đừng làm

Tài liệu này tồn tại vì phần lớn lời khuyên "tối ưu image gen" trên mạng viết
cho **Stable Diffusion / SDXL chạy 30 bước**. FLUX.2-klein-9B là model
**rectified-flow đã chưng cất về 4 bước**, nên một số lời khuyên đó vô tác
dụng, và một số **có hại thật sự**.

Đọc bảng này trước khi áp dụng bất cứ hướng dẫn nào.

---

## Bảng tra nhanh

| Kỹ thuật (theo cách nói của SDXL) | Ở đây | Ghi chú |
|---|---|---|
| FP16/BF16 thay FP32 | ✅ **đã có** | `loader.DTYPE = bfloat16`, không có đường nào chạy fp32 |
| Distilled model (Turbo/LCM/Lightning) | ✅ **là chính model** | klein-9B đã distilled; preflight cảnh báo nếu tải nhầm bản base |
| Efficient attention (xformers) | ✅ **đã có, ĐỪNG thêm xformers** | xem §1 |
| `torch.compile` | ✅ **đã có**, mặc định tắt | `tuning.maybe_compile()`, chờ đo — xem §2 |
| TensorRT | ❌ ngoài phạm vi | xem §3 |
| Đổi sampler (DPM++ 2M Karras) | 🔴 **CÓ HẠI** | xem §4 |
| Tiny VAE (TAESD / TAEF2) | ⏸️ **đo trước** | xem §5 |
| Warm model pool | ✅ **đã có** | model resident + warm-up + cache embed — §6 |
| Continuous batching / Triton | ❌ ngoài phạm vi | xem §7 |
| **FP8** | 🧪 **chưa đo — ứng viên số 1 để nhanh hơn** | xem §8 |
| **VAE tiling / slicing** | 🧪 **mới thêm, mặc định tắt** | xem §9 |
| **Nén text encoder (NF4)** | ✅ **đã có, MẶC ĐỊNH BẬT** | đòn bẩy VRAM lớn nhất ở bản 9B — xem §0 |

---

## 0. Ngân sách VRAM trên L4 — đọc trước mọi thứ khác

Bản 9B có **hai** component lớn, và chúng nén bằng hai knob khác nhau:

| Component | bf16 | Knob | Mặc định |
|---|---:|---|---|
| text_encoder Qwen3-8B | **16.4 GiB** | `text_encoder_quantization` | `nf4` → ~5.0 GiB |
| transformer | ~18 GiB | `quantization` | `gguf` Q4_K_M → 5.9 GiB |
| VAE | 0.3 GiB | — | không nén |

Tổng ở cấu hình mặc định: **~11.2 GiB**, giữ được `offload: "resident"`.

> **"L4 24GB" thực tế là bao nhiêu?** Trên Colab, `nvidia-smi` báo **22.5
> GiB** khả dụng — phần còn lại thuộc về driver và ECC. Mọi ngưỡng trong tài
> liệu này tính theo 22.5, không phải 24.

**Dư VRAM KHÔNG tự biến thành tốc độ.** Card rảnh 8 GiB không làm denoise
nhanh hơn một mili-giây nào; nó chỉ là điều kiện cần để đổi sang một cấu
hình *khác* nhanh hơn. Đúng một đường đổi có thật ở đây — bỏ GGUF sang FP8
— xem §8.

> 🔴 **Cạm bẫy đã làm hỏng một lần deploy thật.** Đặt `quantization: "gguf"`
> nhưng để `text_encoder_quantization: "bf16"` thì tổng là **22.6 GiB** —
> vừa đúng VRAM khả dụng của L4, nên service load xong, báo "Ready", rồi OOM
> ở lượt denoise đầu tiên. GGUF **không** chạm tới text encoder. Ở bản 4B
> (text encoder chỉ 8 GiB) thì một knob là đủ; ở 9B thì không.

Toàn bộ con số "~16 GiB tổng, L4 dư VRAM" trong các bản tài liệu cũ thuộc về
**klein-4B** và không còn đúng.

---

## 0b. Số đo thật đầu tiên trên L4 (2026-09-21)

Colab L4 22.5 GiB, cấu hình mặc định (transformer GGUF Q4_K_M + text encoder
NF4 + resident), Gradio t2i, prompt đã nằm trong cache embed:

```
⚡ GPU 18.1s = denoise 17.2s (4.30s/bước) + text 0.0s (cache)
             + prep 0.4s + decode 0.5s     | 1024×1024 · 4 bước
VRAM: 14.4 / 22.5 GiB
```

| Giai đoạn | Thời gian | Tỉ trọng | Kết luận |
|---|---:|---:|---|
| denoise | 17.2s | **95%** | mọi nỗ lực tối ưu phải nhắm vào đây |
| decode (VAE) | 0.5s | 3% | **TAEF2 (§5) không đáng** — ngưỡng là 20% |
| text encode | 0.0s | 0% | cache embed (§6) đang chạy đúng; không cần đụng |
| prep | 0.4s | 2% | nhiễu |

Ba kết luận đóng lại ba hướng:

1. **§5 TAEF2: đóng.** 3% thì có xoá sạch VAE cũng chỉ còn 17.6s.
2. **§6 cache embed: đóng.** Đã 0.0s, không còn gì để lấy.
3. **§8 FP8: vẫn mở** — nhưng xem phần ngay dưới trước, vì hoá ra
   4.30s/bước KHÔNG phải do GGUF.

### ✅ Đã giải: thủ phạm là ĐỘ DÀI PROMPT, không phải tầng service

Cùng L4, cùng GGUF Q4_K_M + NF4, cùng 1024²/4 bước:

| Đường chạy | Prompt | Thời gian | denoise/bước |
|---|---|---:|---:|
| `spike_flux2.py` (pipeline trực tiếp) | ngắn | 10.97s | ~2.5s |
| `benchmark.py` (**đường service**) | ngắn | **10.82s** | **2.46s** |
| Gradio (đường service) | demo, ~1840 ký tự | **18.1s** | **4.30s** |

Hai dòng đầu **trùng nhau trong sai số**. Nên:

- **Tầng service KHÔNG tốn gì.** `generate()`, `callback_on_step_end`,
  wrapper — tất cả cộng lại dưới 0.2s. Giả thuyết "callback đo đạc làm chậm
  chính thứ đang đo" đã bị bác bỏ bằng số.
- **Prompt dài là toàn bộ 7.7s.** +75% trên MỖI bước denoise.

Vì sao đắt đến thế: FLUX.2 là MMDiT, text token và image token đi **chung**
một chuỗi joint-attention ở **mọi bước**. Prompt dài không chỉ tốn thêm một
lần lúc encode (`text` vẫn 0.0s nhờ cache) — nó làm nặng thêm từng bước
denoise, 4 lần mỗi ảnh.

> 💡 **Đây là đòn bẩy lớn nhất đo được trong dự án, và nó miễn phí.** Lớn
> hơn FP8, lớn hơn `torch.compile`, lớn hơn TAEF2 — cộng lại. Prompt demo
> trong `ui/prompts.py` chưa từng được tinh chỉnh cho FLUX.2 (chính
> `docs/architecture.md` ghi "⚠️ chưa tinh chỉnh lại cho FLUX.2").

Đo lại sau khi cắt prompt:

```bash
python scripts/benchmark.py --mode t2i                 # prompt ngắn
python scripts/benchmark.py --mode t2i --long-prompt   # prompt cỡ tab demo
```

⚠️ Cắt prompt **đổi ảnh ra**, không phải một tối ưu trong suốt. Phải xem
bằng mắt xem chất lượng có giữ được không — §4 của chính tài liệu này cảnh
báo đừng đánh đổi mù.

### 🔴 FP8 qua optimum-quanto: TRƯỢT

```
RuntimeError: A is not contiguous
optimum/quanto/tensor/weights/marlin/fp8/qbits.py:37
```

Kernel Marlin FP8 của quanto đòi input contiguous; `x_embedder` của
`Flux2Transformer2DModel` đưa vào một view không contiguous. Bug của quanto,
không phải của FP8 — và diffusers 0.40 đã deprecate cả `QuantoConfig` lẫn
`QuantoQuantizer`. Backend FP8 đã chuyển sang **torchao** (§8).

---

## 1. Attention — đã dùng SDPA, ĐỪNG thêm xformers

`enable_xformers_memory_efficient_attention()` nhắm vào API `AttnProcessor`
**cũ** của diffusers. `Flux2Transformer2DModel` không dùng API đó; nó gọi
`dispatch_attention_fn`, và backend mặc định là `native` →
`torch.nn.functional.scaled_dot_product_attention`.

Nói cách khác: **SDPA đã bật sẵn**. Cài thêm xformers chỉ thêm một dependency
nặng mà không có đường nào chạm tới nó.

Muốn thử backend khác thì **không cần sửa code** — đổi một biến môi trường:

```bash
DIFFUSERS_ATTN_BACKEND=native          # mặc định (SDPA)
DIFFUSERS_ATTN_BACKEND=_native_flash   # ép nhánh flash của SDPA
DIFFUSERS_ATTN_BACKEND=flash           # cần cài flash-attn
```

`make benchmark` in ra backend đang dùng, nên so sánh là chuyện của một lần
chạy. Lưu ý SDPA của PyTorch vốn đã tự chọn kernel flash khi đủ điều kiện,
nên khoảng cách thường nhỏ.

## 2. `torch.compile` — đã có, bật sau khi đo

`tuning.maybe_compile()` dùng `Module.compile()` tại chỗ với
`mode="max-autotune-no-cudagraphs"` và `dynamic=True`.

Hai rào chắn đã cài sẵn và **đều là giới hạn thật**:

- **GGUF không compile được** ([diffusers#10795](https://github.com/huggingface/diffusers/issues/10795)) — hàm tự bỏ qua và ghi log.
- **`dynamic=True` là bắt buộc**: chiều ảnh ra suy từ tỉ lệ ảnh người dùng
  nên gần như mỗi request một shape mới. Compile static sẽ recompile lại từ
  đầu cho từng shape — chi phí đó lớn hơn phần tiết kiệm.

Bật bằng `compile_transformer: true`, nhưng **đo trước** với
`TORCH_LOGS=recompiles` để xác nhận số lần recompile hội tụ về 0 sau vài
request đầu.

## 3. TensorRT — ngoài phạm vi, không phải vì lười

Các hướng dẫn TensorRT cho "Stable Diffusion" build engine cho UNet của SD —
kiến trúc khác hẳn. Cho FLUX.2 thì đường TensorRT thực tế là NVFP4, mà NVFP4
cần **Blackwell (sm_100+)**; L4 là Ada (sm_89) nên không dùng được.

Nếu sau này đổi sang Blackwell thì đây là hướng đáng quay lại.

## 4. 🔴 ĐỪNG đổi sampler

Đây là lời khuyên **có hại** khi bê từ SDXL sang.

"Đổi Euler sang DPM++ 2M Karras để giảm 30 bước xuống 20" đúng với model
khuếch tán thường. FLUX.2-klein là **rectified flow đã chưng cất**: nó được
huấn luyện để nhảy đúng theo lịch nhiễu của
`FlowMatchEulerDiscreteScheduler` khai trong `model_index.json`, trong đúng
4 bước. Thay scheduler khác là phá bỏ chính thứ khiến nó nhanh — ảnh ra sẽ
hỏng, không phải "nhanh hơn một chút".

Cùng lý do: **đừng hạ `num_steps` xuống dưới 4**, và tăng lên trên 4 cũng
gần như không cải thiện gì ngoài việc tốn thêm thời gian.

Backend cũ (Qwen Lightning) cũng đã tự chế scheduler config và đó là một
nguồn lỗi; bản này cố ý **đọc scheduler từ chính repo model**.

## 5. ⏸️ Tiny VAE (TAEF2) — đo trước, đừng tích hợp mù

[TAEF2](https://huggingface.co/madebyollin/taef2) có thật cho FLUX.2, nhưng:

- **chưa được tích hợp vào diffusers** — phải vendor `taesd.py` cộng một hàm
  chuyển state-dict cộng một class wrapper;
- **có đánh đổi chất lượng nhìn thấy được** theo chính model card.

Trước khi trả cái giá đó, hãy biết VAE decode đang chiếm bao nhiêu:

```bash
make benchmark      # in bảng "Tỉ trọng từng giai đoạn"
```

Ngưỡng quyết định: `decode` **dưới ~10%** thì không đáng; **trên ~20%** thì
đáng cân nhắc.

> ⚠️ **Đừng suy từ con số của SDXL.** Người ta hay trích "VAE = 15%" — con số
> đó tính cho model **30 bước**. Ở đây denoise chỉ có **4 bước**, nên mọi chi
> phí *cố định* (text encode, VAE decode) chiếm tỉ trọng **lớn hơn hẳn**.
> Rất có thể `decode` ở đây đáng kể hơn ở SDXL, chứ không phải ít hơn. Đó
> chính là lý do phải đo thay vì suy.

## 6. Warm model pool — đã có đủ

| Việc | Ở đâu |
|---|---|
| Nạp model MỘT LẦN lúc khởi động | `Flux2KleinGenerator.__init__`, chạy trước khi `/readyz` trả 200 |
| Pre-warm bằng ảnh giả | `tuning.warmup()` — chạy **cả hai** đường t2i và edit |
| Cache prompt-embed | `inference._EMBED_CACHE`, LRU, sức chứa qua `embed_cache_size` |
| Giữ resident trên VRAM | `placement.place()` với `offload: "resident"` |

**Về gợi ý "cache embed bằng Redis":** cache ở đây nằm **trong tiến trình**,
cố ý. Một lượt tra dict mất micro-giây; một vòng qua Redis mất mili-giây
cộng chi phí serialize một tensor vài MB. Cache ngoài chỉ đáng khi có
**nhiều process** cùng phục vụ và muốn chia sẻ — cấu hình hiện tại là một
process một GPU, nên không có ai để chia sẻ cùng. Nếu sau này bật
multi-process (xem [MULTI_PROCESS_WORKERS.md](MULTI_PROCESS_WORKERS.md)) thì
hãy quay lại câu hỏi này.

Muốn tăng tỉ lệ trúng cache: nâng `embed_cache_size`. Mỗi entry chỉ vài MB.

## 7. Continuous batching / Triton — ngoài phạm vi

RabbitMQ **là hợp đồng với BE/App**, không phải một lựa chọn kỹ thuật nội
bộ: schema inbound/outbound, 4 queue theo tier, DLQ, audit trail đều dựa
vào nó. Thay bằng Triton/BentoML/Ray Serve là đổi hợp đồng của cả tổ chức.

Những gì hướng dẫn đó khuyên "nếu bắt buộc giữ RabbitMQ" thì **đã làm rồi**:

- ✅ `prefetch_count: 1`
- ✅ consumer giữ model resident
- ⚠️ ảnh trả về **qua URL** (GCS/CDN) ở đường mặc định. Có một nhánh base64
  (`provider: "openai"` → `result.openai.b64_json`) nhưng đó là hợp đồng BE
  yêu cầu, không phải mặc định.

**Batching thật thì sao?** `Flux2KleinPipeline` có `num_images_per_prompt`
và nhận `prompt: list[str]`. Lý do ban đầu khiến batching bất khả thi
(kernel INT4 của Nunchaku không batch được) **đã biến mất cùng Nunchaku**.
Nhưng gom nhiều request khác nhau vào một batch đòi hỏi chúng **cùng shape**
— mà shape ở đây suy từ tỉ lệ ảnh người dùng. Đây là một thay đổi kiến trúc
thật, không phải một cờ bật. Xem MULTI_PROCESS_WORKERS.md.

## 8. 🧪 FP8 — đường đổi VRAM-dư-lấy-tốc-độ, CHƯA ĐO

`quantization: "fp8"` dùng **torchao** với
`Float8DynamicActivationFloat8WeightConfig` — lượng tử hoá cả activation
nên matmul chạy thật trên tensor core FP8 của Ada.

> Backend cũ `optimum-quanto` đã **trượt** trên L4 ("A is not contiguous",
> xem §0b) và bị diffusers deprecate. Giữ lại sau cờ `backend="quanto"` chỉ
> để tái lập kết quả, đừng chọn.

Đây là lựa chọn đáng thử nhất khi card còn dư VRAM, vì nó thắng GGUF ở
**hai** điểm cùng lúc:

1. **Không giải nén trong forward.** L4 là Ada (sm_89) nên có tensor core
   FP8 thật. GGUF Q4_K_M phải bung weight về bf16 ở mỗi lần forward, mỗi
   bước, mỗi request — đổi dung lượng lấy băng thông, mà băng thông
   (300 GB/s trên L4) mới là nút cổ chai.
2. **`torch.compile` chạy được.** GGUF thì không (§2, diffusers#10795). Nên
   FP8 mở khoá luôn cả §2, hai tối ưu cộng dồn chứ không phải chọn một.

Ngân sách VRAM (text encoder NF4, trần 22.5 GiB):

| transformer | Weight | Vừa? | Giải nén trong forward? | compile? |
|---|---:|---|---|---|
| `gguf` Q4_K_M | ~11.2 GiB | ✅ | **có** | ❌ |
| `fp8` | ~14.3 GiB | ✅ | không | ✅ |
| `bf16` | ~23.3 GiB | ❌ | không | ✅ |

FP8 vừa thoải mái, và đúng chỗ 8 GiB đang bỏ không hiện nay.

**Nhưng chưa ai đo nó trên phần cứng nào.** Trước khi bật trên production:

```bash
uv sync --extra fp8      # kéo torchao (+ quanto/ninja cho nhánh cũ)
python scripts/spike_flux2.py --modes gguf fp8              # so trực tiếp
python scripts/spike_flux2.py --modes fp8 --compile         # cộng thêm §2
```

Ba câu hỏi phải trả lời: (1) có nhanh hơn GGUF thật không, (2) ảnh ra có suy
giảm nhìn thấy được không, (3) compile có hội tụ (số lần recompile → 0) hay
recompile mỗi shape.

## 9. 🧪 VAE tiling / slicing — mới thêm, mặc định tắt

`AutoencoderKLFlux2` hỗ trợ cả hai; diffusers để tắt, và ở đây cũng để tắt.

- `vae_tiling`: decode latent theo ô thay vì cả ảnh. Đổi một chút thời gian
  lấy **VRAM đỉnh thấp hơn hẳn** ở bước decode.
- `vae_slicing`: decode từng ảnh một khi batch > 1. Hiện batch luôn = 1 nên
  **không có tác dụng**; để sẵn cho lúc bật batching.

Mặc định tắt vì cấu hình mặc định (transformer GGUF + text encoder NF4,
~11 GiB) đang **dư VRAM chứ không thiếu**. Bật khi: để text encoder ở bf16,
hoặc chạy trên 1024²+, hoặc nhồi nhiều process lên một card, hoặc gặp OOM
lúc decode
([diffusers#13079](https://github.com/huggingface/diffusers/issues/13079)).

Lưu ý `Flux2KleinPipeline` **không có** `enable_vae_tiling()` ở cấp pipeline
(khác nhiều pipeline khác của diffusers) — phải gọi trên `pipeline.vae`.
`loader._apply_vae_memory_options()` đã lo việc đó.

---

## Thứ tự nên làm khi cần nhanh hơn

0. **Kiểm VRAM trước.** Nếu đang OOM thì không phải bài toán tốc độ — xem
   §0. `text_encoder_quantization` là knob đầu tiên cần nhìn.
1. **Đo trước.** `make benchmark` — đọc bảng tỉ trọng giai đoạn. Tối ưu thứ
   không phải nút cổ chai là tốn công vô ích.
2. **Cắt ngắn prompt.** ĐÒN BẨY LỚN NHẤT đã đo: prompt demo dài làm mỗi
   bước denoise đắt thêm **75%** (§0b). Miễn phí về mặt kỹ thuật, nhưng đổi
   ảnh ra nên phải xem bằng mắt.
3. **Hạ `output_area`.** Chi phí denoise tỉ lệ với số token latent, tức với
   diện tích ảnh. 832² thay 1024² là đòn bẩy lớn thứ hai và không tốn gì.
4. **Nếu `denoise` áp đảo và card còn dư VRAM → thử `fp8`** (§8). Đây là
   cách duy nhất VRAM dư đổi được thành tốc độ, và nó mở khoá luôn bước 5.
5. **Bật `compile_transformer`** sau khi xác nhận không recompile liên tục.
   Chỉ có tác dụng ở nhánh `bf16`/`fp8`, KHÔNG có ở `gguf`.
6. Nếu `text` chiếm tỉ trọng lớn → **nâng `embed_cache_size`**.
7. Nếu `decode` > 20% → cân nhắc **TAEF2** (§5).
8. Nếu cần throughput chứ không phải latency → đọc MULTI_PROCESS_WORKERS.md.
