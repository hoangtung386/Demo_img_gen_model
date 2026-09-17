# Tốc độ sinh ảnh — cái gì đắt, và đòn bẩy nào có thật

Tài liệu này giải thích **chi phí của một tấm ảnh nằm ở đâu** trong
HiDream-O1-Image, rồi liệt kê từng đòn bẩy đã được cài vào code: nó sửa cái
gì, nó đổi lấy cái gì, và bật/tắt bằng knob nào.

> Các bảng số trong §1 là **đo thật, trên RTX 3080** — không phải trên L4 và
> không phải trên card của bạn. Chúng cho biết chi phí phân bố THEO TỈ LỆ NÀO
> giữa các phần, chứ không dự đoán được thời gian trên phần cứng khác. Chạy
> `python scripts/benchmark.py --compare` trên đúng GPU của bạn để có số
> thật, và `--profile` khi tổng không khớp với những gì §1 giải thích được.

---

## 1. Chi phí nằm ở đâu

Model là Qwen3-VL 8B đã sửa thành **pixel-level unified transformer**: 36
layer, hidden 4096, 32 head / 8 KV-head, không VAE. Ảnh đi thẳng vào chuỗi
token dưới dạng patch 32×32.

| Độ phân giải | Token ảnh | Chuỗi (kèm text) |
| :-- | --: | --: |
| 2048×2048 | 4096 | ~4150 |
| 1440×2560 | 3600 | ~3650 |
| 1024×1024 | 1024 | ~1080 |

Một lượt sinh ảnh mặc định = **50 bước × 2 forward pass (CFG) = 100 forward
pass**. Đây là con số phải nhớ: `num_steps` là thứ hiện trên UI,
`Recipe.forward_passes` mới là thứ GPU phải trả.

Trong mỗi forward pass, thời gian chia vào ba chỗ:

| Chỗ | Vì sao đắt |
| :-- | :-- |
| **Matmul của MLP/projection** | Phần "công việc thật", và là phần **lớn nhất** — xem bảng đo bên dưới |
| **Giải nén trọng số SDNQ** | Mỗi lớp Linear giải nén **toàn bộ** ma trận của nó rồi mới gọi `F.linear`. ~8B tham số × 100 lần một ảnh. Thuần băng thông bộ nhớ, không sinh thông tin gì mới |
| **Attention** | Mask 4D dày đặc chặn backend flash của SDPA (xem §2.1). O(S²) nên nó phình theo bình phương độ phân giải — nhỏ ở seq ngắn, đáng kể ở 2048² và ở đường editing |

### Số đo, không phải ước lượng

Đo trên **RTX 3080** (Ampere sm_86, 760 GB/s) với đúng kích thước của model
(hidden 4096, intermediate 12288, 32/8 head, head_dim 128):

**Bảy phép Linear của một decoder layer, seq 1810:**

| | ms/layer | ×72 (36 layer × 2 pass) |
| :-- | --: | --: |
| bf16 thuần (`dequantize: bf16`) | 15.11 | 1.09 s/bước |
| SDNQ uint4, scale fp32 (nguyên bản) | 20.54 | 1.48 s/bước |
| SDNQ uint4, scale bf16 (mặc định mới) | 20.02 | 1.44 s/bước |

**Attention một layer:**

| seq | mask 4D | SDPA hai lượt | |
| --: | --: | --: | --: |
| 1810 | 2.33 ms | 1.20 ms | 1.94× |
| 3600 | 8.18 ms | 4.04 ms | 2.03× |
| 4150 | 9.97 ms | 5.25 ms | 1.90× |

**Đọc bảng cho đúng.** Phần Linear ăn 1.44 s/bước, attention ăn 0.17 s/bước
ở seq 1810 — tức attention chỉ là **~10%** tổng. Gỡ mask 4D cho 1.9× **trên
phần đó**, nên đóng góp vào tổng chỉ ~5%. Ở seq 3600 (đường editing, xem
dưới) attention lên ~17% và đóng góp thành ~9%.

Đòn bẩy lớn hơn là giải nén: 20.02 → 15.11 ms là **1.33×** trên phần chiếm
90% thời gian.

> ⚠️ **Hai cảnh báo về bảng trên.**
>
> 1. Số này đo trên 3080, **không phải L4**. Tỉ lệ sẽ KHÁC: 3080 có 760
>    GB/s, L4 chỉ có 300 GB/s. Giải nén là việc bị chặn bởi băng thông còn
>    matmul bị chặn bởi tính toán, nên trên L4 phần giải nén đắt hơn tương
>    đối — `dequantize: bf16` đáng giá **hơn** ở đó, không phải ít hơn.
> 2. Máy đo không có C compiler nên `torch.compile` của SDNQ không chạy được
>    và nó rơi về kernel eager. Trong image runtime có compiler thì SDNQ fuse
>    được phép giải nén, và khoảng cách 1.33× sẽ **thu hẹp**.
>
> Kết luận: chạy `--compare` trên chính card của bạn. Bảng này để bạn biết
> phải nhìn vào đâu, không phải để trích dẫn.

### Khi tổng thời gian không khớp

Cộng bảng trên lại cho seq 1810 ra ~1.6 s/bước trên 3080. Nếu card của bạn
báo một con số cách xa dự đoán đó nhiều hơn mức chênh lệch phần cứng giải
thích được, thì có thứ khác đang ăn thời gian — đừng đoán, hỏi thẳng GPU:

```bash
python scripts/benchmark.py --profile
```

In ra 15 kernel CUDA tốn nhiều thời gian nhất. `*gemm*` / `*cutlass*` là
tính toán thật; `*elementwise*` / `*copy*` / `*cat*` chiếm nhiều nghĩa là
đang trả tiền cho giải nén trọng số hoặc cho việc dựng/expand mask.

**Một chỗ hay bị bỏ sót:** đường **editing** (có ảnh tham chiếu) nối ref
patch vào chuỗi — `vinputs = cat([z, ref_patches])`. Với đúng 1 ảnh,
`max_size = max(h, w)`, tức ref patch nhiều gần bằng ảnh đích và **chuỗi dài
gấp đôi** so với cùng kích thước ở t2i. Attention thì O(S²). Một request
editing "1278×1417" đắt hơn nhiều so với vẻ ngoài của nó — và `ref_max_size`
(§2.5) là knob duy nhất chạm được vào nó.

---

## 2. Các đòn bẩy

Xếp theo **mức độ chắc chắn**, không theo mức tiết kiệm: §2.1 và §2.2 không
đổi ảnh ra nên bật được ngay; từ §2.3 trở đi đều là đánh đổi chất lượng.
Theo số đo ở §1 thì §2.2 (giải nén) đóng góp nhiều hơn §2.1 (attention) —
nhưng tỉ lệ đó phụ thuộc băng thông của card và độ dài chuỗi, nên hãy đo
thay vì tin thứ tự ở đây.

### 2.1 Đường attention — `attention_mode` (mặc định `auto`, ĐÃ BẬT)

Đường gốc của upstream dựng một mask `[batch, 1, seq, seq]` bằng float rồi
đưa vào `scaled_dot_product_attention`. Vấn đề không phải kích thước mask
(~34MB) mà là: **một `attn_mask` tường minh cấm PyTorch chọn backend
flash.** SDPA rơi xuống mem-efficient, nơi mask phải expand ra
`[batch, heads, seq, seq]` — ở 2048² là ~1.1GB — và copy contiguous **lại từ
đầu ở mỗi layer, mỗi bước**: 36 layer × 100 forward pass = 3600 lần.

Nhưng mask đó có cấu trúc. Token AR (text) là một **tiền tố liên tục**,
causal với nhau; mọi token gen (tms + ảnh đích + ref patch) nằm sau và nhìn
thấy tất cả. Vậy nó tách được thành hai lời gọi SDPA **không mask**:

```python
out_ar  = sdpa(q[:n_ar], k[:n_ar], v[:n_ar], is_causal=True)   # ~40 token
out_gen = sdpa(q[n_ar:], k,        v,        is_causal=False)  # phần còn lại
```

Kết quả toán học **giống hệt** mask cũ (`tests/test_speed_path.py` so trực
tiếp hai bên ở fp32), nhưng cả hai lời gọi đều vào được backend flash, và
không còn tensor `[B, heads, S, S]` nào được materialize.

Đây là cùng chiến lược mà `_run_decoder_flash` của upstream dùng — chỉ khác
là **không cần wheel `flash-attn`**, nên nó chạy được trên image runtime
hiện tại thay vì chỉ trên máy đã build được flash-attn khớp torch + CUDA.

| Giá trị | Nghĩa |
| :-- | :-- |
| `auto` / `sdpa` | Tách hai lượt SDPA. **Mặc định.** |
| `mask` | Đường 4D-mask gốc. Chỉ dùng để đối chiếu numerics |
| `flash` | Wheel `flash-attn` (phải tự cài). Rơi về `sdpa` nếu import không được |

### 2.2 Giải nén trọng số một lần — `dequantize` (mặc định `auto`)

SDNQ là quantizer *dequant-on-the-fly*: `quantized_linear_forward()` gọi
`F.linear(input, sdnq_dequantizer(weight, scale, zero_point))` — nghĩa là
**toàn bộ** ma trận trọng số được giải nén lại ở **mỗi** forward pass. Một
tấm ảnh = 100 lần giải nén lại cùng một bộ trọng số.

Trên card đủ VRAM, giải nén **một lần lúc nạp** xoá sạch chi phí này:

| `dequantize` | VRAM | Chi phí giải nén mỗi forward pass |
| :-- | :-- | :-- |
| `bf16` | ~18 GiB | không còn |
| `off` | ~11 GiB | đầy đủ |
| `auto` | tự đo lúc nạp: bật `bf16` nếu còn đủ chỗ (cần thêm ~7 GiB + 3 GiB biên) | |

**L4 24GB → `auto` sẽ chọn `bf16`.** Card < 20GB sẽ tự giữ dạng nén.

Khi **không** giải nén được, loader vẫn hạ `scale`/`zero_point` từ fp32
xuống bf16: checkpoint lưu `dequantize_fp32: true`, mà dtype của scale quyết
định dtype của cả phép giải nén — tức là mỗi forward pass ghi ra một bản
fp32 của toàn bộ trọng số rồi mới ép về bf16, cho một độ chính xác vô nghĩa
trên trọng số 4-bit.

**Nhưng đừng kỳ vọng nhiều:** đo được 20.54 → 20.02 ms/layer, tức **2.5%**.
Lý thuyết nói phải hơn thế (cắt đôi lưu lượng của phần giải nén), nên hoặc
kernel eager của SDNQ không bị chặn bởi băng thông như dự đoán, hoặc phần
này vốn đã nhỏ so với matmul. Nó miễn phí nên cứ bật, nhưng nó **không thay
thế** được `dequantize: bf16`. Đặt `IMG_SDNQ_FP32_SCALES=true` để quay lại
fp32 nếu cần đối chiếu.

> **Vì sao không bật `use_quantized_matmul` của SDNQ (int8, 2× throughput
> trên L4)?** Vì layout trọng số lưu trên đĩa phụ thuộc cờ đó: với
> `use_quantized_matmul=True`, `dequantize_asymmetric()` lưu ma trận đã
> transpose. Checkpoint của WaveCut được quantize với cờ **tắt**, nên không
> lật được sau khi nạp — phải quantize lại từ bản gốc. Đó là một việc riêng,
> không phải một knob.

### 2.3 Khoảng CFG — `cfg_interval_start` / `cfg_interval_end` (mặc định tắt)

CFG là **đúng 50%** chi phí: mỗi bước chạy hai forward pass (cond + uncond).
[Kynkäänniemi et al. 2024](https://arxiv.org/abs/2404.07724) chỉ ra guidance
gần như vô dụng ở hai đầu quỹ đạo khuếch tán — đầu thì ảnh còn là nhiễu,
cuối thì chỉ còn tinh chỉnh tần số cao.

Ngoài khoảng `[start, end]` (chuẩn hoá theo chỉ số bước), mỗi bước chỉ chạy
nhánh cond → **một** forward pass.

| Cấu hình | Forward pass (50 bước) | Nhanh hơn |
| :-- | --: | --: |
| `[0.0, 1.0]` (mặc định) | 100 | — |
| `[0.0, 0.8]` | 91 | ~9% |
| `[0.0, 0.6]` | 81 | ~19% |
| `[0.1, 0.7]` | 81 | ~19% |

Đây là knob **đổi chất lượng lấy tốc độ** nên mặc định tắt. Thu hẹp dần và
so ảnh bằng mắt trước khi đẩy lên PRD.

### 2.4 Số bước — `num_steps`

Scheduler mặc định là `FlowUniPCMultistepScheduler` — một solver **bậc cao**.
UniPC hội tụ ở số bước thấp hơn hẳn Euler; 50 bước là giá trị an toàn của
upstream, không phải ngưỡng cần thiết. Hạ xuống 28–35 bước thường gần như
không phân biệt được, và nó **tuyến tính**: 50 → 30 bước là nhanh hơn 1.67×.

Đây là đòn bẩy rẻ nhất để thử trước tiên, và không cần sửa code — chỉ đổi
`num_steps` trong `base.yaml`.

### 2.5 Độ phân giải — `snap_resolution` (mặc định `true`)

Model snap mọi yêu cầu về một trong 11 độ phân giải cứng, nhỏ nhất
**2048×2048**. `snap_resolution: false` bỏ chặn đó và sinh đúng kích thước
yêu cầu (làm tròn xuống bội số 32).

Chi phí tỉ lệ với số token cho phần linear, và **bình phương** số token cho
phần attention:

| Kích thước | Token | Chi phí tương đối |
| :-- | --: | --: |
| 2048×2048 | 4096 | 1× |
| 1408×1408 | 1936 | ~0.45× |
| 1024×1024 | 1024 | ~0.22× |

> ⚠️ **Ngoài phân bố huấn luyện.** Model chỉ được train ở 11 độ phân giải
> đó. Chất lượng tụt rõ, không chỉ tụt độ nét. Dùng cho bản nháp / preview,
> đừng để bật ở PRD.

Với đường **editing**, knob này còn hạ luôn trần của ảnh tham chiếu
(`ref_max_size`, upstream ghim cứng 2048) — đó là cách duy nhất hạ chi phí
đường editing, vì kích thước ảnh ra được suy từ ảnh vào.

### 2.6 `torch.compile` — `compile_model` (mặc định tắt)

Sau khi trọng số đã ở bf16 (§2.2), Inductor có đất để fuse. Nhưng nó biên
dịch lại cho **mỗi hình dạng đầu vào mới**, mà độ phân giải thì đổi theo tỉ
lệ ảnh người dùng (11 giá trị). Chỉ có lãi cho worker chạy dài ở **một**
kích thước cố định. Lượt đầu ở mỗi độ phân giải tốn vài phút.

### 2.7 Warm-up — `warmup_steps` (mặc định 2)

Không phải tối ưu inference mà là tối ưu **thời gian khởi động**. Bản trước
warm-up bằng một lượt sinh ảnh đầy đủ 50 bước ở 2048² — hàng phút, chỉ để
vứt ảnh đi. Chi phí cần trả trước (CUDA context, allocator pool, dequant lần
đầu, autotune) là chi phí **một lần**, không nhân theo số bước, nên 2 bước
trả trước đúng những gì 50 bước trả trước.

Độ phân giải thì vẫn giữ đúng của config: một phần chi phí lần đầu phụ thuộc
hình dạng đầu vào.

---

## 3. Bắt đầu từ đâu

**Bước 1 — đo đường cũ và đường mới cạnh nhau:**

```bash
python scripts/benchmark.py --compare
```

In ra bảng 4 dòng (`off`/`bf16` × `mask`/`sdpa`). Dòng `off` + `mask` là
hành vi trước khi tối ưu.

**Bước 2 — mặc định mới đã đủ chưa.** `attention_mode: auto` +
`dequantize: auto` không đổi ảnh ra (ngoài sai số dấu phẩy động) nên không
cần đánh giá chất lượng. Nếu đủ nhanh thì dừng ở đây.

**Bước 3 — nếu vẫn cần nhanh hơn**, theo thứ tự "rẻ về chất lượng" trước:

1. `num_steps: 30` (§2.4) — tuyến tính, dễ đánh giá nhất
2. `cfg_interval_end: 0.7` (§2.3) — thêm ~19%
3. `snap_resolution: false` + `width`/`height` nhỏ hơn (§2.5) — nhanh nhất,
   nhưng đây là bước đánh đổi chất lượng thật sự

**Bước 4 — cập nhật SLA của queue.** Đây là hệ quả dễ bỏ sót nhất:

```yaml
rabbitmq:
  avg_inference_seconds: 90.0   # ← số benchmark in ra
```

Để sai giá trị này là cách nhanh nhất làm backlog queue phình ra. Xem
[QUEUE_SERVICE.md](QUEUE_SERVICE.md).

---

## 4. Những gì KHÔNG phải đòn bẩy

| | Vì sao không |
| :-- | :-- |
| `cudnn.benchmark = True` | Autotune theo **từng** shape rồi mới cache. Shape đổi giữa các request nên chi phí autotune không bao giờ khấu hao hết |
| Gộp cond + uncond thành batch 2 | Hai mẫu có độ dài text khác nhau → phải pad → phải có mask trở lại, tức là huỷ đúng cái §2.1 vừa gỡ bỏ |
| Cache prompt-embeds | Prompt và ảnh được tokenize chung một chuỗi mỗi lượt; không có API nào trả embeds tái dùng được |
| Bản `dev` 28 bước | Nhanh hơn ~3.5× nhưng WaveCut chỉ lượng tử hoá bản `full`. Muốn `dev` thì phải tự quant hoặc chạy BF16 |
| FP8 trên L4 | L4 (sm_89) có tensor core FP8, nhưng dùng được nó nghĩa là quantize lại model — một việc riêng, không phải một knob |

---

## 5. Bảng knob

Mọi knob đọc được từ **cả ba** nguồn: `config_setup/base.yaml` khối `app:`
(cho Gradio) và khối `processor:` (cho queue worker), hoặc biến môi trường.
Env thắng YAML.

| Knob | Env | Mặc định | §    |
| :-- | :-- | :-- | :-- |
| `attention_mode` | `IMG_ATTENTION_MODE` | `auto` | 2.1 |
| `dequantize` | `IMG_DEQUANTIZE` | `auto` | 2.2 |
| — | `IMG_SDNQ_FP32_SCALES` | `false` | 2.2 |
| `cfg_interval_start` | `IMG_CFG_INTERVAL_START` | `0.0` | 2.3 |
| `cfg_interval_end` | `IMG_CFG_INTERVAL_END` | `1.0` | 2.3 |
| `num_steps` | `IMG_NUM_STEPS` | `50` | 2.4 |
| `snap_resolution` | `IMG_SNAP_RESOLUTION` | `true` | 2.5 |
| `compile_model` | `IMG_COMPILE` | `false` | 2.6 |
| `warmup_steps` | `IMG_WARMUP_STEPS` | `2` | 2.7 |

Thêm `--profile` cho benchmark khi cần biết thời gian đi vào kernel nào.

Knob nào đổi ảnh ra thì mặc định của nó là **hành vi cũ** —
`tests/test_speed_path.py::test_default_settings_do_not_change_behaviour`
giữ điều đó.
