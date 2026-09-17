from dataclasses import dataclass, field


@dataclass(frozen=True)
class RabbitMQConfig:
    host: str
    port: int
    user: str
    password: str
    vhost: str
    exchange: str
    queues_in: dict[str, str]
    queue_out: str
    queue_out_key: str
    # prefetch_count: số message tối đa broker gửi cho mỗi consumer mà chưa
    # ack. Mặc định 0 → service auto-tune theo SLA + avg inference time:
    #   prefetch = clamp(target_sla_seconds / avg_inference_seconds, 1,
    # prefetch_max)
    # Đặt số dương >0 để override (vd cố định 1 cho service GPU rất nặng).
    # Reconnect backoff policy (exponential + jitter):
    # delay = min(reconnect_initial_seconds * 2^attempt, reconnect_max_seconds)
    #         * (1 ± reconnect_jitter)
    # Reset attempt khi connect thành công.
    reconnect_initial_seconds: float = 1.0
    reconnect_max_seconds: float = 60.0
    reconnect_jitter: float = 0.2
    # Broker healthcheck — thread ping broker port mỗi N giây để flip /readyz.
    # 0 = tắt healthcheck (default; service vẫn reconnect, chỉ là /readyz
    # không reflect broker state).
    broker_healthcheck_interval_seconds: float = 15.0
    broker_healthcheck_timeout_seconds: float = 3.0
    prefetch_count: int = 0
    # Auto-tune knobs (chỉ dùng khi prefetch_count == 0):
    #   target_sla_seconds: backlog tối đa 1 consumer có thể "ôm" mà vẫn đảm
    #     bảo message cuối được xử lý trong khoảng thời gian này.
    #   avg_inference_seconds: ước lượng latency mỗi inference. HiDream-O1
    #     full = 50 bước x 2 forward pass (CFG) ở 2048^2. ĐO THẬT rồi điền:
    #     để sai số này là cách nhanh nhất làm backlog phình và message
    #     timeout hàng loạt. Bản Qwen Lightning cũ là ~8s; bản này chậm hơn
    #     một bậc độ lớn.
    #   prefetch_max: trần cứng — 1 container không bao giờ giữ quá ngần này
    #     in-flight. Ảnh giờ rất đắt nên để thấp.
    target_sla_seconds: float = 600.0
    avg_inference_seconds: float = 90.0
    prefetch_max: int = 4
    heartbeat: int = 600
    # Retry / DLQ topology.
    #
    # Chỉ có TTL CỐ ĐỊNH (`retry_initial_delay_ms`), không có backoff mũ.
    # Hai knob multiplier/max_delay từng tồn tại ở đây nhưng không code
    # nào đọc — đã xoá để config không hứa thứ nó không làm.
    retry_exchange_suffix: str = ".retry"
    dlq_suffix: str = ".dlq"
    retry_max_attempts: int = (
        0  # canonical: no-retry, DLQ ngay (tránh loop vô tận)
    )
    retry_initial_delay_ms: int = 5000
    # Topology mode:
    #   "declare" (default): full topology — main + .retry + .dlq exchanges,
    #     each inbound queue + retry sibling (TTL) + DLQ sibling, with
    #     x-dead-letter-exchange args. Failure routing via broker DLX.
    #   "declare_minimal": exchange + N inbound queue + outbound queue + bind.
    #     KHÔNG DLX args, KHÔNG retry/.dlq siblings. Tương thích broker do BE
    #     pre-create queue legacy không có DLX. Failure path: reply qua
    #     queue_out + ack original; audit trên disk.
    #   "passive": queues đã tồn tại, service chỉ verify existence.
    topology_mode: str = "declare"

    def effective_prefetch(self) -> int:
        """
        Trả về prefetch_count thực tế broker sẽ dùng.

        - Nếu prefetch_count > 0: tôn trọng giá trị config (manual override).
        - Nếu prefetch_count == 0: tự tính sao cho 1 consumer giữ đủ message
          để xử lý trong target_sla_seconds dựa trên avg_inference_seconds,
          cap bằng prefetch_max.

        Công thức:
          n = ceil(target_sla_seconds / avg_inference_seconds)
          prefetch = clamp(n, 1, prefetch_max)
        """
        if self.prefetch_count > 0:
            return self.prefetch_count
        avg = max(self.avg_inference_seconds, 0.001)  # tránh div zero
        n = int(-(-self.target_sla_seconds // avg))  # ceil
        return max(1, min(n, self.prefetch_max))


@dataclass(frozen=True)
class StorageBackendConfig:
    """
    Per-destination storage config. Service có thể khai báo nhiều backend
    trong `storage.backends[]` và route theo `storage_index` của inbound
    message.

    `cdn_base_url` là switch giữa 2 cách trả URL cho client:
        - Rỗng ("") → fallback signed URL của bucket (TTL =
          `signed_url_ttl_minutes`). Dùng cho bucket private.
        - Có giá trị → trả URL CDN public dạng
          `{cdn_base_url}/{object_key}`. Dùng cho bucket có Cloud CDN /
          custom domain phía trước, không cần signature.
    """

    credentials_path: str
    bucket: str
    folder: str  # env/tenant root, vd "prd/perm"
    feature: str  # service stable label
    signed_url_ttl_minutes: int = 15
    cdn_base_url: str = ""  # rỗng → signed URL


@dataclass(frozen=True)
class StorageConfig:
    """
    Top-level storage config. `backend` chọn implementation
    (gcs/local). `backends[]` là list destination song song; inbound
    message có `storage_index` chỉ định dùng backend nào (default
    `default_index`).
    """

    backend: str
    backends: tuple[StorageBackendConfig, ...]
    default_index: int = 0


@dataclass(frozen=True)
class ProcessorConfig:
    """
    HiDream-O1-Image (SDNQ 4-bit) inference knobs.

    Map thẳng vào imagegen.config.Settings khi nạp model. Chỉ giữ các field
    service queue thực sự dùng (không có server_name/port của Gradio).
    `device` rỗng → auto-select (cuda:1 nếu >=2 GPU, else cuda:0).

    `width`/`height` là kích thước YÊU CẦU. Model snap về một trong 11 độ
    phân giải cứng (PREDEFINED_RESOLUTIONS), nhỏ nhất 2048x2048 — đặt nhỏ
    hơn KHÔNG cho ảnh nhỏ hơn, chỉ đổi tỉ lệ khung.
    """

    type: str = "hidream_o1"
    device: str = ""
    model_path: str = ""  # rỗng → theo IMG_MODEL_PATH / base.yaml
    model_type: str = "full"  # "full" (50 bước) | "dev" (28 bước)
    num_steps: int = 50
    guidance_scale: float = 5.0  # 0.0 cho model_type=dev (đã chưng cất)
    shift: float = 3.0
    scheduler_name: str = "default"  # "default" | "flow_match" | "flash"
    width: int = 2048
    height: int = 2048
    warmup: bool = True

    # --- Tốc độ (gương của imagegen.config.Settings) ---
    # Xem docs/PERFORMANCE.md cho ý nghĩa và đánh đổi của từng knob.
    attention_mode: str = "auto"   # auto | sdpa | mask | flash
    dequantize: str = "auto"       # auto | bf16 | off
    compile_model: bool = False
    cfg_interval_start: float = 0.0
    cfg_interval_end: float = 1.0
    snap_resolution: bool = True
    warmup_steps: int = 2


@dataclass(frozen=True)
class WorkerConfig:
    num_ai_workers: int = 1
    in_queue_capacity: int = 2
    publish_queue_capacity: int = 5000
    basic_tier_weights: tuple = (5, 3, 2)
    # Idempotency cache size (per-process LRU). Catches duplicate request_id
    # khi BE/App publish lại trong burst. Đặt 0 để tắt idempotency check.
    recent_requests_capacity: int = 1000


@dataclass(frozen=True)
class ObservabilityConfig:
    """Logging behavior. Logs go to stdout (no file)."""

    log_level: str = "INFO"
    log_payload_max_bytes: int = 500  # 0 disables payload preview entirely
    strip_url_query: bool = True


@dataclass(frozen=True)
class AuditConfig:
    """
    7-day per-request audit storage (input/output/error/RMQ files).

    Layout:
        <root_dir>/<service_name>/<Y>/<M>/<D>/{success,fail,error}/<request_id>/[attempt_N/]
        <root_dir>/<service_name>/rmq_errors/<Y>/<M>/<D>/<event>_<ts>.json

    `staging_dir` is the local working directory for downloads + AI output
    while a request is in flight. Files there are deleted after each request.
    """

    service_name: str = "model_gen_img"
    root_dir: str = "data_predict"
    staging_dir: str = "data/staging"
    retention_days: int = 7
    cleanup_interval_hours: int = 6
    enabled: bool = True
    save_input_image: bool = True
    save_output_image: bool = True
    compute_checksum: bool = True


@dataclass(frozen=True)
class Settings:
    service_version: str
    rabbitmq: RabbitMQConfig
    storage: StorageConfig
    processor: ProcessorConfig
    worker: WorkerConfig
    audit: AuditConfig = field(default_factory=AuditConfig)
    observability: ObservabilityConfig = field(
        default_factory=ObservabilityConfig
    )
