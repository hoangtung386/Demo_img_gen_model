import os
from pathlib import Path

import yaml

from .schema import (
    AuditConfig,
    ObservabilityConfig,
    ProcessorConfig,
    RabbitMQConfig,
    Settings,
    StorageBackendConfig,
    StorageConfig,
    WorkerConfig,
)


def _build_storage(sto: dict) -> StorageConfig:
    """Parse storage section. Hỗ trợ 2 dạng:

    Dạng mới (multi-backend):
        storage:
          backend: gcs
          default_index: 0
          backends:
            - { credentials_path, bucket, folder, feature, ... }
            - { ... }

    Dạng cũ (single backend, để backward compat):
        storage:
          backend: gcs
          credentials_path: ...
          bucket: ...
          folder: ...
          feature: ...
    Loader tự promote thành backends=[<single>].
    """
    raw_backends = sto.get("backends")
    if raw_backends is None:
        # Legacy single-backend → wrap thành 1 entry
        raw_backends = [
            {
                "credentials_path": sto["credentials_path"],
                "bucket": sto["bucket"],
                "folder": sto["folder"],
                "feature": sto["feature"],
                "signed_url_ttl_minutes": sto.get(
                    "signed_url_ttl_minutes", 15
                ),
                "cdn_base_url": sto.get("cdn_base_url", ""),
            }
        ]
    if not raw_backends:
        raise ValueError("storage.backends must contain at least 1 entry")

    backends = tuple(
        StorageBackendConfig(
            credentials_path=str(b["credentials_path"]),
            bucket=str(b["bucket"]),
            folder=str(b["folder"]),
            feature=str(b["feature"]),
            signed_url_ttl_minutes=int(b.get("signed_url_ttl_minutes", 15)),
            cdn_base_url=str(b.get("cdn_base_url", "")),
        )
        for b in raw_backends
    )
    return StorageConfig(
        backend=str(sto.get("backend", "gcs")),
        backends=backends,
        default_index=int(sto.get("default_index", 0)),
    )


def _project_root() -> Path:
    # queue_service/config/loader.py → parents[4] = project root
    #   [0]=config [1]=queue_service [2]=imagegen [3]=src [4]=root
    return Path(__file__).resolve().parents[4]


# Env override cho khối `rabbitmq`. Mục đích chính là để secret (password)
# không bắt buộc phải nằm trong file: docker-compose/k8s inject qua env, file
# yaml chỉ giữ phần không nhạy cảm. Cũng là cách trỏ smoke test sang broker
# khác mà không phải sửa base.yaml của production.
_RMQ_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "IMG_RMQ_HOST": ("host", str),
    "IMG_RMQ_PORT": ("port", int),
    "IMG_RMQ_USER": ("user", str),
    "IMG_RMQ_PASSWORD": ("password", str),
    "IMG_RMQ_VHOST": ("vhost", str),
    "IMG_RMQ_EXCHANGE": ("exchange", str),
}


def _apply_env_overrides(cfg: dict) -> None:
    """Ghi đè khối rabbitmq bằng IMG_RMQ_* nếu biến có giá trị."""
    rmq = cfg.setdefault("rabbitmq", {})
    for env_name, (key, caster) in _RMQ_ENV_OVERRIDES.items():
        raw = os.environ.get(env_name, "").strip()
        if raw:
            rmq[key] = caster(raw)


def load_config(configs_dir: Path | None = None) -> Settings:
    """Load typed Settings từ `config_setup/base.yaml`.

    Service dùng 1 file duy nhất `base.yaml` chia 2 phần:
      PHẦN 1 (BE/DevOps): rabbitmq + storage — chung mọi AI service.
      PHẦN 2 (AI internal): processor + worker + observability + audit
        — đặc trưng từng service, BE để mặc định.

    Hai lối đi ra khỏi file cố định đó:
      • `IMG_QUEUE_CONFIG` — đường dẫn tới file yaml khác (smoke test dùng
        config_setup/smoke.yaml để chạy echo processor + storage local).
      • `IMG_RMQ_*` — ghi đè từng field của khối rabbitmq, để password không
        phải nằm trong file.
    """
    override = os.environ.get("IMG_QUEUE_CONFIG", "").strip()
    if override:
        base_path = Path(override)
    else:
        configs_dir = configs_dir or _project_root() / "config_setup"
        base_path = configs_dir / "base.yaml"

    if not base_path.is_file():
        # Fail rõ ràng ngay ở dòng đầu log: thiếu file này là lỗi deploy hay
        # gặp nhất (base.yaml bị .gitignore nên KHÔNG có sẵn sau khi clone),
        # và nếu chỉ để FileNotFoundError trần thì trên container nó hiện ra
        # như một traceback vô nghĩa rồi service chết im.
        raise FileNotFoundError(
            f"Không tìm thấy config queue service: {base_path}\n"
            "  • Trên host: cp config_setup/base.example.yaml "
            "config_setup/base.yaml "
            "rồi điền rabbitmq + storage.\n"
            "  • Trong Docker: compose mount "
            "./config_setup:/app/config_setup:ro — "
            "file phải nằm trên HOST trước khi `docker compose up`.\n"
            "  • Smoke test: IMG_QUEUE_CONFIG=/app/config_setup/smoke.yaml"
        )

    with open(base_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    _apply_env_overrides(cfg)
    _require_broker(cfg, base_path)
    return _build_settings(cfg)


def _require_broker(cfg: dict, source: Path) -> None:
    """Chặn sớm cấu hình broker rỗng.

    Không có bước này, `host=""` đi thẳng vào pika và service rơi vào vòng
    reconnect vô tận với log mơ hồ — nhìn từ ngoài giống hệt "container chạy
    bình thường nhưng queue_out không có gì".
    """
    rmq = cfg.get("rabbitmq") or {}
    missing = [
        k
        for k in ("host", "user", "password", "vhost")
        if not str(rmq.get(k, "")).strip()
    ]
    if missing:
        raise ValueError(
            f"Config broker thiếu {missing} (đọc từ {source}).\n"
            "  Điền vào file, hoặc export IMG_RMQ_HOST / IMG_RMQ_USER / "
            "IMG_RMQ_PASSWORD / IMG_RMQ_VHOST trước khi chạy."
        )


def _build_settings(cfg: dict) -> Settings:
    rmq = cfg["rabbitmq"]
    sto = cfg["storage"]
    proc = cfg.get("processor", {})
    wrk = cfg.get("worker", {})
    aud = cfg.get("audit", {})
    obs = cfg.get("observability", {})

    return Settings(
        service_version=cfg.get("service_version", "v0.0.0"),
        rabbitmq=RabbitMQConfig(
            host=rmq["host"],
            port=int(rmq["port"]),
            user=rmq["user"],
            password=rmq["password"],
            vhost=rmq["vhost"],
            exchange=rmq["exchange"],
            queues_in=dict(rmq["queues_in"]),
            queue_out=rmq["queue_out"],
            queue_out_key=rmq["queue_out_key"],
            reconnect_initial_seconds=float(
                rmq.get("reconnect_initial_seconds", 1.0)
            ),
            reconnect_max_seconds=float(
                rmq.get("reconnect_max_seconds", 60.0)
            ),
            reconnect_jitter=float(rmq.get("reconnect_jitter", 0.2)),
            broker_healthcheck_interval_seconds=float(
                rmq.get("broker_healthcheck_interval_seconds", 15.0)
            ),
            broker_healthcheck_timeout_seconds=float(
                rmq.get("broker_healthcheck_timeout_seconds", 3.0)
            ),
            prefetch_count=int(rmq.get("prefetch_count", 0)),
            target_sla_seconds=float(rmq.get("target_sla_seconds", 600.0)),
            avg_inference_seconds=float(
                rmq.get("avg_inference_seconds", 90.0)
            ),
            prefetch_max=int(rmq.get("prefetch_max", 4)),
            heartbeat=int(rmq.get("heartbeat", 600)),
            retry_exchange_suffix=str(
                rmq.get("retry_exchange_suffix", ".retry")
            ),
            dlq_suffix=str(rmq.get("dlq_suffix", ".dlq")),
            retry_max_attempts=int(rmq.get("retry_max_attempts", 0)),
            retry_initial_delay_ms=int(
                rmq.get("retry_initial_delay_ms", 5000)
            ),
            topology_mode=str(rmq.get("topology_mode", "declare")),
        ),
        storage=_build_storage(sto),
        processor=ProcessorConfig(
            type=str(proc.get("type", "hidream_o1")),
            device=str(proc.get("device", "") or ""),
            model_path=str(proc.get("model_path", "") or ""),
            model_type=str(proc.get("model_type", "full")).lower(),
            num_steps=int(proc.get("num_steps", 50)),
            guidance_scale=float(proc.get("guidance_scale", 5.0)),
            shift=float(proc.get("shift", 3.0)),
            scheduler_name=str(proc.get("scheduler_name", "default")).lower(),
            width=int(proc.get("width", 2048)),
            height=int(proc.get("height", 2048)),
            warmup=bool(proc.get("warmup", True)),
            attention_mode=str(
                proc.get("attention_mode", "auto") or "auto"
            ).lower(),
            dequantize=str(proc.get("dequantize", "auto") or "auto").lower(),
            compile_model=bool(proc.get("compile_model", False)),
            cfg_interval_start=float(proc.get("cfg_interval_start", 0.0)),
            cfg_interval_end=float(proc.get("cfg_interval_end", 1.0)),
            snap_resolution=bool(proc.get("snap_resolution", True)),
            warmup_steps=int(proc.get("warmup_steps", 2)),
        ),
        worker=WorkerConfig(
            num_ai_workers=int(wrk.get("num_ai_workers", 1)),
            in_queue_capacity=int(wrk.get("in_queue_capacity", 2)),
            publish_queue_capacity=int(
                wrk.get("publish_queue_capacity", 5000)
            ),
            basic_tier_weights=tuple(wrk.get("basic_tier_weights", [5, 3, 2])),
            recent_requests_capacity=int(
                wrk.get("recent_requests_capacity", 1000)
            ),
        ),
        observability=ObservabilityConfig(
            log_level=str(obs.get("log_level", "INFO")),
            log_payload_max_bytes=int(obs.get("log_payload_max_bytes", 500)),
            strip_url_query=bool(obs.get("strip_url_query", True)),
        ),
        audit=AuditConfig(
            service_name=str(aud.get("service_name", "model_gen_img")),
            root_dir=str(aud.get("root_dir", "data_predict")),
            staging_dir=str(aud.get("staging_dir", "data/staging")),
            retention_days=int(aud.get("retention_days", 7)),
            cleanup_interval_hours=int(aud.get("cleanup_interval_hours", 6)),
            enabled=bool(aud.get("enabled", True)),
            save_input_image=bool(aud.get("save_input_image", True)),
            save_output_image=bool(aud.get("save_output_image", True)),
            compute_checksum=bool(aud.get("compute_checksum", True)),
        ),
    )
