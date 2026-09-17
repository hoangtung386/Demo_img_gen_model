"""Load test: bắn message liên tục vào 4 inbound queue của queue service.

Mục tiêu là đo BROKER + đường vào, không đo GPU: script chỉ publish (và tuỳ
chọn drain `queue_out`), nên chạy được từ máy dev không có card.

    # Smoke: 30s, 5 msg/s, tự tạo topology trên vhost trống.
    # Password lấy từ env IMG_RMQ_PASSWORD (hoặc config_setup/base.yaml) —
    # đừng viết secret vào lệnh, nó nằm lại trong shell history.
    export IMG_RMQ_PASSWORD='...'
    python scripts/loadtest_queue.py --host <broker-host> --vhost gen-image \
        --user <username> --declare --rate 5 --duration 30

    # Đẩy tải: 200 msg/s trong 2 phút, 4 connection, tắt publisher confirm
    python scripts/loadtest_queue.py --rate 200 --duration 120 --workers 4
    --no-confirm

    # Đo end-to-end (cần worker đang chạy): đọc luôn queue_out để tính latency
    python scripts/loadtest_queue.py --rate 2 --duration 300 --drain-out

Tỉ lệ message giữa các tier lấy theo `--mix premium:basic_1:basic_2:basic_3`,
mặc định `1:5:3:2` — phần basic khớp `worker.basic_tier_weights` trong
config_setup/base.yaml, tức đúng tỉ lệ WRR mà PriorityRouter dùng để lấy
message ra. Premium để 1 phần (~9%) cho có traffic ưu tiên; đặt `--mix 0:5:3:2`
nếu chỉ muốn test nhánh basic, hoặc `--mix 1:0:0:0` để test riêng premium.

Payload dựng theo đúng hợp đồng ở
imagegen/queue_service/messaging/schemas.py (_REQUIRED_FIELDS +
validate_inbound). Sửa schema bên đó thì phải sửa `build_payload()` ở đây,
nếu không worker sẽ reply BAD_REQUEST cho toàn bộ tải test.

Script cố tình CHỈ phụ thuộc pika + pyyaml (không import imagegen) để
chạy được trong venv nhẹ trên máy bắn tải — package imagegen kéo theo
torch qua queue_service/__init__.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pika
from pika.exceptions import AMQPError, NackError, UnroutableError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config_setup" / "base.yaml"
EXAMPLE_CONFIG = PROJECT_ROOT / "config_setup" / "base.example.yaml"

TIERS = ("premium", "basic_1", "basic_2", "basic_3")

# Prompt ngắn, không dấu — chỉ để message hợp lệ và khác nhau giữa các lần
# bắn. Nội dung không ảnh hưởng tới tải broker.
PROMPTS = (
    "a cozy scandinavian living room, warm afternoon light",
    "portrait of a young woman, soft studio lighting, film grain",
    "a red sports car on a coastal road at sunset",
    "cyberpunk street market at night, neon reflections on wet asphalt",
    "minimalist product shot of a ceramic coffee cup on marble",
    "watercolor illustration of a mountain village in autumn",
)


# ────────────────────────── config ──────────────────────────


def load_rmq_section(path: Path) -> dict:
    """Đọc khối `rabbitmq` từ base.yaml. File thiếu → trả dict rỗng.

    Không dùng queue_service.config.load_config vì loader đó validate cả
    storage/processor (bắt buộc bucket, credentials...) — thừa với một tool
    chỉ cần host + tên queue.
    """
    import yaml

    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    section = data.get("rabbitmq") or {}
    return section if isinstance(section, dict) else {}


def resolve(cli_value, env_name: str, cfg: dict, cfg_key: str, default):
    """Thứ tự ưu tiên: CLI > biến môi trường > base.yaml > default.

    Secret (password) vì thế không cần nằm trong repo: export IMG_RMQ_PASSWORD
    hoặc để trong base.yaml (đã bị .gitignore chặn).
    """
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    if cfg.get(cfg_key) not in (None, ""):
        return cfg[cfg_key]
    return default


# ────────────────────────── phân phối tier ──────────────────────────


def smooth_wrr(weights: dict[str, int]) -> list[str]:
    """Trải weights thành một chu kỳ smooth weighted round-robin.

    Với 1:5:3:2 thì kết quả là chu kỳ 11 phần tử xen kẽ (basic_1, basic_2,
    basic_1, basic_3, ...) chứ không phải 5 basic_1 liền nhau rồi tới 3
    basic_2. Quan trọng với test tải: bắn từng cụm đồng tier sẽ tạo burst giả,
    làm sai cả số liệu backlog lẫn hành vi ưu tiên của consumer.

    Thuật toán smooth WRR của nginx: mỗi vòng cộng weight vào current, chọn
    thằng current lớn nhất rồi trừ đi tổng weight.
    """
    active = {k: w for k, w in weights.items() if w > 0}
    if not active:
        raise ValueError("mix phải có ít nhất một tier với weight > 0")
    total = sum(active.values())
    current = dict.fromkeys(active, 0)
    cycle: list[str] = []
    for _ in range(total):
        for tier, weight in active.items():
            current[tier] += weight
        pick = max(current, key=lambda t: current[t])
        current[pick] -= total
        cycle.append(pick)
    return cycle


def parse_mix(raw: str) -> dict[str, int]:
    parts = raw.split(":")
    if len(parts) != len(TIERS):
        raise argparse.ArgumentTypeError(
            f"--mix cần đúng {len(TIERS)} số theo thứ tự {':'.join(TIERS)}, "
            f"nhận {raw!r}"
        )
    try:
        values = [int(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--mix chỉ nhận số nguyên, nhận {raw!r}"
        ) from None
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("--mix không nhận số âm")
    if sum(values) == 0:
        raise argparse.ArgumentTypeError("--mix phải có tổng > 0")
    return dict(zip(TIERS, values, strict=True))


# ────────────────────────── payload ──────────────────────────


def build_payload(
    msg_id: str, images: list[str], seed: int, num_steps: int
) -> dict:
    """Một inbound message hợp lệ theo validate_inbound().

    `images` rỗng → text-to-image (QwenImagePipeline); có URL → image-edit,
    ảnh CUỐI là ảnh nền. Các field bắt buộc: _id, os, firebase_token, appid,
    country (được phép rỗng), device_id, prompt.
    """
    payload = {
        "_id": msg_id,
        "os": random.choice(("android", "ios")),
        "firebase_token": f"loadtest-token-{uuid.uuid4().hex[:12]}",
        "appid": "com.example.loadtest",
        "country": "VN",
        "device_id": f"loadtest-{uuid.uuid4().hex[:16]}",
        "prompt": random.choice(PROMPTS),
        "seed": seed,
        "num_steps": num_steps,
        "storage_index": 0,
    }
    if images:
        payload["images"] = images
    return payload


# ────────────────────────── thống kê ──────────────────────────


@dataclass
class Stats:
    """Counter dùng chung giữa các publisher thread + thread drain."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    published: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(TIERS, 0)
    )
    failed: int = 0
    unroutable: int = 0
    publish_latencies: list[float] = field(default_factory=list)
    # Bản đồ _id -> thời điểm publish, chỉ giữ khi --drain-out để tính e2e.
    inflight: dict[str, float] = field(default_factory=dict)
    replies: int = 0
    reply_status: dict[int, int] = field(default_factory=dict)
    e2e_latencies: list[float] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    def record_publish(
        self, tier: str, latency: float, msg_id: str, track: bool
    ) -> None:
        with self.lock:
            self.published[tier] += 1
            # Cap mẫu latency: test 30 phút ở 500 msg/s là 900k điểm, giữ hết
            # chỉ để tính percentile là phí RAM vô ích.
            if len(self.publish_latencies) < 200_000:
                self.publish_latencies.append(latency)
            if track:
                self.inflight[msg_id] = time.perf_counter()

    def record_failure(self, unroutable: bool = False) -> None:
        with self.lock:
            self.failed += 1
            if unroutable:
                self.unroutable += 1

    def record_reply(self, msg_id: str, status_code: int) -> None:
        with self.lock:
            self.replies += 1
            self.reply_status[status_code] = (
                self.reply_status.get(status_code, 0) + 1
            )
            sent = self.inflight.pop(msg_id, None)
            if sent is not None:
                self.e2e_latencies.append(time.perf_counter() - sent)

    def total(self) -> int:
        return sum(self.published.values())

    def snapshot(self) -> tuple[int, dict[str, int], int, int]:
        with self.lock:
            return (
                self.total(),
                dict(self.published),
                self.failed,
                self.replies,
            )


def percentiles(samples: list[float]) -> str:
    if not samples:
        return "n/a"
    ordered = sorted(samples)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
        return ordered[idx]

    return (
        f"p50={pct(0.5) * 1000:.1f}ms p95={pct(0.95) * 1000:.1f}ms "
        f"p99={pct(0.99) * 1000:.1f}ms max={ordered[-1] * 1000:.1f}ms "
        f"avg={statistics.fmean(ordered) * 1000:.1f}ms"
    )


# ────────────────────────── broker ──────────────────────────


def connection_params(args, heartbeat: int = 60) -> pika.ConnectionParameters:
    return pika.ConnectionParameters(
        host=args.host,
        port=args.port,
        virtual_host=args.vhost,
        credentials=pika.PlainCredentials(args.user, args.password),
        heartbeat=heartbeat,
        blocked_connection_timeout=60,
    )


def declare_topology(channel, args) -> None:
    """Bản sao tối giản của topology._declare_minimal.

    Chỉ exchange direct + 4 inbound queue + 1 outbound queue + bind, KHÔNG
    DLX args — giống hệt `topology_mode: declare_minimal` của service, nên
    declare bằng tool này rồi service start lên sẽ không dính
    PRECONDITION_FAILED 406 vì lệch argument.
    """
    channel.exchange_declare(
        exchange=args.exchange, exchange_type="direct", durable=True
    )
    channel.queue_declare(queue=args.queue_out, durable=True)
    channel.queue_bind(
        exchange=args.exchange,
        queue=args.queue_out,
        routing_key=args.queue_out_key,
    )
    for queue_name in args.queues.values():
        channel.queue_declare(queue=queue_name, durable=True)
        channel.queue_bind(
            exchange=args.exchange, queue=queue_name, routing_key=queue_name
        )


def purge_queues(conn, args) -> None:
    """Purge từng queue trên MỘT channel riêng.

    queue_purge vào queue không tồn tại trả NOT_FOUND 404 và broker đóng luôn
    channel — dùng chung một channel thì queue thứ hai trở đi sẽ fail theo dây
    chuyền dù nó vẫn tồn tại.
    """
    targets = list(args.queues.values()) + (
        [args.queue_out] if args.purge_out else []
    )
    for queue_name in targets:
        channel = conn.channel()
        try:
            result = channel.queue_purge(queue=queue_name)
            count = getattr(result.method, "message_count", "?")
            print(f"[purge] {queue_name}: {count} message đã xoá")
        except (
            AMQPError
        ) as exc:  # queue chưa tồn tại → bỏ qua, không phải lỗi test
            print(f"[purge] {queue_name}: bỏ qua ({exc.__class__.__name__})")
        finally:
            if channel.is_open:
                channel.close()


# ────────────────────────── worker publish ──────────────────────────


def publish_worker(
    worker_id: int, args, stats: Stats, stop: threading.Event, budget
) -> None:
    """Một thread = một connection.

    pika KHÔNG thread-safe nên không share channel giữa các thread.

    Mỗi worker đi riêng một chu kỳ WRR nhưng lệch pha theo worker_id, để N
    worker cộng lại vẫn ra đúng tỉ lệ --mix mà không cùng lúc bắn vào một tier.
    """
    cycle = smooth_wrr(args.mix)
    pos = worker_id % len(cycle)
    params = connection_params(args)
    conn = pika.BlockingConnection(params)
    channel = conn.channel()
    if not args.no_confirm:
        # Publisher confirm: basic_publish block tới khi broker ack. Chậm hơn
        # nhưng đây mới là con số "broker nhận thật", nên để mặc định bật.
        channel.confirm_delivery()

    interval = 0.0 if args.rate <= 0 else args.workers / args.rate
    next_send = time.perf_counter()
    try:
        while not stop.is_set():
            seq = budget()
            if seq is None:
                break

            tier = cycle[pos % len(cycle)]
            pos += 1
            queue_name = args.queues[tier]
            msg_id = f"lt-{args.run_id}-{seq:08d}"
            images = (
                args.images
                if (args.images and random.random() < args.edit_ratio)
                else []
            )
            body = json.dumps(
                build_payload(msg_id, images, args.seed, args.num_steps)
            ).encode()

            if interval:
                now = time.perf_counter()
                if next_send > now:
                    stop.wait(next_send - now)
                    if stop.is_set():
                        break
                # Nhịp tính theo mốc tuyệt đối: nếu một lần publish bị chậm,
                # các lần sau bù lại thay vì trôi dần khỏi rate mục tiêu.
                next_send = max(next_send + interval, now - interval)

            started = time.perf_counter()
            try:
                channel.basic_publish(
                    exchange=args.publish_exchange,
                    routing_key=queue_name,
                    body=body,
                    properties=pika.BasicProperties(
                        # persistent, giống publisher của service
                        delivery_mode=2,
                        content_type="application/json",
                        message_id=msg_id,
                        timestamp=int(time.time()),
                    ),
                    mandatory=not args.no_confirm,
                )
            except UnroutableError:
                # Không có queue nào bind với routing key này → chạy tiếp cho
                # hết test nhưng đếm riêng, vì đây là lỗi cấu hình chứ không
                # phải quá tải.
                stats.record_failure(unroutable=True)
                continue
            except (NackError, AMQPError, OSError) as exc:
                stats.record_failure()
                if stop.is_set():
                    break
                print(
                    f"[worker-{worker_id}] publish lỗi: {exc!r} → reconnect",
                    file=sys.stderr,
                )
                try:
                    conn.close()
                except Exception:
                    pass
                conn = pika.BlockingConnection(params)
                channel = conn.channel()
                if not args.no_confirm:
                    channel.confirm_delivery()
                continue

            stats.record_publish(
                tier, time.perf_counter() - started, msg_id, args.drain_out
            )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def drain_worker(args, stats: Stats, stop: threading.Event) -> None:
    """Đọc queue_out để đo end-to-end + phổ status_code.

    CẢNH BÁO: consume ở đây là LẤY MẤT message khỏi queue_out. Chỉ bật trên
    môi trường test, đừng chạy khi BE thật đang đọc queue đó.
    """
    params = connection_params(args)
    conn = pika.BlockingConnection(params)
    channel = conn.channel()
    channel.basic_qos(prefetch_count=50)
    try:
        for method, _props, body in channel.consume(
            queue=args.queue_out, inactivity_timeout=0.5, auto_ack=True
        ):
            if stop.is_set():
                break
            if method is None:  # inactivity tick — vòng lại để check stop
                continue
            try:
                data = json.loads(body)
                stats.record_reply(
                    str(data.get("_id", "")), int(data.get("status_code", -1))
                )
            except (ValueError, TypeError):
                stats.record_reply("", -1)
    except Exception as exc:
        if not stop.is_set():
            print(f"[drain] lỗi: {exc!r}", file=sys.stderr)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ────────────────────────── main ──────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bắn tải vào inbound queue của gen-image queue service.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    conn = p.add_argument_group("broker (CLI > env IMG_RMQ_* > base.yaml)")
    conn.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="file config lấy mặc định",
    )
    conn.add_argument("--host")
    conn.add_argument("--port", type=int)
    conn.add_argument("--user")
    conn.add_argument("--password")
    conn.add_argument("--vhost")
    conn.add_argument("--exchange")

    load = p.add_argument_group("tải")
    load.add_argument(
        "--rate", type=float, default=10.0, help="msg/s tổng; 0 = bắn hết sức"
    )
    load.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="giây; 0 = chạy tới khi đủ --count",
    )
    load.add_argument(
        "--count",
        type=int,
        default=0,
        help="tổng số message; 0 = không giới hạn",
    )
    load.add_argument(
        "--workers",
        type=int,
        default=1,
        help="số connection publish song song",
    )
    load.add_argument(
        "--mix",
        type=parse_mix,
        default="1:5:3:2",
        help="tỉ lệ premium:basic_1:basic_2:basic_3",
    )
    load.add_argument(
        "--no-confirm",
        action="store_true",
        help=(
            "tắt publisher confirm (nhanh hơn nhiều, không chắc "
            "broker đã nhận)"
        ),
    )

    payload = p.add_argument_group("payload")
    payload.add_argument(
        "--image-url",
        action="append",
        dest="images",
        default=[],
        help="URL ảnh điều kiện; lặp nhiều lần. Ảnh CUỐI là ảnh nền",
    )
    payload.add_argument(
        "--edit-ratio",
        type=float,
        default=1.0,
        help="tỉ lệ message kèm ảnh (chỉ có tác dụng khi có --image-url)",
    )
    payload.add_argument(
        "--seed", type=int, default=-1, help="-1 = random mỗi lần sinh ảnh"
    )
    payload.add_argument(
        "--num-steps",
        type=int,
        default=0,
        help="0 = dùng default của processor",
    )

    ops = p.add_argument_group("topology / đo đạc")
    ops.add_argument(
        "--declare",
        action="store_true",
        help=(
            "tạo exchange + queue (tương đương topology_mode declare_minimal)"
        ),
    )
    ops.add_argument(
        "--direct-to-queue",
        action="store_true",
        help="publish qua default exchange thay vì exchange của service",
    )
    ops.add_argument(
        "--purge",
        action="store_true",
        help="xoá sạch 4 inbound queue TRƯỚC khi bắn",
    )
    ops.add_argument(
        "--purge-out",
        action="store_true",
        help="purge cả queue_out (đi kèm --purge)",
    )
    ops.add_argument(
        "--drain-out",
        action="store_true",
        help="consume queue_out để đo e2e — LẤY MẤT message của BE",
    )
    ops.add_argument(
        "--drain-grace",
        type=float,
        default=15.0,
        help="giây chờ thêm sau khi ngừng bắn, cho reply cuối về (tối đa 60)",
    )
    ops.add_argument(
        "--report-interval",
        type=float,
        default=5.0,
        help="giây giữa 2 dòng progress",
    )
    ops.add_argument(
        "--dry-run",
        action="store_true",
        help="in kế hoạch + payload mẫu rồi thoát",
    )

    args = p.parse_args(argv)
    if isinstance(args.mix, str):  # default không đi qua type=parse_mix
        args.mix = parse_mix(args.mix)
    return args


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_rmq_section(args.config)
    if not cfg and args.config == DEFAULT_CONFIG:
        # base.yaml là file secret, không có trong repo. Lấy tên queue/exchange
        # từ base.example.yaml để tool vẫn chạy được ngay sau khi clone.
        cfg = load_rmq_section(EXAMPLE_CONFIG)

    args.host = resolve(args.host, "IMG_RMQ_HOST", cfg, "host", "localhost")
    args.port = int(resolve(args.port, "IMG_RMQ_PORT", cfg, "port", 5672))
    args.user = resolve(args.user, "IMG_RMQ_USER", cfg, "user", "guest")
    args.password = resolve(
        args.password, "IMG_RMQ_PASSWORD", cfg, "password", "guest"
    )
    args.vhost = resolve(args.vhost, "IMG_RMQ_VHOST", cfg, "vhost", "/")
    args.exchange = resolve(
        args.exchange,
        "IMG_RMQ_EXCHANGE",
        cfg,
        "exchange",
        "predict-ai-gen-image",
    )

    queues_in = cfg.get("queues_in") or {}
    args.queues = {
        tier: queues_in.get(
            tier, f"gen-image-queue-in-{tier.replace('_', '-tier-')}"
        )
        for tier in TIERS
    }
    args.queue_out = cfg.get("queue_out", "gen-image-queue-out")
    args.queue_out_key = cfg.get("queue_out_key", args.queue_out)
    # Default exchange ("") route thẳng theo tên queue — dùng khi không được
    # phép tạo/đụng vào exchange của service.
    args.publish_exchange = "" if args.direct_to_queue else args.exchange
    args.run_id = uuid.uuid4().hex[:8]

    if args.workers < 1:
        raise SystemExit("--workers phải >= 1")
    if args.duration <= 0 and args.count <= 0:
        raise SystemExit(
            "phải đặt --duration > 0 hoặc --count > 0, nếu không test không "
            "bao giờ dừng"
        )
    if args.images and not 0.0 <= args.edit_ratio <= 1.0:
        raise SystemExit("--edit-ratio phải nằm trong [0, 1]")
    return args


def make_budget(args, stop: threading.Event):
    """Cấp số thứ tự message; trả None khi đã đủ --count."""
    counter = iter(range(1, args.count + 1)) if args.count > 0 else None
    lock = threading.Lock()
    unlimited = [0]

    def budget():
        if counter is None:
            with lock:
                unlimited[0] += 1
                return unlimited[0]
        with lock:
            try:
                return next(counter)
            except StopIteration:
                stop.set()
                return None

    return budget


def main(argv: list[str] | None = None) -> int:
    args = finalize_args(parse_args(argv))
    cycle = smooth_wrr(args.mix)

    payload_label = (
        f"image-edit {len(args.images)} ảnh"
        if args.images
        else "text-to-image"
    )
    plan = (
        f"broker   : "
        f"amqp://{args.user}@{args.host}:{args.port}{'/' + args.vhost}\n"
        f"exchange : {args.publish_exchange or '(default)'} → routing_key = "
        f"tên queue\n"
        "queues   : "
        + ", ".join(f"{t}={q}" for t, q in args.queues.items())
        + "\n"
        "mix      : "
        + ":".join(str(args.mix[t]) for t in TIERS)
        + f"  (chu kỳ {len(cycle)}: {' '.join(cycle)})\n"
        f"tải      : rate={args.rate or 'max'} msg/s, workers={args.workers}, "
        f"duration={args.duration or '∞'}s, count={args.count or '∞'}, "
        f"confirm={'off' if args.no_confirm else 'on'}\n"
        f"payload  : {payload_label}"
        f", seed={args.seed}, num_steps={args.num_steps or 'default'}"
    )
    print(plan)
    if args.dry_run:
        sample = build_payload(
            f"lt-{args.run_id}-00000001",
            args.images,
            args.seed,
            args.num_steps,
        )
        print(
            "\npayload mẫu:\n"
            + json.dumps(sample, ensure_ascii=False, indent=2)
        )
        return 0

    stop = threading.Event()
    stats = Stats()

    def on_signal(_sig, _frm):
        print("\n[main] nhận tín hiệu dừng, đang đóng connection...")
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if args.declare or args.purge:
        conn = pika.BlockingConnection(connection_params(args, heartbeat=30))
        channel = conn.channel()
        if args.declare:
            declare_topology(channel, args)
            print(
                f"[declare] exchange={args.exchange} + {len(args.queues)} "
                f"inbound + {args.queue_out}"
            )
        if args.purge:
            print("[purge] CẢNH BÁO: xoá message đang nằm trong queue")
            purge_queues(conn, args)
        conn.close()

    drain_thread = None
    if args.drain_out:
        print(
            f"[drain] consume {args.queue_out} — message sẽ KHÔNG còn cho BE "
            f"đọc"
        )
        drain_thread = threading.Thread(
            target=drain_worker, args=(args, stats, stop), daemon=True
        )
        drain_thread.start()

    budget = make_budget(args, stop)
    threads = [
        threading.Thread(
            target=publish_worker,
            args=(i, args, stats, stop, budget),
            daemon=True,
        )
        for i in range(args.workers)
    ]
    started = time.perf_counter()
    for t in threads:
        t.start()

    deadline = started + args.duration if args.duration > 0 else float("inf")
    last_report = started
    last_total = 0
    while any(t.is_alive() for t in threads):
        now = time.perf_counter()
        if now >= deadline:
            stop.set()
        if now - last_report >= args.report_interval:
            total, per_tier, failed, replies = stats.snapshot()
            window = now - last_report
            print(
                f"[{now - started:6.1f}s] pub={total:>7} "
                f"({(total - last_total) / window:7.1f}/s) "
                f"fail={failed} | "
                + " ".join(f"{t}={per_tier[t]}" for t in TIERS)
                + (f" | out={replies}" if args.drain_out else "")
            )
            last_report, last_total = now, total
        stop.wait(0.2)

    for t in threads:
        t.join(timeout=10)

    if drain_thread is not None:
        # Cho worker thật kịp trả nốt reply của những message cuối.
        grace = min(args.drain_grace, 60.0)
        if grace > 0:
            print(f"[drain] chờ thêm {grace:.0f}s cho reply cuối...")
            time.sleep(grace)
        stop.set()
        drain_thread.join(timeout=5)

    elapsed = time.perf_counter() - started
    total, per_tier, failed, replies = stats.snapshot()
    print("\n" + "─" * 72)
    print(
        f"Tổng       : {total} message trong {elapsed:.1f}s → "
        f"{total / max(elapsed, 1e-9):.1f} msg/s"
    )
    print(f"Thất bại   : {failed} (unroutable={stats.unroutable})")
    for tier in TIERS:
        share = per_tier[tier] / total * 100 if total else 0.0
        target = args.mix[tier] / sum(args.mix.values()) * 100
        print(
            f"  {tier:<9}: {per_tier[tier]:>7}  {share:5.1f}%  (mix đặt "
            f"{target:5.1f}%)"
        )
    print(f"Publish    : {percentiles(stats.publish_latencies)}")
    if args.drain_out:
        print(f"Reply      : {replies} message trên {args.queue_out}")
        if stats.reply_status:
            print(
                "  status   : "
                + ", ".join(
                    f"{k}×{v}" for k, v in sorted(stats.reply_status.items())
                )
            )
        print(f"  e2e      : {percentiles(stats.e2e_latencies)}")
        print(f"  chưa reply: {len(stats.inflight)}")
    print("─" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
