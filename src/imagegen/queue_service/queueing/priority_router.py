import queue
from dataclasses import dataclass, field

PREMIUM_LABEL = "premium"


def basic_label(idx: int) -> str:
    return f"basic_{idx + 1}"


@dataclass
class TieredQueues:
    """
    Priority queues:
      - premium: drained first, exhaustively
      - basic: weighted round-robin theo weights
    """

    premium: queue.Queue
    basic: list[queue.Queue]
    weights: tuple[int, ...]
    basic_labels: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.basic) != len(self.weights):
            raise ValueError("basic queues count != weights count")
        object.__setattr__(
            self,
            "basic_labels",
            tuple(basic_label(i) for i in range(len(self.basic))),
        )

    def sizes(self) -> dict[str, int]:
        out = {PREMIUM_LABEL: self.premium.qsize()}
        for label, q in zip(self.basic_labels, self.basic, strict=True):
            out[label] = q.qsize()
        return out


class PriorityRouter:
    def __init__(self, queues: TieredQueues):
        self.queues = queues
        # Pre-compute zip target so the hot loop avoids re-allocating tuples.
        self._basic_plan = tuple(
            zip(queues.basic_labels, queues.basic, queues.weights, strict=True)
        )

    def get_next_batch(self) -> list[tuple[str, object]]:
        """
        Trả về list (label, envelope) để worker xử lý tuần tự.
        - Nếu premium có item → chỉ lấy 1 từ premium.
        - Ngược lại → lấy theo weighted round-robin từ basic.
        """
        try:
            return [(PREMIUM_LABEL, self.queues.premium.get_nowait())]
        except queue.Empty:
            pass

        batch: list[tuple[str, object]] = []
        for label, q, weight in self._basic_plan:
            for _ in range(weight):
                try:
                    batch.append((label, q.get_nowait()))
                except queue.Empty:
                    break
        return batch
