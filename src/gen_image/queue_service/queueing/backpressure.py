import queue
from typing import Any


def put_blocking(q: queue.Queue, item: Any) -> None:
    """
    Put an item into the queue, blocking when full. The queue's maxsize provides
    backpressure: consumers stop consuming from the broker until the worker drains.
    """
    q.put(item, block=True)
