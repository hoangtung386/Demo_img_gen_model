from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..messaging import FailureKind, InboundMessage, OutboundMessage

__all__ = ["FailureKind", "Pipeline", "PipelineResult"]


@dataclass
class PipelineResult:
    """Pipeline outcome: reply for client + classification for the worker."""

    reply: OutboundMessage
    failure: FailureKind = FailureKind.NONE

    @property
    def succeeded(self) -> bool:
        return self.failure is FailureKind.NONE


class Pipeline(ABC):
    """End-to-end orchestration: download → generate → upload → reply."""

    @abstractmethod
    def execute(
        self,
        msg: InboundMessage,
        worker_id: int = 0,
        *,
        attempt: int = 0,
        rmq_headers: dict[str, Any] | None = None,
    ) -> PipelineResult: ...
