from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class TraceEvent:
    name: str
    started_at: float
    finished_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, name: str, callback: Any, **metadata: Any) -> Any:
        started_at = perf_counter()
        result = callback()
        finished_at = perf_counter()
        self.events.append(
            TraceEvent(name=name, started_at=started_at, finished_at=finished_at, metadata=metadata)
        )
        return result
