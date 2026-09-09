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
    succeeded: bool = True
    error_type: str | None = None


class TraceRecorder:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(self, name: str, callback: Any, **metadata: Any) -> Any:
        started_at = perf_counter()
        try:
            result = callback()
        except BaseException as error:
            self.events.append(
                TraceEvent(
                    name=name,
                    started_at=started_at,
                    finished_at=perf_counter(),
                    metadata=metadata,
                    succeeded=False,
                    error_type=type(error).__name__,
                )
            )
            raise
        self.events.append(
            TraceEvent(
                name=name,
                started_at=started_at,
                finished_at=perf_counter(),
                metadata=metadata,
            )
        )
        return result
