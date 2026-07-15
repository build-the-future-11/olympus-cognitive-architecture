from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from pydantic import Field

from olympus.core.schemas import StrictModel


class SpecialistTask(StrictModel):
    name: str
    payload: str
    dependencies: list[str] = Field(default_factory=list)


class SpecialistResult(StrictModel):
    name: str
    output: str
    confidence: float


class NeuralDivisionReassembly:
    def execute(
        self,
        tasks: list[SpecialistTask],
        specialists: dict[str, Callable[[str], SpecialistResult]],
    ) -> tuple[list[SpecialistResult], SpecialistResult]:
        with ThreadPoolExecutor(max_workers=min(4, len(tasks) or 1)) as pool:
            futures = [pool.submit(specialists[task.name], task.payload) for task in tasks]
            results = [future.result() for future in futures]
        combined_text = " | ".join(f"{result.name}: {result.output}" for result in results)
        consistency = sum(result.confidence for result in results) / max(1, len(results))
        final = SpecialistResult(
            name="reassembly",
            output=combined_text,
            confidence=round(consistency, 4),
        )
        return results, final
