from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from graphlib import CycleError, TopologicalSorter

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


class DependencyAwareDivisionReassembly:
    """Run named specialist callables in dependency order and combine their results."""

    def execute(
        self,
        tasks: list[SpecialistTask],
        specialists: dict[str, Callable[[str], SpecialistResult]],
    ) -> tuple[list[SpecialistResult], SpecialistResult]:
        names = [task.name for task in tasks]
        if len(names) != len(set(names)):
            raise ValueError("specialist task names must be unique")
        unknown_specialists = sorted(set(names) - specialists.keys())
        if unknown_specialists:
            raise ValueError(f"missing specialists: {', '.join(unknown_specialists)}")
        known = set(names)
        missing_dependencies = sorted(
            {dependency for task in tasks for dependency in task.dependencies} - known
        )
        if missing_dependencies:
            raise ValueError(f"missing task dependencies: {', '.join(missing_dependencies)}")

        sorter = TopologicalSorter({task.name: set(task.dependencies) for task in tasks})
        try:
            sorter.prepare()
        except CycleError as error:
            raise ValueError("specialist task dependencies contain a cycle") from error

        tasks_by_name = {task.name: task for task in tasks}
        completed: dict[str, SpecialistResult] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(tasks) or 1)) as pool:
            while sorter.is_active():
                ready = tuple(sorter.get_ready())
                futures = {
                    name: pool.submit(specialists[name], tasks_by_name[name].payload)
                    for name in ready
                }
                for name in ready:
                    completed[name] = futures[name].result()
                    sorter.done(name)
        results = [completed[name] for name in names]
        combined_text = " | ".join(f"{result.name}: {result.output}" for result in results)
        consistency = sum(result.confidence for result in results) / max(1, len(results))
        final = SpecialistResult(
            name="reassembly",
            output=combined_text,
            confidence=round(consistency, 4),
        )
        return results, final


# Compatibility name retained for existing callers. The implementation is a
# dependency-aware task scheduler; it does not claim to contain a neural model.
NeuralDivisionReassembly = DependencyAwareDivisionReassembly
