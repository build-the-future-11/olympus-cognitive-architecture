from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from olympus.demos import (
    run_ambiguous_interpretation_demo,
    run_forge_demo,
    run_representation_demo,
    run_retrodiction_demo,
    run_transform_demo,
)


@dataclass(slots=True)
class EvaluationResult:
    name: str
    passed: bool
    detail: str


class EvaluationHarness:
    def run(self) -> list[EvaluationResult]:
        results = [
            run_ambiguous_interpretation_demo(),
            run_retrodiction_demo(),
            run_transform_demo(),
            run_representation_demo(),
            run_forge_demo(),
        ]
        return [
            EvaluationResult(
                name=name,
                passed=cast(bool, payload["passed"]),
                detail=cast(str, payload["detail"]),
            )
            for demo in results
            for name, payload in demo.items()
        ]
