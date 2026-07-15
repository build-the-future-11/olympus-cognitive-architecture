from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class RetrodictedState:
    candidate_state: str
    compatibility: float
    invariant_passed: bool
    forward_match: bool


class RetrodictiveIdentityNetwork:
    def infer(
        self,
        current_state: str,
        generators: list[Callable[[str], str]],
        forward_simulator: Callable[[str], str],
        invariant_checker: Callable[[str], bool],
    ) -> list[RetrodictedState]:
        results: list[RetrodictedState] = []
        for generator in generators:
            candidate = generator(current_state)
            invariant_passed = invariant_checker(candidate)
            forward_match = forward_simulator(candidate) == current_state
            compatibility = 0.3
            compatibility += 0.4 if invariant_passed else 0.0
            compatibility += 0.3 if forward_match else 0.0
            results.append(
                RetrodictedState(
                    candidate_state=candidate,
                    compatibility=round(compatibility, 4),
                    invariant_passed=invariant_passed,
                    forward_match=forward_match,
                )
            )
        return sorted(results, key=lambda item: item.compatibility, reverse=True)


def code_bug_generators() -> list[Callable[[str], str]]:
    return [
        lambda code: code.replace("return total / count", "return total / (count - 1)"),
        lambda code: code.replace("return total / count", "return total / count if count else 0"),
        lambda code: code.replace("total += value", "total =+ value"),
    ]
