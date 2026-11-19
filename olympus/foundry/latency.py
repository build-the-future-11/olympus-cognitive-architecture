"""Repeated, interleaved CPU latency measurements for comparable workloads."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PairedLatency:
    warmup_rounds: int
    measured_rounds: int
    reference_ms: tuple[float, ...]
    candidate_ms: tuple[float, ...]
    first_in_round: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "reference_median_ms": statistics.median(self.reference_ms),
            "candidate_median_ms": statistics.median(self.candidate_ms),
            "reference_p90_ms": _percentile(self.reference_ms, 0.9),
            "candidate_p90_ms": _percentile(self.candidate_ms, 0.9),
            "timing_clock": "perf_counter_ns",
            "order_policy": "alternate_reference_first_and_candidate_first",
            "warmup_included_in_samples": False,
        }


def _percentile(values: tuple[float, ...], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def measure_paired_latency(
    reference: Callable[[], object],
    candidate: Callable[[], object],
    *,
    warmup_rounds: int = 2,
    measured_rounds: int = 7,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> PairedLatency:
    """Measure the same synchronous workload on each implementation.

    Callers are responsible for equal input shapes, thread settings, inference
    mode, and any needed device synchronization. The foundry caller uses CPU
    forward passes. This function makes no kernel-speedup or quality claim.
    Exceptions propagate; a failed invocation never becomes a successful sample.
    """
    for name, value, minimum in (
        ("warmup_rounds", warmup_rounds, 1),
        ("measured_rounds", measured_rounds, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    methods = {"reference": reference, "candidate": candidate}
    for index in range(warmup_rounds):
        order = ("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")
        for name in order:
            methods[name]()
    samples: dict[str, list[float]] = {"reference": [], "candidate": []}
    first: list[str] = []
    for index in range(measured_rounds):
        order = ("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")
        first.append(order[0])
        for name in order:
            start = clock()
            methods[name]()
            elapsed_ns = clock() - start
            if elapsed_ns < 0:
                raise ValueError("latency clock moved backwards")
            samples[name].append(elapsed_ns / 1_000_000)
    return PairedLatency(
        warmup_rounds=warmup_rounds,
        measured_rounds=measured_rounds,
        reference_ms=tuple(samples["reference"]),
        candidate_ms=tuple(samples["candidate"]),
        first_in_round=tuple(first),
    )
