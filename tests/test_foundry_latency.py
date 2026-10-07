from __future__ import annotations

import pytest

from olympus.foundry.latency import measure_paired_latency


def test_warmup_is_excluded_and_execution_order_alternates() -> None:
    now = 0
    calls: list[str] = []
    counts = {"reference": 0, "candidate": 0}

    def operation(name: str, steady_ns: int) -> None:
        nonlocal now
        calls.append(name)
        now += 100_000_000 if counts[name] < 2 else steady_ns
        counts[name] += 1

    timing = measure_paired_latency(
        lambda: operation("reference", 2_000_000),
        lambda: operation("candidate", 3_000_000),
        warmup_rounds=2,
        measured_rounds=4,
        clock=lambda: now,
    )
    assert calls == ["reference", "candidate", "candidate", "reference"] * 3
    assert timing.reference_ms == (2.0,) * 4
    assert timing.candidate_ms == (3.0,) * 4
    assert timing.first_in_round == ("reference", "candidate", "reference", "candidate")
    report = timing.as_dict()
    assert report["reference_median_ms"] == 2.0
    assert report["candidate_p90_ms"] == 3.0
    assert report["warmup_included_in_samples"] is False


@pytest.mark.parametrize(
    "options",
    [
        {"warmup_rounds": 0},
        {"warmup_rounds": True},
        {"warmup_rounds": 1.5},
        {"measured_rounds": 1},
        {"measured_rounds": False},
        {"measured_rounds": 2.5},
    ],
)
def test_invalid_measurement_protocol_cannot_emit_a_report(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        measure_paired_latency(lambda: None, lambda: None, **options)  # type: ignore[arg-type]


def test_failed_workload_does_not_become_a_latency_sample() -> None:
    def fail() -> None:
        raise RuntimeError("workload failed")

    with pytest.raises(RuntimeError, match="workload failed"):
        measure_paired_latency(lambda: None, fail)


def test_backwards_clock_is_rejected() -> None:
    clock = iter([2, 1])
    with pytest.raises(ValueError, match="backwards"):
        measure_paired_latency(lambda: None, lambda: None, clock=lambda: next(clock))
