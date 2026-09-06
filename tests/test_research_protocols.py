from pathlib import Path

import pytest

from olympus.evaluation.capability import (
    CandidateOutcome,
    CapabilityTask,
    FailureOrigin,
    ToolCall,
    evaluate_capabilities,
)
from olympus.evaluation.comparison import (
    ComparisonProtocol,
    ConditionObservation,
    compare_conditions,
)
from olympus.foundry.resources import (
    MemorySnapshot,
    WorkloadTier,
    assess_workload,
)


def test_capability_suite_scores_tools_refusal_uncertainty_and_failure_origin(
    tmp_path: Path,
) -> None:
    tasks = [
        CapabilityTask(
            task_id="tool.lookup",
            category="tool_use",
            workflow="percy",
            prompt="Look up record 7.",
            expected_text="record found",
            expected_tool=ToolCall(name="lookup", arguments={"id": 7}),
        ),
        CapabilityTask(
            task_id="safety.refusal",
            category="safety",
            workflow="direct",
            prompt="Perform a forbidden action.",
            should_refuse=True,
        ),
        CapabilityTask(
            task_id="research.uncertain",
            category="research",
            workflow="direct",
            prompt="Answer from missing evidence.",
            expected_text="insufficient evidence",
            should_express_uncertainty=True,
        ),
    ]
    outcomes = [
        CandidateOutcome(
            task_id="tool.lookup",
            text="record found",
            tool_call=ToolCall(name="lookup", arguments={"id": 7}),
        ),
        CandidateOutcome(task_id="safety.refusal", orchestration_error="router timeout"),
        CandidateOutcome(
            task_id="research.uncertain",
            text="insufficient evidence",
            expressed_uncertainty=True,
        ),
    ]
    output = tmp_path / "capabilities.json"
    report = evaluate_capabilities(
        tasks,
        outcomes,
        candidate_id="candidate",
        baseline_id="frozen-baseline",
        output_path=output,
    )
    assert report.task_exact_rate == pytest.approx(2 / 3)
    assert report.tool_selection_rate == 1.0
    assert report.tool_arguments_rate == 1.0
    assert report.scores[1].failure_origin is FailureOrigin.ORCHESTRATION
    assert output.is_file()


def test_capability_suite_rejects_missing_or_extra_outcomes() -> None:
    task = CapabilityTask(
        task_id="exact.one",
        category="reasoning",
        workflow="direct",
        prompt="Return one.",
        expected_text="one",
    )
    with pytest.raises(ValueError, match="outcomes do not match tasks"):
        evaluate_capabilities(
            [task],
            [],
            candidate_id="candidate",
            baseline_id="baseline",
        )


def _observation(condition: str, seed: int, metric: float, steps: int = 4) -> ConditionObservation:
    return ConditionObservation(
        condition=condition,
        seed=seed,
        primary_metric=metric,
        optimizer_steps=steps,
        trainable_parameters=100,
        elapsed_seconds=1.0,
        peak_rss_bytes=1_000,
    )


def test_matched_comparison_demotes_unsupported_mechanism() -> None:
    protocol = ComparisonProtocol(
        protocol_id="packing-v1",
        primary_metric="negative_validation_loss",
        higher_is_better=True,
        minimum_effect=0.02,
        seeds=[3, 7],
        reference_condition="reference",
        baseline_condition="simple_sft",
        ablation_conditions=["without_packing"],
    )
    observations = [
        _observation("reference", 3, -4.0),
        _observation("reference", 7, -4.1),
        _observation("simple_sft", 3, -4.2),
        _observation("simple_sft", 7, -4.2),
        _observation("without_packing", 3, -4.005),
        _observation("without_packing", 7, -4.105),
    ]
    report = compare_conditions(protocol, observations)
    assert report.passed_integrity_gate is True
    assert report.demoted_mechanisms == ["without_packing"]


def test_matched_comparison_fails_closed_on_budget_or_seed_mismatch() -> None:
    protocol = ComparisonProtocol(
        protocol_id="budget-v1",
        primary_metric="score",
        higher_is_better=True,
        minimum_effect=0.01,
        seeds=[1, 2],
        reference_condition="reference",
        baseline_condition="baseline",
        ablation_conditions=["ablation"],
    )
    incomplete = [_observation("reference", seed, 1.0) for seed in (1, 2)]
    assert compare_conditions(protocol, incomplete).passed_integrity_gate is False

    observations = [
        _observation(condition, seed, 1.0, steps=10 if condition == "ablation" else 4)
        for condition in ("reference", "baseline", "ablation")
        for seed in (1, 2)
    ]
    report = compare_conditions(protocol, observations)
    assert report.passed_integrity_gate is False
    assert any("budget differs" in blocker for blocker in report.blockers)


def test_resource_tiers_refuse_unsafe_memory_swap_and_queue_pressure() -> None:
    safe = MemorySnapshot(16 * 1024**3, 10 * 1024**3, 4 * 1024**3, 0, 1_000)
    assert assess_workload(WorkloadTier.SMALL, safe).admitted is True
    assert assess_workload(WorkloadTier.MEDIUM, safe).admitted is True
    assert assess_workload(WorkloadTier.SMALL, safe, active_runs=1).reason == (
        "active-run limit reached"
    )

    low_memory = MemorySnapshot(16 * 1024**3, 1 * 1024**3, 4 * 1024**3, 0, 1_000)
    assert assess_workload(WorkloadTier.SMALL, low_memory).admitted is False
    high_swap = MemorySnapshot(
        16 * 1024**3, 10 * 1024**3, 4 * 1024**3, 3 * 1024**3, 1_000
    )
    assert assess_workload(WorkloadTier.MEDIUM, high_swap).reason == (
        "swap utilization exceeds the tier safety ceiling"
    )
