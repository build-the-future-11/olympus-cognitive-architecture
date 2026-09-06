from __future__ import annotations

import statistics
from collections import defaultdict

from pydantic import Field, model_validator

from olympus.core.schemas import StrictModel


class ComparisonProtocol(StrictModel):
    protocol_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,127}$")
    primary_metric: str
    higher_is_better: bool
    minimum_effect: float = Field(ge=0.0)
    seeds: list[int] = Field(min_length=2)
    reference_condition: str
    baseline_condition: str
    ablation_conditions: list[str] = Field(min_length=1)
    budget_tolerance_fraction: float = Field(default=0.05, ge=0.0, le=0.25)

    @model_validator(mode="after")
    def validate_conditions(self) -> ComparisonProtocol:
        conditions = [
            self.reference_condition,
            self.baseline_condition,
            *self.ablation_conditions,
        ]
        if len(conditions) != len(set(conditions)):
            raise ValueError("comparison condition names must be unique")
        if len(self.seeds) != len(set(self.seeds)):
            raise ValueError("comparison seeds must be unique")
        return self


class ConditionObservation(StrictModel):
    condition: str
    seed: int
    primary_metric: float
    optimizer_steps: int = Field(ge=0)
    trainable_parameters: int = Field(gt=0)
    elapsed_seconds: float = Field(ge=0.0)
    peak_rss_bytes: int = Field(ge=0)

    @property
    def budget_units(self) -> int:
        return self.optimizer_steps * self.trainable_parameters


class ConditionSummary(StrictModel):
    condition: str
    seeds: list[int]
    metric_mean: float
    metric_stdev: float
    budget_mean: float
    elapsed_mean_seconds: float
    peak_rss_bytes: int
    delta_from_reference: float
    budget_matched: bool
    verdict: str


class ComparisonReport(StrictModel):
    schema_version: int = 1
    protocol: ComparisonProtocol
    summaries: list[ConditionSummary]
    demoted_mechanisms: list[str]
    passed_integrity_gate: bool
    blockers: list[str]


def compare_conditions(
    protocol: ComparisonProtocol, observations: list[ConditionObservation]
) -> ComparisonReport:
    grouped: dict[str, list[ConditionObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.condition].append(observation)
    required = {
        protocol.reference_condition,
        protocol.baseline_condition,
        *protocol.ablation_conditions,
    }
    blockers: list[str] = []
    for condition in sorted(required):
        actual_seeds = sorted(item.seed for item in grouped.get(condition, []))
        if actual_seeds != sorted(protocol.seeds):
            blockers.append(
                f"{condition}: expected seeds {sorted(protocol.seeds)}, got {actual_seeds}"
            )
    if blockers:
        return ComparisonReport(
            protocol=protocol,
            summaries=[],
            demoted_mechanisms=[],
            passed_integrity_gate=False,
            blockers=blockers,
        )

    reference_rows = grouped[protocol.reference_condition]
    reference_metric = statistics.mean(item.primary_metric for item in reference_rows)
    reference_budget = statistics.mean(item.budget_units for item in reference_rows)
    summaries: list[ConditionSummary] = []
    demoted: list[str] = []
    for condition in sorted(required):
        rows = grouped[condition]
        metric = statistics.mean(item.primary_metric for item in rows)
        budget = statistics.mean(item.budget_units for item in rows)
        budget_delta = abs(budget - reference_budget) / max(reference_budget, 1.0)
        budget_matched = budget_delta <= protocol.budget_tolerance_fraction
        signed_delta = metric - reference_metric
        reference_contribution = (
            reference_metric - metric
            if protocol.higher_is_better
            else metric - reference_metric
        )
        if condition == protocol.reference_condition:
            verdict = "reference"
        elif not budget_matched:
            verdict = "invalid_budget_mismatch"
            blockers.append(f"{condition}: budget differs from reference by {budget_delta:.2%}")
        elif (
            condition in protocol.ablation_conditions
            and reference_contribution < protocol.minimum_effect
        ):
            verdict = "mechanism_not_supported"
            demoted.append(condition)
        elif condition in protocol.ablation_conditions:
            verdict = "mechanism_supported"
        else:
            verdict = "baseline_comparison"
        summaries.append(
            ConditionSummary(
                condition=condition,
                seeds=sorted(item.seed for item in rows),
                metric_mean=metric,
                metric_stdev=(
                    statistics.stdev(item.primary_metric for item in rows)
                    if len(rows) > 1
                    else 0.0
                ),
                budget_mean=budget,
                elapsed_mean_seconds=statistics.mean(item.elapsed_seconds for item in rows),
                peak_rss_bytes=max(item.peak_rss_bytes for item in rows),
                delta_from_reference=signed_delta,
                budget_matched=budget_matched,
                verdict=verdict,
            )
        )
    return ComparisonReport(
        protocol=protocol,
        summaries=summaries,
        demoted_mechanisms=demoted,
        passed_integrity_gate=not blockers,
        blockers=blockers,
    )
