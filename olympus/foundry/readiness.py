"""Fail-closed large-scale readiness gates for BU1LD model families.

This module evaluates evidence receipts.  It never trains a model and never
turns a reference implementation, unit test, or development fixture into a
capability or production claim.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import Field, model_validator

from olympus.core.schemas import StrictModel
from olympus.foundry.data_pipeline import ALLOWED_LICENSES, verify_dataset_manifest
from olympus.models.specifications import BU1LD_MODEL_SPECS, ModelFamilyId, model_spec

REQUIRED_CONTROLS = frozenset(
    {"base_model", "matched_alternative", "retrieval_ablation", "tool_ablation", "routing_ablation"}
)
REQUIRED_CAPABILITIES = frozenset(
    {
        "reasoning",
        "instruction_following",
        "tool_completion",
        "structured_output",
        "long_context",
        "domain_held_out",
        "distribution_shift_and_adversarial",
        "calibration_and_abstention",
    }
)
REQUIRED_SAFETY_SUITES = frozenset(
    {
        "prompt_injection",
        "data_exfiltration",
        "harmful_output_policy",
        "tool_permission_boundaries",
        "tenant_and_memory_isolation",
    }
)
REQUIRED_DOCUMENTS = frozenset(
    {"model_card", "evaluation_report", "load_and_cost_report", "safety_and_rollback_report"}
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


class ReadinessThresholds(StrictModel):
    schema_version: Literal[1] = 1
    minimum_training_records: int = Field(default=10_000, gt=0)
    minimum_protected_records: int = Field(default=1_200, gt=0)
    minimum_protected_records_per_category: int = Field(default=100, gt=0)
    minimum_independent_seeds: int = Field(default=3, ge=3)
    minimum_primary_effect: float = Field(default=0.02, gt=0)
    minimum_capability_score: float = Field(default=0.80, ge=0, le=1)
    minimum_calibration_score: float = Field(default=0.70, ge=0, le=1)
    maximum_p95_latency_ms: float = Field(default=2_000, gt=0)
    maximum_p99_latency_ms: float = Field(default=4_000, gt=0)
    minimum_throughput_requests_per_second: float = Field(default=1.0, gt=0)
    maximum_error_rate: float = Field(default=0.001, ge=0, le=1)
    minimum_soak_hours: float = Field(default=24, gt=0)
    maximum_cost_per_million_tokens_usd: float = Field(default=100, gt=0)


class ArtifactIdentity(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetEvidence(StrictModel):
    manifest: ArtifactIdentity
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_records: int = Field(ge=0)
    protected_records: int = Field(ge=0)
    minimum_protected_records_per_category: int = Field(ge=0)
    license_approved: bool
    provenance_complete: bool
    exact_and_normalized_dedup_passed: bool
    contamination_scan_passed: bool
    pii_and_secret_scan_passed: bool
    splits_frozen_before_training: bool
    capability_coverage_complete: bool


class TrainingRunEvidence(StrictModel):
    seed: int = Field(ge=0)
    receipt: ArtifactIdentity
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed: bool
    finite_metrics: bool
    protected_accesses_before_freeze: int = Field(ge=0)


class ComparativeEvaluation(StrictModel):
    control: str = Field(min_length=1)
    receipt: ArtifactIdentity
    effect: float
    confidence_interval_low: float
    confidence_interval_high: float
    no_material_regressions: bool


class SuiteEvaluation(StrictModel):
    suite: str = Field(min_length=1)
    receipt: ArtifactIdentity
    score: float = Field(ge=0, le=1)
    passed: bool
    critical_failures: int = Field(ge=0)


class ProductionEvidence(StrictModel):
    receipt: ArtifactIdentity
    p50_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    p99_latency_ms: float = Field(ge=0)
    throughput_requests_per_second: float = Field(ge=0)
    error_rate: float = Field(ge=0, le=1)
    cost_per_million_tokens_usd: float = Field(ge=0)
    soak_hours: float = Field(ge=0)
    failure_injection_passed: bool
    queue_cancel_retry_backpressure_passed: bool
    rollback_passed: bool
    zero_downtime_upgrade_passed: bool
    drift_failure_cost_monitoring_passed: bool

    @model_validator(mode="after")
    def validate_percentiles(self) -> ProductionEvidence:
        if not self.p50_latency_ms <= self.p95_latency_ms <= self.p99_latency_ms:
            raise ValueError("latency percentiles must satisfy p50 <= p95 <= p99")
        return self


class ReproductionEvidence(StrictModel):
    receipt: ArtifactIdentity
    independent_operator: str = Field(min_length=1)
    clean_environment_identity: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metrics_reproduced_within_tolerance: bool


class CandidateReadinessEvidence(StrictModel):
    family_id: ModelFamilyId
    candidate_id: str = Field(min_length=1)
    dataset: DatasetEvidence | None = None
    training_runs: list[TrainingRunEvidence] = Field(default_factory=list)
    controls: list[ComparativeEvaluation] = Field(default_factory=list)
    capabilities: list[SuiteEvaluation] = Field(default_factory=list)
    safety: list[SuiteEvaluation] = Field(default_factory=list)
    production: ProductionEvidence | None = None
    reproduction: ReproductionEvidence | None = None
    family_gate_results: dict[str, bool] = Field(default_factory=dict)
    documents: dict[str, ArtifactIdentity] = Field(default_factory=dict)


class ReadinessEvidenceSet(StrictModel):
    schema_version: Literal[1] = 1
    candidates: list[CandidateReadinessEvidence]

    @model_validator(mode="after")
    def unique_families(self) -> ReadinessEvidenceSet:
        families = [candidate.family_id for candidate in self.candidates]
        if len(families) != len(set(families)):
            raise ValueError("readiness evidence contains duplicate model families")
        return self


class ReadinessGate(StrictModel):
    gate: str
    passed: bool
    observed: str
    requirement: str


class FamilyReadinessReport(StrictModel):
    family_id: ModelFamilyId
    candidate_id: str
    gates: list[ReadinessGate]
    ready_for_large_scale: bool
    blockers: list[str]


class PortfolioReadinessReport(StrictModel):
    schema_version: Literal[1] = 1
    thresholds_sha256: str
    evidence_sha256: str
    family_reports: list[FamilyReadinessReport]
    ready_families: int
    total_families: int
    all_families_ready: bool


def _gate(name: str, passed: bool, observed: object, requirement: str) -> ReadinessGate:
    return ReadinessGate(gate=name, passed=passed, observed=str(observed), requirement=requirement)


def _identity_valid(identity: ArtifactIdentity | None, root: Path) -> bool:
    if identity is None:
        return False
    path = Path(identity.path)
    if not path.is_absolute():
        path = root / path
    return path.is_file() and _sha256(path) == identity.sha256


def _all_receipts_valid(items: Sequence[object], root: Path) -> bool:
    return bool(items) and all(
        _identity_valid(getattr(item, "receipt", None), root) for item in items
    )


def evaluate_family_readiness(
    evidence: CandidateReadinessEvidence,
    thresholds: ReadinessThresholds,
    *,
    evidence_root: Path,
) -> FamilyReadinessReport:
    dataset = evidence.dataset
    data_receipt_ok = dataset is not None and _identity_valid(dataset.manifest, evidence_root)
    data_quality_ok = dataset is not None and all(
        (
            dataset.license_approved,
            dataset.provenance_complete,
            dataset.exact_and_normalized_dedup_passed,
            dataset.contamination_scan_passed,
            dataset.pii_and_secret_scan_passed,
            dataset.splits_frozen_before_training,
            dataset.capability_coverage_complete,
        )
    )
    data_scale_ok = (
        dataset is not None
        and dataset.training_records >= thresholds.minimum_training_records
        and dataset.protected_records >= thresholds.minimum_protected_records
        and dataset.minimum_protected_records_per_category
        >= thresholds.minimum_protected_records_per_category
    )

    runs = evidence.training_runs
    run_identities = {
        (
            run.dataset_manifest_sha256,
            run.base_model_sha256,
            run.training_config_sha256,
        )
        for run in runs
    }
    seeds_ok = (
        len({run.seed for run in runs}) >= thresholds.minimum_independent_seeds
        and len(run_identities) == 1
        and _all_receipts_valid(runs, evidence_root)
        and all(
            run.completed and run.finite_metrics and run.protected_accesses_before_freeze == 0
            for run in runs
        )
    )
    if dataset is not None:
        seeds_ok = seeds_ok and all(
            run.dataset_manifest_sha256 == dataset.dataset_manifest_sha256 for run in runs
        )

    controls = {item.control: item for item in evidence.controls}
    controls_ok = (
        REQUIRED_CONTROLS <= controls.keys()
        and _all_receipts_valid(evidence.controls, evidence_root)
        and all(
            item.confidence_interval_low >= thresholds.minimum_primary_effect
            and item.no_material_regressions
            for item in controls.values()
        )
    )
    capabilities = {item.suite: item for item in evidence.capabilities}
    capability_ok = (
        REQUIRED_CAPABILITIES <= capabilities.keys()
        and _all_receipts_valid(evidence.capabilities, evidence_root)
        and all(
            item.passed
            and item.critical_failures == 0
            and item.score
            >= (
                thresholds.minimum_calibration_score
                if item.suite == "calibration_and_abstention"
                else thresholds.minimum_capability_score
            )
            for item in capabilities.values()
        )
    )
    safety = {item.suite: item for item in evidence.safety}
    safety_ok = (
        REQUIRED_SAFETY_SUITES <= safety.keys()
        and _all_receipts_valid(evidence.safety, evidence_root)
        and all(item.passed and item.critical_failures == 0 for item in safety.values())
    )
    production = evidence.production
    production_ok = (
        production is not None
        and _identity_valid(production.receipt, evidence_root)
        and production.p95_latency_ms <= thresholds.maximum_p95_latency_ms
        and production.p99_latency_ms <= thresholds.maximum_p99_latency_ms
        and production.throughput_requests_per_second
        >= thresholds.minimum_throughput_requests_per_second
        and production.error_rate <= thresholds.maximum_error_rate
        and production.cost_per_million_tokens_usd <= thresholds.maximum_cost_per_million_tokens_usd
        and production.soak_hours >= thresholds.minimum_soak_hours
        and production.failure_injection_passed
        and production.queue_cancel_retry_backpressure_passed
        and production.rollback_passed
        and production.zero_downtime_upgrade_passed
        and production.drift_failure_cost_monitoring_passed
    )
    reproduction = evidence.reproduction
    reproduction_ok = (
        reproduction is not None
        and _identity_valid(reproduction.receipt, evidence_root)
        and reproduction.metrics_reproduced_within_tolerance
        and dataset is not None
        and reproduction.dataset_manifest_sha256 == dataset.dataset_manifest_sha256
        and any(reproduction.checkpoint_sha256 == run.checkpoint_sha256 for run in runs)
    )
    required_family_gates = set(model_spec(evidence.family_id).capability_gates)
    family_gates_ok = required_family_gates <= evidence.family_gate_results.keys() and all(
        evidence.family_gate_results.get(name) is True for name in required_family_gates
    )
    documents_ok = REQUIRED_DOCUMENTS <= evidence.documents.keys() and all(
        _identity_valid(evidence.documents[name], evidence_root) for name in REQUIRED_DOCUMENTS
    )

    gates = [
        _gate(
            "data_receipt_integrity",
            data_receipt_ok,
            data_receipt_ok,
            "Manifest exists and hash matches.",
        ),
        _gate(
            "data_quality",
            data_quality_ok,
            data_quality_ok,
            (
                "Rights, provenance, deduplication, contamination, PII, frozen splits, "
                "and coverage all pass."
            ),
        ),
        _gate(
            "data_scale",
            data_scale_ok,
            (
                f"train={dataset.training_records if dataset else 0}, "
                f"protected={dataset.protected_records if dataset else 0}"
            ),
            (
                f"Train >= {thresholds.minimum_training_records}; protected >= "
                f"{thresholds.minimum_protected_records}; each category >= "
                f"{thresholds.minimum_protected_records_per_category}."
            ),
        ),
        _gate(
            "three_seed_training",
            seeds_ok,
            f"runs={len(runs)}, unique_seeds={len({run.seed for run in runs})}",
            (
                "At least three completed, finite, identically configured seeds with "
                "immutable receipts and no pre-freeze protected access."
            ),
        ),
        _gate(
            "strong_controls",
            controls_ok,
            sorted(controls),
            (
                f"All controls {sorted(REQUIRED_CONTROLS)} pass with confidence-interval "
                f"lower bound >= {thresholds.minimum_primary_effect} and no material regression."
            ),
        ),
        _gate(
            "capability_evaluation",
            capability_ok,
            sorted(capabilities),
            (
                f"All capability suites {sorted(REQUIRED_CAPABILITIES)} meet frozen score "
                "floors without critical failures."
            ),
        ),
        _gate(
            "safety_evaluation",
            safety_ok,
            sorted(safety),
            f"All safety suites {sorted(REQUIRED_SAFETY_SUITES)} pass without critical failures.",
        ),
        _gate(
            "production_qualification",
            production_ok,
            "present" if production else "missing",
            (
                "Latency, throughput, cost, 24-hour soak, failure handling, rollback, "
                "upgrades, and monitoring meet thresholds."
            ),
        ),
        _gate(
            "independent_reproduction",
            reproduction_ok,
            "present" if reproduction else "missing",
            (
                "A clean independent operator reproduces the exact source, data, "
                "checkpoint, and metrics."
            ),
        ),
        _gate(
            "family_specific_gates",
            family_gates_ok,
            f"passed={sum(evidence.family_gate_results.values())}/{len(required_family_gates)}",
            "Every capability gate in the canonical family specification passes.",
        ),
        _gate(
            "release_documentation",
            documents_ok,
            sorted(evidence.documents),
            f"Hash-bound documents required: {sorted(REQUIRED_DOCUMENTS)}.",
        ),
    ]
    blockers = [gate.gate for gate in gates if not gate.passed]
    return FamilyReadinessReport(
        family_id=evidence.family_id,
        candidate_id=evidence.candidate_id,
        gates=gates,
        ready_for_large_scale=not blockers,
        blockers=blockers,
    )


def evaluate_portfolio_readiness(
    evidence_path: Path,
    thresholds_path: Path,
    output_path: Path,
) -> PortfolioReadinessReport:
    evidence = ReadinessEvidenceSet.model_validate_json(evidence_path.read_bytes())
    thresholds = ReadinessThresholds.model_validate_json(thresholds_path.read_bytes())
    candidates = {candidate.family_id: candidate for candidate in evidence.candidates}
    reports = []
    for spec in BU1LD_MODEL_SPECS:
        candidate = candidates.get(spec.family_id) or CandidateReadinessEvidence(
            family_id=spec.family_id,
            candidate_id=f"{spec.family_id}-no-candidate",
        )
        reports.append(
            evaluate_family_readiness(candidate, thresholds, evidence_root=evidence_path.parent)
        )
    ready = sum(report.ready_for_large_scale for report in reports)
    report = PortfolioReadinessReport(
        thresholds_sha256=_sha256(thresholds_path),
        evidence_sha256=_sha256(evidence_path),
        family_reports=reports,
        ready_families=ready,
        total_families=len(reports),
        all_families_ready=ready == len(reports),
    )
    _atomic_json(output_path, report.model_dump(mode="json"))
    return report


def audit_instruction_dataset(manifest_path: Path, output_path: Path) -> DatasetEvidence:
    """Create a scale-aware readiness receipt for a verified instruction dataset."""

    manifest = verify_dataset_manifest(manifest_path)
    if manifest.manifest_sha256 is None:
        raise ValueError("verified dataset manifest has no identity")
    split_by_name = {split.name: split for split in manifest.splits}
    train = split_by_name["train"]
    protected = split_by_name["test"]
    minimum_category = min(protected.categories.values())
    evidence = DatasetEvidence(
        manifest=ArtifactIdentity(
            path=os.path.relpath(manifest_path.resolve(), output_path.resolve().parent),
            sha256=_sha256(manifest_path),
        ),
        dataset_manifest_sha256=manifest.manifest_sha256,
        training_records=train.records,
        protected_records=protected.records,
        minimum_protected_records_per_category=minimum_category,
        license_approved=manifest.source.license in ALLOWED_LICENSES,
        provenance_complete=bool(
            manifest.source.uri
            and manifest.source.owner
            and manifest.source.sha256
            and manifest.source.record_count
        ),
        exact_and_normalized_dedup_passed=(
            manifest.quality.duplicate_ids == 0 and manifest.quality.normalized_duplicates == 0
        ),
        contamination_scan_passed=manifest.quality.split_leakage_hits == 0,
        pii_and_secret_scan_passed=manifest.quality.pii_or_secret_hits == 0,
        splits_frozen_before_training=bool(manifest.manifest_sha256),
        capability_coverage_complete=not manifest.quality.missing_category_split_pairs,
    )
    _atomic_json(output_path, evidence.model_dump(mode="json"))
    return evidence


def blocker_counts(report: PortfolioReadinessReport) -> dict[str, int]:
    """Return deterministic portfolio blocker totals for dashboards and CI."""

    return dict(
        sorted(
            Counter(blocker for item in report.family_reports for blocker in item.blockers).items()
        )
    )
