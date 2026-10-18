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
FROZEN_THRESHOLDS_SHA256 = "368073c460c05f24a874a4fcc115224b9b93bad1bdbae51e4642114c7d2dcf70"
PROTOCOL_ID: Literal["olympus-six-family-readiness-v1"] = "olympus-six-family-readiness-v1"
ReceiptKind = Literal[
    "training",
    "control",
    "capability",
    "safety",
    "production",
    "reproduction",
    "family_gate",
]
DOCUMENT_HEADER_PREFIX = "<!-- olympus-readiness:"
DOCUMENT_HEADER_SUFFIX = "-->"


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


class EvidenceReceiptEnvelope(StrictModel):
    schema_version: Literal[1] = 1
    protocol_id: Literal["olympus-six-family-readiness-v1"]
    kind: ReceiptKind
    candidate_id: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    producer: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class ReleaseDocumentHeader(StrictModel):
    protocol_id: Literal["olympus-six-family-readiness-v1"]
    document_type: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    effect: float = Field(allow_inf_nan=False)
    confidence_interval_low: float = Field(allow_inf_nan=False)
    confidence_interval_high: float = Field(allow_inf_nan=False)
    no_material_regressions: bool

    @model_validator(mode="after")
    def validate_interval(self) -> ComparativeEvaluation:
        if not self.confidence_interval_low <= self.effect <= self.confidence_interval_high:
            raise ValueError("control confidence interval must contain the observed effect")
        return self


class SuiteEvaluation(StrictModel):
    suite: str = Field(min_length=1)
    receipt: ArtifactIdentity
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score: float = Field(ge=0, le=1)
    passed: bool
    critical_failures: int = Field(ge=0)


class FamilyGateEvidence(StrictModel):
    gate: str = Field(min_length=1)
    receipt: ArtifactIdentity
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool


class ProductionEvidence(StrictModel):
    receipt: ArtifactIdentity
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
    source_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    candidate_checkpoint_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training_operator: str | None = Field(default=None, min_length=1)
    dataset: DatasetEvidence | None = None
    training_runs: list[TrainingRunEvidence] = Field(default_factory=list)
    controls: list[ComparativeEvaluation] = Field(default_factory=list)
    capabilities: list[SuiteEvaluation] = Field(default_factory=list)
    safety: list[SuiteEvaluation] = Field(default_factory=list)
    production: ProductionEvidence | None = None
    reproduction: ReproductionEvidence | None = None
    family_gate_results: list[FamilyGateEvidence] = Field(default_factory=list)
    documents: dict[str, ArtifactIdentity] = Field(default_factory=dict)


class ReadinessEvidenceSet(StrictModel):
    schema_version: Literal[1] = 1
    protocol_id: Literal["olympus-six-family-readiness-v1"]
    candidates: list[CandidateReadinessEvidence]

    @model_validator(mode="after")
    def unique_candidate_identities(self) -> ReadinessEvidenceSet:
        families = [candidate.family_id for candidate in self.candidates]
        if len(families) != len(set(families)):
            raise ValueError("readiness evidence contains duplicate model families")
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("readiness evidence contains duplicate candidate IDs")
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
    protocol_id: Literal["olympus-six-family-readiness-v1"] = PROTOCOL_ID
    thresholds_sha256: str
    evidence_sha256: str
    evidence_schema_sha256: str
    evaluator_sha256: str
    decision_code_sha256: str
    family_reports: list[FamilyReadinessReport]
    ready_families: int
    total_families: int
    all_families_ready: bool


def readiness_evidence_schema_bytes() -> bytes:
    """Return the canonical JSON Schema bytes used for evidence interchange."""

    return (
        json.dumps(
            ReadinessEvidenceSet.model_json_schema(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def write_readiness_evidence_schema(output_path: Path) -> None:
    _atomic_json(output_path, ReadinessEvidenceSet.model_json_schema())


def _decision_code_sha256() -> str:
    paths = (
        Path(__file__),
        Path(__file__).with_name("data_pipeline.py"),
        Path(__file__).parents[1] / "models/specifications.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(str(len(payload)).encode())
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def _gate(name: str, passed: bool, observed: object, requirement: str) -> ReadinessGate:
    return ReadinessGate(gate=name, passed=passed, observed=str(observed), requirement=requirement)


def _contained_artifact_path(identity: ArtifactIdentity, root: Path) -> Path | None:
    path = Path(identity.path)
    if path.is_absolute() or path.is_symlink():
        return None
    root = root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        return None
    return resolved


def _artifact_bytes(identity: ArtifactIdentity | None, root: Path) -> bytes | None:
    if identity is None:
        return None
    path = _contained_artifact_path(identity, root)
    if path is None:
        return None
    try:
        payload = path.read_bytes()
    except OSError:
        return None
    return payload if hashlib.sha256(payload).hexdigest() == identity.sha256 else None


def _identity_valid(identity: ArtifactIdentity | None, root: Path) -> bool:
    return _artifact_bytes(identity, root) is not None


def _identity_path(identity: ArtifactIdentity, root: Path) -> Path | None:
    if not _identity_valid(identity, root):
        return None
    return _contained_artifact_path(identity, root)


def _receipt_valid(
    identity: ArtifactIdentity,
    root: Path,
    *,
    kind: ReceiptKind,
    candidate_id: str,
    checkpoint_sha256: str,
    dataset_manifest_sha256: str,
    source_commit: str,
    producer: str | None = None,
) -> bool:
    payload = _artifact_bytes(identity, root)
    if payload is None:
        return False
    try:
        receipt = EvidenceReceiptEnvelope.model_validate_json(payload)
    except ValueError:
        return False
    return all(
        (
            receipt.kind == kind,
            receipt.candidate_id == candidate_id,
            receipt.checkpoint_sha256 == checkpoint_sha256,
            receipt.dataset_manifest_sha256 == dataset_manifest_sha256,
            receipt.source_commit == source_commit,
            producer is None or receipt.producer.casefold() == producer.casefold(),
        )
    )


def _item_receipts_valid(
    items: Sequence[
        TrainingRunEvidence | ComparativeEvaluation | SuiteEvaluation | FamilyGateEvidence
    ],
    root: Path,
    *,
    kind: ReceiptKind,
    candidate_id: str,
    source_commit: str,
    producer: str | None = None,
) -> bool:
    return bool(items) and all(
        _receipt_valid(
            item.receipt,
            root,
            kind=kind,
            candidate_id=candidate_id,
            checkpoint_sha256=item.checkpoint_sha256,
            dataset_manifest_sha256=item.dataset_manifest_sha256,
            source_commit=source_commit,
            producer=producer,
        )
        for item in items
    )


def _dataset_receipt_valid(dataset: DatasetEvidence | None, root: Path) -> bool:
    if dataset is None:
        return False
    manifest_path = _identity_path(dataset.manifest, root)
    if manifest_path is None:
        return False
    try:
        manifest = verify_dataset_manifest(
            manifest_path,
            expected_manifest_file_sha256=dataset.manifest.sha256,
            require_contained_splits=True,
        )
    except OSError, ValueError:
        return False
    splits = {split.name: split for split in manifest.splits}
    train = splits["train"]
    protected = splits["test"]
    return all(
        (
            manifest.manifest_sha256 == dataset.dataset_manifest_sha256,
            train.records == dataset.training_records,
            protected.records == dataset.protected_records,
            min(protected.categories.values()) == dataset.minimum_protected_records_per_category,
            dataset.license_approved == (manifest.source.license in ALLOWED_LICENSES),
            dataset.provenance_complete
            == bool(
                manifest.source.uri
                and manifest.source.owner
                and manifest.source.sha256
                and manifest.source.record_count
            ),
            dataset.exact_and_normalized_dedup_passed
            == (
                manifest.quality.duplicate_ids == 0 and manifest.quality.normalized_duplicates == 0
            ),
            dataset.contamination_scan_passed == (manifest.quality.split_leakage_hits == 0),
            dataset.pii_and_secret_scan_passed == (manifest.quality.pii_or_secret_hits == 0),
            dataset.splits_frozen_before_training == bool(manifest.manifest_sha256),
            dataset.capability_coverage_complete
            == (not manifest.quality.missing_category_split_pairs),
        )
    )


def _names_are_unique(items: Sequence[object], attribute: str) -> bool:
    names = [str(getattr(item, attribute)) for item in items]
    return len(names) == len(set(names))


def _bound_to_candidate(
    items: Sequence[object], candidate_sha: str | None, dataset_sha: str | None
) -> bool:
    return (
        candidate_sha is not None
        and dataset_sha is not None
        and all(
            getattr(item, "checkpoint_sha256", None) == candidate_sha
            and getattr(item, "dataset_manifest_sha256", None) == dataset_sha
            for item in items
        )
    )


def _documents_bound_to_candidate(evidence: CandidateReadinessEvidence, root: Path) -> bool:
    checkpoint_sha = evidence.candidate_checkpoint_sha256
    dataset_sha = evidence.dataset.dataset_manifest_sha256 if evidence.dataset is not None else None
    if (
        checkpoint_sha is None
        or dataset_sha is None
        or evidence.source_commit is None
        or set(evidence.documents) != REQUIRED_DOCUMENTS
    ):
        return False
    for name in REQUIRED_DOCUMENTS:
        payload = _artifact_bytes(evidence.documents[name], root)
        if payload is None:
            return False
        try:
            content = payload.decode("utf-8")
        except UnicodeError:
            return False
        lines = content.splitlines()
        if not lines or not any(line.strip() for line in lines[1:]):
            return False
        first_line = lines[0]
        if not (
            first_line.startswith(DOCUMENT_HEADER_PREFIX)
            and first_line.endswith(DOCUMENT_HEADER_SUFFIX)
        ):
            return False
        header_payload = first_line[
            len(DOCUMENT_HEADER_PREFIX) : -len(DOCUMENT_HEADER_SUFFIX)
        ].strip()
        try:
            header = ReleaseDocumentHeader.model_validate_json(header_payload)
        except ValueError:
            return False
        if not all(
            (
                header.document_type == name,
                header.candidate_id == evidence.candidate_id,
                header.source_commit == evidence.source_commit,
                header.checkpoint_sha256 == checkpoint_sha,
                header.dataset_manifest_sha256 == dataset_sha,
            )
        ):
            return False
    return True


def evaluate_family_readiness(
    evidence: CandidateReadinessEvidence,
    thresholds: ReadinessThresholds,
    *,
    evidence_root: Path,
) -> FamilyReadinessReport:
    dataset = evidence.dataset
    data_receipt_ok = _dataset_receipt_valid(dataset, evidence_root)
    data_quality_ok = (
        data_receipt_ok
        and dataset is not None
        and all(
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
    )
    data_scale_ok = (
        data_receipt_ok
        and dataset is not None
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
        data_receipt_ok
        and evidence.source_commit is not None
        and evidence.training_operator is not None
        and len({run.seed for run in runs}) >= thresholds.minimum_independent_seeds
        and len({run.seed for run in runs}) == len(runs)
        and len({run.checkpoint_sha256 for run in runs}) == len(runs)
        and len(run_identities) == 1
        and _item_receipts_valid(
            runs,
            evidence_root,
            kind="training",
            candidate_id=evidence.candidate_id,
            source_commit=evidence.source_commit,
            producer=evidence.training_operator,
        )
        and all(
            run.completed and run.finite_metrics and run.protected_accesses_before_freeze == 0
            for run in runs
        )
    )
    if dataset is not None:
        seeds_ok = seeds_ok and all(
            run.dataset_manifest_sha256 == dataset.dataset_manifest_sha256 for run in runs
        )
    seeds_ok = seeds_ok and evidence.candidate_checkpoint_sha256 in {
        run.checkpoint_sha256 for run in runs
    }

    controls = {item.control: item for item in evidence.controls}
    dataset_sha = dataset.dataset_manifest_sha256 if dataset is not None else None
    controls_ok = (
        set(controls) == REQUIRED_CONTROLS
        and _names_are_unique(evidence.controls, "control")
        and _item_receipts_valid(
            evidence.controls,
            evidence_root,
            kind="control",
            candidate_id=evidence.candidate_id,
            source_commit=evidence.source_commit or "",
        )
        and _bound_to_candidate(
            evidence.controls, evidence.candidate_checkpoint_sha256, dataset_sha
        )
        and all(
            item.confidence_interval_low >= thresholds.minimum_primary_effect
            and item.no_material_regressions
            for item in controls.values()
        )
    )
    capabilities = {item.suite: item for item in evidence.capabilities}
    capability_ok = (
        set(capabilities) == REQUIRED_CAPABILITIES
        and _names_are_unique(evidence.capabilities, "suite")
        and _item_receipts_valid(
            evidence.capabilities,
            evidence_root,
            kind="capability",
            candidate_id=evidence.candidate_id,
            source_commit=evidence.source_commit or "",
        )
        and _bound_to_candidate(
            evidence.capabilities, evidence.candidate_checkpoint_sha256, dataset_sha
        )
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
        set(safety) == REQUIRED_SAFETY_SUITES
        and _names_are_unique(evidence.safety, "suite")
        and _item_receipts_valid(
            evidence.safety,
            evidence_root,
            kind="safety",
            candidate_id=evidence.candidate_id,
            source_commit=evidence.source_commit or "",
        )
        and _bound_to_candidate(evidence.safety, evidence.candidate_checkpoint_sha256, dataset_sha)
        and all(item.passed and item.critical_failures == 0 for item in safety.values())
    )
    production = evidence.production
    production_ok = (
        production is not None
        and _receipt_valid(
            production.receipt,
            evidence_root,
            kind="production",
            candidate_id=evidence.candidate_id,
            checkpoint_sha256=production.checkpoint_sha256,
            dataset_manifest_sha256=dataset_sha or "",
            source_commit=evidence.source_commit or "",
        )
        and production.checkpoint_sha256 == evidence.candidate_checkpoint_sha256
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
        data_receipt_ok
        and reproduction is not None
        and _receipt_valid(
            reproduction.receipt,
            evidence_root,
            kind="reproduction",
            candidate_id=evidence.candidate_id,
            checkpoint_sha256=reproduction.checkpoint_sha256,
            dataset_manifest_sha256=reproduction.dataset_manifest_sha256,
            source_commit=evidence.source_commit or "",
            producer=reproduction.independent_operator,
        )
        and reproduction.metrics_reproduced_within_tolerance
        and evidence.training_operator is not None
        and reproduction.source_commit == evidence.source_commit
        and reproduction.independent_operator.casefold() != evidence.training_operator.casefold()
        and dataset is not None
        and reproduction.dataset_manifest_sha256 == dataset.dataset_manifest_sha256
        and reproduction.checkpoint_sha256 == evidence.candidate_checkpoint_sha256
    )
    required_family_gates = set(model_spec(evidence.family_id).capability_gates)
    family_gate_results = {item.gate: item for item in evidence.family_gate_results}
    family_gates_ok = (
        set(family_gate_results) == required_family_gates
        and _names_are_unique(evidence.family_gate_results, "gate")
        and _item_receipts_valid(
            evidence.family_gate_results,
            evidence_root,
            kind="family_gate",
            candidate_id=evidence.candidate_id,
            source_commit=evidence.source_commit or "",
        )
        and _bound_to_candidate(
            evidence.family_gate_results,
            evidence.candidate_checkpoint_sha256,
            dataset_sha,
        )
        and all(family_gate_results[name].passed for name in required_family_gates)
    )
    documents_ok = _documents_bound_to_candidate(evidence, evidence_root)

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
            (
                f"passed={sum(item.passed for item in evidence.family_gate_results)}"
                f"/{len(required_family_gates)}"
            ),
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
    for path, label in ((evidence_path, "evidence"), (thresholds_path, "thresholds")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} input must be a regular non-symlink file")
    if output_path.resolve() in {evidence_path.resolve(), thresholds_path.resolve()}:
        raise ValueError("readiness output cannot overwrite an input")
    thresholds_sha = _sha256(thresholds_path)
    if thresholds_sha != FROZEN_THRESHOLDS_SHA256:
        raise ValueError(
            "readiness thresholds do not match the frozen v1 policy; create a versioned protocol"
        )
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
        thresholds_sha256=thresholds_sha,
        evidence_sha256=_sha256(evidence_path),
        evidence_schema_sha256=hashlib.sha256(readiness_evidence_schema_bytes()).hexdigest(),
        evaluator_sha256=_sha256(Path(__file__)),
        decision_code_sha256=_decision_code_sha256(),
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
