from __future__ import annotations

import hashlib
import json
from pathlib import Path

from olympus.foundry.readiness import (
    REQUIRED_CAPABILITIES,
    REQUIRED_CONTROLS,
    REQUIRED_DOCUMENTS,
    REQUIRED_SAFETY_SUITES,
    ArtifactIdentity,
    CandidateReadinessEvidence,
    ComparativeEvaluation,
    DatasetEvidence,
    ProductionEvidence,
    ReadinessEvidenceSet,
    ReadinessThresholds,
    ReproductionEvidence,
    SuiteEvaluation,
    TrainingRunEvidence,
    evaluate_family_readiness,
    evaluate_portfolio_readiness,
)
from olympus.models.specifications import model_spec


def _artifact(tmp_path: Path, name: str) -> ArtifactIdentity:
    path = tmp_path / name
    path.write_text(name, encoding="utf-8")
    return ArtifactIdentity(path=name, sha256=hashlib.sha256(name.encode()).hexdigest())


def _passing_candidate(tmp_path: Path) -> CandidateReadinessEvidence:
    manifest = _artifact(tmp_path, "manifest.json")
    shared = {
        "dataset_manifest_sha256": "d" * 64,
        "base_model_sha256": "a" * 64,
        "training_config_sha256": "b" * 64,
        "completed": True,
        "finite_metrics": True,
        "protected_accesses_before_freeze": 0,
    }
    runs = [
        TrainingRunEvidence(
            seed=seed,
            receipt=_artifact(tmp_path, f"training-{seed}.json"),
            checkpoint_sha256=(str(seed) * 64)[:64],
            **shared,
        )
        for seed in (1, 2, 3)
    ]
    controls = [
        ComparativeEvaluation(
            control=name,
            receipt=_artifact(tmp_path, f"control-{name}.json"),
            effect=0.04,
            confidence_interval_low=0.03,
            confidence_interval_high=0.05,
            no_material_regressions=True,
        )
        for name in REQUIRED_CONTROLS
    ]
    capabilities = [
        SuiteEvaluation(
            suite=name,
            receipt=_artifact(tmp_path, f"capability-{name}.json"),
            score=0.9,
            passed=True,
            critical_failures=0,
        )
        for name in REQUIRED_CAPABILITIES
    ]
    safety = [
        SuiteEvaluation(
            suite=name,
            receipt=_artifact(tmp_path, f"safety-{name}.json"),
            score=1.0,
            passed=True,
            critical_failures=0,
        )
        for name in REQUIRED_SAFETY_SUITES
    ]
    return CandidateReadinessEvidence(
        family_id="hermes",
        candidate_id="hermes-test",
        dataset=DatasetEvidence(
            manifest=manifest,
            dataset_manifest_sha256="d" * 64,
            training_records=10_000,
            protected_records=1_200,
            minimum_protected_records_per_category=100,
            license_approved=True,
            provenance_complete=True,
            exact_and_normalized_dedup_passed=True,
            contamination_scan_passed=True,
            pii_and_secret_scan_passed=True,
            splits_frozen_before_training=True,
            capability_coverage_complete=True,
        ),
        training_runs=runs,
        controls=controls,
        capabilities=capabilities,
        safety=safety,
        production=ProductionEvidence(
            receipt=_artifact(tmp_path, "production.json"),
            p50_latency_ms=100,
            p95_latency_ms=500,
            p99_latency_ms=900,
            throughput_requests_per_second=10,
            error_rate=0,
            cost_per_million_tokens_usd=10,
            soak_hours=24,
            failure_injection_passed=True,
            queue_cancel_retry_backpressure_passed=True,
            rollback_passed=True,
            zero_downtime_upgrade_passed=True,
            drift_failure_cost_monitoring_passed=True,
        ),
        reproduction=ReproductionEvidence(
            receipt=_artifact(tmp_path, "reproduction.json"),
            independent_operator="external-lab",
            clean_environment_identity="container-sha256:abc",
            source_commit="c" * 40,
            dataset_manifest_sha256="d" * 64,
            checkpoint_sha256=runs[0].checkpoint_sha256,
            metrics_reproduced_within_tolerance=True,
        ),
        family_gate_results={name: True for name in model_spec("hermes").capability_gates},
        documents={name: _artifact(tmp_path, f"{name}.md") for name in REQUIRED_DOCUMENTS},
    )


def test_complete_hash_bound_evidence_passes(tmp_path: Path) -> None:
    report = evaluate_family_readiness(
        _passing_candidate(tmp_path), ReadinessThresholds(), evidence_root=tmp_path
    )
    assert report.ready_for_large_scale
    assert report.blockers == []
    assert all(gate.passed for gate in report.gates)


def test_tamper_and_weak_effect_fail_closed(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    (tmp_path / "training-1.json").write_text("tampered", encoding="utf-8")
    candidate.controls[0].confidence_interval_low = 0.019
    report = evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert not report.ready_for_large_scale
    assert "three_seed_training" in report.blockers
    assert "strong_controls" in report.blockers


def test_empty_portfolio_reports_every_family_and_gate(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.json"
    thresholds_path = tmp_path / "thresholds.json"
    output_path = tmp_path / "report.json"
    evidence_path.write_text(
        ReadinessEvidenceSet(candidates=[]).model_dump_json(), encoding="utf-8"
    )
    thresholds_path.write_text(ReadinessThresholds().model_dump_json(), encoding="utf-8")
    report = evaluate_portfolio_readiness(evidence_path, thresholds_path, output_path)
    assert report.total_families == 6
    assert report.ready_families == 0
    assert not report.all_families_ready
    assert all(len(item.gates) == 11 for item in report.family_reports)
    assert json.loads(output_path.read_text())["all_families_ready"] is False
