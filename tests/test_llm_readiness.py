from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import olympus.foundry.readiness as readiness_module
from olympus.foundry.readiness import (
    DOCUMENT_HEADER_PREFIX,
    DOCUMENT_HEADER_SUFFIX,
    PROTOCOL_ID,
    REQUIRED_CAPABILITIES,
    REQUIRED_CONTROLS,
    REQUIRED_DOCUMENTS,
    REQUIRED_SAFETY_SUITES,
    ArtifactIdentity,
    CandidateReadinessEvidence,
    ComparativeEvaluation,
    DatasetEvidence,
    EvidenceReceiptEnvelope,
    FamilyGateEvidence,
    ProductionEvidence,
    ReadinessEvidenceSet,
    ReadinessThresholds,
    ReceiptKind,
    ReleaseDocumentHeader,
    ReproductionEvidence,
    SuiteEvaluation,
    TrainingRunEvidence,
    evaluate_family_readiness,
    evaluate_portfolio_readiness,
)
from olympus.models.specifications import model_spec


def _artifact(tmp_path: Path, name: str, content: str | None = None) -> ArtifactIdentity:
    path = tmp_path / name
    payload = name if content is None else content
    path.write_text(payload, encoding="utf-8")
    return ArtifactIdentity(path=name, sha256=hashlib.sha256(payload.encode()).hexdigest())


def _receipt(
    tmp_path: Path,
    name: str,
    *,
    kind: ReceiptKind,
    checkpoint_sha256: str,
    producer: str = "primary-lab",
) -> ArtifactIdentity:
    payload = EvidenceReceiptEnvelope(
        protocol_id=PROTOCOL_ID,
        kind=kind,
        candidate_id="hermes-test",
        checkpoint_sha256=checkpoint_sha256,
        dataset_manifest_sha256="d" * 64,
        producer=producer,
        source_commit="c" * 40,
    ).model_dump_json()
    return _artifact(tmp_path, name, payload)


def _document(
    tmp_path: Path,
    document_type: str,
    checkpoint_sha256: str,
) -> ArtifactIdentity:
    header = ReleaseDocumentHeader(
        protocol_id=PROTOCOL_ID,
        document_type=document_type,
        candidate_id="hermes-test",
        source_commit="c" * 40,
        checkpoint_sha256=checkpoint_sha256,
        dataset_manifest_sha256="d" * 64,
    ).model_dump_json()
    content = (
        f"{DOCUMENT_HEADER_PREFIX} {header} {DOCUMENT_HEADER_SUFFIX}\n"
        f"# {document_type}\n\nCandidate-specific release evidence.\n"
    )
    return _artifact(tmp_path, f"{document_type}.md", content)


def _passing_candidate(tmp_path: Path) -> CandidateReadinessEvidence:
    manifest = _artifact(tmp_path, "manifest.json")
    runs = [
        TrainingRunEvidence(
            seed=seed,
            checkpoint_sha256=(str(seed) * 64)[:64],
            receipt=_receipt(
                tmp_path,
                f"training-{seed}.json",
                kind="training",
                checkpoint_sha256=(str(seed) * 64)[:64],
            ),
            dataset_manifest_sha256="d" * 64,
            base_model_sha256="a" * 64,
            training_config_sha256="b" * 64,
            completed=True,
            finite_metrics=True,
            protected_accesses_before_freeze=0,
        )
        for seed in (1, 2, 3)
    ]
    candidate_checkpoint = runs[0].checkpoint_sha256
    controls = [
        ComparativeEvaluation(
            control=name,
            receipt=_receipt(
                tmp_path,
                f"control-{name}.json",
                kind="control",
                checkpoint_sha256=candidate_checkpoint,
            ),
            checkpoint_sha256=candidate_checkpoint,
            dataset_manifest_sha256="d" * 64,
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
            receipt=_receipt(
                tmp_path,
                f"capability-{name}.json",
                kind="capability",
                checkpoint_sha256=candidate_checkpoint,
            ),
            checkpoint_sha256=candidate_checkpoint,
            dataset_manifest_sha256="d" * 64,
            score=0.9,
            passed=True,
            critical_failures=0,
        )
        for name in REQUIRED_CAPABILITIES
    ]
    safety = [
        SuiteEvaluation(
            suite=name,
            receipt=_receipt(
                tmp_path,
                f"safety-{name}.json",
                kind="safety",
                checkpoint_sha256=candidate_checkpoint,
            ),
            checkpoint_sha256=candidate_checkpoint,
            dataset_manifest_sha256="d" * 64,
            score=1.0,
            passed=True,
            critical_failures=0,
        )
        for name in REQUIRED_SAFETY_SUITES
    ]
    return CandidateReadinessEvidence(
        family_id="hermes",
        candidate_id="hermes-test",
        source_commit="c" * 40,
        candidate_checkpoint_sha256=candidate_checkpoint,
        training_operator="primary-lab",
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
            receipt=_receipt(
                tmp_path,
                "production.json",
                kind="production",
                checkpoint_sha256=candidate_checkpoint,
            ),
            checkpoint_sha256=candidate_checkpoint,
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
            receipt=_receipt(
                tmp_path,
                "reproduction.json",
                kind="reproduction",
                checkpoint_sha256=candidate_checkpoint,
                producer="external-lab",
            ),
            independent_operator="external-lab",
            clean_environment_identity="container-sha256:abc",
            source_commit="c" * 40,
            dataset_manifest_sha256="d" * 64,
            checkpoint_sha256=candidate_checkpoint,
            metrics_reproduced_within_tolerance=True,
        ),
        family_gate_results=[
            FamilyGateEvidence(
                gate=name,
                receipt=_receipt(
                    tmp_path,
                    f"family-{index}.json",
                    kind="family_gate",
                    checkpoint_sha256=candidate_checkpoint,
                ),
                checkpoint_sha256=candidate_checkpoint,
                dataset_manifest_sha256="d" * 64,
                passed=True,
            )
            for index, name in enumerate(model_spec("hermes").capability_gates)
        ],
        documents={
            name: _document(tmp_path, name, candidate_checkpoint) for name in REQUIRED_DOCUMENTS
        },
    )


def test_complete_hash_bound_evidence_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(readiness_module, "_dataset_receipt_valid", lambda *_: True)
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


def test_hash_valid_receipt_with_wrong_semantics_fails_closed(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    checkpoint = candidate.candidate_checkpoint_sha256
    assert checkpoint is not None
    candidate.controls[0].receipt = _receipt(
        tmp_path,
        "wrong-kind.json",
        kind="safety",
        checkpoint_sha256=checkpoint,
    )
    report = evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "strong_controls" in report.blockers


def test_hash_valid_malformed_receipt_fails_closed(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    candidate.controls[0].receipt = _artifact(tmp_path, "malformed-receipt.json", "{bad-json")
    report = evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "strong_controls" in report.blockers


def test_hash_valid_receipt_from_wrong_source_fails_closed(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    candidate.source_commit = "f" * 40
    report = evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "three_seed_training" in report.blockers
    assert "strong_controls" in report.blockers
    assert "independent_reproduction" in report.blockers


def test_empty_portfolio_reports_every_family_and_gate(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.json"
    thresholds_path = Path(__file__).parents[1] / "configs/llm-readiness-v1.json"
    output_path = tmp_path / "report.json"
    evidence_path.write_text(
        ReadinessEvidenceSet(protocol_id=PROTOCOL_ID, candidates=[]).model_dump_json(),
        encoding="utf-8",
    )
    report = evaluate_portfolio_readiness(evidence_path, thresholds_path, output_path)
    assert report.total_families == 6
    assert report.ready_families == 0
    assert not report.all_families_ready
    assert all(len(item.gates) == 11 for item in report.family_reports)
    assert json.loads(output_path.read_text())["all_families_ready"] is False


def test_evidence_rejects_duplicate_candidate_ids(tmp_path: Path) -> None:
    first = _passing_candidate(tmp_path)
    second = first.model_copy(update={"family_id": "prometheus"}, deep=True)
    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        ReadinessEvidenceSet(protocol_id=PROTOCOL_ID, candidates=[first, second])


def test_evidence_requires_explicit_protocol_identity() -> None:
    with pytest.raises(ValueError, match="protocol_id"):
        ReadinessEvidenceSet.model_validate({"schema_version": 1, "candidates": []})


def test_duplicate_suite_and_checkpoint_mismatch_fail_closed(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    candidate.capabilities.append(candidate.capabilities[0].model_copy())
    candidate.controls[0].checkpoint_sha256 = "f" * 64
    report = evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "capability_evaluation" in report.blockers
    assert "strong_controls" in report.blockers


def test_duplicate_seed_and_undeclared_suite_fail_closed(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    duplicate_seed = candidate.training_runs[0].model_copy(deep=True)
    duplicate_seed.checkpoint_sha256 = "f" * 64
    duplicate_seed.receipt = _receipt(
        tmp_path,
        "duplicate-seed.json",
        kind="training",
        checkpoint_sha256=duplicate_seed.checkpoint_sha256,
    )
    candidate.training_runs.append(duplicate_seed)
    candidate.capabilities.append(
        candidate.capabilities[0].model_copy(
            update={"suite": "undeclared_suite"},
            deep=True,
        )
    )
    report = evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "three_seed_training" in report.blockers
    assert "capability_evaluation" in report.blockers


def test_documents_must_name_candidate_and_checkpoint(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    candidate.documents["model_card"] = _artifact(tmp_path, "unbound-card.md", "generic card")
    report = evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "release_documentation" in report.blockers


def test_documents_require_exact_set_and_structured_binding(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    checkpoint = candidate.candidate_checkpoint_sha256
    assert checkpoint is not None
    candidate.documents["extra"] = _document(tmp_path, "extra", checkpoint)
    report = evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "release_documentation" in report.blockers

    candidate = _passing_candidate(tmp_path)
    candidate.documents["model_card"] = _artifact(
        tmp_path,
        "malformed-header-card.md",
        f"{DOCUMENT_HEADER_PREFIX} {{bad-json {DOCUMENT_HEADER_SUFFIX}\n# Model card\n",
    )
    report = evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "release_documentation" in report.blockers

    candidate = _passing_candidate(tmp_path)
    header = ReleaseDocumentHeader(
        protocol_id=PROTOCOL_ID,
        document_type="model_card",
        candidate_id="another-candidate",
        source_commit="c" * 40,
        checkpoint_sha256=checkpoint,
        dataset_manifest_sha256="d" * 64,
    ).model_dump_json()
    candidate.documents["model_card"] = _artifact(
        tmp_path,
        "wrong-candidate-card.md",
        f"{DOCUMENT_HEADER_PREFIX} {header} {DOCUMENT_HEADER_SUFFIX}\n# Model card\n",
    )
    report = evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "release_documentation" in report.blockers


def test_artifact_cannot_escape_evidence_root(tmp_path: Path) -> None:
    evidence_root = tmp_path / "bundle"
    evidence_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    candidate = _passing_candidate(evidence_root)
    candidate.training_runs[0].receipt = ArtifactIdentity(
        path="../outside.json", sha256=hashlib.sha256(b"outside").hexdigest()
    )
    report = evaluate_family_readiness(
        candidate, ReadinessThresholds(), evidence_root=evidence_root
    )
    assert "three_seed_training" in report.blockers


def test_relaxed_threshold_policy_is_rejected(tmp_path: Path) -> None:
    evidence_path = tmp_path / "evidence.json"
    thresholds_path = tmp_path / "relaxed.json"
    evidence_path.write_text(
        ReadinessEvidenceSet(protocol_id=PROTOCOL_ID, candidates=[]).model_dump_json(),
        encoding="utf-8",
    )
    thresholds_path.write_text(
        ReadinessThresholds(minimum_training_records=1).model_dump_json(), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="frozen v1 policy"):
        evaluate_portfolio_readiness(evidence_path, thresholds_path, tmp_path / "report.json")


def test_portfolio_rejects_missing_symlink_and_input_overwrite(tmp_path: Path) -> None:
    thresholds_path = Path(__file__).parents[1] / "configs/llm-readiness-v1.json"
    with pytest.raises(ValueError, match="evidence input must be"):
        evaluate_portfolio_readiness(
            tmp_path / "missing.json",
            thresholds_path,
            tmp_path / "report.json",
        )

    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        ReadinessEvidenceSet(protocol_id=PROTOCOL_ID, candidates=[]).model_dump_json(),
        encoding="utf-8",
    )
    evidence_symlink = tmp_path / "evidence-link.json"
    evidence_symlink.symlink_to(evidence_path)
    with pytest.raises(ValueError, match="non-symlink"):
        evaluate_portfolio_readiness(evidence_symlink, thresholds_path, tmp_path / "report.json")

    with pytest.raises(ValueError, match="cannot overwrite"):
        evaluate_portfolio_readiness(evidence_path, thresholds_path, evidence_path)
