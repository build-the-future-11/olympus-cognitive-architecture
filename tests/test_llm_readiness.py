from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import olympus.foundry.readiness as readiness_module
from olympus.foundry.readiness import (
    DOCUMENT_HEADER_PREFIX,
    DOCUMENT_HEADER_SUFFIX,
    PROTOCOL_ID,
    REQUIRED_CONTROLS,
    REQUIRED_DOCUMENT_SECTIONS,
    REQUIRED_DOCUMENTS,
    ArtifactIdentity,
    CandidateReadinessEvidence,
    ComparativeEvaluation,
    DatasetAttestationEnvelope,
    DatasetAttestationKind,
    DatasetEvidence,
    EvidenceReceiptEnvelope,
    FamilyGateEvidence,
    FamilyReadinessReport,
    PortfolioReadinessReport,
    ProductionEvidence,
    ReadinessEvidenceSet,
    ReadinessThresholds,
    ReceiptAuthorizationKind,
    ReceiptKind,
    ReceiptTrustPolicy,
    ReleaseDocumentHeader,
    ReproductionEvidence,
    SuiteEvaluation,
    TrainingRunEvidence,
    TrustedReceiptSigner,
    evaluate_portfolio_readiness,
    receipt_signature_payload,
)
from olympus.models.specifications import model_spec

_ALL_RECEIPT_KINDS: list[ReceiptAuthorizationKind] = [
    "training",
    "control",
    "capability",
    "safety",
    "production",
    "reproduction",
    "family_gate",
    "rights_review",
    "dataset_freeze",
]


def _private_key(producer: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(hashlib.sha256(producer.encode()).digest())


def _trusted_signer(
    producer: str,
    allowed_kinds: list[ReceiptAuthorizationKind] | None = None,
) -> TrustedReceiptSigner:
    public_key = _private_key(producer).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return TrustedReceiptSigner(
        producer=producer,
        key_id=hashlib.sha256(public_key).hexdigest(),
        public_key_base64=base64.b64encode(public_key).decode(),
        allowed_kinds=allowed_kinds or _ALL_RECEIPT_KINDS,
    )


_TEST_TRUST_POLICY = ReceiptTrustPolicy(
    signers=[
        _trusted_signer("primary-lab"),
        _trusted_signer("external-lab"),
        _trusted_signer("data-governance"),
    ]
)


def _sign_receipt(
    receipt: EvidenceReceiptEnvelope | DatasetAttestationEnvelope,
) -> EvidenceReceiptEnvelope | DatasetAttestationEnvelope:
    signature = _private_key(receipt.producer).sign(receipt_signature_payload(receipt))
    return receipt.model_copy(update={"signature_base64": base64.b64encode(signature).decode()})


def _evaluate_family_readiness(
    evidence: CandidateReadinessEvidence,
    thresholds: ReadinessThresholds,
    *,
    evidence_root: Path,
    trust_policy: ReceiptTrustPolicy = _TEST_TRUST_POLICY,
) -> FamilyReadinessReport:
    return readiness_module.evaluate_family_readiness(
        evidence,
        thresholds,
        evidence_root=evidence_root,
        trust_policy=trust_policy,
    )


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
    evidence_id: str,
    claim_sha256: str,
    producer: str = "primary-lab",
) -> ArtifactIdentity:
    signer = _trusted_signer(producer)
    receipt = EvidenceReceiptEnvelope(
        protocol_id=PROTOCOL_ID,
        kind=kind,
        candidate_id="hermes-test",
        checkpoint_sha256=checkpoint_sha256,
        dataset_manifest_sha256="d" * 64,
        producer=producer,
        source_commit="c" * 40,
        evidence_id=evidence_id,
        claim_sha256=claim_sha256,
        signer_key_id=signer.key_id,
        signature_base64=base64.b64encode(bytes(64)).decode(),
    )
    payload = _sign_receipt(receipt).model_dump_json()
    return _artifact(tmp_path, name, payload)


def _placeholder() -> ArtifactIdentity:
    return ArtifactIdentity(path="pending", sha256="0" * 64)


def _bind_receipt(
    tmp_path: Path,
    name: str,
    *,
    kind: ReceiptKind,
    evidence: (
        TrainingRunEvidence
        | ComparativeEvaluation
        | SuiteEvaluation
        | FamilyGateEvidence
        | ProductionEvidence
        | ReproductionEvidence
    ),
    evidence_id: str,
    producer: str = "primary-lab",
) -> ArtifactIdentity:
    return _receipt(
        tmp_path,
        name,
        kind=kind,
        checkpoint_sha256=evidence.checkpoint_sha256,
        evidence_id=evidence_id,
        claim_sha256=readiness_module._canonical_claim_sha256(evidence),
        producer=producer,
    )


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
        + "".join(
            f"\n## {section}\n\nVerified candidate-specific evidence.\n"
            for section in sorted(REQUIRED_DOCUMENT_SECTIONS.get(document_type, frozenset()))
        )
    )
    return _artifact(tmp_path, f"{document_type}.md", content)


def _dataset_attestation(
    tmp_path: Path,
    name: str,
    kind: DatasetAttestationKind,
) -> ArtifactIdentity:
    signer = _trusted_signer("data-governance")
    attestation = DatasetAttestationEnvelope(
        protocol_id=PROTOCOL_ID,
        kind=kind,
        dataset_manifest_sha256="d" * 64,
        producer="data-governance",
        source_commit="c" * 40,
        approved=True,
        signer_key_id=signer.key_id,
        signature_base64=base64.b64encode(bytes(64)).decode(),
    )
    payload = _sign_receipt(attestation).model_dump_json()
    return _artifact(tmp_path, name, payload)


def _passing_candidate(tmp_path: Path) -> CandidateReadinessEvidence:
    manifest = _artifact(tmp_path, "manifest.json")
    runs = []
    for seed in (1, 2, 3):
        run = TrainingRunEvidence(
            seed=seed,
            checkpoint_sha256=(str(seed) * 64)[:64],
            receipt=_placeholder(),
            dataset_manifest_sha256="d" * 64,
            base_model_sha256="a" * 64,
            training_config_sha256="b" * 64,
            completed=True,
            finite_metrics=True,
            protected_accesses_before_freeze=0,
        )
        run.receipt = _bind_receipt(
            tmp_path,
            f"training-{seed}.json",
            kind="training",
            evidence=run,
            evidence_id=f"seed:{seed}",
        )
        runs.append(run)
    candidate_checkpoint = runs[0].checkpoint_sha256
    capability_policy, safety_policy, production_policy, _ = readiness_module._load_policy_bundle(
        Path(__file__).parents[1] / "configs", ReadinessThresholds()
    )
    controls = []
    for name in REQUIRED_CONTROLS:
        control = ComparativeEvaluation(
            control=name,
            receipt=_placeholder(),
            checkpoint_sha256=candidate_checkpoint,
            dataset_manifest_sha256="d" * 64,
            effect=0.04,
            confidence_interval_low=0.03,
            confidence_interval_high=0.05,
            evaluated_examples=capability_policy.minimum_examples_per_suite,
            bootstrap_resamples=capability_policy.statistics.paired_bootstrap_resamples,
            confidence_level=capability_policy.statistics.confidence_level,
            multiple_comparison_correction=(
                capability_policy.statistics.multiple_comparison_correction
            ),
            worst_category_delta=0.0,
            no_material_regressions=True,
        )
        control.receipt = _bind_receipt(
            tmp_path,
            f"control-{name}.json",
            kind="control",
            evidence=control,
            evidence_id=name,
        )
        controls.append(control)
    capabilities = []
    for capability_spec in capability_policy.suites:
        capability = SuiteEvaluation(
            suite=capability_spec.id,
            receipt=_placeholder(),
            checkpoint_sha256=candidate_checkpoint,
            dataset_manifest_sha256="d" * 64,
            metric=capability_spec.metric,
            evaluated_cases=capability_policy.minimum_examples_per_suite,
            score=0.9,
            passed=True,
            critical_failures=0,
        )
        capability.receipt = _bind_receipt(
            tmp_path,
            f"capability-{capability_spec.id}.json",
            kind="capability",
            evidence=capability,
            evidence_id=capability_spec.id,
        )
        capabilities.append(capability)
    safety = []
    for safety_spec in safety_policy.suites:
        safety_result = SuiteEvaluation(
            suite=safety_spec.id,
            receipt=_placeholder(),
            checkpoint_sha256=candidate_checkpoint,
            dataset_manifest_sha256="d" * 64,
            metric=safety_spec.pass_rule,
            evaluated_cases=safety_spec.minimum_cases,
            case_groups=safety_spec.required_case_groups,
            score=1.0,
            passed=True,
            critical_failures=0,
        )
        safety_result.receipt = _bind_receipt(
            tmp_path,
            f"safety-{safety_spec.id}.json",
            kind="safety",
            evidence=safety_result,
            evidence_id=safety_spec.id,
        )
        safety.append(safety_result)
    production = ProductionEvidence(
        receipt=_placeholder(),
        checkpoint_sha256=candidate_checkpoint,
        p50_latency_ms=100,
        p95_latency_ms=500,
        p99_latency_ms=900,
        throughput_requests_per_second=10,
        error_rate=0,
        cost_per_million_tokens_usd=10,
        soak_hours=24,
        environment_identity="staging-image-sha256:abc",
        request_count=production_policy.workload.minimum_requests,
        traffic_phases=production_policy.workload.traffic_phases,
        request_classes=production_policy.workload.request_classes,
        failure_injections=production_policy.failure_injections,
        uncaught_process_crashes=0,
        lost_accepted_requests=0,
        failure_injection_passed=True,
        queue_cancel_retry_backpressure_passed=True,
        rollback_passed=True,
        zero_downtime_upgrade_passed=True,
        drift_failure_cost_monitoring_passed=True,
    )
    production.receipt = _bind_receipt(
        tmp_path,
        "production.json",
        kind="production",
        evidence=production,
        evidence_id="production",
    )
    reproduction = ReproductionEvidence(
        receipt=_placeholder(),
        independent_operator="external-lab",
        clean_environment_identity="container-sha256:abc",
        source_commit="c" * 40,
        dataset_manifest_sha256="d" * 64,
        checkpoint_sha256=candidate_checkpoint,
        metrics_reproduced_within_tolerance=True,
    )
    reproduction.receipt = _bind_receipt(
        tmp_path,
        "reproduction.json",
        kind="reproduction",
        evidence=reproduction,
        evidence_id="reproduction",
        producer="external-lab",
    )
    family_gate_results = []
    for index, name in enumerate(model_spec("hermes").capability_gates):
        family_gate = FamilyGateEvidence(
            gate=name,
            receipt=_placeholder(),
            checkpoint_sha256=candidate_checkpoint,
            dataset_manifest_sha256="d" * 64,
            passed=True,
        )
        family_gate.receipt = _bind_receipt(
            tmp_path,
            f"family-{index}.json",
            kind="family_gate",
            evidence=family_gate,
            evidence_id=name,
        )
        family_gate_results.append(family_gate)
    dataset = DatasetEvidence(
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
        rights_receipt=_dataset_attestation(
            tmp_path, "dataset-rights.json", "rights_review"
        ),
        freeze_receipt=_dataset_attestation(
            tmp_path, "dataset-freeze.json", "dataset_freeze"
        ),
    )
    return CandidateReadinessEvidence(
        family_id="hermes",
        candidate_id="hermes-test",
        source_commit="c" * 40,
        candidate_checkpoint_sha256=candidate_checkpoint,
        training_operator="primary-lab",
        dataset=dataset,
        training_runs=runs,
        controls=controls,
        capabilities=capabilities,
        safety=safety,
        production=production,
        reproduction=reproduction,
        family_gate_results=family_gate_results,
        documents={
            name: _document(tmp_path, name, candidate_checkpoint) for name in REQUIRED_DOCUMENTS
        },
    )


def test_complete_hash_bound_evidence_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(readiness_module, "_dataset_receipt_valid", lambda *_: True)
    report = _evaluate_family_readiness(
        _passing_candidate(tmp_path), ReadinessThresholds(), evidence_root=tmp_path
    )
    assert report.ready_for_large_scale
    assert report.blockers == []
    assert all(gate.passed for gate in report.gates)


def test_tamper_and_weak_effect_fail_closed(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    (tmp_path / "training-1.json").write_text("tampered", encoding="utf-8")
    candidate.controls[0].confidence_interval_low = 0.019
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
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
        evidence_id=candidate.controls[0].control,
        claim_sha256=readiness_module._canonical_claim_sha256(candidate.controls[0]),
    )
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "strong_controls" in report.blockers


def test_forged_signature_fails_even_when_artifact_hash_matches(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    receipt_path = tmp_path / candidate.controls[0].receipt.path
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    signature = bytearray(base64.b64decode(payload["signature_base64"]))
    signature[0] ^= 1
    payload["signature_base64"] = base64.b64encode(signature).decode()
    candidate.controls[0].receipt = _artifact(
        tmp_path,
        "forged-control.json",
        json.dumps(payload, sort_keys=True),
    )
    report = _evaluate_family_readiness(
        candidate, ReadinessThresholds(), evidence_root=tmp_path
    )
    assert "strong_controls" in report.blockers


def test_signer_must_be_authorized_for_receipt_kind(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    restricted_policy = ReceiptTrustPolicy(
        signers=[
            _trusted_signer("primary-lab", ["training"]),
            _trusted_signer("external-lab"),
            _trusted_signer("data-governance"),
        ]
    )
    report = _evaluate_family_readiness(
        candidate,
        ReadinessThresholds(),
        evidence_root=tmp_path,
        trust_policy=restricted_policy,
    )
    assert "strong_controls" in report.blockers
    assert "capability_evaluation" in report.blockers


def test_trust_policy_rejects_key_reuse_across_producers() -> None:
    signer = _trusted_signer("primary-lab")
    reused = signer.model_copy(update={"producer": "impostor-lab"})
    with pytest.raises(ValueError, match="cannot represent multiple producer"):
        ReceiptTrustPolicy(signers=[signer, reused])


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"public_key_base64": "not-base64"}, "canonical base64"),
        (
            {"public_key_base64": base64.b64encode(bytes(31)).decode()},
            "32-byte Ed25519",
        ),
        ({"key_id": "0" * 64}, "public-key SHA-256"),
        ({"allowed_kinds": ["training", "training"]}, "must be unique"),
    ],
)
def test_trust_policy_rejects_malformed_signer_material(
    update: dict[str, object], message: str
) -> None:
    payload = _trusted_signer("primary-lab").model_dump()
    payload.update(update)
    with pytest.raises(ValueError, match=message):
        TrustedReceiptSigner.model_validate(payload)


def test_signed_receipt_cannot_change_producer_identity(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    receipt_path = tmp_path / candidate.controls[0].receipt.path
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["producer"] = "external-lab"
    candidate.controls[0].receipt = _artifact(
        tmp_path,
        "producer-substitution.json",
        json.dumps(payload, sort_keys=True),
    )
    report = _evaluate_family_readiness(
        candidate, ReadinessThresholds(), evidence_root=tmp_path
    )
    assert "strong_controls" in report.blockers


def test_hash_valid_malformed_receipt_fails_closed(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    candidate.controls[0].receipt = _artifact(tmp_path, "malformed-receipt.json", "{bad-json")
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "strong_controls" in report.blockers


def test_receipt_binds_exact_claim_and_cannot_be_reused(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    candidate.controls[0].effect = 0.045
    candidate.controls[1].receipt = candidate.controls[0].receipt
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "strong_controls" in report.blockers


def test_dataset_quality_requires_rights_and_freeze_attestations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(readiness_module, "_dataset_receipt_valid", lambda *_: True)
    candidate = _passing_candidate(tmp_path)
    assert candidate.dataset is not None
    candidate.dataset.freeze_receipt = None
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "data_receipt_integrity" not in report.blockers
    assert "data_quality" in report.blockers


def test_frozen_suite_and_production_workload_details_are_enforced(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    candidate.controls[0].bootstrap_resamples = 10
    candidate.controls[0].receipt = _bind_receipt(
        tmp_path,
        "control-underpowered.json",
        kind="control",
        evidence=candidate.controls[0],
        evidence_id=candidate.controls[0].control,
    )
    candidate.capabilities[0].evaluated_cases = 1
    candidate.capabilities[0].receipt = _bind_receipt(
        tmp_path,
        "capability-too-small.json",
        kind="capability",
        evidence=candidate.capabilities[0],
        evidence_id=candidate.capabilities[0].suite,
    )
    candidate.safety[0].case_groups = ["invented"]
    candidate.safety[0].receipt = _bind_receipt(
        tmp_path,
        "safety-wrong-groups.json",
        kind="safety",
        evidence=candidate.safety[0],
        evidence_id=candidate.safety[0].suite,
    )
    assert candidate.production is not None
    candidate.production.request_count = 99_999
    candidate.production.receipt = _bind_receipt(
        tmp_path,
        "production-too-small.json",
        kind="production",
        evidence=candidate.production,
        evidence_id="production",
    )
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "strong_controls" in report.blockers
    assert "capability_evaluation" in report.blockers
    assert "safety_evaluation" in report.blockers
    assert "production_qualification" in report.blockers


def test_report_rejects_inconsistent_ready_count(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    generated = tmp_path / "report.json"
    evaluate_portfolio_readiness(
        root / "research/llm_readiness_v1/CURRENT_EVIDENCE.json",
        root / "configs/llm-readiness-v1.json",
        generated,
    )
    payload = json.loads(generated.read_text())
    payload["ready_families"] = 1
    with pytest.raises(ValueError, match="ready count"):
        PortfolioReadinessReport.model_validate(payload)


def test_hash_valid_receipt_from_wrong_source_fails_closed(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    candidate.source_commit = "f" * 40
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
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
    trust_path = Path(__file__).parents[1] / "configs/llm-readiness-trust-v1.json"
    assert report.trust_policy_sha256 == hashlib.sha256(trust_path.read_bytes()).hexdigest()


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
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
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
        evidence_id=f"seed:{duplicate_seed.seed}",
        claim_sha256=readiness_module._canonical_claim_sha256(duplicate_seed),
    )
    candidate.training_runs.append(duplicate_seed)
    candidate.capabilities.append(
        candidate.capabilities[0].model_copy(
            update={"suite": "undeclared_suite"},
            deep=True,
        )
    )
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "three_seed_training" in report.blockers
    assert "capability_evaluation" in report.blockers


def test_documents_must_name_candidate_and_checkpoint(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    candidate.documents["model_card"] = _artifact(tmp_path, "unbound-card.md", "generic card")
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "release_documentation" in report.blockers


def test_documents_require_exact_set_and_structured_binding(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    checkpoint = candidate.candidate_checkpoint_sha256
    assert checkpoint is not None
    candidate.documents["extra"] = _document(tmp_path, "extra", checkpoint)
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "release_documentation" in report.blockers

    candidate = _passing_candidate(tmp_path)
    header = ReleaseDocumentHeader(
        protocol_id=PROTOCOL_ID,
        document_type="model_card",
        candidate_id="hermes-test",
        source_commit="c" * 40,
        checkpoint_sha256=checkpoint,
        dataset_manifest_sha256="d" * 64,
    ).model_dump_json()
    candidate.documents["model_card"] = _artifact(
        tmp_path,
        "empty-model-card.md",
        f"{DOCUMENT_HEADER_PREFIX} {header} {DOCUMENT_HEADER_SUFFIX}\n# Model card\n",
    )
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "release_documentation" in report.blockers

    candidate = _passing_candidate(tmp_path)
    candidate.documents["model_card"] = _artifact(
        tmp_path,
        "malformed-header-card.md",
        f"{DOCUMENT_HEADER_PREFIX} {{bad-json {DOCUMENT_HEADER_SUFFIX}\n# Model card\n",
    )
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
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
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
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
    report = _evaluate_family_readiness(
        candidate, ReadinessThresholds(), evidence_root=evidence_root
    )
    assert "three_seed_training" in report.blockers


def test_artifact_parent_symlink_is_rejected(tmp_path: Path) -> None:
    candidate = _passing_candidate(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    source = tmp_path / candidate.controls[0].receipt.path
    copied = real / "control.json"
    copied.write_bytes(source.read_bytes())
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    candidate.controls[0].receipt = ArtifactIdentity(
        path="linked/control.json",
        sha256=hashlib.sha256(copied.read_bytes()).hexdigest(),
    )
    report = _evaluate_family_readiness(candidate, ReadinessThresholds(), evidence_root=tmp_path)
    assert "strong_controls" in report.blockers


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


def test_tampered_frozen_suite_policy_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    for filename in (
        "llm-readiness-v1.json",
        "llm-readiness-capability-v1.json",
        "llm-readiness-safety-v1.json",
        "llm-readiness-production-v1.json",
    ):
        (tmp_path / filename).write_bytes((root / "configs" / filename).read_bytes())
    capability_path = tmp_path / "llm-readiness-capability-v1.json"
    capability_path.write_bytes(capability_path.read_bytes() + b"\n")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        ReadinessEvidenceSet(protocol_id=PROTOCOL_ID, candidates=[]).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="policy hash mismatch"):
        evaluate_portfolio_readiness(
            evidence_path,
            tmp_path / "llm-readiness-v1.json",
            tmp_path / "report.json",
        )


def test_portfolio_hashes_the_exact_evidence_bytes_it_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_path = tmp_path / "evidence.json"
    original_payload = (
        ReadinessEvidenceSet(protocol_id=PROTOCOL_ID, candidates=[]).model_dump_json().encode()
    )
    evidence_path.write_bytes(original_payload)
    original_read_bytes = Path.read_bytes
    reads = 0

    def changing_read(path: Path) -> bytes:
        nonlocal reads
        payload = original_read_bytes(path)
        if path == evidence_path:
            reads += 1
            path.write_bytes(payload + b"\n")
        return payload

    monkeypatch.setattr(Path, "read_bytes", changing_read)
    report = evaluate_portfolio_readiness(
        evidence_path,
        Path(__file__).parents[1] / "configs/llm-readiness-v1.json",
        tmp_path / "report.json",
    )
    assert reads == 1
    assert report.evidence_sha256 == hashlib.sha256(original_payload).hexdigest()


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
