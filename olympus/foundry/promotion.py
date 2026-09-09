from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from olympus.core.schemas import StrictModel
from olympus.foundry.attestation import (
    PromotionEvidenceSubject,
    parse_strict_json,
    read_evidence_bytes,
    verify_promotion_attestations,
)
from olympus.foundry.data_pipeline import DatasetManifestV2, verify_dataset_manifest
from olympus.foundry.eval_suite import HeldOutEvaluation
from olympus.foundry.quantization import QuantizationReport


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class GateResult(StrictModel):
    gate: str
    passed: bool
    observed: str
    requirement: str


class PromotionReport(StrictModel):
    schema_version: int = 1
    requested_model_id: str
    checkpoint_sha256: str
    gates: list[GateResult]
    passed: bool
    status: str
    release_manifest_path: str | None
    blockers: list[str]


def _gate(name: str, passed: bool, observed: object, requirement: str) -> GateResult:
    return GateResult(
        gate=name,
        passed=passed,
        observed=str(observed),
        requirement=requirement,
    )


def _workflow_floor(evaluation: HeldOutEvaluation) -> float:
    return min(score.exact_match_rate for score in evaluation.workflow_scores)


def record_invalid_promotion_attempt(
    output_path: Path, *, requested_model_id: str, error_type: str
) -> Path:
    """Record an invalid attempt separately; never mistake a historical release for it."""
    path = output_path.parent / f"{output_path.name}.runs" / uuid4().hex / "invalid-input.json"
    _atomic_json(path, {
        "schema_version": 1,
        "requested_model_id": requested_model_id,
        "status": "INVALID_INPUT",
        "passed": False,
        "release_manifest_path": None,
        "error_type": error_type,
        "detail": "Promotion evidence is invalid or unreadable. No new decision was produced.",
    })
    return path


def _has_markdown_heading(document: str, heading: str) -> bool:
    target = heading.strip().casefold()
    return any(
        line.lstrip().startswith("#")
        and line.lstrip("# ").strip().casefold() == target
        for line in document.splitlines()
    )


def evaluate_promotion(
    *,
    requested_model_id: str,
    checkpoint_path: Path,
    dataset_manifest_path: Path,
    evaluation_path: Path,
    quantization_report_path: Path,
    model_card_path: Path,
    output_path: Path,
    approved_base_license: str,
    serving_verification: dict[str, Any] | None = None,
    attestation_bundle_path: Path | None = None,
    attestation_trust_store_path: Path | None = None,
) -> PromotionReport:
    inputs = [
        checkpoint_path, dataset_manifest_path, evaluation_path,
        quantization_report_path, model_card_path,
    ]
    inputs.extend(path for path in (attestation_bundle_path, attestation_trust_store_path) if path)
    if any(output_path.resolve() == path.resolve() for path in inputs):
        raise ValueError("promotion output cannot overwrite an input")
    checkpoint_sha = _sha256(checkpoint_path)
    manifest: DatasetManifestV2 = verify_dataset_manifest(dataset_manifest_path)
    manifest_sha = manifest.manifest_sha256
    if manifest_sha is None:
        raise ValueError("verified dataset manifest is missing its content hash")
    evaluation_bytes = read_evidence_bytes(evaluation_path)
    quantization_bytes = read_evidence_bytes(quantization_report_path)
    evaluation = HeldOutEvaluation.model_validate(parse_strict_json(evaluation_bytes))
    quantization = QuantizationReport.model_validate(parse_strict_json(quantization_bytes))
    train = next(split for split in manifest.splits if split.name == "train")
    test = next(split for split in manifest.splits if split.name == "test")
    minimum_category_records = min(test.categories.values())
    card_bytes = (
        read_evidence_bytes(model_card_path, limit=1_000_000)
        if model_card_path.is_file() else b""
    )
    model_card = card_bytes.decode("utf-8")
    model_card_sha = hashlib.sha256(card_bytes).hexdigest()
    serving = serving_verification or {}
    allowed_license = approved_base_license in {"Apache-2.0", "MIT", "CC-BY-4.0"}
    quantized_artifact = Path(quantization.artifact_path)
    if not quantized_artifact.is_absolute():
        quantized_artifact = quantization_report_path.parent / quantized_artifact
    if output_path.resolve() == quantized_artifact.resolve():
        raise ValueError("promotion output cannot overwrite a quantized artifact")
    quantized_artifact_matches = (
        quantized_artifact.is_file()
        and _sha256(quantized_artifact) == quantization.artifact_sha256
    )
    evidence_subject = PromotionEvidenceSubject(
        requested_model_id=requested_model_id,
        checkpoint_sha256=checkpoint_sha,
        dataset_manifest_sha256=manifest_sha,
        evaluation_sha256=hashlib.sha256(evaluation_bytes).hexdigest(),
        quantization_report_sha256=hashlib.sha256(quantization_bytes).hexdigest(),
        model_card_sha256=model_card_sha,
        serving_verification_sha256=_canonical_sha256(serving),
        approved_base_license=approved_base_license,
    )
    attestation = verify_promotion_attestations(
        bundle_path=attestation_bundle_path,
        trust_store_path=attestation_trust_store_path,
        expected_subject=evidence_subject,
    )
    trusted_policy_sha = os.environ.get("OLYMPUS_PROMOTION_TRUST_SHA256", "")
    policy_pinned = (
        re.fullmatch(r"[0-9a-f]{64}", trusted_policy_sha) is not None
        and trusted_policy_sha == attestation.trust_store_sha256
    )
    gates = [
        _gate(
            "checkpoint_identity",
            evaluation.checkpoint_sha256 == checkpoint_sha
            and quantization.source_checkpoint_sha256 == checkpoint_sha,
            checkpoint_sha,
            "Evaluation and quantization evidence must reference the exact checkpoint hash.",
        ),
        _gate(
            "dataset_identity",
            manifest_sha == evaluation.dataset_manifest_sha256
            and manifest_sha == quantization.dataset_manifest_sha256
            and evaluation.test_records == test.records
            and {item.category: item.records for item in evaluation.category_results}
            == test.categories,
            manifest_sha,
            "Evaluation must reference the exact immutable dataset manifest.",
        ),
        _gate(
            "base_license",
            allowed_license,
            approved_base_license,
            "Base-model rights must be explicitly approved for training and distribution.",
        ),
        _gate(
            "training_scale",
            train.records >= 10_000,
            train.records,
            "At least 10,000 reviewed SFT training records are required for Hermes Alpha.",
        ),
        _gate(
            "held_out_scale",
            test.records >= 1_200 and minimum_category_records >= 100,
            f"total={test.records}, minimum_per_category={minimum_category_records}",
            "At least 1,200 held-out records and 100 per category are required.",
        ),
        _gate(
            "loss_and_regression",
            evaluation.overall_candidate_loss < evaluation.overall_baseline_loss
            and evaluation.regression_count == 0,
            (
                f"baseline={evaluation.overall_baseline_loss:.6f}, "
                f"candidate={evaluation.overall_candidate_loss:.6f}, "
                f"regressions={evaluation.regression_count}"
            ),
            "Candidate loss must improve and no capability category may regress over 5%.",
        ),
        _gate(
            "task_quality",
            evaluation.exact_match_rate >= 0.80
            and evaluation.format_compliance_rate >= 0.95
            and _workflow_floor(evaluation) >= 0.75,
            (
                f"exact={evaluation.exact_match_rate:.3f}, "
                f"format={evaluation.format_compliance_rate:.3f}, "
                f"workflow_floor={_workflow_floor(evaluation):.3f}"
            ),
            "Exact match >=80%, format compliance >=95%, workflow floor >=75%.",
        ),
        _gate(
            "quantized_artifact_identity",
            quantized_artifact_matches,
            (
                f"path={quantized_artifact}, "
                f"reported_sha256={quantization.artifact_sha256}"
            ),
            "The quantized artifact must exist and match the hash in its quality report.",
        ),
        _gate(
            "quantization_quality",
            abs(quantization.loss_change_fraction) <= 0.02
            and quantization.context_limit_enforced
            and quantization.tool_exact_match_rate >= 0.75,
            (
                f"loss_change={quantization.loss_change_fraction:.4f}, "
                f"tool_exact={quantization.tool_exact_match_rate:.3f}"
            ),
            "Quantized loss change <=2%, enforced context, and tool exact match >=75%.",
        ),
        _gate(
            "serving_reproducibility",
            serving.get("passed") is True
            and serving.get("checkpoint_sha256") == checkpoint_sha,
            serving or "missing",
            "Fresh-process API, CLI, and web serving must pass against the exact checkpoint.",
        ),
        _gate(
            "model_card",
            bool(model_card)
            and checkpoint_sha in model_card
            and approved_base_license in model_card
            and _has_markdown_heading(model_card, "License")
            and _has_markdown_heading(model_card, "Intended use")
            and _has_markdown_heading(model_card, "Limitations"),
            str(model_card_path),
            "Published card must name the exact hash, license, intended use, and limitations.",
        ),
        _gate(
            "independent_evidence_authority",
            attestation.passed and policy_pinned,
            (
                f"verified_kinds={attestation.verified_kinds}, "
                f"verified_issuers={attestation.verified_issuers}, "
                f"operator_policy_pinned={policy_pinned}, "
                f"issues={attestation.issues}"
            ),
            (
                "Evaluation, quantization, license review, and fresh-process serving must each "
                "have a live, Ed25519-signed attestation over the exact candidate evidence from "
                "operator-trusted independent issuers."
            ),
        ),
    ]
    passed = all(gate.passed for gate in gates)
    run_directory = output_path.parent / f"{output_path.name}.runs" / uuid4().hex
    release_manifest = run_directory / "release-manifest.json"
    release_manifest_path = str(release_manifest.resolve()) if passed else None
    blockers = [gate.gate for gate in gates if not gate.passed]
    report = PromotionReport(
        requested_model_id=requested_model_id,
        checkpoint_sha256=checkpoint_sha,
        gates=gates,
        passed=passed,
        status="PROMOTED" if passed else "NOT_PROMOTED",
        release_manifest_path=release_manifest_path,
        blockers=blockers,
    )
    run_report = run_directory / "promotion.json"
    _atomic_json(run_report, report.model_dump(mode="json"))
    if passed:
        _atomic_json(
            release_manifest,
            {
                "schema_version": 1,
                "model_id": requested_model_id,
                "checkpoint_sha256": checkpoint_sha,
                "dataset_manifest_sha256": manifest_sha,
                "evaluation_sha256": evidence_subject.evaluation_sha256,
                "quantization_report_sha256": evidence_subject.quantization_report_sha256,
                "model_card_sha256": model_card_sha,
                "base_license": approved_base_license,
                "serving_verification_sha256": evidence_subject.serving_verification_sha256,
                "attestation_bundle_sha256": attestation.bundle_sha256,
                "attestation_trust_store_sha256": attestation.trust_store_sha256,
                "attestation_issuers": attestation.verified_issuers,
                "promotion_report_sha256": _sha256(run_report),
            },
        )
    _atomic_json(output_path, report.model_dump(mode="json"))
    return report
