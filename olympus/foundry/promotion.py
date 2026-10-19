from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from olympus.core.schemas import StrictModel
from olympus.foundry.data_pipeline import (
    REQUIRED_CATEGORIES,
    DatasetManifestV2,
    SplitDescriptor,
    verify_dataset_manifest,
)
from olympus.foundry.eval_suite import HeldOutEvaluation
from olympus.foundry.quantization import QuantizationReport

# This is the contract of olympus-held-out-capabilities-v1, matching the producer
# in eval_suite._workflow_scores. Changes to its population require a new suite.
_WORKFLOW_CATEGORIES = {
    "tool_workflow": {"tool_use", "agent_behavior"},
    "result_comparison": {"reasoning", "self_correction", "research"},
    "multi_step_planning": {"planning", "safety"},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
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
    return min((score.exact_match_rate for score in evaluation.workflow_scores), default=0.0)


def _close(left: float, right: float) -> bool:
    """Allow serialization rounding, without changing any scientific threshold."""
    return (
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
    )


def _nonnegative_finite(*values: float) -> bool:
    return all(math.isfinite(value) and value >= 0.0 for value in values)


def _evaluation_integrity(evaluation: HeldOutEvaluation, test: SplitDescriptor) -> list[str]:
    issues: list[str] = []
    if evaluation.schema_version != 1 or evaluation.suite != "olympus-held-out-capabilities-v1":
        issues.append("unsupported evaluation schema or suite")
    if not evaluation.baseline_identity.strip():
        issues.append("evaluation must identify its frozen baseline")
    if evaluation.test_records != test.records:
        issues.append("evaluated record count does not match the verified test split")
    categories = [result.category for result in evaluation.category_results]
    if (
        len(categories) != len(set(categories))
        or set(categories) != set(test.categories)
        or set(test.categories) != set(REQUIRED_CATEGORIES)
    ):
        issues.append("evaluation must contain every required category exactly once")
    regressions = 0
    for result in evaluation.category_results:
        if result.records != test.categories.get(result.category):
            issues.append(f"category record count mismatch: {result.category}")
        if not _nonnegative_finite(result.baseline_loss, result.candidate_loss):
            issues.append(f"invalid category loss: {result.category}")
            continue
        change = (result.candidate_loss - result.baseline_loss) / max(result.baseline_loss, 1e-12)
        regressed = change > 0.05
        regressions += regressed
        if not _close(result.loss_change_fraction, change) or result.regressed != regressed:
            issues.append(f"inconsistent category regression evidence: {result.category}")
    if sum(result.records for result in evaluation.category_results) != test.records:
        issues.append("category record totals do not cover the verified test split")
    if evaluation.regression_count != regressions:
        issues.append("regression count does not match measured category losses")
    for label, loss, perplexity in (
        (
            "baseline",
            evaluation.overall_baseline_loss,
            evaluation.overall_baseline_perplexity,
        ),
        (
            "candidate",
            evaluation.overall_candidate_loss,
            evaluation.overall_candidate_perplexity,
        ),
    ):
        if not _nonnegative_finite(loss) or not _close(perplexity, math.exp(min(loss, 20.0))):
            issues.append(f"invalid or inconsistent overall {label} loss/perplexity")
    if not _nonnegative_finite(
        evaluation.exact_match_rate,
        evaluation.format_compliance_rate,
        evaluation.median_generation_latency_ms,
        evaluation.generated_bytes_per_second,
    ):
        issues.append("evaluation contains a nonfinite or negative measurement")
    names = [score.name for score in evaluation.workflow_scores]
    if len(names) != len(set(names)) or set(names) != set(_WORKFLOW_CATEGORIES):
        issues.append("evaluation must contain every required workflow exactly once")
    for score in evaluation.workflow_scores:
        expected_categories = _WORKFLOW_CATEGORIES.get(score.name)
        if expected_categories is None:
            continue
        if (
            len(score.categories) != len(set(score.categories))
            or set(score.categories) != expected_categories
            or score.examples != sum(test.categories.get(name, 0) for name in expected_categories)
        ):
            issues.append(f"workflow population does not match its definition: {score.name}")
        if not _nonnegative_finite(score.exact_match_rate, score.nonempty_rate):
            issues.append(f"invalid workflow measurement: {score.name}")
    smoke_passed = (
        evaluation.overall_candidate_loss < evaluation.overall_baseline_loss
        and regressions == 0
        and evaluation.format_compliance_rate >= 0.8
    )
    if evaluation.passed_smoke_quality_gate != smoke_passed:
        issues.append("smoke quality flag contradicts its recorded measurements")
    return issues


def _quantization_integrity(report: QuantizationReport, evaluation: HeldOutEvaluation) -> list[str]:
    issues: list[str] = []
    if report.schema_version != 1 or report.format != f"olympus-symmetric-int{report.bits}-v1":
        issues.append("unsupported quantization schema or format")
    if report.max_context_tokens <= 0:
        issues.append("quantization context capacity must be positive")
    if report.loss_normalization != evaluation.loss_normalization:
        issues.append("quantization and evaluation use different loss normalization")
    if not _nonnegative_finite(
        report.source_loss,
        report.quantized_loss,
        report.source_latency_ms,
        report.quantized_latency_ms,
        report.startup_latency_ms,
        report.tool_exact_match_rate,
    ):
        issues.append("quantization contains a nonfinite or negative measurement")
        return issues
    # The two producers repeat a float32 CPU measurement. This consistency
    # tolerance is separate from the unchanged 2% quantization-quality bound.
    if not math.isclose(
        report.source_loss, evaluation.overall_candidate_loss, rel_tol=1e-6, abs_tol=1e-8
    ):
        issues.append("quantization source loss does not match the evaluated checkpoint")
    change = (report.quantized_loss - report.source_loss) / max(report.source_loss, 1e-12)
    if not _close(report.loss_change_fraction, change):
        issues.append("quantization loss change contradicts its measured losses")
    if not _close(
        report.size_reduction_fraction, 1.0 - report.quantized_bytes / report.float_weights_bytes
    ):
        issues.append("quantization size reduction contradicts its recorded byte counts")
    quality_passed = abs(change) <= 0.02 and report.context_limit_enforced
    if report.passed_quality_gate != quality_passed:
        issues.append("quantization quality flag contradicts loss or context evidence")
    return issues


def _quantized_artifact_identity(
    report: QuantizationReport, path: Path
) -> tuple[GateResult, str | None]:
    requirement = "The quantized artifact must exist and match its recorded SHA-256 and byte count."
    try:
        payload = path.read_bytes()
    except OSError as error:
        return (
            _gate(
                "quantized_artifact_identity", False, f"unreadable artifact: {error}", requirement
            ),
            None,
        )
    digest = hashlib.sha256(payload).hexdigest()
    return (
        _gate(
            "quantized_artifact_identity",
            digest == report.artifact_sha256 and len(payload) == report.quantized_bytes,
            f"sha256={digest}, bytes={len(payload)}",
            requirement,
        ),
        digest,
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
) -> PromotionReport:
    """Check completeness, arithmetic, and byte identities of supplied evidence.

    These checks do not independently reproduce the submitted capability,
    serving, or rights evidence. Their existing proof gates remain mandatory.
    """
    checkpoint_sha = _sha256(checkpoint_path)
    manifest: DatasetManifestV2 = verify_dataset_manifest(dataset_manifest_path)
    evaluation_bytes = evaluation_path.read_bytes()
    quantization_bytes = quantization_report_path.read_bytes()
    evaluation = HeldOutEvaluation.model_validate_json(evaluation_bytes)
    quantization = QuantizationReport.model_validate_json(quantization_bytes)
    train = next(split for split in manifest.splits if split.name == "train")
    test = next(split for split in manifest.splits if split.name == "test")
    minimum_category_records = min(test.categories.values(), default=0)
    model_card_bytes = model_card_path.read_bytes() if model_card_path.is_file() else b""
    model_card = model_card_bytes.decode("utf-8")
    serving = dict(serving_verification or {})
    quantized_path = Path(quantization.artifact_path)
    if not quantized_path.is_absolute():
        quantized_path = quantization_report_path.parent / quantized_path
    release_manifest = output_path.with_name("release-manifest.json")
    inputs = {
        path.resolve()
        for path in (
            checkpoint_path,
            dataset_manifest_path,
            evaluation_path,
            quantization_report_path,
            model_card_path,
            quantized_path,
        )
    }
    if (
        output_path.resolve() == release_manifest.resolve()
        or output_path.resolve() in inputs
        or release_manifest.resolve() in inputs
    ):
        raise ValueError("promotion outputs must not replace input evidence or each other")
    evaluation_issues = _evaluation_integrity(evaluation, test)
    quantization_issues = _quantization_integrity(quantization, evaluation)
    artifact_gate, artifact_sha = _quantized_artifact_identity(quantization, quantized_path)
    workflow_floor = _workflow_floor(evaluation)
    allowed_license = approved_base_license in {"Apache-2.0", "MIT", "CC-BY-4.0"}
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
            manifest.manifest_sha256 == evaluation.dataset_manifest_sha256,
            manifest.manifest_sha256,
            "Evaluation must reference the exact immutable dataset manifest.",
        ),
        _gate(
            "evaluation_integrity",
            not evaluation_issues,
            "; ".join(evaluation_issues) or "complete, consistent evaluation evidence",
            "Required categories, workflows, counts, and derived metrics must match "
            "the verified test split.",
        ),
        _gate(
            "quantization_integrity",
            not quantization_issues,
            "; ".join(quantization_issues) or "consistent quantization measurements",
            "Quantization flags and derived metrics must match finite measurements "
            "of the evaluated checkpoint.",
        ),
        artifact_gate,
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
            and workflow_floor >= 0.75,
            (
                f"exact={evaluation.exact_match_rate:.3f}, "
                f"format={evaluation.format_compliance_rate:.3f}, "
                f"workflow_floor={workflow_floor:.3f}"
            ),
            "Exact match >=80%, format compliance >=95%, workflow floor >=75%.",
        ),
        _gate(
            "quantization_quality",
            not quantization_issues
            and quantization.passed_quality_gate
            and abs(quantization.loss_change_fraction) <= 0.02
            and quantization.context_limit_enforced
            and quantization.max_context_tokens > 0
            and quantization.tool_exact_match_rate >= 0.75,
            (
                f"loss_change={quantization.loss_change_fraction:.4f}, "
                f"tool_exact={quantization.tool_exact_match_rate:.3f}"
            ),
            "Absolute quantized loss change <=2%, positive enforced context, "
            "and tool exact match >=75%.",
        ),
        _gate(
            "serving_reproducibility",
            serving.get("passed") is True and serving.get("checkpoint_sha256") == checkpoint_sha,
            serving or "missing",
            "Fresh-process API, CLI, and web serving must pass against the exact checkpoint.",
        ),
        _gate(
            "model_card",
            bool(model_card)
            and checkpoint_sha in model_card
            and "Limitations" in model_card
            and "License" in model_card,
            str(model_card_path),
            "Published card must name the exact hash, license, intended use, and limitations.",
        ),
    ]
    passed = all(gate.passed for gate in gates)
    release_manifest_path: str | None = None
    if passed:
        _atomic_json(
            release_manifest,
            {
                "schema_version": 1,
                "model_id": requested_model_id,
                "checkpoint_sha256": checkpoint_sha,
                "dataset_manifest_sha256": manifest.manifest_sha256,
                "evaluation_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
                "quantization_report_sha256": hashlib.sha256(quantization_bytes).hexdigest(),
                "quantized_artifact_sha256": artifact_sha,
                "quantized_artifact_bytes": quantization.quantized_bytes,
                "model_card_sha256": hashlib.sha256(model_card_bytes).hexdigest(),
                "base_license": approved_base_license,
            },
        )
        release_manifest_path = str(release_manifest.resolve())
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
    _atomic_json(output_path, report.model_dump(mode="json"))
    return report
