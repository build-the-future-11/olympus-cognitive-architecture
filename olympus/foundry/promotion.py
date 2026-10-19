from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from olympus.core.schemas import StrictModel
from olympus.foundry.artifacts import validate_output_paths
from olympus.foundry.data_pipeline import (
    REQUIRED_CATEGORIES,
    DatasetManifestV2,
    SplitDescriptor,
    verify_dataset_manifest,
)
from olympus.foundry.eval_suite import WORKFLOW_CATEGORIES, HeldOutEvaluation
from olympus.foundry.quantization import (
    QuantizationReport,
    _model_from_quantized_payload,
    _read_quantized_payload,
)
from olympus.foundry.sft import (
    SFTConfig,
    TinyModelConfig,
    _checkpoint_training_config,
    _read_checkpoint,
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


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


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_numbers(child) for child in value.values())
    if isinstance(value, list | tuple):
        return all(_finite_numbers(child) for child in value)
    return True


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9)


def _evaluation_consistent(evaluation: HeldOutEvaluation, test: SplitDescriptor) -> bool:
    """Admit complete suite evidence without changing its historical loss objective."""
    if (
        evaluation.schema_version != 1
        or evaluation.suite != "olympus-held-out-capabilities-v1"
        or not _finite_numbers(evaluation.model_dump())
        or evaluation.test_records != test.records
        or set(test.categories) != set(REQUIRED_CATEGORIES)
        or len(evaluation.category_results) != len(test.categories)
        or {item.category for item in evaluation.category_results} != set(test.categories)
        or evaluation.overall_baseline_loss < 0
        or evaluation.overall_candidate_loss < 0
    ):
        return False
    for item in evaluation.category_results:
        change = (item.candidate_loss - item.baseline_loss) / max(item.baseline_loss, 1e-12)
        if (
            item.records != test.categories[item.category]
            or not _close(item.loss_change_fraction, change)
            or item.regressed != (change > 0.05)
        ):
            return False
    if evaluation.regression_count != sum(item.regressed for item in evaluation.category_results):
        return False
    if len(evaluation.workflow_scores) != len(WORKFLOW_CATEGORIES) or {
        item.name for item in evaluation.workflow_scores
    } != set(WORKFLOW_CATEGORIES):
        return False
    for workflow in evaluation.workflow_scores:
        categories = WORKFLOW_CATEGORIES[workflow.name]
        if (
            len(workflow.categories) != len(categories)
            or set(workflow.categories) != set(categories)
            or workflow.examples != sum(test.categories[category] for category in categories)
            or workflow.exact_match_rate > workflow.nonempty_rate
        ):
            return False
    expected_smoke = (
        evaluation.overall_candidate_loss < evaluation.overall_baseline_loss
        and evaluation.regression_count == 0
        and evaluation.format_compliance_rate >= 0.8
    )
    return (
        evaluation.passed_smoke_quality_gate == expected_smoke
        and _close(
            evaluation.overall_baseline_perplexity,
            math.exp(min(evaluation.overall_baseline_loss, 20.0)),
        )
        and _close(
            evaluation.overall_candidate_perplexity,
            math.exp(min(evaluation.overall_candidate_loss, 20.0)),
        )
    )


def _quantization_consistent(
    report: QuantizationReport, evaluation: HeldOutEvaluation, training: SFTConfig
) -> bool:
    if (
        report.schema_version != 1
        or report.format != f"olympus-symmetric-int{report.bits}-v1"
        or not _finite_numbers(report.model_dump())
        or min(
            report.source_loss,
            report.quantized_loss,
            report.source_latency_ms,
            report.quantized_latency_ms,
            report.startup_latency_ms,
        )
        < 0
        or report.max_context_tokens != training.model.max_sequence_tokens
        or report.loss_normalization != evaluation.loss_normalization
        or report.loss_normalization != training.loss_normalization
        or training.mode != "full"
        or not _close(report.source_loss, evaluation.overall_candidate_loss)
    ):
        return False
    change = (report.quantized_loss - report.source_loss) / max(report.source_loss, 1e-12)
    expected_passed = abs(change) <= 0.02 and report.context_limit_enforced
    return (
        _close(report.loss_change_fraction, change)
        and _close(
            report.size_reduction_fraction,
            1.0 - report.quantized_bytes / report.float_weights_bytes,
        )
        and report.passed_quality_gate == expected_passed
    )


def _admit_quantized_artifact(
    report: QuantizationReport, report_path: Path, checkpoint_sha: str, model: TinyModelConfig
) -> tuple[dict[str, Any] | None, str]:
    artifact_path = Path(report.artifact_path)
    if not artifact_path.is_absolute():
        artifact_path = report_path.parent / artifact_path
    try:
        payload, artifact_sha, artifact_size = _read_quantized_payload(artifact_path)
        if (
            artifact_sha != report.artifact_sha256
            or artifact_size != report.quantized_bytes
            or payload["bits"] != report.bits
            or payload["format"] != report.format
            or payload.get("source_checkpoint_sha256") != checkpoint_sha
            or TinyModelConfig.model_validate(payload["model_config"]) != model
        ):
            raise ValueError("quantized artifact identity or metadata does not match its evidence")
        # The same admitted snapshot must also be loadable by the actual runtime.
        _model_from_quantized_payload(payload)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        return None, f"{type(error).__name__}: {error}"
    return {
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": artifact_sha,
        "artifact_bytes": artifact_size,
        "artifact_format": payload["format"],
        "artifact_bits": payload["bits"],
    }, artifact_sha


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
    checkpoint, checkpoint_sha = _read_checkpoint(checkpoint_path)
    training = _checkpoint_training_config(checkpoint)
    if TinyModelConfig.model_validate(checkpoint["model_config"]) != training.model:
        raise ValueError("checkpoint model configuration does not match training configuration")
    manifest: DatasetManifestV2 = verify_dataset_manifest(dataset_manifest_path)
    evaluation_raw = evaluation_path.read_bytes()
    quantization_raw = quantization_report_path.read_bytes()
    evaluation = HeldOutEvaluation.model_validate_json(evaluation_raw)
    quantization = QuantizationReport.model_validate_json(quantization_raw)
    artifact_path = Path(quantization.artifact_path)
    if not artifact_path.is_absolute():
        artifact_path = quantization_report_path.parent / artifact_path
    release_manifest = output_path.with_name("release-manifest.json")
    validate_output_paths(
        [output_path, release_manifest],
        [
            checkpoint_path,
            dataset_manifest_path,
            evaluation_path,
            quantization_report_path,
            model_card_path,
            artifact_path,
        ],
    )
    train = next(split for split in manifest.splits if split.name == "train")
    test = next(split for split in manifest.splits if split.name == "test")
    minimum_category_records = min(test.categories.values())
    model_card_raw = model_card_path.read_bytes() if model_card_path.is_file() else b""
    model_card = model_card_raw.decode("utf-8")
    serving = serving_verification or {}
    allowed_license = approved_base_license in {"Apache-2.0", "MIT", "CC-BY-4.0"}
    evaluation_consistent = _evaluation_consistent(evaluation, test)
    quantization_consistent = _quantization_consistent(quantization, evaluation, training)
    artifact, artifact_observed = _admit_quantized_artifact(
        quantization, quantization_report_path, checkpoint_sha, training.model
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
            manifest.manifest_sha256 == evaluation.dataset_manifest_sha256
            and checkpoint.get("dataset_manifest_sha256") == manifest.manifest_sha256,
            manifest.manifest_sha256,
            "Checkpoint and evaluation must reference the exact immutable dataset manifest.",
        ),
        _gate(
            "evaluation_evidence",
            evaluation_consistent,
            f"complete_and_consistent={evaluation_consistent}",
            "Category/workflow coverage, counts, finite metrics, and summary gates must agree.",
        ),
        _gate(
            "quantization_evidence",
            quantization_consistent,
            f"consistent={quantization_consistent}",
            "Source loss, objective, context, size, loss change, and quality flag must agree.",
        ),
        _gate(
            "quantized_artifact_identity",
            artifact is not None,
            artifact_observed,
            "Actual loadable artifact bytes, hash, size, format, bits, and source must match.",
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
            "quantization_quality",
            quantization_consistent
            and quantization.passed_quality_gate
            and abs(quantization.loss_change_fraction) <= 0.02
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
        assert artifact is not None
        _atomic_json(
            release_manifest,
            {
                "schema_version": 1,
                "model_id": requested_model_id,
                "checkpoint_sha256": checkpoint_sha,
                "dataset_manifest_sha256": manifest.manifest_sha256,
                "evaluation_sha256": _sha256(evaluation_raw),
                "quantization_report_sha256": _sha256(quantization_raw),
                "model_card_sha256": _sha256(model_card_raw),
                "base_license": approved_base_license,
                **artifact,
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
