from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from olympus.core.schemas import StrictModel
from olympus.foundry.data_pipeline import DatasetManifestV2, verify_dataset_manifest
from olympus.foundry.eval_suite import HeldOutEvaluation
from olympus.foundry.quantization import QuantizationReport


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
    checkpoint_sha = _sha256(checkpoint_path)
    manifest: DatasetManifestV2 = verify_dataset_manifest(dataset_manifest_path)
    evaluation = HeldOutEvaluation.model_validate_json(evaluation_path.read_bytes())
    quantization = QuantizationReport.model_validate_json(quantization_report_path.read_bytes())
    train = next(split for split in manifest.splits if split.name == "train")
    test = next(split for split in manifest.splits if split.name == "test")
    minimum_category_records = min(test.categories.values())
    model_card = model_card_path.read_text(encoding="utf-8") if model_card_path.is_file() else ""
    serving = serving_verification or {}
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
            quantization.passed_quality_gate
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
            and "Limitations" in model_card
            and "License" in model_card,
            str(model_card_path),
            "Published card must name the exact hash, license, intended use, and limitations.",
        ),
    ]
    passed = all(gate.passed for gate in gates)
    release_manifest_path: str | None = None
    if passed:
        release_manifest = output_path.with_name("release-manifest.json")
        _atomic_json(
            release_manifest,
            {
                "schema_version": 1,
                "model_id": requested_model_id,
                "checkpoint_sha256": checkpoint_sha,
                "dataset_manifest_sha256": manifest.manifest_sha256,
                "evaluation_sha256": _sha256(evaluation_path),
                "quantization_report_sha256": _sha256(quantization_report_path),
                "model_card_sha256": _sha256(model_card_path),
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
