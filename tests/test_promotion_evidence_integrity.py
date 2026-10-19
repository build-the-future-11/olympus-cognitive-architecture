"""Artificial promotion-evidence contracts; no training or protected evaluation.

The manifest verifier is stubbed with one complete, artificial metadata snapshot
so the fixed Hermes scale gates can be exercised without creating a scientific
dataset. Every other gate, file read, digest, and publication path is real.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from olympus.foundry import promotion
from olympus.foundry.data_pipeline import REQUIRED_CATEGORIES, DatasetManifestV2
from olympus.foundry.eval_suite import CategoryEvaluation, HeldOutEvaluation, WorkflowScore
from olympus.foundry.promotion import PromotionReport, evaluate_promotion
from olympus.foundry.quantization import QuantizationReport


@dataclass
class PromotionCase:
    paths: dict[str, Path]
    manifest: DatasetManifestV2
    evaluation: HeldOutEvaluation
    quantization: QuantizationReport

    def run(
        self,
        *,
        serving_verification: dict[str, Any] | None = None,
        approved_base_license: str = "Apache-2.0",
        output_path: Path | None = None,
    ) -> PromotionReport:
        serving = (
            {
                "passed": True,
                "checkpoint_sha256": self.evaluation.checkpoint_sha256,
            }
            if serving_verification is None
            else serving_verification
        )
        return evaluate_promotion(
            requested_model_id="artificial-promotion-fixture",
            checkpoint_path=self.paths["checkpoint"],
            dataset_manifest_path=self.paths["manifest"],
            evaluation_path=self.paths["evaluation"],
            quantization_report_path=self.paths["quantization"],
            model_card_path=self.paths["card"],
            output_path=output_path or self.paths["report"],
            approved_base_license=approved_base_license,
            serving_verification=serving,
        )

    def write_evaluation(self, payload: dict[str, Any]) -> None:
        self.paths["evaluation"].write_text(json.dumps(payload), encoding="utf-8")

    def write_quantization(self, payload: dict[str, Any]) -> None:
        self.paths["quantization"].write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PromotionCase:
    paths = {
        "checkpoint": tmp_path / "artificial-checkpoint.bin",
        "manifest": tmp_path / "manifest.json",
        "evaluation": tmp_path / "evaluation.json",
        "quantization": tmp_path / "quantization.json",
        "artifact": tmp_path / "artificial-int4.bin",
        "card": tmp_path / "MODEL_CARD.md",
        "report": tmp_path / "promotion.json",
        "release": tmp_path / "release-manifest.json",
    }
    paths["checkpoint"].write_bytes(b"artificial checkpoint for evidence validation only")
    paths["artifact"].write_bytes(b"artificial artifact bytes".ljust(40, b"."))
    checkpoint_sha = hashlib.sha256(paths["checkpoint"].read_bytes()).hexdigest()
    train_categories = {name: 833 for name in REQUIRED_CATEGORIES}
    train_categories[REQUIRED_CATEGORIES[0]] += 4
    test_categories = {name: 100 for name in REQUIRED_CATEGORIES}
    manifest = DatasetManifestV2.model_validate(
        {
            "dataset_id": "artificial-promotion-contract",
            "version": "1",
            "source": {
                "uri": "test://artificial-promotion-contract",
                "sha256": "1" * 64,
                "license": "MIT",
                "owner": "test fixture",
                "record_count": 11_212,
            },
            "transformations": ["artificial metadata fixture; no data consumed"],
            "split_policy": "artificial",
            "deduplication": "artificial",
            "quality_policy": "artificial",
            "contamination_policy": "artificial",
            "pii_policy": "artificial",
            "splits": [
                {
                    "name": name,
                    "path": f"{name}.jsonl",
                    "sha256": "2" * 64,
                    "records": sum(categories.values()),
                    "categories": categories,
                }
                for name, categories in (
                    ("train", train_categories),
                    ("validation", {name: 1 for name in REQUIRED_CATEGORIES}),
                    ("test", test_categories),
                )
            ],
            "quality": {
                "input_records": 11_212,
                "accepted_records": 11_212,
                "duplicate_ids": 0,
                "normalized_duplicates": 0,
                "pii_or_secret_hits": 0,
                "split_leakage_hits": 0,
                "missing_category_split_pairs": [],
                "mean_prompt_characters": 20,
                "mean_response_characters": 20,
            },
        }
    )
    manifest.manifest_sha256 = hashlib.sha256(
        manifest.canonical_bytes(include_hash=False)
    ).hexdigest()
    paths["manifest"].write_bytes(manifest.canonical_bytes())
    monkeypatch.setattr(promotion, "verify_dataset_manifest", lambda _: manifest)
    workflow_categories = {
        "tool_workflow": ["tool_use", "agent_behavior"],
        "result_comparison": ["reasoning", "self_correction", "research"],
        "multi_step_planning": ["planning", "safety"],
    }
    evaluation = HeldOutEvaluation(
        checkpoint_path=str(paths["checkpoint"]),
        checkpoint_sha256=checkpoint_sha,
        dataset_manifest_sha256=manifest.manifest_sha256 or "",
        baseline_identity="artificial-frozen-control",
        test_records=1_200,
        overall_baseline_loss=2.0,
        overall_candidate_loss=1.0,
        overall_baseline_perplexity=math.exp(2.0),
        overall_candidate_perplexity=math.exp(1.0),
        category_results=[
            CategoryEvaluation(
                category=name,
                records=100,
                baseline_loss=2.0,
                candidate_loss=1.0,
                loss_change_fraction=-0.5,
                regressed=False,
            )
            for name in REQUIRED_CATEGORIES
        ],
        regression_count=0,
        exact_match_rate=0.9,
        format_compliance_rate=0.98,
        workflow_scores=[
            WorkflowScore(
                name=name,
                categories=categories,
                examples=sum(test_categories[category] for category in categories),
                exact_match_rate=0.9,
                nonempty_rate=1.0,
            )
            for name, categories in workflow_categories.items()
        ],
        median_generation_latency_ms=10,
        generated_bytes_per_second=100,
        peak_rss_bytes=1_000,
        passed_smoke_quality_gate=True,
        decision="artificial gate control; no capability conclusion",
    )
    quantization = QuantizationReport(
        format="olympus-symmetric-int4-v1",
        bits=4,
        source_checkpoint_sha256=checkpoint_sha,
        artifact_path=str(paths["artifact"]),
        artifact_sha256=hashlib.sha256(paths["artifact"].read_bytes()).hexdigest(),
        float_weights_bytes=100,
        quantized_bytes=40,
        size_reduction_fraction=0.6,
        source_loss=1.0,
        quantized_loss=1.01,
        loss_change_fraction=0.01,
        source_latency_ms=10,
        quantized_latency_ms=8,
        startup_latency_ms=5,
        max_context_tokens=1_024,
        context_limit_enforced=True,
        tool_exact_match_rate=0.8,
        peak_rss_bytes=1_000,
        passed_quality_gate=True,
    )
    paths["evaluation"].write_text(evaluation.model_dump_json(), encoding="utf-8")
    paths["quantization"].write_text(quantization.model_dump_json(), encoding="utf-8")
    paths["card"].write_text(
        f"# Artificial fixture\nCheckpoint: {checkpoint_sha}\nLicense: Apache-2.0\n"
        "Intended use: engineering gate tests.\n"
        "Limitations: no trained model or capability claim.\n",
        encoding="utf-8",
    )
    return PromotionCase(paths, manifest, evaluation, quantization)


def _assert_blocked(case: PromotionCase, gate: str) -> PromotionReport:
    report = case.run()
    assert report.status == "NOT_PROMOTED"
    assert report.passed is False
    assert gate in report.blockers
    assert report.release_manifest_path is None
    assert not case.paths["release"].exists()
    assert json.loads(case.paths["report"].read_text())["status"] == "NOT_PROMOTED"
    return report


def test_complete_consistent_artificial_evidence_passes_without_lowering_gates(
    case: PromotionCase,
) -> None:
    report = case.run()
    assert report.passed
    assert report.blockers == []
    assert report.status == "PROMOTED"
    release = json.loads(case.paths["release"].read_text())
    assert release["quantized_artifact_sha256"] == case.quantization.artifact_sha256
    assert release["quantized_artifact_bytes"] == len(case.paths["artifact"].read_bytes())


def test_existing_task_and_tool_quality_boundaries_are_unchanged(case: PromotionCase) -> None:
    evaluation = case.evaluation.model_dump()
    evaluation.update(exact_match_rate=0.8, format_compliance_rate=0.95)
    for workflow in evaluation["workflow_scores"]:
        workflow["exact_match_rate"] = 0.75
    case.write_evaluation(evaluation)
    quantization = case.quantization.model_dump()
    quantization["tool_exact_match_rate"] = 0.75
    case.write_quantization(quantization)
    assert case.run().passed


def test_matching_v2_loss_normalization_is_supported(case: PromotionCase) -> None:
    evaluation = case.evaluation.model_dump()
    quantization = case.quantization.model_dump()
    evaluation["loss_normalization"] = "supervised_token_mean_v2"
    quantization["loss_normalization"] = "supervised_token_mean_v2"
    case.write_evaluation(evaluation)
    case.write_quantization(quantization)
    assert case.run().passed


@pytest.mark.parametrize(
    "changes",
    [
        {"quantized_loss": 10.0, "loss_change_fraction": 9.0},
        {"context_limit_enforced": False},
        {"max_context_tokens": 0},
        {"max_context_tokens": -16},
        {"loss_change_fraction": -0.5},
        {"source_loss": 2.0, "quantized_loss": 2.02},
        {"source_loss": float("inf")},
        {"quantized_loss": float("nan")},
        {"loss_change_fraction": float("nan")},
        {"size_reduction_fraction": -0.6},
        {"source_latency_ms": -1.0},
        {"quantized_latency_ms": float("inf")},
        {"startup_latency_ms": float("nan")},
        {"loss_normalization": "supervised_token_mean_v2"},
        {"format": "unrecognized-int4"},
        {"schema_version": 9},
    ],
)
def test_passing_quantization_flag_cannot_override_contradictory_measurements(
    case: PromotionCase, changes: dict[str, Any]
) -> None:
    payload = case.quantization.model_dump()
    payload.update(changes)
    assert payload["passed_quality_gate"] is True
    case.write_quantization(payload)
    _assert_blocked(case, "quantization_integrity")


@pytest.mark.parametrize("failure", ["missing", "tampered", "digest", "byte_count"])
def test_quantized_artifact_is_verified_before_release(case: PromotionCase, failure: str) -> None:
    payload = case.quantization.model_dump()
    if failure == "missing":
        case.paths["artifact"].unlink()
    elif failure == "tampered":
        case.paths["artifact"].write_bytes(b"corruption".ljust(40, b"."))
    elif failure == "digest":
        payload["artifact_sha256"] = "0" * 64
    else:
        payload["quantized_bytes"] = 39
        payload["size_reduction_fraction"] = 0.61
    case.write_quantization(payload)
    _assert_blocked(case, "quantized_artifact_identity")


def test_unreadable_quantized_artifact_fails_closed(
    case: PromotionCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    read = Path.read_bytes

    def deny_artifact(path: Path) -> bytes:
        if path == case.paths["artifact"]:
            raise PermissionError("artificial unreadable quantized artifact")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", deny_artifact)
    _assert_blocked(case, "quantized_artifact_identity")


def test_relative_quantized_artifact_path_is_relative_to_its_report(case: PromotionCase) -> None:
    payload = case.quantization.model_dump()
    payload["artifact_path"] = case.paths["artifact"].name
    case.write_quantization(payload)
    assert case.run().passed


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_total",
        "missing_categories",
        "missing_category",
        "duplicate_category",
        "category_count",
        "category_regression",
        "category_fraction",
        "category_nonfinite",
        "regression_count",
        "missing_workflows",
        "missing_workflow",
        "duplicate_workflow",
        "unknown_workflow",
        "workflow_categories",
        "workflow_duplicate_categories",
        "workflow_count",
        "schema",
        "suite",
        "baseline_identity",
        "negative_loss",
        "nonfinite_loss",
        "perplexity",
        "smoke_flag",
        "nonfinite_latency",
    ],
)
def test_incomplete_or_inconsistent_evaluation_cannot_pass(
    case: PromotionCase, defect: str
) -> None:
    payload = case.evaluation.model_dump()
    categories = payload["category_results"]
    workflows = payload["workflow_scores"]
    if defect == "wrong_total":
        payload["test_records"] = 1
    elif defect == "missing_categories":
        payload["category_results"] = []
    elif defect == "missing_category":
        categories.pop()
    elif defect == "duplicate_category":
        categories[-1] = categories[0]
    elif defect == "category_count":
        categories[0]["records"] = 1
    elif defect == "category_regression":
        categories[0].update(candidate_loss=3.0, loss_change_fraction=0.5, regressed=False)
    elif defect == "category_fraction":
        categories[0]["loss_change_fraction"] = 0.0
    elif defect == "category_nonfinite":
        categories[0]["baseline_loss"] = float("inf")
    elif defect == "regression_count":
        payload["regression_count"] = 1
    elif defect == "missing_workflows":
        payload["workflow_scores"] = []
    elif defect == "missing_workflow":
        workflows.pop()
    elif defect == "duplicate_workflow":
        workflows.append(workflows[0])
    elif defect == "unknown_workflow":
        workflows[0]["name"] = "unrecognized-workflow"
    elif defect == "workflow_categories":
        workflows[0]["categories"] = ["planning"]
    elif defect == "workflow_duplicate_categories":
        workflows[0]["categories"].append(workflows[0]["categories"][0])
    elif defect == "workflow_count":
        workflows[0]["examples"] = 1
    elif defect == "schema":
        payload["schema_version"] = 9
    elif defect == "suite":
        payload["suite"] = "training-only"
    elif defect == "baseline_identity":
        payload["baseline_identity"] = " "
    elif defect == "negative_loss":
        payload["overall_candidate_loss"] = -1.0
    elif defect == "nonfinite_loss":
        payload["overall_baseline_loss"] = float("inf")
    elif defect == "perplexity":
        payload["overall_candidate_perplexity"] = 99.0
    elif defect == "smoke_flag":
        payload["passed_smoke_quality_gate"] = False
    else:
        payload["median_generation_latency_ms"] = float("inf")
    case.write_evaluation(payload)
    _assert_blocked(case, "evaluation_integrity")


def test_a_consistent_negative_result_remains_not_promoted(case: PromotionCase) -> None:
    payload = case.quantization.model_dump()
    payload.update(quantized_loss=1.1, loss_change_fraction=0.1, passed_quality_gate=False)
    case.write_quantization(payload)
    report = _assert_blocked(case, "quantization_quality")
    assert "quantization_integrity" not in report.blockers


def test_repeated_float32_loss_measurement_can_differ_by_rounding(case: PromotionCase) -> None:
    payload = case.quantization.model_dump()
    payload["source_loss"] = 1.0000001
    payload["quantized_loss"] = payload["source_loss"] * 1.01
    case.write_quantization(payload)
    assert case.run().passed


@pytest.mark.parametrize("target", ["evaluation", "quantization", "card", "artifact"])
def test_release_hashes_identify_the_same_evidence_bytes_that_were_checked(
    case: PromotionCase, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    source = case.paths[target]
    original = source.read_bytes()
    read = Path.read_bytes
    reads: list[bytes] = []

    def replace_after_read(path: Path) -> bytes:
        payload = read(path)
        if path == source:
            reads.append(payload)
            source.write_bytes(b"changed after the evidence snapshot")
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    assert case.run().passed
    release = json.loads(case.paths["release"].read_text())
    key = {
        "evaluation": "evaluation_sha256",
        "quantization": "quantization_report_sha256",
        "card": "model_card_sha256",
        "artifact": "quantized_artifact_sha256",
    }[target]
    assert release[key] == hashlib.sha256(original).hexdigest()
    assert reads == [original]


@pytest.mark.parametrize(
    "target",
    ["checkpoint", "manifest", "evaluation", "quantization", "card", "artifact", "release"],
)
def test_output_cannot_overwrite_its_evidence_or_release_manifest(
    case: PromotionCase, target: str
) -> None:
    preserved = {
        key: path.read_bytes()
        for key, path in case.paths.items()
        if key not in {"report", "release"}
    }
    with pytest.raises(ValueError, match="must not replace"):
        case.run(output_path=case.paths[target])
    assert not case.paths["release"].exists()
    assert all(case.paths[key].read_bytes() == payload for key, payload in preserved.items())


@pytest.mark.parametrize("proof", ["serving", "license", "card", "checkpoint", "dataset"])
def test_original_proof_gates_remain_required(case: PromotionCase, proof: str) -> None:
    options: dict[str, Any] = {}
    expected_gate = {
        "serving": "serving_reproducibility",
        "license": "base_license",
        "card": "model_card",
        "checkpoint": "checkpoint_identity",
        "dataset": "dataset_identity",
    }[proof]
    if proof == "serving":
        options["serving_verification"] = {}
    elif proof == "license":
        options["approved_base_license"] = "unapproved"
    elif proof == "card":
        case.paths["card"].write_text("incomplete artificial card", encoding="utf-8")
    else:
        payload = case.evaluation.model_dump()
        key = "checkpoint_sha256" if proof == "checkpoint" else "dataset_manifest_sha256"
        payload[key] = "0" * 64
        case.write_evaluation(payload)
    report = case.run(**options)
    assert report.status == "NOT_PROMOTED"
    assert expected_gate in report.blockers
    assert not case.paths["release"].exists()


def test_failed_report_publication_keeps_prior_bytes_and_removes_temporary(
    case: PromotionCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = case.evaluation.model_dump()
    evaluation["exact_match_rate"] = 0.1
    case.write_evaluation(evaluation)
    output = case.paths["report"]
    output.write_bytes(b"previous retained report")
    replace = Path.replace

    def fail_report(path: Path, target: Path) -> Path:
        if target == output:
            raise OSError("artificial report publication failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_report)
    with pytest.raises(OSError, match="publication failure"):
        case.run()
    assert output.read_bytes() == b"previous retained report"
    assert list(output.parent.glob(f".{output.name}.*")) == []
    assert not case.paths["release"].exists()
