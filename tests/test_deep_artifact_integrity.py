"""Snapshot provenance and fail-closed promotion using artificial CPU-only inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from olympus.foundry import eval_suite, promotion, quantization
from olympus.foundry.quantization import QuantizationBits


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _replace_after_read(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    mutation: str,
) -> None:
    original = Path.read_bytes
    changed = False

    def reading(path: Path) -> bytes:
        nonlocal changed
        raw = original(path)
        if path == target and not changed:
            changed = True
            if mutation == "delete":
                path.unlink()
            else:
                path.write_bytes(b"Different bytes after the admitted snapshot.")
        return raw

    monkeypatch.setattr(Path, "read_bytes", reading)


@pytest.mark.parametrize("role", ["candidate", "baseline"])
@pytest.mark.parametrize("mutation", ["replace", "delete"])
def test_evaluation_hashes_the_consumed_checkpoint_snapshot(
    deep_artifact_dataset: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    mutation: str,
) -> None:
    checkpoint: Path = deep_artifact_dataset["checkpoint"]
    target = checkpoint
    baseline: Path | None = None
    if role == "baseline":
        baseline = tmp_path / "artificial-baseline.pt"
        baseline.write_bytes(checkpoint.read_bytes())
        target = baseline
    expected = _sha(target.read_bytes())
    _replace_after_read(monkeypatch, target, mutation)
    report = eval_suite.evaluate_checkpoint(
        checkpoint,
        deep_artifact_dataset["manifest_path"],
        tmp_path / "evaluation.json",
        base_checkpoint=baseline,
        max_generation_tokens=1,
    )
    if role == "candidate":
        assert report.checkpoint_sha256 == expected
    else:
        assert report.baseline_identity == f"checkpoint:{expected}"


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("mutation", ["replace", "delete"])
def test_quantization_hashes_the_consumed_source_checkpoint(
    deep_artifact_dataset: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bits: QuantizationBits,
    mutation: str,
) -> None:
    checkpoint: Path = deep_artifact_dataset["checkpoint"]
    expected = _sha(checkpoint.read_bytes())
    _replace_after_read(monkeypatch, checkpoint, mutation)
    report = quantization.quantize_checkpoint(
        checkpoint,
        deep_artifact_dataset["manifest_path"],
        tmp_path / "quantized",
        bits=bits,
    )
    artifact = torch.load(report.artifact_path, map_location="cpu", weights_only=True)
    assert report.source_checkpoint_sha256 == expected
    assert artifact["source_checkpoint_sha256"] == expected


@pytest.mark.parametrize("mutation", ["replace", "delete"])
def test_quantization_report_hashes_the_artifact_snapshot_it_scored(
    deep_artifact_dataset: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    output = tmp_path / "quantized"
    artifact = output / "model-int4.pt"
    original_latency = quantization._latency
    consumed: list[bytes] = []

    def changing_latency(*args: Any, **kwargs: Any) -> float:
        if not consumed:
            consumed.append(artifact.read_bytes())
            if mutation == "delete":
                artifact.unlink()
            else:
                artifact.write_bytes(b"Artifact replaced after it was loaded for scoring.")
        return original_latency(*args, **kwargs)

    monkeypatch.setattr(quantization, "_latency", changing_latency)
    report = quantization.quantize_checkpoint(
        deep_artifact_dataset["checkpoint"],
        deep_artifact_dataset["manifest_path"],
        output,
        bits=4,
    )
    assert report.artifact_sha256 == _sha(consumed[0])
    assert report.quantized_bytes == len(consumed[0])


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_bytes()))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _assert_blocked(evidence: dict[str, Any], gate: str) -> None:
    report = promotion.evaluate_promotion(**evidence)
    assert report.status == "NOT_PROMOTED"
    assert gate in report.blockers
    assert report.release_manifest_path is None
    assert not evidence["output_path"].with_name("release-manifest.json").exists()


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "digest", "size"])
def test_promotion_requires_actual_artifact_hash_and_size(
    deep_promotion_evidence: dict[str, Any],
    mutation: str,
) -> None:
    report_path = deep_promotion_evidence["quantization_report_path"]
    report = _read_json(report_path)
    artifact = Path(report["artifact_path"])
    if mutation == "missing":
        artifact.unlink()
    elif mutation == "corrupt":
        artifact.write_bytes(artifact.read_bytes() + b"changed")
    elif mutation == "digest":
        report["artifact_sha256"] = "0" * 64
    else:
        report["quantized_bytes"] += 1
    _write_json(report_path, report)
    _assert_blocked(deep_promotion_evidence, "quantized_artifact_identity")


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "format",
        "bits",
        "source",
        "model_config",
        "nonfinite",
        "missing_tensor",
        "unreadable",
        "encoded_dtype",
        "encoded_count",
        "floating_flag",
        "negative_scale",
    ],
)
def test_promotion_checks_artifact_metadata_and_real_loadability(
    deep_promotion_evidence: dict[str, Any],
    mutation: str,
) -> None:
    report_path = deep_promotion_evidence["quantization_report_path"]
    report = _read_json(report_path)
    artifact = Path(report["artifact_path"])
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if mutation == "schema":
        payload["schema_version"] = 2
    elif mutation == "format":
        payload["format"] = "different-format"
    elif mutation == "bits":
        payload["bits"] = 8
    elif mutation == "source":
        payload["source_checkpoint_sha256"] = "0" * 64
    elif mutation == "model_config":
        payload["model_config"]["dropout"] = 0.5
    elif mutation == "nonfinite":
        tensor = next(value for value in payload["tensors"].values() if value["floating"])
        tensor["scale"].fill_(float("nan"))
    elif mutation == "missing_tensor":
        payload["tensors"].pop(next(iter(payload["tensors"])))
    elif mutation in {"encoded_dtype", "encoded_count", "floating_flag", "negative_scale"}:
        tensor = next(value for value in payload["tensors"].values() if value["floating"])
        if mutation == "encoded_dtype":
            tensor["quantized"] = tensor["quantized"].float()
        elif mutation == "encoded_count":
            tensor["count"] += 1
        elif mutation == "floating_flag":
            tensor["floating"] = False
        else:
            tensor["scale"].mul_(-1)
    if mutation == "unreadable":
        artifact.write_bytes(b"Not a torch artifact.")
    else:
        torch.save(payload, artifact)
    raw = artifact.read_bytes()
    report.update(
        {
            "artifact_sha256": _sha(raw),
            "quantized_bytes": len(raw),
            "size_reduction_fraction": 1.0 - len(raw) / report["float_weights_bytes"],
        }
    )
    _write_json(report_path, report)
    _assert_blocked(deep_promotion_evidence, "quantized_artifact_identity")


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_categories",
        "missing_category",
        "duplicate_category",
        "test_count",
        "category_count",
        "category_change",
        "regression_flag",
        "regression_count",
        "empty_workflows",
        "missing_workflow",
        "duplicate_workflow",
        "workflow_categories",
        "workflow_count",
        "workflow_nonempty",
        "nonfinite",
        "perplexity",
        "smoke_flag",
    ],
)
def test_promotion_rejects_incomplete_or_contradictory_evaluation(
    deep_promotion_evidence: dict[str, Any],
    mutation: str,
) -> None:
    path = deep_promotion_evidence["evaluation_path"]
    report = _read_json(path)
    if mutation == "empty_categories":
        report["category_results"] = []
    elif mutation == "missing_category":
        report["category_results"].pop()
    elif mutation == "duplicate_category":
        report["category_results"][-1] = report["category_results"][0]
    elif mutation == "test_count":
        report["test_records"] = 1
    elif mutation == "category_count":
        report["category_results"][0]["records"] = 1
    elif mutation == "category_change":
        report["category_results"][0]["loss_change_fraction"] = 0.5
    elif mutation == "regression_flag":
        report["category_results"][0]["regressed"] = True
    elif mutation == "regression_count":
        report["regression_count"] = 1
    elif mutation == "empty_workflows":
        report["workflow_scores"] = []
    elif mutation == "missing_workflow":
        report["workflow_scores"].pop()
    elif mutation == "duplicate_workflow":
        report["workflow_scores"][-1] = report["workflow_scores"][0]
    elif mutation == "workflow_categories":
        report["workflow_scores"][0]["categories"] = ["planning", "safety"]
    elif mutation == "workflow_count":
        report["workflow_scores"][0]["examples"] = 1
    elif mutation == "workflow_nonempty":
        report["workflow_scores"][0]["nonempty_rate"] = 0.1
    elif mutation == "nonfinite":
        report["overall_baseline_loss"] = float("inf")
    elif mutation == "perplexity":
        report["overall_candidate_perplexity"] = 1.0
    else:
        report["passed_smoke_quality_gate"] = False
    _write_json(path, report)
    _assert_blocked(deep_promotion_evidence, "evaluation_evidence")


@pytest.mark.parametrize(
    "mutation",
    [
        "context_flag",
        "context_size",
        "loss_change",
        "quantized_loss",
        "source_loss",
        "size_fraction",
        "negative_latency",
        "nonfinite",
        "normalization",
        "quality_flag",
    ],
)
def test_promotion_rejects_contradictory_quantization_metrics(
    deep_promotion_evidence: dict[str, Any],
    mutation: str,
) -> None:
    path = deep_promotion_evidence["quantization_report_path"]
    report = _read_json(path)
    updates: dict[str, dict[str, Any]] = {
        "context_flag": {"context_limit_enforced": False},
        "context_size": {"max_context_tokens": 1_024},
        "loss_change": {"loss_change_fraction": 0.5},
        "quantized_loss": {"quantized_loss": 1.5, "loss_change_fraction": 0.5},
        "source_loss": {"source_loss": 2.0},
        "size_fraction": {"size_reduction_fraction": 0.0},
        "negative_latency": {"source_latency_ms": -1.0},
        "nonfinite": {"source_loss": float("nan")},
        "normalization": {"loss_normalization": "supervised_token_mean_v2"},
        "quality_flag": {"passed_quality_gate": False},
    }
    report.update(updates[mutation])
    _write_json(path, report)
    _assert_blocked(deep_promotion_evidence, "quantization_evidence")


@pytest.mark.parametrize(
    "field,identity",
    [
        ("evaluation_path", "evaluation_sha256"),
        ("quantization_report_path", "quantization_report_sha256"),
        ("model_card_path", "model_card_sha256"),
    ],
)
def test_release_hashes_the_report_and_card_snapshots_used_by_its_gates(
    deep_promotion_evidence: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    identity: str,
) -> None:
    target = deep_promotion_evidence[field]
    expected = _sha(target.read_bytes())
    original_gate = promotion._gate
    changed = False

    def changing_gate(*args: Any, **kwargs: Any) -> promotion.GateResult:
        nonlocal changed
        if not changed:
            changed = True
            target.write_bytes(b"Different bytes after evidence admission.")
        return original_gate(*args, **kwargs)

    monkeypatch.setattr(promotion, "_gate", changing_gate)
    report = promotion.evaluate_promotion(**deep_promotion_evidence)
    assert report.status == "PROMOTED"
    release = _read_json(Path(report.release_manifest_path or ""))
    assert release[identity] == expected


def test_legacy_report_defaults_and_relative_artifact_paths_remain_admissible(
    deep_promotion_evidence: dict[str, Any],
) -> None:
    for key in ("evaluation_path", "quantization_report_path"):
        path = deep_promotion_evidence[key]
        report = _read_json(path)
        report.pop("loss_normalization")
        if key == "quantization_report_path":
            report["artifact_path"] = Path(report["artifact_path"]).name
        _write_json(path, report)
    result = promotion.evaluate_promotion(**deep_promotion_evidence)
    assert result.status == "PROMOTED"


def test_quantization_rejects_replacement_artifact_metadata_before_scoring(
    deep_artifact_dataset: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_save = quantization._atomic_torch_save

    def changing_save(path: Path, payload: object) -> None:
        original_save(path, payload)
        if path.name == "model-int4.pt":
            replacement = dict(cast(dict[str, Any], payload))
            replacement["source_checkpoint_sha256"] = "0" * 64
            torch.save(replacement, path)

    monkeypatch.setattr(quantization, "_atomic_torch_save", changing_save)
    output = tmp_path / "changed-artifact"
    with pytest.raises(ValueError, match="changed before evaluation admission"):
        quantization.quantize_checkpoint(
            deep_artifact_dataset["checkpoint"],
            deep_artifact_dataset["manifest_path"],
            output,
            bits=4,
        )
    assert not (output / "int4-report.json").exists()


@pytest.mark.parametrize("bits", [3, 4.0, True])
def test_quantization_rejects_unsupported_bits_before_any_output(
    deep_artifact_dataset: dict[str, Any],
    tmp_path: Path,
    bits: Any,
) -> None:
    output = tmp_path / "invalid-bits"
    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        quantization.quantize_checkpoint(
            deep_artifact_dataset["checkpoint"],
            deep_artifact_dataset["manifest_path"],
            output,
            bits=bits,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "target", ["checkpoint_path", "evaluation_path", "artifact", "release", "symlink"]
)
def test_promotion_cannot_overwrite_admitted_inputs_or_its_release_manifest(
    deep_promotion_evidence: dict[str, Any], tmp_path: Path, target: str
) -> None:
    evidence = dict(deep_promotion_evidence)
    quantized = _read_json(evidence["quantization_report_path"])
    artifact = Path(quantized["artifact_path"])
    if target == "artifact":
        destination = artifact
    elif target == "release":
        destination = evidence["output_path"].with_name("release-manifest.json")
    elif target == "symlink":
        destination = tmp_path / "artifact-alias.json"
        destination.symlink_to(artifact)
    else:
        destination = evidence[target]
    evidence["output_path"] = destination
    protected = [
        evidence["checkpoint_path"],
        evidence["evaluation_path"],
        evidence["quantization_report_path"],
        artifact,
    ]
    before = {path: path.read_bytes() for path in protected}
    with pytest.raises(ValueError, match="distinct from input artifacts"):
        promotion.evaluate_promotion(**evidence)
    assert {path: path.read_bytes() for path in protected} == before
    assert not destination.with_name("release-manifest.json").exists()
    if target == "symlink":
        assert destination.is_symlink()


def test_promotion_requires_fresh_attempt_without_erasing_a_previous_release(
    deep_promotion_evidence: dict[str, Any],
) -> None:
    report = promotion.evaluate_promotion(**deep_promotion_evidence)
    release = Path(report.release_manifest_path or "")
    before_release = release.read_bytes()
    before_report = deep_promotion_evidence["output_path"].read_bytes()
    repeat = {**deep_promotion_evidence, "approved_base_license": "unapproved"}
    with pytest.raises(ValueError, match="fresh output path"):
        promotion.evaluate_promotion(**repeat)
    assert release.read_bytes() == before_release
    assert deep_promotion_evidence["output_path"].read_bytes() == before_report


@pytest.mark.parametrize("target", ["checkpoint", "manifest", "earlier-report"])
def test_evaluation_preserves_input_and_predecessor_files(
    deep_artifact_dataset: dict[str, Any], tmp_path: Path, target: str
) -> None:
    if target == "earlier-report":
        destination = tmp_path / "earlier-report.json"
        destination.write_bytes(b"Retained predecessor report, including negative results.")
    else:
        destination = deep_artifact_dataset[
            "checkpoint" if target == "checkpoint" else "manifest_path"
        ]
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="distinct from input artifacts|fresh output path"):
        eval_suite.evaluate_checkpoint(
            deep_artifact_dataset["checkpoint"],
            deep_artifact_dataset["manifest_path"],
            destination,
        )
    assert destination.read_bytes() == before


@pytest.mark.parametrize("filename", ["float32-model.pt", "model-int4.pt", "int4-report.json"])
def test_quantization_preserves_existing_outputs_before_writing_any_artifact(
    deep_artifact_dataset: dict[str, Any], tmp_path: Path, filename: str
) -> None:
    output = tmp_path / "previous-quantization"
    output.mkdir()
    destination = output / filename
    before = b"Retained predecessor artifact."
    destination.write_bytes(before)
    with pytest.raises(ValueError, match="fresh output path"):
        quantization.quantize_checkpoint(
            deep_artifact_dataset["checkpoint"],
            deep_artifact_dataset["manifest_path"],
            output,
            bits=4,
        )
    assert destination.read_bytes() == before
    assert list(output.iterdir()) == [destination]
