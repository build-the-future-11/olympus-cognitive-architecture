"""Artificial inputs for deep Foundry integrity checks; no training or capability evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from olympus.foundry import promotion
from olympus.foundry.data_pipeline import (
    REQUIRED_CATEGORIES,
    prepare_instruction_dataset,
    sha256_bytes,
)
from olympus.foundry.eval_suite import (
    CategoryEvaluation,
    HeldOutEvaluation,
    WorkflowScore,
)
from olympus.foundry.quantization import QuantizationReport, _quantize_tensor
from olympus.foundry.sft import SFTConfig, TinyCausalLM, TinyModelConfig


@pytest.fixture
def deep_artifact_dataset(tmp_path: Path) -> dict[str, Any]:
    rows = [
        {
            "id": f"{split}-{category}",
            "category": category,
            "prompt": f"Question {split}_{category}?",
            "response": f"Answer {split}_{category}",
            "license": "MIT",
            "source": "artificial://deep-artifact-integrity",
            "split": split,
        }
        for split in ("train", "validation", "test")
        for category in REQUIRED_CATEGORIES
    ]
    source = tmp_path / "artificial-source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prepared = tmp_path / "prepared"
    manifest = prepare_instruction_dataset(source, prepared)
    config = SFTConfig(
        loss_normalization="legacy_batch_mean_v1",
        mixed_precision=False,
        pack_sequences=False,
        model=TinyModelConfig(width=16, layers=1, heads=2, max_sequence_tokens=128),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(731)
        model = TinyCausalLM(config.model)
    payload = {
        "schema_version": 3,
        "loss_normalization": config.loss_normalization,
        "training_config": config.model_dump(mode="json"),
        "model_config": config.model.model_dump(mode="json"),
        "dataset_manifest_sha256": manifest.manifest_sha256,
        "model_state": model.state_dict(),
    }
    checkpoint = tmp_path / "artificial-checkpoint.pt"
    torch.save(payload, checkpoint)
    return {
        "manifest": manifest,
        "manifest_path": prepared / "manifest.json",
        "config": config,
        "model": model,
        "payload": payload,
        "checkpoint": checkpoint,
    }


@pytest.fixture
def deep_promotion_evidence(
    deep_artifact_dataset: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    # Only scale admission is substituted. All artifact bytes/metadata are real;
    # all supplied quality numbers are explicit artificial gate-test inputs.
    manifest = deep_artifact_dataset["manifest"]
    splits = []
    for split in manifest.splits:
        if split.name == "train":
            split = split.model_copy(update={"records": 10_000})
        elif split.name == "test":
            split = split.model_copy(
                update={
                    "records": 1_200,
                    "categories": {key: 100 for key in REQUIRED_CATEGORIES},
                }
            )
        splits.append(split)
    manifest = manifest.model_copy(update={"splits": splits, "manifest_sha256": None})
    manifest.manifest_sha256 = sha256_bytes(manifest.canonical_bytes(include_hash=False))
    manifest_path = tmp_path / "artificial-scale-manifest.json"
    manifest_path.write_bytes(manifest.canonical_bytes())
    monkeypatch.setattr(promotion, "verify_dataset_manifest", lambda _: manifest)

    config: SFTConfig = deep_artifact_dataset["config"]
    model: TinyCausalLM = deep_artifact_dataset["model"]
    checkpoint: Path = deep_artifact_dataset["checkpoint"]
    payload = dict(deep_artifact_dataset["payload"])
    payload["dataset_manifest_sha256"] = manifest.manifest_sha256
    torch.save(payload, checkpoint)
    checkpoint_sha = sha256_bytes(checkpoint.read_bytes())
    artifact = tmp_path / "artificial-int4.pt"
    torch.save(
        {
            "schema_version": 1,
            "format": "olympus-symmetric-int4-v1",
            "bits": 4,
            "model_config": config.model.model_dump(mode="json"),
            "source_checkpoint_sha256": checkpoint_sha,
            "tensors": {
                name: _quantize_tensor(value, 4) for name, value in model.state_dict().items()
            },
        },
        artifact,
    )
    float_path = tmp_path / "artificial-float.pt"
    torch.save(model.state_dict(), float_path)
    float_bytes = float_path.stat().st_size
    artifact_bytes = artifact.stat().st_size
    evaluation = HeldOutEvaluation(
        loss_normalization=config.loss_normalization,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_sha,
        dataset_manifest_sha256=manifest.manifest_sha256,
        baseline_identity="artificial-frozen",
        test_records=1_200,
        overall_baseline_loss=2.0,
        overall_candidate_loss=1.0,
        overall_baseline_perplexity=math.exp(2.0),
        overall_candidate_perplexity=math.exp(1.0),
        category_results=[
            CategoryEvaluation(
                category=category,
                records=100,
                baseline_loss=2.0,
                candidate_loss=1.0,
                loss_change_fraction=-0.5,
                regressed=False,
            )
            for category in REQUIRED_CATEGORIES
        ],
        regression_count=0,
        exact_match_rate=0.9,
        format_compliance_rate=0.98,
        workflow_scores=[
            WorkflowScore(
                name=name,
                categories=list(categories),
                examples=100 * len(categories),
                exact_match_rate=0.9,
                nonempty_rate=1.0,
            )
            for name, categories in {
                "tool_workflow": ("tool_use", "agent_behavior"),
                "result_comparison": ("reasoning", "self_correction", "research"),
                "multi_step_planning": ("planning", "safety"),
            }.items()
        ],
        median_generation_latency_ms=10,
        generated_bytes_per_second=100,
        peak_rss_bytes=1_000,
        passed_smoke_quality_gate=True,
        decision="Artificial promotion-admission fixture; not model capability evidence.",
    )
    evaluation_path = tmp_path / "artificial-evaluation.json"
    evaluation_path.write_text(evaluation.model_dump_json(), encoding="utf-8")
    quantization = QuantizationReport(
        loss_normalization=config.loss_normalization,
        format="olympus-symmetric-int4-v1",
        bits=4,
        source_checkpoint_sha256=checkpoint_sha,
        artifact_path=str(artifact),
        artifact_sha256=sha256_bytes(artifact.read_bytes()),
        float_weights_bytes=float_bytes,
        quantized_bytes=artifact_bytes,
        size_reduction_fraction=1.0 - artifact_bytes / float_bytes,
        source_loss=1.0,
        quantized_loss=1.01,
        loss_change_fraction=0.01,
        source_latency_ms=10,
        quantized_latency_ms=8,
        startup_latency_ms=5,
        max_context_tokens=config.model.max_sequence_tokens,
        context_limit_enforced=True,
        tool_exact_match_rate=0.8,
        peak_rss_bytes=1_000,
        passed_quality_gate=True,
    )
    quantization_path = tmp_path / "artificial-quantization.json"
    quantization_path.write_text(quantization.model_dump_json(), encoding="utf-8")
    card = tmp_path / "ARTIFICIAL_MODEL_CARD.md"
    card.write_text(
        f"# Artificial fixture\n\nCheckpoint: {checkpoint_sha}\n\nLicense: MIT\n\nLimitations\n"
        "For engineering gate tests only. Not a trained or released Olympus model.\n",
        encoding="utf-8",
    )
    return {
        "requested_model_id": "artificial-integrity-fixture",
        "checkpoint_path": checkpoint,
        "dataset_manifest_path": manifest_path,
        "evaluation_path": evaluation_path,
        "quantization_report_path": quantization_path,
        "model_card_path": card,
        "output_path": tmp_path / "artificial-promotion.json",
        "approved_base_license": "MIT",
        "serving_verification": {"passed": True, "checkpoint_sha256": checkpoint_sha},
    }
