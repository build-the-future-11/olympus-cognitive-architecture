"""Admission rules for the durable, infrastructure-only bigram runtime."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from olympus.foundry.bigram import (
    BigramCheckpoint,
    CharacterBigramModel,
    uniform_baseline_metrics,
)
from olympus.foundry.schemas import (
    ArtifactStatus,
    CheckpointRecord,
    DatasetRecord,
    EvaluationRecord,
    ExperimentRecord,
    ModelRecord,
)

BIGRAM_FAMILY = "FoundryVerificationBigram"
BIGRAM_FORMAT = "olympus-character-bigram-v1"
BIGRAM_SUITE = "held-out-character-model-v1"


def read_registered_dataset(record: DatasetRecord) -> str:
    raw = Path(record.materialized_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record.sha256:
        raise ValueError("dataset content no longer matches the registry hash")
    text = raw.decode("utf-8")
    if len(raw) != record.byte_count or len(text) != record.character_count:
        raise ValueError("registered dataset counts do not match its bytes")
    return text


def load_registered_checkpoint(
    record: CheckpointRecord, experiment: ExperimentRecord, dataset: DatasetRecord
) -> tuple[bytes, CharacterBigramModel]:
    """Hash and parse one snapshot; never relabel a later file read as that model."""
    if not Path(record.path).is_file():
        raise ValueError("missing artifact")
    raw = Path(record.path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record.sha256:
        raise ValueError("checkpoint hash mismatch")
    if record.checkpoint_id != f"ckpt_{record.sha256[:16]}" or len(raw) != record.byte_count:
        raise ValueError("checkpoint identity or size mismatch")
    if record.format != BIGRAM_FORMAT:
        raise ValueError("unsupported checkpoint format")
    if (
        record.experiment_id != experiment.experiment_id
        or experiment.dataset_id != dataset.dataset_id
        or experiment.dataset_version != dataset.version
        or experiment.dataset_sha256 != dataset.sha256
        or experiment.model_family != BIGRAM_FAMILY
    ):
        raise ValueError("checkpoint experiment or dataset identity mismatch")
    checkpoint = BigramCheckpoint.from_bytes(raw)
    if checkpoint.schema_version != 1 or checkpoint.model_type != "character-bigram":
        raise ValueError("unsupported checkpoint schema or model type")
    expected_provenance = {
        "experiment_id": experiment.experiment_id,
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.version,
        "dataset_sha256": dataset.sha256,
        "code_commit": experiment.code_commit,
    }
    if any(checkpoint.training_provenance.get(k) != v for k, v in expected_provenance.items()):
        raise ValueError("checkpoint training provenance mismatch")
    model = CharacterBigramModel(checkpoint)
    if (
        checkpoint.seed != experiment.seed
        or checkpoint.smoothing != experiment.config.get("smoothing")
        or checkpoint.training_characters != experiment.config.get("training_characters")
        or sum(sum(row) for row in checkpoint.counts) != checkpoint.training_characters - 1
    ):
        raise ValueError("checkpoint training configuration or count total mismatch")
    return raw, model


def validate_model_identity(model: ModelRecord, checkpoint: CheckpointRecord) -> None:
    if model.status is not ArtifactStatus.VERIFIED:
        raise ValueError("only verified checkpoints can be promoted to the model registry")
    if model.family != BIGRAM_FAMILY or model.runtime != BIGRAM_FORMAT:
        raise ValueError("unsupported model family or runtime; bigram verification only")
    if (
        model.checkpoint_id != checkpoint.checkpoint_id
        or model.model_id != f"foundry-verification-bigram-{checkpoint.sha256[:8]}"
        or model.version != checkpoint.sha256[:12]
    ):
        raise ValueError("model identity is not bound to its checkpoint")


def evaluation_passes(
    evaluation: EvaluationRecord,
    checkpoint: CheckpointRecord,
    experiment: ExperimentRecord,
    model: CharacterBigramModel,
) -> bool:
    if (
        evaluation.checkpoint_id != checkpoint.checkpoint_id
        or evaluation.dataset_sha256 != experiment.dataset_sha256
        or evaluation.suite != BIGRAM_SUITE
    ):
        raise ValueError("evaluation checkpoint, dataset or suite mismatch")
    expected_transitions = experiment.config.get("evaluation_characters", 0) - 1
    for metrics in (evaluation.baseline_metrics, evaluation.candidate_metrics):
        required = {
            "negative_log_likelihood",
            "perplexity",
            "top1_accuracy",
            "evaluated_transitions",
        }
        if set(metrics) != required or not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError("evaluation metrics are missing or nonfinite")
        if (
            expected_transitions <= 0
            or metrics["evaluated_transitions"] != expected_transitions
            or not 0 <= metrics["top1_accuracy"] <= 1
            or metrics["negative_log_likelihood"] < 0
            or not math.isclose(
                math.log(metrics["perplexity"]),
                metrics["negative_log_likelihood"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("evaluation metrics or transition counts are inconsistent")
    uniform = uniform_baseline_metrics("ab", len(model.checkpoint.alphabet))
    if not math.isclose(
        evaluation.baseline_metrics["perplexity"], uniform.perplexity
    ) or not math.isclose(evaluation.baseline_metrics["top1_accuracy"], uniform.top1_accuracy):
        raise ValueError("evaluation does not use the frozen uniform baseline")
    passed = all(
        evaluation.candidate_metrics[key] < evaluation.baseline_metrics[key]
        for key in ("negative_log_likelihood", "perplexity")
    )
    identity = json.dumps(
        {
            "checkpoint_id": checkpoint.checkpoint_id,
            "dataset_sha256": experiment.dataset_sha256,
            "baseline": evaluation.baseline_metrics,
            "candidate": evaluation.candidate_metrics,
            "passed": evaluation.passed,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if (
        evaluation.passed != passed
        or evaluation.evaluation_id != f"eval_{hashlib.sha256(identity).hexdigest()[:16]}"
    ):
        raise ValueError("evaluation decision or identity contradicts its metrics")
    return passed
