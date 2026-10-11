"""Hash-bound rate-distortion training for the encoder-quantization pathway."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from olympus.core.schemas import StrictModel
from olympus.models.encoder_quantization import (
    EncoderQuantizationConfig,
    QuantizedAutoencoder,
)


class FeatureTensorArtifact(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shape: list[int] = Field(min_length=2, max_length=3)

    @model_validator(mode="after")
    def valid_artifact(self) -> FeatureTensorArtifact:
        relative = PurePosixPath(self.path)
        if relative.is_absolute() or ".." in relative.parts or self.path in {"", "."}:
            raise ValueError("feature path must be a contained relative path")
        if any(dimension < 1 for dimension in self.shape):
            raise ValueError("feature tensor dimensions must be positive")
        return self


class EncoderDatasetManifest(StrictModel):
    schema_version: int = 1
    license: str = Field(min_length=1)
    train_source_ids: list[str] = Field(min_length=1)
    evaluation_source_ids: list[str] = Field(min_length=1)
    train_features: FeatureTensorArtifact
    evaluation_features: FeatureTensorArtifact

    @model_validator(mode="after")
    def disjoint_sources(self) -> EncoderDatasetManifest:
        if set(self.train_source_ids) & set(self.evaluation_source_ids):
            raise ValueError("training and evaluation source identities must be disjoint")
        if len(set(self.train_source_ids)) != len(self.train_source_ids):
            raise ValueError("training source identities must be unique")
        if len(set(self.evaluation_source_ids)) != len(self.evaluation_source_ids):
            raise ValueError("evaluation source identities must be unique")
        return self


class EncoderQuantizationTrainingConfig(StrictModel):
    seed: int = Field(default=20261011, ge=0, le=2**32 - 1)
    steps: int = Field(default=1_000, ge=1, le=10_000_000)
    batch_size: int = Field(default=8, ge=1, le=4_096)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=0.1)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0, le=100.0)
    encoder: EncoderQuantizationConfig


class EncoderQuantizationEvaluation(StrictModel):
    examples: int = Field(gt=0)
    reconstruction_mse: float = Field(ge=0.0)
    quantization_error: float = Field(ge=0.0)
    codebook_utilization: float = Field(ge=0.0, le=1.0)


class EncoderQuantizationTrainingReport(StrictModel):
    schema_version: int = 1
    status: str
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_steps: int = Field(ge=1)
    final_training_loss: float
    checkpoint_path: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation: EncoderQuantizationEvaluation
    promotion_authorized: bool = False
    claim_boundary: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def feature_tensor_artifact(path: Path, *, root: Path) -> FeatureTensorArtifact:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(resolved_root):
        raise ValueError("feature tensor must be a contained regular file")
    content = resolved.read_bytes()
    loaded: Any = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(loaded, Tensor) or loaded.ndim not in {2, 3}:
        raise ValueError("feature artifact must contain one rank-2 or rank-3 tensor")
    if not loaded.is_floating_point():
        raise ValueError("feature artifact must contain floating-point values")
    return FeatureTensorArtifact(
        path=resolved.relative_to(resolved_root).as_posix(),
        sha256=_sha256(content),
        shape=list(loaded.shape),
    )


def write_encoder_dataset_manifest(
    output: Path,
    *,
    license_name: str,
    train_source_ids: list[str],
    evaluation_source_ids: list[str],
    train_features: Path,
    evaluation_features: Path,
) -> EncoderDatasetManifest:
    root = output.resolve().parent
    manifest = EncoderDatasetManifest(
        license=license_name,
        train_source_ids=train_source_ids,
        evaluation_source_ids=evaluation_source_ids,
        train_features=feature_tensor_artifact(train_features, root=root),
        evaluation_features=feature_tensor_artifact(evaluation_features, root=root),
    )
    _atomic_json(output, manifest.model_dump(mode="json"))
    return manifest


def _load_features(root: Path, artifact: FeatureTensorArtifact) -> Tensor:
    path = root / artifact.path
    if path.is_symlink():
        raise ValueError("feature tensor must not be a symlink")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("feature tensor escapes the manifest root")
    content = resolved.read_bytes()
    if _sha256(content) != artifact.sha256:
        raise ValueError(f"feature tensor digest changed: {artifact.path}")
    loaded: Any = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(loaded, Tensor) or list(loaded.shape) != artifact.shape:
        raise ValueError(f"feature tensor shape changed: {artifact.path}")
    features = loaded.float()
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"feature tensor contains non-finite values: {artifact.path}")
    return features


def _evaluate(model: QuantizedAutoencoder, features: Tensor) -> EncoderQuantizationEvaluation:
    model.eval()
    with torch.inference_mode():
        output = model(features)
        return EncoderQuantizationEvaluation(
            examples=features.shape[0],
            reconstruction_mse=float(
                nn.functional.mse_loss(output.reconstruction, features).item()
            ),
            quantization_error=float(output.quantization.quantization_error.item()),
            codebook_utilization=float(output.quantization.utilization.item()),
        )


def train_encoder_quantizer(
    manifest_path: Path,
    output_directory: Path,
    *,
    config: EncoderQuantizationTrainingConfig,
) -> EncoderQuantizationTrainingReport:
    """Train and evaluate a residual-VQ encoder without authorizing promotion."""

    manifest_bytes = manifest_path.read_bytes()
    manifest = EncoderDatasetManifest.model_validate_json(manifest_bytes)
    root = manifest_path.resolve().parent
    train_features = _load_features(root, manifest.train_features)
    evaluation_features = _load_features(root, manifest.evaluation_features)
    if train_features.shape[-1] != config.encoder.input_dim:
        raise ValueError("training feature width does not match encoder input")
    if evaluation_features.shape[-1] != config.encoder.input_dim:
        raise ValueError("evaluation feature width does not match encoder input")
    if train_features.shape[0] != len(manifest.train_source_ids):
        raise ValueError("training feature rows must match training source identities")
    if evaluation_features.shape[0] != len(manifest.evaluation_source_ids):
        raise ValueError("evaluation feature rows must match evaluation source identities")
    if config.batch_size > train_features.shape[0]:
        raise ValueError("batch size must not exceed training examples")
    report_path = output_directory / "report.json"
    checkpoint_path = output_directory / "checkpoint.pt"
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError("refusing to overwrite an existing encoder run")

    torch.manual_seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    model = QuantizedAutoencoder(config.encoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    final_loss = math.nan
    model.train()
    for _step in range(config.steps):
        indices = torch.randint(train_features.shape[0], (config.batch_size,), generator=generator)
        batch = train_features[indices]
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        reconstruction = nn.functional.mse_loss(output.reconstruction, batch)
        loss = reconstruction + output.quantization.loss
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite encoder-quantization loss")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite encoder-quantization gradient")
        optimizer.step()
        final_loss = float(loss.detach().item())

    evaluation = _evaluate(model, evaluation_features)
    checkpoint = {
        "schema_version": 1,
        "dataset_manifest_sha256": _sha256(manifest_bytes),
        "config": config.model_dump(mode="json"),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "completed_steps": config.steps,
        "final_training_loss": final_loss,
        "evaluation": evaluation.model_dump(mode="json"),
    }
    _atomic_torch_save(checkpoint_path, checkpoint)
    report = EncoderQuantizationTrainingReport(
        status="COMPLETED_DEVELOPMENT_NOT_PROMOTED",
        dataset_manifest_sha256=_sha256(manifest_bytes),
        completed_steps=config.steps,
        final_training_loss=final_loss,
        checkpoint_path=str(checkpoint_path.resolve()),
        checkpoint_sha256=_sha256(checkpoint_path.read_bytes()),
        evaluation=evaluation,
        promotion_authorized=False,
        claim_boundary=(
            "Rate-distortion execution is not evidence of downstream task quality, a trained "
            "BU1LD family, or target-scale quantized inference."
        ),
    )
    _atomic_json(report_path, report.model_dump(mode="json"))
    return report
