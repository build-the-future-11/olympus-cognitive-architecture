"""Hash-bound training and held-out comparison for the JEPA upscaler."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any, cast

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from olympus.core.schemas import StrictModel
from olympus.upscaler.jepa import JEPAUpscaler, JEPAUpscalerConfig, jepa_upscaler_loss


class TensorArtifact(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shape: tuple[int, int, int, int]

    @model_validator(mode="after")
    def contained_path(self) -> TensorArtifact:
        relative = PurePosixPath(self.path)
        if relative.is_absolute() or ".." in relative.parts or self.path in {"", "."}:
            raise ValueError("tensor path must be a contained relative path")
        if any(dimension < 1 for dimension in self.shape):
            raise ValueError("tensor dimensions must be positive")
        return self


class SuperResolutionDatasetManifest(StrictModel):
    schema_version: int = 1
    license: str = Field(min_length=1)
    train_source_ids: list[str] = Field(min_length=1)
    evaluation_source_ids: list[str] = Field(min_length=1)
    train_low_resolution: TensorArtifact
    train_high_resolution: TensorArtifact
    evaluation_low_resolution: TensorArtifact
    evaluation_high_resolution: TensorArtifact

    @model_validator(mode="after")
    def disjoint_sources(self) -> SuperResolutionDatasetManifest:
        if set(self.train_source_ids) & set(self.evaluation_source_ids):
            raise ValueError("training and evaluation source identities must be disjoint")
        if len(set(self.train_source_ids)) != len(self.train_source_ids):
            raise ValueError("training source identities must be unique")
        if len(set(self.evaluation_source_ids)) != len(self.evaluation_source_ids):
            raise ValueError("evaluation source identities must be unique")
        return self


class JEPAUpscalerTrainingConfig(StrictModel):
    seed: int = Field(default=20261011, ge=0, le=2**32 - 1)
    steps: int = Field(default=1_000, ge=1, le=10_000_000)
    batch_size: int = Field(default=4, ge=1, le=1_024)
    learning_rate: float = Field(default=2e-4, gt=0.0, le=0.1)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0, le=100.0)
    mask_ratio: float = Field(default=0.5, ge=0.0, le=0.95)
    mask_block_size: int = Field(default=2, ge=1, le=64)
    model: JEPAUpscalerConfig = Field(default_factory=JEPAUpscalerConfig)


class UpscalerEvaluation(StrictModel):
    examples: int = Field(gt=0)
    model_psnr: float
    bicubic_psnr: float
    psnr_improvement: float
    model_edge_error: float
    bicubic_edge_error: float
    model_ssim: float
    bicubic_ssim: float
    model_frequency_error: float
    bicubic_frequency_error: float


class JEPAUpscalerTrainingReport(StrictModel):
    schema_version: int = 3
    status: str
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_steps: int = Field(ge=1)
    final_training_loss: float
    final_loss_components: dict[str, float]
    checkpoint_path: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation: UpscalerEvaluation
    promotion_authorized: bool = False
    claim_boundary: str


def _sha256_bytes(payload: bytes) -> str:
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


def tensor_artifact(path: Path, *, root: Path) -> TensorArtifact:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(resolved_root):
        raise ValueError("tensor artifact must be a contained regular file")
    content = resolved.read_bytes()
    loaded: Any = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(loaded, Tensor) or loaded.ndim != 4:
        raise ValueError("super-resolution tensor artifact must contain one NCHW tensor")
    return TensorArtifact(
        path=resolved.relative_to(resolved_root).as_posix(),
        sha256=_sha256_bytes(content),
        shape=cast(tuple[int, int, int, int], tuple(loaded.shape)),
    )


def write_super_resolution_manifest(
    output: Path,
    *,
    license_name: str,
    train_source_ids: list[str],
    evaluation_source_ids: list[str],
    train_low_resolution: Path,
    train_high_resolution: Path,
    evaluation_low_resolution: Path,
    evaluation_high_resolution: Path,
) -> SuperResolutionDatasetManifest:
    root = output.resolve().parent
    manifest = SuperResolutionDatasetManifest(
        license=license_name,
        train_source_ids=train_source_ids,
        evaluation_source_ids=evaluation_source_ids,
        train_low_resolution=tensor_artifact(train_low_resolution, root=root),
        train_high_resolution=tensor_artifact(train_high_resolution, root=root),
        evaluation_low_resolution=tensor_artifact(evaluation_low_resolution, root=root),
        evaluation_high_resolution=tensor_artifact(evaluation_high_resolution, root=root),
    )
    _atomic_json(output, manifest.model_dump(mode="json"))
    return manifest


def _load_tensor(root: Path, artifact: TensorArtifact) -> Tensor:
    path = root / artifact.path
    if path.is_symlink():
        raise ValueError("dataset tensor must not be a symlink")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("dataset tensor escapes the manifest root")
    content = resolved.read_bytes()
    if _sha256_bytes(content) != artifact.sha256:
        raise ValueError(f"dataset tensor digest changed: {artifact.path}")
    loaded: Any = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(loaded, Tensor) or tuple(loaded.shape) != artifact.shape:
        raise ValueError(f"dataset tensor shape changed: {artifact.path}")
    tensor = loaded.float()
    if not bool(torch.isfinite(tensor).all()) or tensor.min() < 0.0 or tensor.max() > 1.0:
        raise ValueError(f"dataset tensor must contain finite values in [0, 1]: {artifact.path}")
    return tensor


def _validate_pair(low: Tensor, high: Tensor, config: JEPAUpscalerConfig, label: str) -> None:
    if low.shape[0] != high.shape[0] or low.shape[1] != config.input_channels:
        raise ValueError(f"{label} tensors have incompatible examples or channels")
    expected = (low.shape[-2] * config.scale, low.shape[-1] * config.scale)
    if high.shape[1] != config.input_channels or high.shape[-2:] != expected:
        raise ValueError(f"{label} high-resolution tensor does not match configured scale")


def _edge_error(prediction: Tensor, target: Tensor) -> Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    return nn.functional.l1_loss(pred_x, target_x) + nn.functional.l1_loss(pred_y, target_y)


def _frequency_error(prediction: Tensor, target: Tensor) -> Tensor:
    prediction_spectrum = torch.fft.rfft2(prediction.float(), norm="ortho").abs()
    target_spectrum = torch.fft.rfft2(target.float(), norm="ortho").abs()
    return nn.functional.l1_loss(torch.log1p(prediction_spectrum), torch.log1p(target_spectrum))


def _ssim(prediction: Tensor, target: Tensor) -> Tensor:
    mean_prediction = nn.functional.avg_pool2d(prediction, 3, stride=1, padding=1)
    mean_target = nn.functional.avg_pool2d(target, 3, stride=1, padding=1)
    variance_prediction = nn.functional.avg_pool2d(prediction.square(), 3, 1, 1)
    variance_prediction = (variance_prediction - mean_prediction.square()).clamp_min(0.0)
    variance_target = nn.functional.avg_pool2d(target.square(), 3, 1, 1)
    variance_target = (variance_target - mean_target.square()).clamp_min(0.0)
    covariance = nn.functional.avg_pool2d(prediction * target, 3, 1, 1)
    covariance = covariance - mean_prediction * mean_target
    numerator = (2.0 * mean_prediction * mean_target + 0.01**2) * (2.0 * covariance + 0.03**2)
    denominator = (mean_prediction.square() + mean_target.square() + 0.01**2) * (
        variance_prediction + variance_target + 0.03**2
    )
    return (numerator / denominator.clamp_min(1e-12)).mean().clamp(-1.0, 1.0)


def _evaluate(model: JEPAUpscaler, low: Tensor, high: Tensor) -> UpscalerEvaluation:
    model.eval()
    with torch.inference_mode():
        prediction = model(low).upscaled
        bicubic = nn.functional.interpolate(
            low, scale_factor=model.config.scale, mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        model_mse = float(nn.functional.mse_loss(prediction, high).item())
        bicubic_mse = float(nn.functional.mse_loss(bicubic, high).item())
        model_psnr = -10.0 * math.log10(max(model_mse, 1e-12))
        bicubic_psnr = -10.0 * math.log10(max(bicubic_mse, 1e-12))
        return UpscalerEvaluation(
            examples=low.shape[0],
            model_psnr=model_psnr,
            bicubic_psnr=bicubic_psnr,
            psnr_improvement=model_psnr - bicubic_psnr,
            model_edge_error=float(_edge_error(prediction, high).item()),
            bicubic_edge_error=float(_edge_error(bicubic, high).item()),
            model_ssim=float(_ssim(prediction, high).item()),
            bicubic_ssim=float(_ssim(bicubic, high).item()),
            model_frequency_error=float(_frequency_error(prediction, high).item()),
            bicubic_frequency_error=float(_frequency_error(bicubic, high).item()),
        )


def _context_mask(
    batch: int,
    height: int,
    width: int,
    *,
    ratio: float,
    block_size: int,
    generator: torch.Generator,
) -> Tensor:
    if ratio == 0.0:
        return torch.zeros((batch, height, width), dtype=torch.bool)
    coarse_height = math.ceil(height / block_size)
    coarse_width = math.ceil(width / block_size)
    coarse = torch.rand((batch, 1, coarse_height, coarse_width), generator=generator) < ratio
    mask = nn.functional.interpolate(coarse.float(), scale_factor=block_size, mode="nearest").to(
        dtype=torch.bool
    )
    mask = mask[:, 0, :height, :width]
    flat = mask.flatten(1)
    for row in flat:
        if not bool(row.any()):
            row[0] = True
        if bool(row.all()):
            row[-1] = False
    return mask


def train_jepa_upscaler(
    manifest_path: Path,
    output_directory: Path,
    *,
    config: JEPAUpscalerTrainingConfig,
) -> JEPAUpscalerTrainingReport:
    """Train once against a source-disjoint manifest and retain a negative result."""

    manifest_bytes = manifest_path.read_bytes()
    manifest = SuperResolutionDatasetManifest.model_validate_json(manifest_bytes)
    root = manifest_path.resolve().parent
    train_low = _load_tensor(root, manifest.train_low_resolution)
    train_high = _load_tensor(root, manifest.train_high_resolution)
    evaluation_low = _load_tensor(root, manifest.evaluation_low_resolution)
    evaluation_high = _load_tensor(root, manifest.evaluation_high_resolution)
    _validate_pair(train_low, train_high, config.model, "training")
    _validate_pair(evaluation_low, evaluation_high, config.model, "evaluation")
    if train_low.shape[0] != len(manifest.train_source_ids):
        raise ValueError("training tensor rows must match training source identities")
    if evaluation_low.shape[0] != len(manifest.evaluation_source_ids):
        raise ValueError("evaluation tensor rows must match evaluation source identities")
    if config.batch_size > train_low.shape[0]:
        raise ValueError("batch size must not exceed training examples")

    report_path = output_directory / "report.json"
    checkpoint_path = output_directory / "checkpoint.pt"
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError("refusing to overwrite an existing upscaler run")
    torch.manual_seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    model = JEPAUpscaler(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    final_loss = math.nan
    final_components: dict[str, float] = {}
    model.train()
    for _step in range(config.steps):
        indices = torch.randint(train_low.shape[0], (config.batch_size,), generator=generator)
        low = train_low[indices]
        high = train_high[indices]
        timestamp = indices.float() / max(train_low.shape[0] - 1, 1)
        context_mask = _context_mask(
            low.shape[0],
            low.shape[2],
            low.shape[3],
            ratio=config.mask_ratio,
            block_size=config.mask_block_size,
            generator=generator,
        )
        optimizer.zero_grad(set_to_none=True)
        losses = jepa_upscaler_loss(
            model(low, high, timestamp=timestamp, context_mask=context_mask), high
        )
        if not torch.isfinite(losses.total):
            raise FloatingPointError("non-finite JEPA upscaler loss")
        losses.total.backward()  # type: ignore[no-untyped-call]
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite JEPA upscaler gradient")
        optimizer.step()
        model.update_target_encoder()
        final_loss = float(losses.total.detach().item())
        final_components = {
            "reconstruction": float(losses.reconstruction.detach().item()),
            "edge": float(losses.edge.detach().item()),
            "joint_embedding": float(losses.joint_embedding.detach().item()),
            "latent_geometry": float(losses.latent_geometry.detach().item()),
            "variance": float(losses.variance.detach().item()),
            "frequency": float(losses.frequency.detach().item()),
            "structure": float(losses.structure.detach().item()),
            "pyramid": float(losses.pyramid.detach().item()),
            "spectral_coherence": float(losses.spectral_coherence.detach().item()),
        }

    evaluation = _evaluate(model, evaluation_low, evaluation_high)
    checkpoint = {
        "schema_version": 3,
        "dataset_manifest_sha256": _sha256_bytes(manifest_bytes),
        "config": config.model_dump(mode="json"),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "completed_steps": config.steps,
        "final_training_loss": final_loss,
        "evaluation": evaluation.model_dump(mode="json"),
    }
    _atomic_torch_save(checkpoint_path, checkpoint)
    checkpoint_sha = _sha256_bytes(checkpoint_path.read_bytes())
    report = JEPAUpscalerTrainingReport(
        status="COMPLETED_DEVELOPMENT_NOT_PROMOTED",
        dataset_manifest_sha256=_sha256_bytes(manifest_bytes),
        completed_steps=config.steps,
        final_training_loss=final_loss,
        final_loss_components=final_components,
        checkpoint_path=str(checkpoint_path.resolve()),
        checkpoint_sha256=checkpoint_sha,
        evaluation=evaluation,
        promotion_authorized=False,
        claim_boundary=(
            "A completed local training/evaluation path is not evidence of natural-image "
            "superiority, a promoted model, or any BU1LD family capability."
        ),
    )
    _atomic_json(report_path, report.model_dump(mode="json"))
    return report


def load_jepa_upscaler_checkpoint(
    checkpoint_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_dataset_manifest_sha256: str | None = None,
) -> tuple[JEPAUpscaler, JEPAUpscalerTrainingConfig]:
    """Load an immutable development checkpoint with optional digest pinning."""

    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("checkpoint must be a regular non-symlink file")
    content = checkpoint_path.read_bytes()
    observed_sha256 = _sha256_bytes(content)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError("checkpoint digest does not match the expected SHA-256")
    payload: Any = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {2, 3}:
        raise ValueError("checkpoint has an unsupported schema")
    dataset_manifest_sha256 = payload.get("dataset_manifest_sha256")
    if (
        not isinstance(dataset_manifest_sha256, str)
        or len(dataset_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in dataset_manifest_sha256)
    ):
        raise ValueError("checkpoint has an invalid dataset manifest identity")
    if (
        expected_dataset_manifest_sha256 is not None
        and dataset_manifest_sha256 != expected_dataset_manifest_sha256
    ):
        raise ValueError("checkpoint dataset manifest does not match the expected SHA-256")
    raw_config = payload.get("config")
    raw_state = payload.get("model_state")
    if not isinstance(raw_config, dict) or not isinstance(raw_state, dict):
        raise ValueError("checkpoint is missing its configuration or model state")
    config = JEPAUpscalerTrainingConfig.model_validate(raw_config)
    model = JEPAUpscaler(config.model)
    model.load_state_dict(cast(dict[str, Tensor], raw_state), strict=True)
    model.eval()
    return model, config
