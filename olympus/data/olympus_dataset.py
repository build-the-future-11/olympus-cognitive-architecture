"""Content-addressed aligned datasets for the composed Olympus system."""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any, Literal, cast

import torch
from pydantic import Field, model_validator
from torch import Tensor

from olympus.core.schemas import StrictModel

_REQUIRED_FIELDS = frozenset(
    {
        "tokens",
        "image",
        "states",
        "actions",
        "allowed_transitions",
        "risk_score",
        "decision_targets",
        "transition_targets",
        "next_state_targets",
    }
)
_OPTIONAL_FIELDS = frozenset(
    {
        "attention_mask",
        "domain_ids",
        "temporal_mask",
        "timestamp",
        "intrinsic_risk_targets",
        "component_reliability_targets",
    }
)
_ALLOWED_LICENSES = frozenset({"LicenseRef-Proprietary", "Apache-2.0", "MIT", "CC-BY-4.0"})


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{serialized}\n".encode()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


class OlympusBatchArtifact(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ids: list[str] = Field(min_length=1)
    shapes: dict[str, tuple[int, ...]]
    dtypes: dict[str, str]

    @model_validator(mode="after")
    def valid_identity(self) -> OlympusBatchArtifact:
        relative = PurePosixPath(self.path)
        if relative.is_absolute() or ".." in relative.parts or self.path in {"", "."}:
            raise ValueError("batch path must be a contained relative path")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("source identities must be unique within a split")
        if any(not identity.strip() for identity in self.source_ids):
            raise ValueError("source identities must not be empty")
        if set(self.shapes) != set(self.dtypes):
            raise ValueError("batch shape and dtype descriptors must name the same fields")
        return self


class OlympusSystemDatasetManifest(StrictModel):
    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    license: str = Field(min_length=1)
    extensions: list[str] = Field(default_factory=list)
    train: OlympusBatchArtifact
    evaluation: OlympusBatchArtifact
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_provenance(self) -> OlympusSystemDatasetManifest:
        if self.license not in _ALLOWED_LICENSES:
            raise ValueError(f"unsupported training license: {self.license}")
        if len(self.extensions) != len(set(self.extensions)):
            raise ValueError("extension names must be unique")
        if any(not name or not name.replace("_", "").isalnum() for name in self.extensions):
            raise ValueError("extension names must contain only letters, digits, and underscores")
        overlap = set(self.train.source_ids) & set(self.evaluation.source_ids)
        if overlap:
            raise ValueError("training and evaluation source identities must be disjoint")
        if self.train.path == self.evaluation.path or self.train.sha256 == self.evaluation.sha256:
            raise ValueError("training and evaluation must not reuse the same batch artifact")
        return self

    def canonical_bytes(self, *, include_hash: bool = True) -> bytes:
        payload = self.model_dump(mode="json")
        if not include_hash:
            payload["manifest_sha256"] = None
        return _canonical_json(payload)


@dataclass(frozen=True, slots=True)
class LoadedOlympusDataset:
    manifest: OlympusSystemDatasetManifest
    train: dict[str, Tensor]
    evaluation: dict[str, Tensor]


def _decode_bundle(content: bytes) -> dict[str, Tensor]:
    loaded: Any = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not all(
        isinstance(name, str) and isinstance(tensor, Tensor) for name, tensor in loaded.items()
    ):
        raise ValueError("dataset batch must contain a string-to-tensor mapping")
    return cast(dict[str, Tensor], loaded)


def _read_bundle(path: Path) -> tuple[bytes, dict[str, Tensor]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("dataset batch must be a regular non-symlink file")
    content = path.read_bytes()
    return content, _decode_bundle(content)


def _validate_bundle(
    tensors: dict[str, Tensor],
    *,
    extensions: list[str],
    source_ids: list[str],
) -> None:
    extension_fields = {f"extension__{name}" for name in extensions}
    names = set(tensors)
    missing = sorted(_REQUIRED_FIELDS - names)
    unexpected = sorted(names - _REQUIRED_FIELDS - _OPTIONAL_FIELDS - extension_fields)
    missing_extensions = sorted(extension_fields - names)
    if missing or unexpected or missing_extensions:
        raise ValueError(
            "dataset fields do not match the Olympus contract; "
            f"missing={missing}, missing_extensions={missing_extensions}, "
            f"unexpected={unexpected}"
        )
    for name, tensor in tensors.items():
        if tensor.layout is not torch.strided:
            raise ValueError(f"dataset tensor must use a dense strided layout: {name}")
    tokens = tensors["tokens"]
    if tokens.ndim != 2 or tokens.dtype not in {torch.int32, torch.int64}:
        raise ValueError("tokens must be a rank-two integer tensor")
    if tokens.shape[1] < 1 or bool((tokens < 0).any()):
        raise ValueError("tokens must have a non-empty sequence and non-negative IDs")
    batch = tokens.shape[0]
    if batch < 1 or batch != len(source_ids):
        raise ValueError("source identities must match the non-empty tensor batch")
    if any(tensor.ndim < 1 or tensor.shape[0] != batch for tensor in tensors.values()):
        raise ValueError("every dataset tensor must share the first batch dimension")
    for name, tensor in tensors.items():
        if any(size < 1 for size in tensor.shape):
            raise ValueError(f"dataset tensor dimensions must be non-empty: {name}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"dataset tensor contains non-finite values: {name}")
    if tensors["image"].ndim != 4 or not tensors["image"].is_floating_point():
        raise ValueError("image must be a floating-point NCHW tensor")
    if bool(((tensors["image"] < 0.0) | (tensors["image"] > 1.0)).any()):
        raise ValueError("image values must fall in [0, 1]")
    if tensors["states"].ndim != 3 or tensors["actions"].ndim != 3:
        raise ValueError("states and actions must be rank-three temporal tensors")
    for name in ("states", "actions", "next_state_targets", "risk_score"):
        if not tensors[name].is_floating_point():
            raise ValueError(f"{name} must use a floating-point dtype")
    if tensors["states"].shape[:2] != tensors["actions"].shape[:2]:
        raise ValueError("state and action tensors must share batch and time dimensions")
    allowed = tensors["allowed_transitions"]
    if allowed.ndim != 2 or allowed.dtype is not torch.bool:
        raise ValueError("allowed transitions must be a rank-two Boolean tensor")
    if not bool(allowed.any(dim=1).all()):
        raise ValueError("every example must allow at least one transition")
    if tensors["risk_score"].shape != (batch,):
        raise ValueError("risk score must have shape [batch]")
    if bool(((tensors["risk_score"] < 0.0) | (tensors["risk_score"] > 1.0)).any()):
        raise ValueError("risk scores must fall in [0, 1]")
    for target_name in ("decision_targets", "transition_targets"):
        target = tensors[target_name]
        if target.shape != (batch,) or target.dtype not in {torch.int32, torch.int64}:
            raise ValueError(f"{target_name} must be a rank-one integer tensor")
        if bool((target < 0).any()):
            raise ValueError(f"{target_name} must contain non-negative class indices")
    transition_targets = tensors["transition_targets"].to(dtype=torch.long)
    if bool((transition_targets >= allowed.shape[1]).any()):
        raise ValueError("transition targets fall outside the allowed-transition width")
    if not bool(allowed.gather(1, transition_targets.unsqueeze(1)).all()):
        raise ValueError("transition targets must be allowed by the host policy")
    if tensors["next_state_targets"].shape != tensors["states"].shape:
        raise ValueError("next-state targets must match the state tensor shape")
    if "attention_mask" in tensors:
        if tensors["attention_mask"].shape != tensors["tokens"].shape:
            raise ValueError("attention mask must match token shape")
        if tensors["attention_mask"].dtype is not torch.bool:
            raise ValueError("attention mask must be Boolean")
        _validate_prefix_mask(tensors["attention_mask"], "attention mask")
    if "temporal_mask" in tensors:
        if tensors["temporal_mask"].shape != tensors["states"].shape[:2]:
            raise ValueError("temporal mask must match state batch and time dimensions")
        if tensors["temporal_mask"].dtype is not torch.bool:
            raise ValueError("temporal mask must be Boolean")
        _validate_prefix_mask(tensors["temporal_mask"], "temporal mask")
    for vector_name in ("domain_ids", "timestamp", "intrinsic_risk_targets"):
        if vector_name in tensors and tensors[vector_name].shape != (batch,):
            raise ValueError(f"{vector_name} must have shape [batch]")
    if "domain_ids" in tensors:
        domain_ids = tensors["domain_ids"]
        if domain_ids.dtype not in {torch.int32, torch.int64} or bool((domain_ids < 0).any()):
            raise ValueError("domain IDs must be non-negative integers")
    for name in ("timestamp", "intrinsic_risk_targets"):
        if name in tensors and not tensors[name].is_floating_point():
            raise ValueError(f"{name} must use a floating-point dtype")
    if "intrinsic_risk_targets" in tensors and bool(
        (
            (tensors["intrinsic_risk_targets"] < 0.0) | (tensors["intrinsic_risk_targets"] > 1.0)
        ).any()
    ):
        raise ValueError("intrinsic-risk targets must fall in [0, 1]")
    if "component_reliability_targets" in tensors:
        reliability = tensors["component_reliability_targets"]
        if reliability.ndim != 2 or not reliability.is_floating_point():
            raise ValueError("component-reliability targets must be a rank-two floating tensor")
        if bool(((reliability < 0.0) | (reliability > 1.0)).any()):
            raise ValueError("component-reliability targets must fall in [0, 1]")
    for name in extensions:
        extension = tensors[f"extension__{name}"]
        if extension.ndim != 2 or not extension.is_floating_point():
            raise ValueError(f"extension {name!r} must be a rank-two floating tensor")


def _validate_prefix_mask(mask: Tensor, name: str) -> None:
    if not bool(mask.any(dim=1).all()):
        raise ValueError(f"every {name} row must contain at least one active position")
    if bool(((~mask).cummax(dim=1).values & mask).any()):
        raise ValueError(f"active positions in {name} must form a contiguous prefix")


def _artifact(
    path: Path,
    *,
    root: Path,
    source_ids: list[str],
    extensions: list[str],
) -> OlympusBatchArtifact:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(resolved_root):
        raise ValueError("dataset batch must be contained by the manifest directory")
    content, tensors = _read_bundle(resolved)
    _validate_bundle(tensors, extensions=extensions, source_ids=source_ids)
    return OlympusBatchArtifact(
        path=resolved.relative_to(resolved_root).as_posix(),
        sha256=_sha256(content),
        source_ids=source_ids,
        shapes={name: tuple(tensor.shape) for name, tensor in sorted(tensors.items())},
        dtypes={name: str(tensor.dtype) for name, tensor in sorted(tensors.items())},
    )


def write_olympus_dataset_manifest(
    output: Path,
    *,
    dataset_id: str,
    version: str,
    license_name: str,
    train_batch: Path,
    evaluation_batch: Path,
    train_source_ids: list[str],
    evaluation_source_ids: list[str],
    extensions: list[str] | None = None,
) -> OlympusSystemDatasetManifest:
    """Bind aligned training and evaluation tensor batches to immutable identities."""

    extension_names = sorted(extensions or [])
    root = output.resolve().parent
    manifest = OlympusSystemDatasetManifest(
        dataset_id=dataset_id,
        version=version,
        license=license_name,
        extensions=extension_names,
        train=_artifact(
            train_batch,
            root=root,
            source_ids=train_source_ids,
            extensions=extension_names,
        ),
        evaluation=_artifact(
            evaluation_batch,
            root=root,
            source_ids=evaluation_source_ids,
            extensions=extension_names,
        ),
    )
    manifest.manifest_sha256 = _sha256(manifest.canonical_bytes(include_hash=False))
    _atomic_write(output, manifest.canonical_bytes())
    return manifest


def _load_artifact(
    root: Path, artifact: OlympusBatchArtifact, extensions: list[str]
) -> dict[str, Tensor]:
    path = root / artifact.path
    if path.is_symlink():
        raise ValueError("dataset batch must not be a symlink")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("dataset batch escapes the manifest directory")
    if not resolved.is_file():
        raise ValueError("dataset batch is missing")
    content = resolved.read_bytes()
    if _sha256(content) != artifact.sha256:
        raise ValueError(f"dataset batch digest changed: {artifact.path}")
    tensors = _decode_bundle(content)
    _validate_bundle(tensors, extensions=extensions, source_ids=artifact.source_ids)
    shapes = {name: tuple(tensor.shape) for name, tensor in sorted(tensors.items())}
    dtypes = {name: str(tensor.dtype) for name, tensor in sorted(tensors.items())}
    if shapes != artifact.shapes or dtypes != artifact.dtypes:
        raise ValueError(f"dataset tensor schema changed: {artifact.path}")
    return tensors


def load_olympus_dataset(manifest_path: Path) -> LoadedOlympusDataset:
    """Verify a complete source-disjoint dataset before returning any tensors."""

    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("dataset manifest must be a regular non-symlink file")
    manifest = OlympusSystemDatasetManifest.model_validate_json(manifest_path.read_bytes())
    expected = _sha256(manifest.canonical_bytes(include_hash=False))
    if manifest.manifest_sha256 != expected:
        raise ValueError("dataset manifest hash mismatch")
    root = manifest_path.resolve().parent
    train = _load_artifact(root, manifest.train, manifest.extensions)
    evaluation = _load_artifact(root, manifest.evaluation, manifest.extensions)
    return LoadedOlympusDataset(manifest=manifest, train=train, evaluation=evaluation)
