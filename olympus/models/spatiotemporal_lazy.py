"""Hash-bound lazy parameter residency planned from space, time, and scale."""

from __future__ import annotations

import hashlib
import io
import math
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, cast

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from olympus.core.schemas import StrictModel


class SpatialBounds(StrictModel):
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    @model_validator(mode="after")
    def ordered(self) -> SpatialBounds:
        if any(low > high for low, high in zip(self.minimum, self.maximum, strict=True)):
            raise ValueError("spatial minimum must not exceed maximum")
        return self


class TemporalBounds(StrictModel):
    start: float
    end: float

    @model_validator(mode="after")
    def ordered(self) -> TemporalBounds:
        if self.start > self.end:
            raise ValueError("temporal start must not exceed end")
        return self


class ParameterShard(StrictModel):
    shard_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)
    tensor_names: list[str] = Field(min_length=1)
    spatial: SpatialBounds | None = None
    temporal: TemporalBounds | None = None
    resolution_min: float = Field(default=0.0, ge=0.0)
    resolution_max: float = Field(default=1.0e30, gt=0.0)
    priority: float = 0.0
    always_load: bool = False

    @model_validator(mode="after")
    def valid_shard(self) -> ParameterShard:
        relative = PurePosixPath(self.path)
        if relative.is_absolute() or ".." in relative.parts or self.path in {"", "."}:
            raise ValueError("shard path must be a contained relative path")
        if self.resolution_min > self.resolution_max:
            raise ValueError("resolution minimum must not exceed maximum")
        if len(set(self.tensor_names)) != len(self.tensor_names):
            raise ValueError("tensor names must be unique within a shard")
        return self


class ParameterManifest(StrictModel):
    schema_version: int = 1
    shards: list[ParameterShard] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_shards(self) -> ParameterManifest:
        identifiers = [shard.shard_id for shard in self.shards]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("shard identifiers must be unique")
        tensor_names = [name for shard in self.shards for name in shard.tensor_names]
        if len(set(tensor_names)) != len(tensor_names):
            raise ValueError("tensor ownership must be unique across shards")
        return self


class SpatiotemporalQuery(StrictModel):
    position: tuple[float, float, float]
    timestamp: float
    resolution: float = Field(gt=0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lookahead_seconds: float = Field(default=0.0, ge=0.0)

    @property
    def predicted_position(self) -> tuple[float, float, float]:
        return cast(
            tuple[float, float, float],
            tuple(
                position + velocity * self.lookahead_seconds
                for position, velocity in zip(self.position, self.velocity, strict=True)
            ),
        )


class PlannedShard(StrictModel):
    shard_id: str
    score: float
    byte_count: int


class ResidencyPlan(StrictModel):
    shards: list[PlannedShard]
    planned_bytes: int = Field(ge=0)
    budget_bytes: int = Field(gt=0)


class ModuleMaterialization(StrictModel):
    loaded_tensors: list[str]
    remaining_meta_tensors: list[str]
    resident_shards: list[str]


def _axis_distance(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _score(shard: ParameterShard, query: SpatiotemporalQuery) -> float:
    score = shard.priority
    if shard.spatial is not None:
        distance = math.sqrt(
            sum(
                _axis_distance(value, low, high) ** 2
                for value, low, high in zip(
                    query.predicted_position,
                    shard.spatial.minimum,
                    shard.spatial.maximum,
                    strict=True,
                )
            )
        )
        diagonal = math.sqrt(
            sum(
                (high - low) ** 2
                for low, high in zip(shard.spatial.minimum, shard.spatial.maximum, strict=True)
            )
        )
        score -= distance / max(diagonal, 1e-9)
    if shard.temporal is not None:
        score -= _axis_distance(query.timestamp, shard.temporal.start, shard.temporal.end)
    if query.resolution < shard.resolution_min:
        score -= abs(math.log2(query.resolution / shard.resolution_min))
    elif query.resolution > shard.resolution_max:
        score -= abs(math.log2(query.resolution / shard.resolution_max))
    return score


class SpatiotemporalParameterStore:
    """Plan and lazily load verified parameter shards under a byte budget."""

    def __init__(self, manifest_path: Path, *, cache_budget_bytes: int) -> None:
        if cache_budget_bytes <= 0:
            raise ValueError("cache budget must be positive")
        self.manifest_path = manifest_path.resolve()
        self.root = self.manifest_path.parent
        self.manifest = ParameterManifest.model_validate_json(self.manifest_path.read_bytes())
        self.cache_budget_bytes = cache_budget_bytes
        self._records = {shard.shard_id: shard for shard in self.manifest.shards}
        self._cache: OrderedDict[str, dict[str, Tensor]] = OrderedDict()
        self._cache_bytes = 0
        self.disk_loads = 0

    @property
    def resident_shards(self) -> tuple[str, ...]:
        return tuple(self._cache)

    def plan(self, query: SpatiotemporalQuery, *, budget_bytes: int) -> ResidencyPlan:
        if budget_bytes <= 0:
            raise ValueError("residency budget must be positive")
        if budget_bytes > self.cache_budget_bytes:
            raise ValueError("residency budget must not exceed the cache budget")
        mandatory = sorted(
            (shard for shard in self.manifest.shards if shard.always_load),
            key=lambda shard: shard.shard_id,
        )
        mandatory_bytes = sum(shard.byte_count for shard in mandatory)
        if mandatory_bytes > budget_bytes:
            raise ValueError("residency budget cannot fit all mandatory shards")
        optional = sorted(
            (
                (shard, _score(shard, query))
                for shard in self.manifest.shards
                if not shard.always_load
            ),
            key=lambda item: (-item[1], item[0].shard_id),
        )
        selected = [
            PlannedShard(
                shard_id=shard.shard_id,
                score=_score(shard, query),
                byte_count=shard.byte_count,
            )
            for shard in mandatory
        ]
        used = mandatory_bytes
        for shard, score in optional:
            if used + shard.byte_count > budget_bytes:
                continue
            selected.append(
                PlannedShard(
                    shard_id=shard.shard_id,
                    score=score,
                    byte_count=shard.byte_count,
                )
            )
            used += shard.byte_count
        return ResidencyPlan(
            shards=selected,
            planned_bytes=used,
            budget_bytes=budget_bytes,
        )

    def _path(self, record: ParameterShard) -> Path:
        path = self.root / record.path
        if path.is_symlink():
            raise ValueError(f"parameter shard {record.shard_id} must not be a symlink")
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"parameter shard {record.shard_id} escapes the manifest root")
        return resolved

    def _load(self, record: ParameterShard) -> dict[str, Tensor]:
        path = self._path(record)
        payload = path.read_bytes()
        if len(payload) != record.byte_count:
            raise ValueError(f"parameter shard {record.shard_id} byte count changed")
        if hashlib.sha256(payload).hexdigest() != record.sha256:
            raise ValueError(f"parameter shard {record.shard_id} digest changed")
        loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or any(
            not isinstance(name, str) or not isinstance(value, Tensor)
            for name, value in loaded.items()
        ):
            raise ValueError(f"parameter shard {record.shard_id} has invalid tensors")
        tensors = cast(dict[str, Tensor], loaded)
        if set(tensors) != set(record.tensor_names):
            raise ValueError(f"parameter shard {record.shard_id} tensor names changed")
        self.disk_loads += 1
        return tensors

    def _admit(self, record: ParameterShard, tensors: dict[str, Tensor]) -> None:
        if record.byte_count > self.cache_budget_bytes:
            raise ValueError(f"parameter shard {record.shard_id} exceeds the cache budget")
        while self._cache and self._cache_bytes + record.byte_count > self.cache_budget_bytes:
            evicted_id, _evicted = self._cache.popitem(last=False)
            self._cache_bytes -= self._records[evicted_id].byte_count
        self._cache[record.shard_id] = tensors
        self._cache.move_to_end(record.shard_id)
        self._cache_bytes += record.byte_count

    def materialize(self, plan: ResidencyPlan) -> dict[str, Tensor]:
        if plan.planned_bytes > plan.budget_bytes:
            raise ValueError("residency plan exceeds its declared budget")
        tensors: dict[str, Tensor] = {}
        for planned in plan.shards:
            record = self._records.get(planned.shard_id)
            if record is None or record.byte_count != planned.byte_count:
                raise ValueError("residency plan does not match the loaded manifest")
            if record.shard_id not in self._cache:
                self._admit(record, self._load(record))
            else:
                self._cache.move_to_end(record.shard_id)
            overlap = tensors.keys() & self._cache[record.shard_id].keys()
            if overlap:
                raise ValueError(f"residency plan contains duplicate tensors: {sorted(overlap)}")
            tensors.update(self._cache[record.shard_id])
        return tensors

    def materialize_module(
        self,
        module: nn.Module,
        plan: ResidencyPlan,
        *,
        require_complete: bool = False,
    ) -> ModuleMaterialization:
        """Assign selected tensors, including into a module initialized on ``meta``."""

        tensors = self.materialize(plan)
        expected = module.state_dict()
        unknown = tensors.keys() - expected.keys()
        if unknown:
            raise ValueError(f"parameter shards contain unknown module tensors: {sorted(unknown)}")
        for name, tensor in tensors.items():
            reference = expected[name]
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise ValueError(f"parameter shard tensor {name} shape or dtype changed")
        missing = sorted(expected.keys() - tensors.keys())
        if require_complete and missing:
            raise ValueError(f"residency plan leaves module tensors unloaded: {missing}")
        module.load_state_dict(tensors, strict=False, assign=True)
        remaining_meta = sorted(
            name for name, tensor in module.state_dict().items() if tensor.is_meta
        )
        return ModuleMaterialization(
            loaded_tensors=sorted(tensors),
            remaining_meta_tensors=remaining_meta,
            resident_shards=list(self.resident_shards),
        )


def build_parameter_manifest(shards: list[ParameterShard], output: Path) -> ParameterManifest:
    """Write a deterministic manifest after validating caller-created shard files."""

    manifest = ParameterManifest(shards=shards)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = manifest.model_dump_json(indent=2) + "\n"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(output)
    return manifest


def parameter_shard_record(
    shard_id: str,
    path: Path,
    *,
    root: Path,
    spatial: SpatialBounds | None = None,
    temporal: TemporalBounds | None = None,
    resolution_min: float = 0.0,
    resolution_max: float = 1.0e30,
    priority: float = 0.0,
    always_load: bool = False,
) -> ParameterShard:
    """Bind an existing tensor dictionary to its immutable shard metadata."""

    resolved_root = root.resolve()
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(resolved_root):
        raise ValueError("parameter shard must be a contained regular file")
    content = resolved.read_bytes()
    payload: Any = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or not payload
        or any(
            not isinstance(name, str) or not isinstance(value, Tensor)
            for name, value in payload.items()
        )
    ):
        raise ValueError("parameter shard must contain a non-empty string-to-tensor mapping")
    relative = resolved.relative_to(resolved_root).as_posix()
    return ParameterShard(
        shard_id=shard_id,
        path=relative,
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
        tensor_names=sorted(cast(dict[str, Tensor], payload)),
        spatial=spatial,
        temporal=temporal,
        resolution_min=resolution_min,
        resolution_max=resolution_max,
        priority=priority,
        always_load=always_load,
    )
