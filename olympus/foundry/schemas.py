from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from olympus.core.schemas import StrictModel


class ArtifactStatus(StrEnum):
    DESIGNED = "DESIGNED"
    RUNNING_EXPERIMENT = "RUNNING_EXPERIMENT"
    CHECKPOINTED = "CHECKPOINTED"
    EVALUATED = "EVALUATED"
    VERIFIED = "VERIFIED"
    NEGATIVE_RESULT = "NEGATIVE_RESULT"
    INCONCLUSIVE = "INCONCLUSIVE"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class DatasetRecord(StrictModel):
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,63}$")
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    materialized_path: str = Field(min_length=1)
    source: str = Field(min_length=1, max_length=2_000)
    owner: str = Field(min_length=1, max_length=200)
    license: str = Field(min_length=1, max_length=200)
    provenance: str = Field(min_length=1, max_length=4_000)
    privacy_classification: Literal["public", "internal", "private", "restricted"]
    synthetic: bool
    generator: str | None = Field(default=None, max_length=500)
    byte_count: int = Field(gt=0)
    character_count: int = Field(gt=0)
    created_at: str

    @model_validator(mode="after")
    def require_synthetic_generator(self) -> DatasetRecord:
        if self.synthetic and not self.generator:
            raise ValueError("synthetic datasets must identify their generator")
        return self


class ExperimentRecord(StrictModel):
    experiment_id: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{2,63}$")
    hypothesis: str = Field(min_length=1, max_length=4_000)
    model_family: str = Field(min_length=1, max_length=100)
    dataset_id: str
    dataset_version: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0, le=2**32 - 1)
    config: dict[str, Any]
    code_commit: str = Field(min_length=1, max_length=64)
    hardware: dict[str, Any]
    status: ArtifactStatus
    failure_reason: str | None = None
    created_at: str
    updated_at: str


class CheckpointRecord(StrictModel):
    checkpoint_id: str = Field(pattern=r"^ckpt_[0-9a-f]{16}$")
    experiment_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str = Field(min_length=1)
    format: str = Field(min_length=1, max_length=100)
    byte_count: int = Field(gt=0)
    metrics: dict[str, float]
    created_at: str


class EvaluationRecord(StrictModel):
    evaluation_id: str = Field(pattern=r"^eval_[0-9a-f]{16}$")
    checkpoint_id: str
    suite: str = Field(min_length=1, max_length=200)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    passed: bool
    decision: str = Field(min_length=1, max_length=2_000)
    created_at: str


class ModelRecord(StrictModel):
    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,127}$")
    family: str = Field(min_length=1, max_length=100)
    version: str = Field(min_length=1, max_length=64)
    checkpoint_id: str
    status: ArtifactStatus
    runtime: str = Field(min_length=1, max_length=100)
    capabilities: list[str]
    intended_use: str = Field(min_length=1, max_length=2_000)
    limitations: list[str]
    created_at: str


class EvidenceRecord(StrictModel):
    sequence: int = Field(gt=0)
    timestamp: str
    entity_type: str
    entity_id: str
    action: str
    payload: dict[str, Any]


class FoundryPipelineResult(StrictModel):
    dataset: DatasetRecord
    experiment: ExperimentRecord
    checkpoint: CheckpointRecord
    evaluation: EvaluationRecord
    model: ModelRecord
    export_path: str


class GenerationResult(StrictModel):
    model: str
    content: str
    finish_reason: Literal["stop", "length"]
    prompt_characters: int = Field(ge=0)
    generated_characters: int = Field(ge=0)
    evidence: dict[str, Any]
