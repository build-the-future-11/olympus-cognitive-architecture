from __future__ import annotations

from pydantic import Field

from olympus.core.schemas import StrictModel


class DatasetSplit(StrictModel):
    train: float
    validation: float
    test: float


class DatasetManifest(StrictModel):
    name: str
    version: str
    split: DatasetSplit
    sources: list[str]
    filters: list[str] = Field(default_factory=list)
    deduplication: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(default_factory=dict)
    licenses: list[str] = Field(default_factory=list)
    contamination_rules: list[str] = Field(default_factory=list)
    transformations: list[str] = Field(default_factory=list)
