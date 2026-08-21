from __future__ import annotations

from pydantic import Field, model_validator

from olympus.core.schemas import StrictModel


class DatasetSplit(StrictModel):
    train: float = Field(ge=0.0, le=1.0)
    validation: float = Field(ge=0.0, le=1.0)
    test: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_total(self) -> DatasetSplit:
        if abs((self.train + self.validation + self.test) - 1.0) > 1e-9:
            raise ValueError("dataset split fractions must sum to 1.0")
        return self


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
