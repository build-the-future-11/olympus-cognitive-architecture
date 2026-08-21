from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from olympus.core.schemas import StrictModel


class DataSourceRecord(StrictModel):
    name: str
    source_url: str
    organization: str
    license: str
    permitted_uses: list[str]
    retrieval_date: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: str
    quality_score: float = Field(ge=0.0, le=1.0)
    language: str
    domain: str
    personal_data: bool
    training_eligible: bool
    removed: bool = False


class DataSourceRegistry(StrictModel):
    sources: list[DataSourceRecord] = Field(default_factory=list)

    def add(self, source: DataSourceRecord) -> None:
        if any(item.name == source.name and not item.removed for item in self.sources):
            raise ValueError(f"active data source already exists: {source.name}")
        self.sources.append(source)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DataSourceRegistry:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
