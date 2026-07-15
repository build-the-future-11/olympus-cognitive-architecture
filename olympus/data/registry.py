from __future__ import annotations

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
    sha256: str
    provenance: str
    quality_score: float
    language: str
    domain: str
    personal_data: bool
    training_eligible: bool
    removed: bool = False


class DataSourceRegistry(StrictModel):
    sources: list[DataSourceRecord] = Field(default_factory=list)

    def add(self, source: DataSourceRecord) -> None:
        self.sources.append(source)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
