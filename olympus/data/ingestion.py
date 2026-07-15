from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from pydantic import Field

from olympus.core.schemas import StrictModel


class IngestedDocument(StrictModel):
    identifier: str
    text: str
    language: str = "unknown"
    chunks: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestionPipeline:
    def ingest_file(self, path: Path, chunk_size: int = 120) -> IngestedDocument:
        text = path.read_text(encoding="utf-8")
        return self._normalize(path.name, text, chunk_size)

    def ingest_http(self, url: str, chunk_size: int = 120) -> IngestedDocument:
        with urllib.request.urlopen(url, timeout=5) as response:
            text = response.read().decode("utf-8")
        return self._normalize(url, text, chunk_size)

    def _normalize(self, identifier: str, text: str, chunk_size: int) -> IngestedDocument:
        normalized = " ".join(text.split())
        chunks = [
            normalized[index : index + chunk_size]
            for index in range(0, len(normalized), chunk_size)
        ]
        return IngestedDocument(
            identifier=identifier,
            text=normalized,
            language="en" if normalized.isascii() else "mixed",
            chunks=chunks,
            metadata={"sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest()},
        )

    def shard(self, documents: list[IngestedDocument], output_path: Path) -> None:
        output_path.write_text(
            json.dumps([document.model_dump(mode="json") for document in documents], indent=2),
            encoding="utf-8",
        )
