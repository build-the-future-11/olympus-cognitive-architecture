from __future__ import annotations

from collections import Counter
from math import sqrt

from olympus.core.modalities import ReferenceEncoder, TextArtifact
from olympus.data.ingestion import IngestedDocument


class RetrievalPlatform:
    def __init__(self) -> None:
        self.encoder = ReferenceEncoder()

    def lexical_search(
        self, query: str, documents: list[IngestedDocument]
    ) -> list[tuple[str, float]]:
        query_tokens = set(query.lower().split())
        scored = []
        for document in documents:
            overlap = query_tokens.intersection(document.text.lower().split())
            scored.append((document.identifier, len(overlap) / max(1, len(query_tokens))))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def semantic_search(
        self, query: str, documents: list[IngestedDocument]
    ) -> list[tuple[str, float]]:
        query_vector = self.encoder.encode(TextArtifact(content=query)).vector
        scored = []
        for document in documents:
            document_vector = self.encoder.encode(TextArtifact(content=document.text)).vector
            numerator = sum(a * b for a, b in zip(query_vector, document_vector, strict=False))
            denominator = (
                sqrt(sum(a * a for a in query_vector) * sum(b * b for b in document_vector)) or 1.0
            )
            scored.append((document.identifier, numerator / denominator))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def contradiction_search(self, claim: str, documents: list[IngestedDocument]) -> list[str]:
        claim_counter = Counter(claim.lower().split())
        contradictors: list[str] = []
        for document in documents:
            text = document.text.lower()
            if any(f"not {word}" in text for word in claim_counter):
                contradictors.append(document.identifier)
        return contradictors
