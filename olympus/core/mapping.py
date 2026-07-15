from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StructuralAnalogy:
    source_features: list[str]
    mapped_features: list[str]
    alignment_score: float
    differences: list[str]
    confidence: float
    literal: bool


class AbstractConcreteMapper:
    def map(self, source: str, target: str) -> StructuralAnalogy:
        source_features = source.lower().replace(",", " ").split()
        target_features = target.lower().replace(",", " ").split()
        overlap = sorted(set(source_features) & set(target_features))
        differences = sorted(set(source_features) ^ set(target_features))
        denominator = max(1, len(set(source_features) | set(target_features)))
        alignment = len(overlap) / denominator
        confidence = min(0.95, 0.35 + alignment)
        literal = alignment > 0.45
        return StructuralAnalogy(
            source_features=source_features,
            mapped_features=target_features,
            alignment_score=round(alignment, 4),
            differences=differences,
            confidence=round(confidence, 4),
            literal=literal,
        )
