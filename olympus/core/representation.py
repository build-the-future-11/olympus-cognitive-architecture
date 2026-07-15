from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field

from olympus.core.schemas import StrictModel


class RepresentationFamily(StrEnum):
    EUCLIDEAN = "euclidean"
    SPHERICAL = "spherical"
    HYPERBOLIC = "hyperbolic"
    GRAPH = "graph"
    TREE = "tree"
    PRODUCT = "product"
    HIERARCHICAL = "hierarchical"
    SET = "set"
    SEQUENCE = "sequence"
    GRID = "grid"
    SIMPLICIAL = "simplicial"
    LOW_RANK = "low_rank"
    DYNAMIC_BASIS = "dynamic_basis"


class RepresentationDescriptor(StrictModel):
    family: RepresentationFamily
    reasons: list[str]
    quality_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class DynamicRepresentationTheory:
    def select(self, sample: dict[str, Any]) -> RepresentationDescriptor:
        edges = sample.get("edges", [])
        sequence = sample.get("sequence", [])
        grid = sample.get("grid", [])
        hierarchy = sample.get("hierarchy_depth", 0)
        if edges and hierarchy:
            return RepresentationDescriptor(
                family=RepresentationFamily.TREE,
                reasons=["graph edges present", "hierarchy depth detected"],
                quality_score=0.89,
                metadata={"depth": hierarchy},
            )
        if edges:
            return RepresentationDescriptor(
                family=RepresentationFamily.GRAPH,
                reasons=["relational edges present"],
                quality_score=0.84,
                metadata={"edge_count": len(edges)},
            )
        if grid:
            return RepresentationDescriptor(
                family=RepresentationFamily.GRID,
                reasons=["two-dimensional regular structure detected"],
                quality_score=0.82,
                metadata={"rows": len(grid)},
            )
        if sequence:
            return RepresentationDescriptor(
                family=RepresentationFamily.SEQUENCE,
                reasons=["ordered temporal data detected"],
                quality_score=0.8,
                metadata={"length": len(sequence)},
            )
        return RepresentationDescriptor(
            family=RepresentationFamily.EUCLIDEAN,
            reasons=["default continuous feature geometry"],
            quality_score=0.72,
            metadata={"feature_count": len(sample)},
        )

    def project(self, values: list[float], descriptor: RepresentationDescriptor) -> list[float]:
        if descriptor.family == RepresentationFamily.SPHERICAL:
            total = sum(abs(value) for value in values) or 1.0
            return [round(value / total, 6) for value in values]
        if descriptor.family in {RepresentationFamily.SEQUENCE, RepresentationFamily.EUCLIDEAN}:
            scale = max(1.0, max(abs(value) for value in values))
            return [round(value / scale, 6) for value in values]
        if descriptor.family == RepresentationFamily.LOW_RANK:
            mean = sum(values) / max(1, len(values))
            return [round(value - mean, 6) for value in values]
        return [round(value, 6) for value in values]
