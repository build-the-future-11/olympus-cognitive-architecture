from collections.abc import Callable
from pathlib import Path

from olympus.core.budget import AdaptiveComputeAllocator
from olympus.core.division import (
    DependencyAwareDivisionReassembly,
    SpecialistResult,
    SpecialistTask,
)
from olympus.core.interpretation import InterpretationCategory, InterpretiveSuperpositionNetwork
from olympus.core.mapping import AbstractConcreteMapper
from olympus.core.modalities import ReferenceEncoder, TextArtifact
from olympus.core.outcome import OutcomeQualityPredictor
from olympus.core.perspective import AbstractPerspectiveGenerator
from olympus.core.representation import DynamicRepresentationTheory, RepresentationFamily
from olympus.core.schemas import BranchScore, ComputeBudget
from olympus.core.workspace import LatentWorkspace


def test_reference_encoder_normalizes_vectors() -> None:
    encoded = ReferenceEncoder().encode(TextArtifact(content="Olympus maps code to meaning."))
    assert encoded.modality == "text"
    assert abs(sum(value * value for value in encoded.vector) - 1.0) < 0.001


def test_workspace_merge_and_diff(tmp_path: Path) -> None:
    first = LatentWorkspace(objective="A", constraints=["safe"], task_state={"step": 1})
    second = LatentWorkspace(objective="A", constraints=["typed"], task_state={"branch": "x"})
    merged = first.merge(second)
    diff = first.diff(merged)
    checkpoint = tmp_path / "workspace.json"
    merged.checkpoint(checkpoint)
    loaded = LatentWorkspace.load(checkpoint)
    assert "constraints" in diff.changed_fields
    assert loaded.task_state["branch"] == "x"


def test_interpretation_and_compute_allocation() -> None:
    network = InterpretiveSuperpositionNetwork()
    branches = network.analyze(
        "The dependency graph looked like a forest canopy.",
        budget=ComputeBudget(branch_budget=4),
    )
    allocator = AdaptiveComputeAllocator()
    schedule = allocator.schedule(
        {branch.category.value: branch.score for branch in branches},
        ComputeBudget(branch_budget=2),
    )
    assert len(branches) == 4
    assert len(schedule) >= 2


def test_temporal_interpretation_is_reachable() -> None:
    branches = InterpretiveSuperpositionNetwork().analyze(
        "Before deployment, verify the history and future plan.",
        budget=ComputeBudget(branch_budget=9),
    )
    temporal = next(
        branch for branch in branches if branch.category == InterpretationCategory.TEMPORAL
    )
    assert temporal.confidence > 0.5


def test_specialist_dependencies_are_enforced() -> None:
    order: list[str] = []

    def specialist(name: str) -> Callable[[str], SpecialistResult]:
        def run(payload: str) -> SpecialistResult:
            order.append(name)
            return SpecialistResult(name=name, output=payload, confidence=1.0)

        return run

    tasks = [
        SpecialistTask(name="evidence", payload="collect"),
        SpecialistTask(name="synthesis", payload="combine", dependencies=["evidence"]),
    ]
    DependencyAwareDivisionReassembly().execute(
        tasks,
        {"evidence": specialist("evidence"), "synthesis": specialist("synthesis")},
    )
    assert order == ["evidence", "synthesis"]


def test_mapping_perspectives_representation_and_outcomes() -> None:
    analogy = AbstractConcreteMapper().map("dependency graph", "forest canopy")
    perspectives = AbstractPerspectiveGenerator().generate(
        "The graph looked alive, almost ecological."
    )
    descriptor = DynamicRepresentationTheory().select({"edges": [("a", "b")], "hierarchy_depth": 2})
    outcome = OutcomeQualityPredictor().predict(
        "Use a novel verifier.", evidence_count=2, tool_count=1
    )
    assert analogy.confidence > 0.3
    assert len(perspectives) == 5
    assert descriptor.family == RepresentationFamily.TREE
    assert outcome.correctness > 0.5


def test_allocator_freezes_excess_branches() -> None:
    allocator = AdaptiveComputeAllocator()
    schedule = allocator.schedule(
        {
            "a": BranchScore(
                plausibility=0.9, information_gain=0.4, usefulness=0.8, uncertainty_value=0.6
            ),
            "b": BranchScore(
                plausibility=0.7, information_gain=0.7, usefulness=0.7, uncertainty_value=0.5
            ),
            "c": BranchScore(
                plausibility=0.3, information_gain=0.8, usefulness=0.4, uncertainty_value=0.8
            ),
        },
        ComputeBudget(branch_budget=2),
    )
    assert any(item.frozen for item in schedule)
