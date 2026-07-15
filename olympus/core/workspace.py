from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pydantic import Field

from olympus.core.schemas import Evidence, Hypothesis, ProvenanceRecord, StrictModel


class WorkspaceDiff(StrictModel):
    changed_fields: list[str]
    old_values: dict[str, Any]
    new_values: dict[str, Any]


class LatentWorkspace(StrictModel):
    objective: str
    task_state: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    uncertainty: float = 0.5
    active_plan: list[str] = Field(default_factory=list)
    memory_references: list[str] = Field(default_factory=list)
    predicted_consequences: list[str] = Field(default_factory=list)
    tools_available: list[str] = Field(default_factory=list)
    branch_activations: dict[str, float] = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    version: int = 1

    def serialize(self) -> str:
        return self.model_dump_json(indent=2)

    def checkpoint(self, path: Path) -> None:
        path.write_text(self.serialize(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LatentWorkspace:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def merge(self, other: LatentWorkspace) -> LatentWorkspace:
        merged = self.model_copy(deep=True)
        merged.task_state.update(other.task_state)
        merged.constraints = sorted(set(self.constraints + other.constraints))
        merged.evidence.extend(other.evidence)
        merged.hypotheses.extend(other.hypotheses)
        merged.active_plan = self.active_plan + [
            step for step in other.active_plan if step not in self.active_plan
        ]
        merged.memory_references = sorted(set(self.memory_references + other.memory_references))
        merged.predicted_consequences = sorted(
            set(self.predicted_consequences + other.predicted_consequences)
        )
        merged.tools_available = sorted(set(self.tools_available + other.tools_available))
        merged.branch_activations.update(other.branch_activations)
        merged.provenance.extend(other.provenance)
        merged.uncertainty = round((self.uncertainty + other.uncertainty) / 2.0, 4)
        merged.version = max(self.version, other.version) + 1
        return merged

    def diff(self, other: LatentWorkspace) -> WorkspaceDiff:
        current = self.model_dump()
        target = other.model_dump()
        changed_fields = sorted(key for key in current if current[key] != target[key])
        return WorkspaceDiff(
            changed_fields=changed_fields,
            old_values={key: current[key] for key in changed_fields},
            new_values={key: target[key] for key in changed_fields},
        )

    def replay(self) -> list[dict[str, Any]]:
        timeline = [{"event": "workspace_initialized", "objective": self.objective, "version": 1}]
        for record in self.provenance:
            timeline.append(
                {
                    "event": record.action,
                    "actor": record.actor,
                    "metadata": record.metadata,
                }
            )
        return timeline

    def export_snapshot(self, path: Path) -> dict[str, Any]:
        payload = cast(dict[str, Any], json.loads(self.serialize()))
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
