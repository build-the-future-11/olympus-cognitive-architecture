from __future__ import annotations

import json
from enum import StrEnum
from graphlib import TopologicalSorter
from typing import Any

import yaml
from pydantic import Field

from olympus.core.schemas import StrictModel


class NodeKind(StrEnum):
    INTERPRET = "interpret"
    VERIFY = "verify"
    MERGE = "merge"
    MEMORY_WRITE = "memory_write"
    TOOL = "tool"
    OUTPUT = "output"


class BehaviorNode(StrictModel):
    id: str
    kind: NodeKind
    description: str
    config: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class BehaviorSpec(StrictModel):
    version: str = "1.0"
    name: str
    description: str
    objective: str
    nodes: list[BehaviorNode]
    outputs: list[str] = Field(default_factory=list)

    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for node in self.nodes:
            lines.append(f'  {node.id}["{node.id}: {node.kind.value}"]')
            for dependency in node.depends_on:
                lines.append(f"  {dependency} --> {node.id}")
        return "\n".join(lines)

    def topological_order(self) -> list[str]:
        graph = {node.id: set(node.depends_on) for node in self.nodes}
        return list(TopologicalSorter(graph).static_order())


class BehaviorLanguage:
    def parse(self, payload: str) -> BehaviorSpec:
        content = (
            yaml.safe_load(payload)
            if payload.strip().startswith(("version:", "name:"))
            else json.loads(payload)
        )
        return BehaviorSpec.model_validate(content)

    def format(self, spec: BehaviorSpec) -> str:
        return yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False)

    def schema(self) -> dict[str, Any]:
        return BehaviorSpec.model_json_schema()

    def migrate(self, payload: dict[str, Any]) -> BehaviorSpec:
        payload.setdefault("version", "1.0")
        payload.setdefault("outputs", ["final_output"])
        return BehaviorSpec.model_validate(payload)

    def semantic_check(self, spec: BehaviorSpec) -> list[str]:
        node_ids = {node.id for node in spec.nodes}
        errors = [
            f"Missing dependency: {dependency}"
            for node in spec.nodes
            for dependency in node.depends_on
            if dependency not in node_ids
        ]
        try:
            spec.topological_order()
        except Exception as error:  # noqa: BLE001
            errors.append(str(error))
        return errors
