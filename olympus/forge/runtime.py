from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from olympus.core.interpretation import InterpretiveSuperpositionNetwork
from olympus.core.verification import CalibrationVerifier, ContradictionVerifier
from olympus.forge.language import BehaviorLanguage, BehaviorSpec, NodeKind
from olympus.memory.store import MemoryRecord, MemoryStore

ToolHandler = Callable[[str, dict[str, Any]], dict[str, Any]]


@dataclass(slots=True)
class RuntimeEvent:
    node_id: str
    kind: str
    payload: dict[str, Any]


@dataclass(slots=True)
class RuntimeResult:
    outputs: dict[str, Any]
    events: list[RuntimeEvent]


class ForgeRuntime:
    """Execute a behavior graph with explicitly injected external effects."""

    def __init__(
        self,
        *,
        tool_handler: ToolHandler | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        self.interpreter = InterpretiveSuperpositionNetwork()
        self.contradiction_verifier = ContradictionVerifier()
        self.calibration_verifier = CalibrationVerifier()
        self.tool_handler = tool_handler
        self.memory = memory

    def execute(self, spec: BehaviorSpec, prompt: str) -> RuntimeResult:
        semantic_errors = BehaviorLanguage().semantic_check(spec)
        if semantic_errors:
            raise ValueError("invalid behavior graph: " + "; ".join(semantic_errors))
        outputs: dict[str, Any] = {"prompt": prompt}
        events: list[RuntimeEvent] = []
        interpretations = self.interpreter.analyze(prompt)

        for node_id in spec.topological_order():
            node = next(item for item in spec.nodes if item.id == node_id)
            if node.kind == NodeKind.INTERPRET:
                outputs["interpretations"] = [
                    branch.model_dump(mode="json") for branch in interpretations
                ]
                events.append(
                    RuntimeEvent(
                        node_id=node.id,
                        kind=node.kind.value,
                        payload={"count": len(interpretations)},
                    )
                )
            elif node.kind == NodeKind.VERIFY:
                contradiction = self.contradiction_verifier.verify(
                    [branch.statement for branch in interpretations]
                )
                calibration = self.calibration_verifier.verify(
                    interpretations[0].confidence, contradiction.passed
                )
                outputs["verification"] = [contradiction.model_dump(), calibration.model_dump()]
                events.append(
                    RuntimeEvent(
                        node_id=node.id,
                        kind=node.kind.value,
                        payload={"passed": contradiction.passed and calibration.passed},
                    )
                )
            elif node.kind == NodeKind.TOOL:
                outputs["tool_result"] = (
                    self.tool_handler(prompt, outputs)
                    if self.tool_handler is not None
                    else {"status": "skipped", "reason": "no tool handler configured"}
                )
                events.append(
                    RuntimeEvent(
                        node_id=node.id, kind=node.kind.value, payload=outputs["tool_result"]
                    )
                )
            elif node.kind == NodeKind.MEMORY_WRITE:
                if self.memory is None:
                    outputs["memory_write"] = {
                        "status": "skipped",
                        "reason": "no memory store configured",
                    }
                else:
                    best = interpretations[0]
                    identity = f"{spec.name}\0{node.id}\0{prompt}".encode()
                    key = f"forge:{hashlib.sha256(identity).hexdigest()}"
                    self.memory.put(
                        MemoryRecord(
                            kind="forge",
                            key=key,
                            value=best.statement,
                            salience=best.confidence,
                            tags=[best.category.value],
                        )
                    )
                    outputs["memory_write"] = {"status": "persisted", "key": key}
                events.append(
                    RuntimeEvent(
                        node_id=node.id, kind=node.kind.value, payload=outputs["memory_write"]
                    )
                )
            elif node.kind == NodeKind.MERGE:
                best = interpretations[0]
                outputs["merged"] = {
                    "statement": best.statement,
                    "confidence": best.confidence,
                    "survivors": len(interpretations),
                }
                events.append(
                    RuntimeEvent(node_id=node.id, kind=node.kind.value, payload=outputs["merged"])
                )
            elif node.kind == NodeKind.OUTPUT:
                outputs["output"] = outputs.get("merged", {})
                events.append(
                    RuntimeEvent(node_id=node.id, kind=node.kind.value, payload=outputs["output"])
                )

        return RuntimeResult(outputs=outputs, events=events)
