from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from olympus.core.interpretation import InterpretiveSuperpositionNetwork
from olympus.core.outcome import OutcomeQualityPredictor
from olympus.core.verification import CalibrationVerifier, ContradictionVerifier
from olympus.forge.language import BehaviorSpec, NodeKind


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
    def __init__(self) -> None:
        self.interpreter = InterpretiveSuperpositionNetwork()
        self.outcome_predictor = OutcomeQualityPredictor()
        self.contradiction_verifier = ContradictionVerifier()
        self.calibration_verifier = CalibrationVerifier()

    def execute(self, spec: BehaviorSpec, prompt: str) -> RuntimeResult:
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
                top_statement = interpretations[0].statement
                contradiction = self.contradiction_verifier.verify([top_statement])
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
                quality = self.outcome_predictor.predict(prompt, evidence_count=1, tool_count=1)
                outputs["tool_result"] = quality.model_dump()
                events.append(
                    RuntimeEvent(
                        node_id=node.id, kind=node.kind.value, payload=outputs["tool_result"]
                    )
                )
            elif node.kind == NodeKind.MEMORY_WRITE:
                outputs["memory_write"] = {
                    "status": "captured",
                    "keys": ["interpretations", "verification"],
                }
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
