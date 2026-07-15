from __future__ import annotations

import ast
from typing import Any

from pydantic import BaseModel, Field

from olympus.core.schemas import StrictModel


class VerificationEvidence(StrictModel):
    verifier: str
    passed: bool
    detail: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SchemaVerifier:
    def verify(self, model: type[BaseModel], payload: dict[str, Any]) -> VerificationEvidence:
        try:
            validated = model.model_validate(payload)
            return VerificationEvidence(
                verifier="schema",
                passed=True,
                detail="Payload conforms to schema.",
                payload=validated.model_dump(),
            )
        except Exception as error:  # noqa: BLE001
            return VerificationEvidence(
                verifier="schema",
                passed=False,
                detail=str(error),
            )


class CodeVerifier:
    def verify(self, code: str) -> VerificationEvidence:
        try:
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            return VerificationEvidence(
                verifier="code",
                passed=True,
                detail="Python source parsed successfully.",
                payload={"functions": functions},
            )
        except SyntaxError as error:
            return VerificationEvidence(
                verifier="code",
                passed=False,
                detail=str(error),
            )


class ContradictionVerifier:
    def verify(self, statements: list[str]) -> VerificationEvidence:
        normalized = {statement.lower().strip() for statement in statements}
        contradiction = any(
            statement.startswith("not ") and statement.removeprefix("not ") in normalized
            for statement in normalized
        )
        return VerificationEvidence(
            verifier="contradiction",
            passed=not contradiction,
            detail="Contradiction detected."
            if contradiction
            else "No direct contradiction detected.",
            payload={"statements": statements},
        )


class CalibrationVerifier:
    def verify(self, confidence: float, passed: bool) -> VerificationEvidence:
        calibrated = not ((confidence > 0.8 and not passed) or (confidence < 0.2 and passed))
        return VerificationEvidence(
            verifier="calibration",
            passed=calibrated,
            detail="Confidence is calibrated."
            if calibrated
            else "Confidence is poorly calibrated.",
            payload={"confidence": confidence, "outcome": passed},
        )
