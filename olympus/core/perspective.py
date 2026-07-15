from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from olympus.core.schemas import StrictModel


class PerspectiveKind(StrEnum):
    CONCEPTUAL = "conceptual"
    SYMBOLIC = "symbolic"
    EMOTIONAL = "emotional"
    THEORETICAL = "theoretical"
    NONLITERAL = "nonliteral"


class Perspective(StrictModel):
    kind: PerspectiveKind
    interpretation: str
    evidence_spans: list[str] = Field(default_factory=list)
    confidence: float
    factual: bool = False


class AbstractPerspectiveGenerator:
    templates = {
        PerspectiveKind.CONCEPTUAL: "Conceptual framing: {text}",
        PerspectiveKind.SYMBOLIC: "Symbolic framing: {text}",
        PerspectiveKind.EMOTIONAL: "Emotional subtext: {text}",
        PerspectiveKind.THEORETICAL: "Theoretical framing: {text}",
        PerspectiveKind.NONLITERAL: "Nonliteral perspective: {text}",
    }

    def generate(self, text: str) -> list[Perspective]:
        spans = [segment.strip() for segment in text.split(",") if segment.strip()] or [text]
        results: list[Perspective] = []
        for index, kind in enumerate(PerspectiveKind):
            confidence = round(max(0.25, 0.72 - (index * 0.08)), 4)
            results.append(
                Perspective(
                    kind=kind,
                    interpretation=self.templates[kind].format(text=text),
                    evidence_spans=spans[:2],
                    confidence=confidence,
                    factual=False,
                )
            )
        return results
