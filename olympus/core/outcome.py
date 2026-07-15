from __future__ import annotations

from pydantic import Field

from olympus.core.schemas import StrictModel


class OutcomeQuality(StrictModel):
    correctness: float = Field(ge=0.0, le=1.0)
    information_gain: float = Field(ge=0.0, le=1.0)
    usefulness: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)
    reversibility: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0, le=1.0)
    latency: float = Field(ge=0.0, le=1.0)
    uncertainty_resolution: float = Field(ge=0.0, le=1.0)


class OutcomeQualityPredictor:
    def predict(self, candidate: str, evidence_count: int, tool_count: int) -> OutcomeQuality:
        correctness = min(1.0, 0.3 + (0.15 * evidence_count))
        information_gain = min(1.0, 0.2 + (0.1 * tool_count))
        usefulness = min(1.0, 0.35 + (0.08 * evidence_count))
        novelty = 0.65 if "novel" in candidate.lower() else 0.4
        risk = 0.7 if "delete" in candidate.lower() else 0.2
        reversibility = 0.2 if risk > 0.5 else 0.8
        cost = min(1.0, 0.1 + (0.05 * tool_count))
        latency = min(1.0, 0.15 + (0.08 * tool_count))
        uncertainty_resolution = min(1.0, (correctness + information_gain) / 2.0)
        return OutcomeQuality(
            correctness=round(correctness, 4),
            information_gain=round(information_gain, 4),
            usefulness=round(usefulness, 4),
            novelty=round(novelty, 4),
            risk=round(risk, 4),
            reversibility=round(reversibility, 4),
            cost=round(cost, 4),
            latency=round(latency, 4),
            uncertainty_resolution=round(uncertainty_resolution, 4),
        )
