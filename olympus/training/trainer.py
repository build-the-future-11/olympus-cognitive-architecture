from __future__ import annotations

from dataclasses import dataclass
from random import Random

from olympus.core.prediction import JEPATrainingResult, train_jepa
from olympus.core.transform import TransformMetrics, train_learned_transform
from olympus.training.synthetic import transform_signals


@dataclass(slots=True)
class TrainingReport:
    seed: int
    jepa: JEPATrainingResult
    transform: TransformMetrics


class OlympusTrainer:
    def run(self, seed: int = 7) -> TrainingReport:
        Random(seed).random()
        jepa = train_jepa()
        transform = train_learned_transform(transform_signals())
        return TrainingReport(seed=seed, jepa=jepa, transform=transform)
