from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from olympus.forge.compiler import NaturalLanguageBehaviorCompiler
from olympus.forge.runtime import ForgeRuntime


@dataclass(slots=True)
class ExperimentSummary:
    trial_count: int
    mean_confidence: float
    mean_survivors: float


class ExperimentController:
    def __init__(self) -> None:
        self.compiler = NaturalLanguageBehaviorCompiler()
        self.runtime = ForgeRuntime()

    def run_trials(self, behavior_text: str, prompt: str, trials: int = 3) -> ExperimentSummary:
        if trials < 1:
            raise ValueError("trials must be positive")
        compilation = self.compiler.compile(behavior_text)
        confidences: list[float] = []
        survivors: list[int] = []
        for _ in range(trials):
            result = self.runtime.execute(compilation.spec, prompt)
            merged = result.outputs["merged"]
            confidences.append(float(merged["confidence"]))
            survivors.append(int(merged["survivors"]))
        return ExperimentSummary(
            trial_count=trials,
            mean_confidence=round(mean(confidences), 4),
            mean_survivors=round(mean(survivors), 4),
        )
