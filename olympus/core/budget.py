from __future__ import annotations

from olympus.core.schemas import BranchScore, ComputeBudget, StrictModel


class ScheduledBranch(StrictModel):
    branch_id: str
    priority: float
    frozen: bool = False
    revived: bool = False


class AdaptiveComputeAllocator:
    def schedule(
        self,
        scores: dict[str, BranchScore],
        budget: ComputeBudget,
    ) -> list[ScheduledBranch]:
        ranked = sorted(scores.items(), key=lambda item: item[1].priority(), reverse=True)
        scheduled: list[ScheduledBranch] = []
        for index, (branch_id, score) in enumerate(ranked):
            if index >= budget.branch_budget:
                scheduled.append(
                    ScheduledBranch(branch_id=branch_id, priority=score.priority(), frozen=True)
                )
                continue
            revived = score.information_gain > 0.7 and score.plausibility < 0.4
            scheduled.append(
                ScheduledBranch(
                    branch_id=branch_id,
                    priority=round(score.priority(), 6),
                    revived=revived,
                )
            )
        return scheduled
