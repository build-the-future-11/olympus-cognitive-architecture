from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import Field

from olympus.core.schemas import BranchScore, ComputeBudget, Evidence, Hypothesis, StrictModel


class InterpretationCategory(StrEnum):
    LITERAL = "literal"
    CONCRETE_RESEMBLANCE = "concrete_resemblance"
    ABSTRACT_CONCRETE_ANALOGY = "abstract_concrete_analogy"
    ABSTRACT_NON_CONCRETE = "abstract_non_concrete"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    SYMBOLIC = "symbolic"
    ADVERSARIAL = "adversarial"
    NULL = "null"


class InterpretiveBranch(StrictModel):
    category: InterpretationCategory
    statement: str
    score: BranchScore
    support: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class InterpretiveSuperpositionNetwork:
    analogy_keywords = {"like", "as", "resembles", "similar", "mirror"}
    symbolic_keywords = {"symbol", "meaning", "story", "dream", "signal"}
    causal_keywords = {"because", "therefore", "caused", "led", "drives"}
    temporal_keywords = {"before", "after", "during", "future", "history"}

    def analyze(
        self,
        prompt: str,
        evidence: list[Evidence] | None = None,
        budget: ComputeBudget | None = None,
    ) -> list[InterpretiveBranch]:
        evidence = evidence or []
        budget = budget or ComputeBudget()
        lowered = prompt.lower()
        tokens = lowered.split()
        counts = Counter(tokens)
        branches = [
            self._make_branch(InterpretationCategory.LITERAL, prompt, counts, evidence),
            self._make_branch(
                InterpretationCategory.CONCRETE_RESEMBLANCE,
                f"Physical resemblance reading: {prompt}",
                counts,
                evidence,
            ),
            self._make_branch(
                InterpretationCategory.ABSTRACT_CONCRETE_ANALOGY,
                f"Analogy reading: {prompt}",
                counts,
                evidence,
            ),
            self._make_branch(
                InterpretationCategory.ABSTRACT_NON_CONCRETE,
                f"Conceptual reading: {prompt}",
                counts,
                evidence,
            ),
            self._make_branch(
                InterpretationCategory.CAUSAL,
                f"Causal reading: {prompt}",
                counts,
                evidence,
            ),
            self._make_branch(
                InterpretationCategory.SYMBOLIC,
                f"Symbolic reading: {prompt}",
                counts,
                evidence,
            ),
            self._make_branch(
                InterpretationCategory.ADVERSARIAL,
                f"Adversarial or deceptive reading: {prompt}",
                counts,
                evidence,
            ),
            self._make_branch(
                InterpretationCategory.NULL,
                "Insufficient structure for a grounded interpretation.",
                counts,
                evidence,
            ),
        ]
        ranked = sorted(branches, key=lambda item: item.score.priority(), reverse=True)
        trimmed = ranked[: budget.branch_budget]
        if len(trimmed) > 1:
            trimmed[-1] = trimmed[-1].model_copy(
                update={"confidence": max(0.15, trimmed[-1].confidence)}
            )
        return trimmed

    def summarize_hypotheses(self, branches: list[InterpretiveBranch]) -> list[Hypothesis]:
        return [
            Hypothesis(
                name=branch.category.value,
                description=branch.statement,
                confidence=branch.confidence,
                evidence=[
                    Evidence(
                        source="interpretive_superposition",
                        detail=support,
                        weight=branch.confidence,
                    )
                    for support in branch.support
                ],
                provenance=["interpretation"],
            )
            for branch in branches
        ]

    def _make_branch(
        self,
        category: InterpretationCategory,
        statement: str,
        counts: Counter[str],
        evidence: list[Evidence],
    ) -> InterpretiveBranch:
        evidence_score = self._score_evidence(category, counts, evidence)
        coherence = min(1.0, 0.35 + evidence_score)
        novelty = min(
            1.0,
            0.2
            + (
                0.3
                if category
                in {
                    InterpretationCategory.SYMBOLIC,
                    InterpretationCategory.ADVERSARIAL,
                }
                else 0.1
            ),
        )
        usefulness = min(1.0, 0.3 + evidence_score)
        uncertainty_value = 1.0 - abs(0.5 - evidence_score)
        compute_cost = 1.0 if category == InterpretationCategory.LITERAL else 1.2
        contradiction_penalty = 0.1 if category == InterpretationCategory.NULL else 0.0
        plausibility = max(0.05, coherence - contradiction_penalty)
        confidence = round((plausibility + usefulness) / 2.0, 4)
        support = [item.detail for item in evidence if item.weight > 0.4]
        if not support:
            support = [f"keyword-match={evidence_score:.2f}"]
        return InterpretiveBranch(
            category=category,
            statement=statement,
            score=BranchScore(
                plausibility=round(plausibility, 4),
                information_gain=round(novelty, 4),
                usefulness=round(usefulness, 4),
                uncertainty_value=round(uncertainty_value, 4),
                compute_cost=compute_cost,
            ),
            support=support,
            confidence=confidence,
        )

    def _score_evidence(
        self,
        category: InterpretationCategory,
        counts: Counter[str],
        evidence: list[Evidence],
    ) -> float:
        token_bonus = 0.0
        if category == InterpretationCategory.ABSTRACT_CONCRETE_ANALOGY:
            token_bonus += (
                0.3 if any(keyword in counts for keyword in self.analogy_keywords) else 0.05
            )
        if category == InterpretationCategory.SYMBOLIC:
            token_bonus += (
                0.25 if any(keyword in counts for keyword in self.symbolic_keywords) else 0.05
            )
        if category == InterpretationCategory.CAUSAL:
            token_bonus += (
                0.25 if any(keyword in counts for keyword in self.causal_keywords) else 0.05
            )
        if category == InterpretationCategory.TEMPORAL:
            token_bonus += (
                0.25 if any(keyword in counts for keyword in self.temporal_keywords) else 0.05
            )
        evidence_bonus = sum(item.weight for item in evidence) / max(1, len(evidence))
        literal_bonus = 0.2 if category == InterpretationCategory.LITERAL else 0.0
        return min(1.0, token_bonus + (0.5 * evidence_bonus) + literal_bonus)
