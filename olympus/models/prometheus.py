from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection
from typing import Annotated, Literal, Self, cast

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn
from torch.nn import functional as F

from olympus.core.schemas import StrictModel
from olympus.models.substrate import (
    EvidenceItem,
    WorkspaceState,
    authorized_workspace_view,
    canonical_sha256,
    validate_workspace_snapshot,
)

Scalar = str | int | float | bool
EvidenceID = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]


class CheckableTransition(StrictModel):
    """An inspectable reasoning step with an objective expected observation."""

    step: int = Field(ge=0)
    operation: str = Field(min_length=1, max_length=500)
    check: Literal["exact", "numeric", "boolean"]
    expected: Scalar
    observed: Scalar
    tolerance: float = Field(default=0.0, ge=0.0)
    evidence_ids: list[EvidenceID] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_check_values(self) -> Self:
        if self.check == "numeric":
            values = (self.expected, self.observed)
            if any(
                isinstance(value, bool) or not isinstance(value, int | float) for value in values
            ):
                raise ValueError("numeric checks require numeric expected and observed values")
            if any(not math.isfinite(float(value)) for value in values):
                raise ValueError("numeric checks require finite values")
        elif self.check == "boolean":
            if not isinstance(self.expected, bool) or not isinstance(self.observed, bool):
                raise ValueError("boolean checks require boolean expected and observed values")
        elif self.tolerance != 0.0:
            raise ValueError("tolerance is only valid for numeric checks")
        return self

    def passes(self) -> bool:
        if self.check == "numeric":
            return math.isclose(
                float(self.expected),
                float(self.observed),
                rel_tol=0.0,
                abs_tol=self.tolerance,
            )
        return type(self.expected) is type(self.observed) and self.expected == self.observed


class HypothesisBranch(StrictModel):
    branch_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
    claim: str = Field(min_length=1, max_length=4_000)
    assumptions: list[str] = Field(default_factory=list)
    transitions: list[CheckableTransition] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def require_ordered_steps(self) -> Self:
        steps = [transition.step for transition in self.transitions]
        if steps != list(range(len(steps))):
            raise ValueError("transition steps must be contiguous and start at zero")
        return self

    def evidence_coverage(self, resolved_evidence_ids: set[str] | frozenset[str]) -> float:
        covered = sum(
            bool(set(transition.evidence_ids) & resolved_evidence_ids)
            for transition in self.transitions
        )
        return covered / len(self.transitions)


class TransitionReceipt(StrictModel):
    """Canonical external observation consumed by deterministic branch checks."""

    schema_version: Literal["prometheus.transition.v1"] = "prometheus.transition.v1"
    workspace_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    model_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    branch_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
    claim_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    operation: str = Field(min_length=1, max_length=500)
    check: Literal["exact", "numeric", "boolean"]
    expected: Scalar
    observed: Scalar
    tolerance: float = Field(default=0.0, ge=0.0)
    issued_by: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def values_are_checkable(self) -> TransitionReceipt:
        CheckableTransition(
            step=0,
            operation=self.operation,
            check=self.check,
            expected=self.expected,
            observed=self.observed,
            tolerance=self.tolerance,
        )
        return self

    def canonical_text(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


class BoundedBranchSet(StrictModel):
    branch_budget: int = Field(ge=1, le=64)
    branches: list[HypothesisBranch] = Field(min_length=1, max_length=64)
    proposer_scores: list[float] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def enforce_budget_and_identity(self) -> Self:
        if len(self.branches) > self.branch_budget:
            raise ValueError("branch set exceeds its declared budget")
        if len(self.branches) != len(self.proposer_scores):
            raise ValueError("every branch must have exactly one proposer score")
        branch_ids = [branch.branch_id for branch in self.branches]
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("branch IDs must be unique")
        return self


class TrainableBranchProposer(nn.Module):
    """Scores explicit candidate branches under a hard branch budget."""

    def __init__(self, context_dim: int, branch_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if min(context_dim, branch_dim, hidden_dim) < 1:
            raise ValueError("model dimensions must be positive")
        self.context_dim = context_dim
        self.branch_dim = branch_dim
        self.context_encoder = nn.Linear(context_dim, hidden_dim)
        self.branch_encoder = nn.Linear(branch_dim, hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, context: Tensor, branch_features: Tensor) -> Tensor:
        if context.ndim != 1 or context.shape[0] != self.context_dim:
            raise ValueError("context must be a one-dimensional context feature vector")
        if branch_features.ndim != 2 or branch_features.shape[1] != self.branch_dim:
            raise ValueError("branch_features must contain one row per candidate branch")
        if branch_features.shape[0] < 1:
            raise ValueError("at least one candidate branch is required")
        if not torch.isfinite(context).all() or not torch.isfinite(branch_features).all():
            raise ValueError("proposer features must be finite")
        encoded_context = self.context_encoder(context).unsqueeze(0)
        hidden = torch.tanh(encoded_context + self.branch_encoder(branch_features))
        return cast(Tensor, self.score_head(hidden).squeeze(-1))

    def propose(
        self,
        context: Tensor,
        candidates: list[HypothesisBranch],
        branch_features: Tensor,
        *,
        branch_budget: int,
    ) -> BoundedBranchSet:
        if branch_budget < 1 or branch_budget > 64:
            raise ValueError("branch_budget must be between 1 and 64")
        if len(candidates) != branch_features.shape[0]:
            raise ValueError("candidates and branch feature rows must have equal length")
        with torch.no_grad():
            logits = self(context, branch_features)
            probabilities = torch.softmax(logits, dim=0)
            count = min(branch_budget, len(candidates))
            indices = torch.topk(logits, k=count).indices.tolist()
        return BoundedBranchSet(
            branch_budget=branch_budget,
            branches=[candidates[index] for index in indices],
            proposer_scores=[float(probabilities[index].item()) for index in indices],
        )


class TrainableProcessVerifier(nn.Module):
    """An independently parameterized validity classifier for reasoning transitions."""

    def __init__(self, context_dim: int, transition_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if min(context_dim, transition_dim, hidden_dim) < 1:
            raise ValueError("model dimensions must be positive")
        self.context_dim = context_dim
        self.transition_dim = transition_dim
        self.network = nn.Sequential(
            nn.Linear(context_dim + transition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: Tensor, transition_features: Tensor) -> Tensor:
        if context.ndim != 1 or context.shape[0] != self.context_dim:
            raise ValueError("context must be a one-dimensional context feature vector")
        if transition_features.ndim != 2 or transition_features.shape[1] != self.transition_dim:
            raise ValueError("transition_features must contain one row per transition")
        if transition_features.shape[0] < 1:
            raise ValueError("at least one transition is required")
        if not torch.isfinite(context).all() or not torch.isfinite(transition_features).all():
            raise ValueError("verifier features must be finite")
        repeated_context = context.unsqueeze(0).expand(transition_features.shape[0], -1)
        return cast(
            Tensor,
            self.network(torch.cat((repeated_context, transition_features), dim=1)).squeeze(-1),
        )


class BranchAssessment(StrictModel):
    branch: HypothesisBranch
    proposer_score: float = Field(ge=0.0, le=1.0)
    verifier_scores: list[float] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def align_transition_scores(self) -> Self:
        if len(self.verifier_scores) != len(self.branch.transitions):
            raise ValueError("every transition must have exactly one verifier score")
        if any(score < 0.0 or score > 1.0 for score in self.verifier_scores):
            raise ValueError("verifier scores must be probabilities")
        return self


class SelectionResult(StrictModel):
    decision: Literal["selected", "abstained"]
    branch_id: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    reason: str

    @model_validator(mode="after")
    def enforce_decision_shape(self) -> Self:
        if (self.decision == "selected") != (self.branch_id is not None):
            raise ValueError("selected decisions require a branch ID; abstentions forbid one")
        return self


class PrometheusSelector:
    def __init__(
        self,
        *,
        minimum_score: float = 0.65,
        minimum_evidence_coverage: float = 0.5,
        trusted_receipt_sha256: Collection[str] = (),
    ) -> None:
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between zero and one")
        if not 0.0 <= minimum_evidence_coverage <= 1.0:
            raise ValueError("minimum_evidence_coverage must be between zero and one")
        if any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in trusted_receipt_sha256
        ):
            raise ValueError("trusted receipt hashes must be lowercase SHA-256 values")
        self.minimum_score = minimum_score
        self.minimum_evidence_coverage = minimum_evidence_coverage
        self.trusted_receipt_sha256 = frozenset(trusted_receipt_sha256)

    def select(
        self, assessments: list[BranchAssessment], workspace: WorkspaceState
    ) -> SelectionResult:
        assessments = [
            BranchAssessment.model_validate(assessment.model_dump(mode="python"))
            for assessment in assessments
        ]
        workspace = validate_workspace_snapshot(workspace)
        view = authorized_workspace_view(workspace)
        evidence_by_id = {item.evidence_id: item for item in view.evidence}
        authorized_ids = set(evidence_by_id)
        eligible: list[tuple[float, BranchAssessment]] = []
        for assessment in assessments:
            branch = assessment.branch
            if not all(transition.passes() for transition in branch.transitions):
                continue
            referenced_ids = {
                evidence_id
                for transition in branch.transitions
                for evidence_id in transition.evidence_ids
            }
            if not referenced_ids.issubset(authorized_ids):
                continue
            verified_receipt_ids = {
                evidence_id
                for transition in branch.transitions
                for evidence_id in transition.evidence_ids
                if evidence_by_id[evidence_id].content_sha256
                in self.trusted_receipt_sha256
                and _receipt_matches_transition(
                    evidence_by_id[evidence_id], transition, branch, workspace
                )
            }
            coverage = branch.evidence_coverage(verified_receipt_ids)
            if coverage < self.minimum_evidence_coverage:
                continue
            verifier_score = sum(assessment.verifier_scores) / len(assessment.verifier_scores)
            score = (0.25 * assessment.proposer_score) + (0.60 * verifier_score) + (0.15 * coverage)
            eligible.append((score, assessment))
        if not eligible:
            return SelectionResult(
                decision="abstained",
                score=0.0,
                reason="no branch passed deterministic checks and evidence coverage",
            )
        best_score, best = max(eligible, key=lambda item: item[0])
        bounded_score = min(1.0, max(0.0, best_score))
        if bounded_score < self.minimum_score:
            return SelectionResult(
                decision="abstained",
                score=bounded_score,
                reason="best checked branch remained below the selection threshold",
            )
        return SelectionResult(
            decision="selected",
            branch_id=best.branch.branch_id,
            score=bounded_score,
            reason="highest-scoring branch passed all deterministic checks",
        )


def proposer_classification_loss(logits: Tensor, target_index: int) -> Tensor:
    if logits.ndim != 1 or logits.numel() < 1:
        raise ValueError("proposer logits must be a non-empty one-dimensional tensor")
    if target_index < 0 or target_index >= logits.numel():
        raise ValueError("target_index is outside the candidate set")
    target = torch.tensor([target_index], device=logits.device)
    return F.cross_entropy(logits.unsqueeze(0), target)


def verifier_process_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("verifier logits and labels must be equally sized vectors")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("verifier labels must be binary")
    return F.binary_cross_entropy_with_logits(logits, labels.to(dtype=logits.dtype))


def branch_diversity_loss(branch_embeddings: Tensor) -> Tensor:
    if branch_embeddings.ndim != 2 or branch_embeddings.shape[0] < 1:
        raise ValueError("branch embeddings must be a non-empty matrix")
    if not torch.isfinite(branch_embeddings).all():
        raise ValueError("branch embeddings must be finite")
    if branch_embeddings.shape[0] == 1:
        return branch_embeddings.sum() * 0.0
    normalized = F.normalize(branch_embeddings, p=2, dim=1)
    similarities = normalized @ normalized.T
    mask = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
    return similarities[mask].square().mean()


def prometheus_training_loss(
    *,
    proposer_logits: Tensor,
    target_branch: int,
    verifier_logits: Tensor,
    verifier_labels: Tensor,
    branch_embeddings: Tensor,
    verifier_weight: float = 1.0,
    diversity_weight: float = 0.1,
) -> Tensor:
    if not all(
        math.isfinite(weight) and weight >= 0.0
        for weight in (verifier_weight, diversity_weight)
    ):
        raise ValueError("loss weights must be finite and non-negative")
    return (
        proposer_classification_loss(proposer_logits, target_branch)
        + verifier_weight * verifier_process_loss(verifier_logits, verifier_labels)
        + diversity_weight * branch_diversity_loss(branch_embeddings)
    )


def verifier_probabilities(logits: Tensor) -> list[float]:
    if logits.ndim != 1:
        raise ValueError("verifier logits must be a vector")
    return [float(value) for value in torch.sigmoid(logits).detach().cpu().tolist()]


def transition_from_receipt(
    *,
    step: int,
    evidence: EvidenceItem,
) -> CheckableTransition:
    """Build a transition only from a canonical, integrity-checked receipt body."""

    if evidence.text is None:
        raise ValueError("transition receipt evidence must include its canonical text")
    receipt = TransitionReceipt.model_validate_json(evidence.text)
    if receipt.canonical_text() != evidence.text:
        raise ValueError("transition receipt evidence must use canonical JSON")
    return CheckableTransition(
        step=step,
        operation=receipt.operation,
        check=receipt.check,
        expected=receipt.expected,
        observed=receipt.observed,
        tolerance=receipt.tolerance,
        evidence_ids=[evidence.evidence_id],
    )


def _receipt_matches_transition(
    evidence: EvidenceItem,
    transition: CheckableTransition,
    branch: HypothesisBranch,
    workspace: WorkspaceState,
) -> bool:
    if evidence.text is None:
        return False
    try:
        receipt = TransitionReceipt.model_validate_json(evidence.text)
    except ValueError:
        return False
    if (
        receipt.canonical_text() != evidence.text
        or receipt.workspace_id != workspace.workspace_id
        or receipt.model_identity_sha256 != canonical_sha256(workspace.model_identity)
        or receipt.branch_id != branch.branch_id
        or receipt.claim_sha256
        != hashlib.sha256(branch.claim.encode("utf-8")).hexdigest()
    ):
        return False
    return (
        receipt.operation,
        receipt.check,
        type(receipt.expected),
        receipt.expected,
        type(receipt.observed),
        receipt.observed,
        receipt.tolerance,
    ) == (
        transition.operation,
        transition.check,
        type(transition.expected),
        transition.expected,
        type(transition.observed),
        transition.observed,
        transition.tolerance,
    )
