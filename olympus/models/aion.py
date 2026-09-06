"""Governed research-loop controller and optional learned router for Aion.

Authorization, protocol freezing, audit, and promotion stay deterministic. The
neural router can rank only transitions that the controller already permits.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import torch
from pydantic import Field, model_validator
from torch import nn
from torch.nn import functional as F

from olympus.core.schemas import StrictModel
from olympus.models.substrate import ensure_finite_json


class ResearchStage(StrEnum):
    QUESTION = "question"
    EVIDENCE_MAP = "evidence_map"
    HYPOTHESES = "hypotheses"
    PREREGISTERED_PROTOCOL = "preregistered_protocol"
    AUTHORIZED_EXECUTION = "authorized_execution"
    AUDIT = "audit"
    REVISE = "revise"
    STOP = "stop"
    HUMAN_REVIEW = "human_review"
    PROMOTION_CANDIDATE = "promotion_candidate"
    EXTERNAL_REPLICATION = "external_replication"


class AuthorityKind(StrEnum):
    HUMAN = "human"
    SYSTEM = "system"
    MODEL = "model"


class Approval(StrictModel):
    approval_id: str = Field(pattern=r"^approval_[0-9a-f]{16,64}$")
    authority: AuthorityKind
    actor: str = Field(min_length=1, max_length=200)
    scope: str = Field(min_length=1, max_length=200)
    run_id: str = Field(pattern=r"^aion_[0-9a-f]{16,64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_head_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def require_aware_expiry(self) -> Approval:
        if self.expires_at is not None and self.expires_at.utcoffset() is None:
            raise ValueError("approval expiry must be timezone-aware")
        return self

    def valid_for(
        self,
        scope: str,
        run_id: str,
        protocol_sha256: str,
        expected_head_sha256: str,
        now: datetime,
    ) -> bool:
        if self.authority is AuthorityKind.MODEL:
            return False
        if (
            self.scope != scope
            or self.run_id != run_id
            or self.protocol_sha256 != protocol_sha256
            or self.expected_head_sha256 != expected_head_sha256
        ):
            return False
        if self.expires_at is not None and self.expires_at <= now:
            return False
        return True


class FrozenProtocol(StrictModel):
    protocol_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    version: int = Field(ge=1)
    question: str = Field(min_length=1, max_length=20_000)
    hypothesis_ids: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(min_length=1)
    falsification_criterion: str = Field(min_length=1, max_length=10_000)
    maximum_steps: int = Field(default=32, ge=1, le=100_000)
    maximum_tool_calls: int = Field(default=16, ge=0, le=100_000)
    frozen_at: datetime
    parent_protocol_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    post_hoc: bool = False

    @model_validator(mode="after")
    def post_hoc_requires_parent(self) -> FrozenProtocol:
        if self.post_hoc and self.parent_protocol_sha256 is None:
            raise ValueError("post-hoc protocols must identify their frozen parent")
        if self.frozen_at.utcoffset() is None:
            raise ValueError("protocol freeze time must be timezone-aware")
        if len(self.hypothesis_ids) != len(set(self.hypothesis_ids)):
            raise ValueError("protocol hypothesis IDs must be unique")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("protocol evidence IDs must be unique")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json", exclude_none=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class AuditFinding(StrictModel):
    audit_id: str = Field(pattern=r"^audit_[0-9a-f]{16,64}$")
    run_id: str = Field(pattern=r"^aion_[0-9a-f]{16,64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audited_head_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_artifact_sha256: list[str] = Field(min_length=1)
    passed: bool
    claim_precision: float = Field(ge=0.0, le=1.0)
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    protected_data_leakage: bool
    unauthorized_effects: bool
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_artifacts(self) -> AuditFinding:
        if len(self.evidence_artifact_sha256) != len(set(self.evidence_artifact_sha256)):
            raise ValueError("audit evidence artifact hashes must be unique")
        if any(
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in self.evidence_artifact_sha256
        ):
            raise ValueError("audit evidence artifacts must be lowercase SHA-256 values")
        return self


class AionReceipt(StrictModel):
    sequence: int = Field(ge=1)
    run_id: str = Field(pattern=r"^aion_[0-9a-f]{16,64}$")
    previous_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    from_stage: ResearchStage
    to_stage: ResearchStage
    actor: str = Field(min_length=1, max_length=200)
    authority: AuthorityKind
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timestamp: datetime
    evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    approval_id: str | None = Field(
        default=None, pattern=r"^approval_[0-9a-f]{16,64}$"
    )
    audit_id: str | None = Field(default=None, pattern=r"^audit_[0-9a-f]{16,64}$")
    tool_calls: int = Field(default=0, ge=0)
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_aware_timestamp(self) -> AionReceipt:
        if self.timestamp.utcoffset() is None:
            raise ValueError("receipt timestamp must be timezone-aware")
        ensure_finite_json(self.metadata, label="receipt metadata")
        return self


class AionRunState(StrictModel):
    run_id: str = Field(pattern=r"^aion_[0-9a-f]{16,64}$")
    stage: ResearchStage = ResearchStage.QUESTION
    protocol: FrozenProtocol
    steps_used: int = Field(default=0, ge=0)
    tool_calls_used: int = Field(default=0, ge=0)
    outcome_inspected: bool = False
    receipts: list[AionReceipt] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_structural_integrity(self) -> AionRunState:
        error = _state_integrity_error(self)
        if error is not None:
            raise ValueError(error)
        return self


_ALLOWED: dict[ResearchStage, frozenset[ResearchStage]] = {
    ResearchStage.QUESTION: frozenset({ResearchStage.EVIDENCE_MAP, ResearchStage.STOP}),
    ResearchStage.EVIDENCE_MAP: frozenset({ResearchStage.HYPOTHESES, ResearchStage.STOP}),
    ResearchStage.HYPOTHESES: frozenset(
        {ResearchStage.PREREGISTERED_PROTOCOL, ResearchStage.REVISE, ResearchStage.STOP}
    ),
    ResearchStage.PREREGISTERED_PROTOCOL: frozenset(
        {ResearchStage.AUTHORIZED_EXECUTION, ResearchStage.REVISE, ResearchStage.STOP}
    ),
    ResearchStage.AUTHORIZED_EXECUTION: frozenset(
        {ResearchStage.AUDIT, ResearchStage.HUMAN_REVIEW, ResearchStage.STOP}
    ),
    ResearchStage.AUDIT: frozenset(
        {
            ResearchStage.REVISE,
            ResearchStage.STOP,
            ResearchStage.HUMAN_REVIEW,
            ResearchStage.PROMOTION_CANDIDATE,
        }
    ),
    ResearchStage.REVISE: frozenset(
        {ResearchStage.EVIDENCE_MAP, ResearchStage.HYPOTHESES, ResearchStage.STOP}
    ),
    ResearchStage.HUMAN_REVIEW: frozenset(
        {ResearchStage.REVISE, ResearchStage.STOP, ResearchStage.PROMOTION_CANDIDATE}
    ),
    ResearchStage.PROMOTION_CANDIDATE: frozenset(
        {ResearchStage.EXTERNAL_REPLICATION, ResearchStage.REVISE, ResearchStage.STOP}
    ),
    ResearchStage.EXTERNAL_REPLICATION: frozenset(
        {ResearchStage.AUDIT, ResearchStage.STOP}
    ),
    ResearchStage.STOP: frozenset(),
}


_APPROVAL_SCOPE: dict[ResearchStage, str] = {
    ResearchStage.AUTHORIZED_EXECUTION: "execute",
    ResearchStage.PROMOTION_CANDIDATE: "promote",
    ResearchStage.EXTERNAL_REPLICATION: "external_replicate",
}


def _receipt_digest(payload: dict[str, Any]) -> str:
    encoded = AionReceipt.model_validate(
        {**payload, "receipt_sha256": "0" * 64}
    ).model_dump(mode="json", exclude={"receipt_sha256"})
    canonical = json.dumps(
        encoded,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _state_integrity_error(state: AionRunState) -> str | None:
    if state.steps_used != len(state.receipts):
        return "steps_used must equal the hash-linked receipt count"
    if state.steps_used > state.protocol.maximum_steps:
        return "protocol step budget exceeded"
    if state.tool_calls_used > state.protocol.maximum_tool_calls:
        return "protocol tool-call budget exceeded"
    if not state.receipts:
        if state.stage is not ResearchStage.QUESTION:
            return "a receipt-free run must remain at the question stage"
        if state.steps_used or state.tool_calls_used or state.outcome_inspected:
            return "a receipt-free run must remain at its genesis counters"
        return None

    previous = "0" * 64
    previous_timestamp: datetime | None = None
    expected_stage = ResearchStage.QUESTION
    tool_calls = 0
    outcome_inspected = False
    for index, receipt in enumerate(state.receipts, start=1):
        if receipt.sequence != index:
            return "receipt sequence is not contiguous"
        if receipt.run_id != state.run_id:
            return "receipt is not bound to the active run"
        if receipt.previous_sha256 != previous:
            return "receipt hash chain is discontinuous"
        if receipt.from_stage is not expected_stage:
            return "receipt stage chain is discontinuous"
        if receipt.to_stage not in _ALLOWED[receipt.from_stage]:
            return "receipt contains an illegal stage transition"
        if receipt.protocol_sha256 != state.protocol.sha256:
            return "receipt is not bound to the active protocol"
        if previous_timestamp is not None and receipt.timestamp < previous_timestamp:
            return "receipt timestamps are not monotonic"
        payload = receipt.model_dump(exclude={"receipt_sha256"})
        if _receipt_digest(payload) != receipt.receipt_sha256:
            return "receipt digest is invalid"
        scope = _APPROVAL_SCOPE.get(receipt.to_stage)
        if scope is not None and receipt.approval_id is None:
            return f"receipt for {scope} is missing its approval reference"
        if receipt.to_stage is ResearchStage.PROMOTION_CANDIDATE and receipt.audit_id is None:
            return "promotion receipt is missing its audit reference"
        previous = receipt.receipt_sha256
        previous_timestamp = receipt.timestamp
        expected_stage = receipt.to_stage
        tool_calls += receipt.tool_calls
        outcome_inspected = outcome_inspected or receipt.to_stage in {
            ResearchStage.AUDIT,
            ResearchStage.HUMAN_REVIEW,
        }
    if state.stage is not expected_stage:
        return "run stage does not match the final receipt"
    if state.tool_calls_used != tool_calls:
        return "tool_calls_used does not equal the receipt total"
    if state.outcome_inspected is not outcome_inspected:
        return "outcome inspection state does not match the receipt history"
    return None


class AionController:
    """Deterministic controller with process-local, hash-linked run heads."""

    genesis_sha256 = "0" * 64

    def __init__(
        self,
        *,
        trusted_approvals: tuple[Approval, ...] | list[Approval] = (),
        trusted_audits: tuple[AuditFinding, ...] | list[AuditFinding] = (),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        approval_ids = [approval.approval_id for approval in trusted_approvals]
        audit_ids = [audit.audit_id for audit in trusted_audits]
        if len(approval_ids) != len(set(approval_ids)):
            raise ValueError("trusted approval IDs must be unique")
        if len(audit_ids) != len(set(audit_ids)):
            raise ValueError("trusted audit IDs must be unique")
        self._trusted_approvals = {
            approval.approval_id: approval.model_copy(deep=True)
            for approval in trusted_approvals
        }
        self._trusted_audits = {
            audit.audit_id: audit.model_copy(deep=True) for audit in trusted_audits
        }
        self._clock = clock or (lambda: datetime.now(UTC))
        self._run_heads: dict[str, str] = {}
        self._run_protocols: dict[str, str] = {}
        self._lock = threading.RLock()

    def create_run(self, run_id: str, protocol: FrozenProtocol) -> AionRunState:
        """Create and anchor a new run at the controller-owned genesis head."""

        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("controller clock must return a timezone-aware time")
        if protocol.frozen_at > now:
            raise ValueError("protocol freeze time cannot be in the future")
        state = AionRunState(run_id=run_id, protocol=protocol)
        with self._lock:
            if run_id in self._run_heads:
                raise ValueError(f"Aion run is already registered: {run_id}")
            self._run_heads[run_id] = self.genesis_sha256
            self._run_protocols[run_id] = protocol.sha256
        return state

    def register_approval(self, approval: Approval) -> None:
        """Register a host-vetted approval record by defensive copy."""

        with self._lock:
            if approval.approval_id in self._trusted_approvals:
                raise ValueError(
                    f"trusted approval ID is already registered: {approval.approval_id}"
                )
            self._trusted_approvals[approval.approval_id] = approval.model_copy(deep=True)

    def register_audit(self, audit: AuditFinding) -> None:
        """Register a host-vetted audit record by defensive copy."""

        with self._lock:
            if audit.audit_id in self._trusted_audits:
                raise ValueError(f"trusted audit ID is already registered: {audit.audit_id}")
            self._trusted_audits[audit.audit_id] = audit.model_copy(deep=True)

    @staticmethod
    def allowed_targets(stage: ResearchStage) -> frozenset[ResearchStage]:
        return _ALLOWED[stage]

    @staticmethod
    def _receipt_digest(payload: dict[str, Any]) -> str:
        return _receipt_digest(payload)

    def transition(
        self,
        state: AionRunState,
        target: ResearchStage,
        *,
        actor: str,
        authority: AuthorityKind,
        evidence_ids: list[str] | None = None,
        approval: Approval | None = None,
        audit: AuditFinding | None = None,
        metadata: dict[str, Any] | None = None,
        tool_calls: int = 0,
    ) -> AionRunState:
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("controller clock must return a timezone-aware time")
        if isinstance(tool_calls, bool) or not isinstance(tool_calls, int) or tool_calls < 0:
            raise ValueError("tool_calls must be a non-negative integer")
        transition_evidence = evidence_ids or []
        if len(transition_evidence) != len(set(transition_evidence)):
            raise ValueError("transition evidence IDs must be unique")
        unknown_evidence = set(transition_evidence) - set(state.protocol.evidence_ids)
        if unknown_evidence:
            raise ValueError(
                f"transition references evidence outside the frozen protocol: "
                f"{sorted(unknown_evidence)}"
            )
        with self._lock:
            if not self._verify_chain_unlocked(state):
                raise ValueError("Aion run state failed chain or authority validation")
            if state.receipts and now < state.receipts[-1].timestamp:
                raise ValueError("controller clock moved backwards relative to the run")
            if target not in self.allowed_targets(state.stage):
                raise ValueError(f"illegal Aion transition: {state.stage} -> {target}")
            if state.steps_used >= state.protocol.maximum_steps:
                raise ValueError("protocol step budget exhausted")
            if state.tool_calls_used + tool_calls > state.protocol.maximum_tool_calls:
                raise ValueError("protocol tool-call budget exhausted")
            if state.outcome_inspected and target is ResearchStage.PREREGISTERED_PROTOCOL:
                raise ValueError("outcomes were inspected; create a versioned post-hoc protocol")

            current_head = (
                state.receipts[-1].receipt_sha256
                if state.receipts
                else self.genesis_sha256
            )

            required_scope = _APPROVAL_SCOPE.get(target)
            if required_scope is not None:
                if approval is None or not approval.valid_for(
                    required_scope,
                    state.run_id,
                    state.protocol.sha256,
                    current_head,
                    now,
                ):
                    raise PermissionError(
                        f"valid non-model approval required for {required_scope}"
                    )
                trusted = self._trusted_approvals.get(approval.approval_id)
                if trusted != approval:
                    raise PermissionError(f"trusted approval required for {required_scope}")
                if approval.actor != actor or approval.authority is not authority:
                    raise PermissionError(
                        "gated transition actor and authority must match its approval"
                    )

            if target is ResearchStage.PROMOTION_CANDIDATE:
                if authority is AuthorityKind.MODEL:
                    raise PermissionError("model authority cannot perform promotion")
                if audit is None:
                    raise ValueError("promotion requires an audit finding")
                if (
                    audit.run_id != state.run_id
                    or audit.protocol_sha256 != state.protocol.sha256
                    or audit.audited_head_sha256 != current_head
                ):
                    raise ValueError("audit does not bind the active run, protocol, and head")
                if (
                    not audit.passed
                    or audit.protected_data_leakage
                    or audit.unauthorized_effects
                ):
                    raise PermissionError("failed or unsafe audit cannot be promoted")
                trusted_audit = self._trusted_audits.get(audit.audit_id)
                if trusted_audit != audit:
                    raise PermissionError("promotion requires a trusted audit finding")
            if target is ResearchStage.AUDIT and audit is not None:
                raise ValueError("audit findings must be registered after entering audit")

            previous = current_head
            payload: dict[str, Any] = {
                "sequence": len(state.receipts) + 1,
                "run_id": state.run_id,
                "previous_sha256": previous,
                "from_stage": state.stage,
                "to_stage": target,
                "actor": actor,
                "authority": authority,
                "protocol_sha256": state.protocol.sha256,
                "timestamp": now,
                "evidence_ids": transition_evidence,
                "metadata": metadata or {},
                "approval_id": approval.approval_id if approval is not None else None,
                "audit_id": audit.audit_id if audit is not None else None,
                "tool_calls": tool_calls,
            }
            receipt = AionReceipt(
                **payload,
                receipt_sha256=self._receipt_digest(payload),
            )
            updated = AionRunState.model_validate(
                {
                    **state.model_dump(),
                    "stage": target,
                    "steps_used": state.steps_used + 1,
                    "tool_calls_used": state.tool_calls_used + tool_calls,
                    "outcome_inspected": state.outcome_inspected
                    or target in {ResearchStage.AUDIT, ResearchStage.HUMAN_REVIEW},
                    "receipts": [*state.receipts, receipt],
                }
            )
            self._run_heads[state.run_id] = receipt.receipt_sha256
            return updated

    def verify_chain(self, state: AionRunState) -> bool:
        with self._lock:
            return self._verify_chain_unlocked(state)

    def _verify_chain_unlocked(self, state: AionRunState) -> bool:
        if _state_integrity_error(state) is not None:
            return False
        expected_head = self._run_heads.get(state.run_id)
        if self._run_protocols.get(state.run_id) != state.protocol.sha256:
            return False
        observed_head = (
            state.receipts[-1].receipt_sha256 if state.receipts else self.genesis_sha256
        )
        if expected_head is None or expected_head != observed_head:
            return False
        for receipt in state.receipts:
            scope = _APPROVAL_SCOPE.get(receipt.to_stage)
            if scope is not None:
                if receipt.approval_id is None:
                    return False
                approval = self._trusted_approvals.get(receipt.approval_id)
                if approval is None or not approval.valid_for(
                    scope,
                    state.run_id,
                    state.protocol.sha256,
                    receipt.previous_sha256,
                    receipt.timestamp,
                ):
                    return False
            if receipt.to_stage is ResearchStage.PROMOTION_CANDIDATE:
                if receipt.audit_id is None:
                    return False
                audit = self._trusted_audits.get(receipt.audit_id)
                if (
                    audit is None
                    or audit.run_id != state.run_id
                    or audit.protocol_sha256 != state.protocol.sha256
                    or audit.audited_head_sha256 != receipt.previous_sha256
                    or not audit.passed
                    or audit.protected_data_leakage
                    or audit.unauthorized_effects
                ):
                    return False
        return True


class AionRouterConfig(StrictModel):
    state_dim: int = Field(default=32, ge=4, le=4_096)
    hidden_dim: int = Field(default=32, ge=4, le=4_096)
    maximum_budget: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def finite_budget(self) -> AionRouterConfig:
        if not math.isfinite(self.maximum_budget):
            raise ValueError("router maximum budget must be finite")
        return self


@dataclass(slots=True)
class AionRouterOutput:
    logits: torch.Tensor
    budget_fraction: torch.Tensor


class AionRouter(nn.Module):
    """Optional learned router whose logits are masked by deterministic policy."""

    stages = tuple(ResearchStage)

    def __init__(self, config: AionRouterConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.stage_head = nn.Linear(config.hidden_dim, len(self.stages))
        self.budget_head = nn.Linear(config.hidden_dim, 1)

    def forward(
        self, state_features: torch.Tensor, allowed_mask: torch.Tensor
    ) -> AionRouterOutput:
        if state_features.ndim != 2 or state_features.shape[-1] != self.config.state_dim:
            raise ValueError("state_features must have shape [batch, state_dim]")
        if state_features.shape[0] == 0 or not torch.isfinite(state_features).all():
            raise ValueError("state_features must be a non-empty finite batch")
        if allowed_mask.shape != (state_features.shape[0], len(self.stages)):
            raise ValueError("allowed_mask shape does not match router stages")
        if torch.any(allowed_mask.sum(dim=1) == 0):
            raise ValueError("each router row requires at least one allowed transition")
        hidden = self.encoder(state_features)
        logits = self.stage_head(hidden).masked_fill(~allowed_mask.to(torch.bool), -torch.inf)
        budget = torch.sigmoid(self.budget_head(hidden)).squeeze(-1)
        return AionRouterOutput(
            logits=logits,
            budget_fraction=budget * self.config.maximum_budget,
        )

    @classmethod
    def allowed_mask(
        cls, stages: list[ResearchStage], *, device: torch.device | None = None
    ) -> torch.Tensor:
        mask = torch.zeros(len(stages), len(cls.stages), dtype=torch.bool, device=device)
        index = {stage: position for position, stage in enumerate(cls.stages)}
        for row, stage in enumerate(stages):
            for target in _ALLOWED[stage]:
                mask[row, index[target]] = True
        return mask


@dataclass(slots=True)
class AionRouterLoss:
    total: torch.Tensor
    transition_cross_entropy: torch.Tensor
    budget_mse: torch.Tensor


def aion_router_loss(
    output: AionRouterOutput,
    target_stages: torch.Tensor,
    target_budget_fraction: torch.Tensor,
    *,
    budget_weight: float = 0.2,
) -> AionRouterLoss:
    if not math.isfinite(budget_weight) or budget_weight < 0.0:
        raise ValueError("budget_weight must be finite and non-negative")
    if target_stages.shape != (output.logits.shape[0],):
        raise ValueError("target_stages shape does not match router batch")
    if target_budget_fraction.shape != output.budget_fraction.shape:
        raise ValueError("target budget shape does not match router batch")
    if target_stages.dtype not in {torch.int32, torch.int64}:
        raise ValueError("target stages must use an integer tensor dtype")
    if torch.any(target_stages < 0) or torch.any(target_stages >= output.logits.shape[1]):
        raise ValueError("target stage index is outside the router stage set")
    if not torch.isfinite(target_budget_fraction).all():
        raise ValueError("target budget fractions must be finite")
    selected = output.logits.gather(1, target_stages[:, None]).squeeze(1)
    if torch.any(torch.isneginf(selected)):
        raise ValueError("a target transition is forbidden by deterministic policy")
    transition_ce = F.cross_entropy(output.logits, target_stages)
    budget_mse = F.mse_loss(output.budget_fraction, target_budget_fraction)
    total = transition_ce + budget_weight * budget_mse
    return AionRouterLoss(
        total=total,
        transition_cross_entropy=transition_ce,
        budget_mse=budget_mse,
    )


class AionComparison(StrictModel):
    learned_valid_experiment_rate: float = Field(ge=0.0, le=1.0)
    deterministic_valid_experiment_rate: float = Field(ge=0.0, le=1.0)
    learned_false_promotion_rate: float = Field(ge=0.0, le=1.0)
    deterministic_false_promotion_rate: float = Field(ge=0.0, le=1.0)
    learned_budget: float = Field(ge=0.0)
    deterministic_budget: float = Field(ge=0.0)

    @model_validator(mode="after")
    def finite_budgets(self) -> AionComparison:
        if not math.isfinite(self.learned_budget) or not math.isfinite(
            self.deterministic_budget
        ):
            raise ValueError("comparison budgets must be finite")
        return self


def learned_router_adds_value(comparison: AionComparison) -> bool:
    """Predeclared Aion falsification rule from the architecture proposal."""

    return (
        comparison.learned_valid_experiment_rate
        > comparison.deterministic_valid_experiment_rate
        and comparison.learned_false_promotion_rate
        <= comparison.deterministic_false_promotion_rate
        and comparison.learned_budget <= comparison.deterministic_budget
    )
