"""Deterministic host-side execution for Aion transition decisions.

The executor accepts a model decision only as a requested transition.  Authority,
affordability, callback registration, and every side effect remain host-owned.
It is deliberately independent of Torch and model loading so a saved trace can
be checked without re-running a model or invoking an action again.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from olympus.core.schemas import StrictModel

ActionHandler = Callable[[], object]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class AionExecutionPlan(StrictModel):
    """Host-owned transition authority and affordability state for one decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    policy_id: str = Field(min_length=1, max_length=256)
    transitions: tuple[str, ...] = Field(min_length=2, max_length=256)
    legal_transitions: tuple[bool, ...] = Field(min_length=2, max_length=256)
    transition_costs: tuple[float, ...] = Field(min_length=2, max_length=256)
    remaining_budget: float = Field(ge=0.0)
    stop_stage: int = Field(ge=0, lt=256)

    @model_validator(mode="after")
    def valid_plan(self) -> AionExecutionPlan:
        if len(set(self.transitions)) != len(self.transitions):
            raise ValueError("transition names must be unique")
        if any(not name or name.strip() != name for name in self.transitions):
            raise ValueError("transition names must be nonempty and trimmed")
        if len(self.legal_transitions) != len(self.transitions):
            raise ValueError("legal transitions must align with transition names")
        if len(self.transition_costs) != len(self.transitions):
            raise ValueError("transition costs must align with transition names")
        if self.stop_stage >= len(self.transitions):
            raise ValueError("stop stage must index a transition")
        if self.transitions[self.stop_stage] != "STOP":
            raise ValueError("stop stage must be named STOP")
        if not math.isfinite(self.remaining_budget):
            raise ValueError("remaining budget must be finite")
        if any(not math.isfinite(cost) or cost < 0.0 for cost in self.transition_costs):
            raise ValueError("transition costs must be finite and nonnegative")
        if self.transition_costs[self.stop_stage] != 0.0:
            raise ValueError("STOP must have zero cost")
        if not self.legal_transitions[self.stop_stage]:
            raise ValueError("STOP must be host-authorized")
        return self

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class AionExecutionDecision(StrictModel):
    """Untrusted model output reduced to a stage index and forced-stop signal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_stage: int = Field(ge=0, lt=256)
    forced_stop: bool

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class AionExecutionTrace(StrictModel):
    """Replayable receipt for one decision; it never embeds executable callbacks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    policy_id: str = Field(min_length=1, max_length=256)
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_stage: int = Field(ge=0, lt=256)
    effective_stage: int = Field(ge=0, lt=256)
    action: str = Field(min_length=1, max_length=256)
    status: Literal["completed", "handler_failed", "stopped"]
    stop_reason: Literal[
        "none",
        "forced_stop",
        "selected_stop",
        "selected_transition_illegal",
        "selected_transition_unaffordable",
        "selected_transition_unregistered",
    ]
    action_executed: bool
    cost: float = Field(ge=0.0)
    budget_before: float = Field(ge=0.0)
    budget_after: float = Field(ge=0.0)
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    handler_error: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def consistent_status(self) -> AionExecutionTrace:
        if not all(
            math.isfinite(value)
            for value in (self.cost, self.budget_before, self.budget_after)
        ):
            raise ValueError("trace costs and budgets must be finite")
        stopped = self.status == "stopped"
        if stopped != (not self.action_executed):
            raise ValueError("stopped traces must not execute an action")
        if stopped != (self.stop_reason != "none"):
            raise ValueError("stop reason must match stopped status")
        if stopped and (self.result_sha256 is not None or self.handler_error is not None):
            raise ValueError("stopped traces cannot contain handler output")
        if not stopped and self.stop_reason != "none":
            raise ValueError("executed traces cannot have a stop reason")
        if self.status == "completed" and (
            self.result_sha256 is None or self.handler_error is not None
        ):
            raise ValueError("completed traces require only a result digest")
        if self.status == "handler_failed" and (
            self.handler_error is None or self.result_sha256 is not None
        ):
            raise ValueError("failed handlers require only a recorded error")
        if stopped and (self.cost != 0.0 or self.budget_after != self.budget_before):
            raise ValueError("stopped traces cannot spend budget")
        if not stopped and not math.isclose(
            self.budget_before - self.cost,
            self.budget_after,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("executed trace budget delta must equal its cost")
        if self.budget_after > self.budget_before:
            raise ValueError("execution cannot increase the budget")
        return self

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class AionExecutionLedger(StrictModel):
    """Immutable receipt for a budget-carrying sequence of Aion decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    policy_id: str = Field(min_length=1, max_length=256)
    initial_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decisions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    requested_decisions: int = Field(ge=1, le=256)
    consumed_decisions: int = Field(ge=1, le=256)
    initial_budget: float = Field(ge=0.0)
    final_budget: float = Field(ge=0.0)
    terminated_by_stop: bool
    traces: tuple[AionExecutionTrace, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def consistent_ledger(self) -> AionExecutionLedger:
        if not math.isfinite(self.initial_budget) or not math.isfinite(self.final_budget):
            raise ValueError("ledger budgets must be finite")
        if self.consumed_decisions != len(self.traces):
            raise ValueError("consumed decision count must equal trace count")
        if self.consumed_decisions > self.requested_decisions:
            raise ValueError("consumed decisions cannot exceed requested decisions")
        if self.traces[0].plan_sha256 != self.initial_plan_sha256:
            raise ValueError("first trace must bind the initial plan")
        if self.traces[0].budget_before != self.initial_budget:
            raise ValueError("first trace must start from the initial budget")
        if self.traces[-1].budget_after != self.final_budget:
            raise ValueError("last trace must end at the final budget")
        if any(trace.policy_id != self.policy_id for trace in self.traces):
            raise ValueError("all traces must use the ledger policy")
        if any(trace.registry_sha256 != self.registry_sha256 for trace in self.traces):
            raise ValueError("all traces must use the ledger registry")
        if any(
            not math.isclose(
                previous.budget_after,
                following.budget_before,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for previous, following in zip(self.traces, self.traces[1:], strict=False)
        ):
            raise ValueError("trace budgets must form one continuous chain")
        stopped = [index for index, trace in enumerate(self.traces) if trace.status == "stopped"]
        if self.terminated_by_stop:
            if stopped != [len(self.traces) - 1]:
                raise ValueError("a stopped ledger must have one terminal STOP trace")
        elif stopped or self.consumed_decisions != self.requested_decisions:
            raise ValueError("a non-stopped ledger must consume every request without STOP")
        return self

    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class HostActionRegistry:
    """Host-registered callbacks; a decision can never invent a callback."""

    handlers: Mapping[str, ActionHandler]

    def __post_init__(self) -> None:
        snapshot = dict(self.handlers)
        if any(not name or name.strip() != name for name in snapshot):
            raise ValueError("registered action names must be nonempty and trimmed")
        if any(not callable(handler) for handler in snapshot.values()):
            raise TypeError("registered action handlers must be callable")
        object.__setattr__(self, "handlers", MappingProxyType(snapshot))

    def handler_for(self, action: str) -> ActionHandler | None:
        return self.handlers.get(action)

    def action_names(self) -> set[str]:
        return set(self.handlers)

    def digest(self) -> str:
        return _digest(sorted(self.handlers))


def _registry_digest(registered_actions: set[str]) -> str:
    return _digest(sorted(registered_actions))


def _decisions_digest(decisions: Sequence[AionExecutionDecision]) -> str:
    return _digest([decision.model_dump(mode="json") for decision in decisions])


def _stop_reason(
    plan: AionExecutionPlan,
    decision: AionExecutionDecision,
    registered_actions: set[str],
) -> Literal[
    "forced_stop",
    "selected_stop",
    "selected_transition_illegal",
    "selected_transition_unaffordable",
    "selected_transition_unregistered",
] | None:
    if decision.selected_stage >= len(plan.transitions):
        raise ValueError("selected stage must index a transition in the host plan")
    if decision.forced_stop:
        return "forced_stop"
    if decision.selected_stage == plan.stop_stage:
        return "selected_stop"
    if not plan.legal_transitions[decision.selected_stage]:
        return "selected_transition_illegal"
    if plan.transition_costs[decision.selected_stage] > plan.remaining_budget:
        return "selected_transition_unaffordable"
    if plan.transitions[decision.selected_stage] not in registered_actions:
        return "selected_transition_unregistered"
    return None


def _stopped_trace(
    plan: AionExecutionPlan,
    decision: AionExecutionDecision,
    reason: Literal[
        "forced_stop",
        "selected_stop",
        "selected_transition_illegal",
        "selected_transition_unaffordable",
        "selected_transition_unregistered",
    ],
    *,
    registry_sha256: str,
) -> AionExecutionTrace:
    return AionExecutionTrace(
        policy_id=plan.policy_id,
        plan_sha256=plan.digest(),
        decision_sha256=decision.digest(),
        registry_sha256=registry_sha256,
        selected_stage=decision.selected_stage,
        effective_stage=plan.stop_stage,
        action=plan.transitions[plan.stop_stage],
        status="stopped",
        stop_reason=reason,
        action_executed=False,
        cost=0.0,
        budget_before=plan.remaining_budget,
        budget_after=plan.remaining_budget,
    )


def execute_aion_decision(
    plan: AionExecutionPlan,
    decision: AionExecutionDecision,
    registry: HostActionRegistry,
) -> AionExecutionTrace:
    """Execute one legal, affordable, host-registered non-STOP transition.

    Illegal, unaffordable, unregistered, forced, and explicit STOP decisions are
    converted to a terminal STOP receipt.  They never call a handler.
    """

    plan = AionExecutionPlan.model_validate(plan.model_dump(mode="python"))
    decision = AionExecutionDecision.model_validate(decision.model_dump(mode="python"))
    registered_actions = registry.action_names()
    reason = _stop_reason(plan, decision, registered_actions)
    if reason is not None:
        return _stopped_trace(
            plan,
            decision,
            reason,
            registry_sha256=registry.digest(),
        )
    cost = plan.transition_costs[decision.selected_stage]
    action = plan.transitions[decision.selected_stage]
    handler = registry.handler_for(action)
    if handler is None:  # pragma: no cover - immutable registry makes this unreachable
        raise RuntimeError("registered action disappeared from immutable registry")
    try:
        result = handler()
        result_digest = _digest(result)
    except Exception as error:
        return AionExecutionTrace(
            policy_id=plan.policy_id,
            plan_sha256=plan.digest(),
            decision_sha256=decision.digest(),
            registry_sha256=registry.digest(),
            selected_stage=decision.selected_stage,
            effective_stage=decision.selected_stage,
            action=action,
            status="handler_failed",
            stop_reason="none",
            action_executed=True,
            cost=cost,
            budget_before=plan.remaining_budget,
            budget_after=plan.remaining_budget - cost,
            handler_error=f"{type(error).__name__}: {error}",
        )
    return AionExecutionTrace(
        policy_id=plan.policy_id,
        plan_sha256=plan.digest(),
        decision_sha256=decision.digest(),
        registry_sha256=registry.digest(),
        selected_stage=decision.selected_stage,
        effective_stage=decision.selected_stage,
        action=action,
        status="completed",
        stop_reason="none",
        action_executed=True,
        cost=cost,
        budget_before=plan.remaining_budget,
        budget_after=plan.remaining_budget - cost,
        result_sha256=result_digest,
    )


def verify_aion_trace(
    plan: AionExecutionPlan,
    decision: AionExecutionDecision,
    trace: AionExecutionTrace,
    *,
    registered_actions: set[str],
) -> None:
    """Check a trace against host inputs without invoking the original action."""

    plan = AionExecutionPlan.model_validate(plan.model_dump(mode="python"))
    decision = AionExecutionDecision.model_validate(decision.model_dump(mode="python"))
    trace = AionExecutionTrace.model_validate(trace.model_dump(mode="python"))
    if trace.policy_id != plan.policy_id or trace.plan_sha256 != plan.digest():
        raise ValueError("trace does not bind the supplied host plan")
    if trace.decision_sha256 != decision.digest():
        raise ValueError("trace does not bind the supplied model decision")
    if trace.registry_sha256 != _registry_digest(registered_actions):
        raise ValueError("trace does not bind the supplied host registry")
    if trace.selected_stage != decision.selected_stage:
        raise ValueError("trace selected stage differs from the model decision")
    reason = _stop_reason(plan, decision, registered_actions)
    if reason is not None:
        expected = _stopped_trace(
            plan,
            decision,
            reason,
            registry_sha256=_registry_digest(registered_actions),
        )
        if trace != expected:
            raise ValueError("STOP trace differs from deterministic host enforcement")
        return
    if trace.status not in {"completed", "handler_failed"}:
        raise ValueError("legal registered transition must contain an execution trace")
    if (
        trace.effective_stage != decision.selected_stage
        or trace.action != plan.transitions[decision.selected_stage]
        or not trace.action_executed
        or trace.cost != plan.transition_costs[decision.selected_stage]
        or trace.budget_before != plan.remaining_budget
        or not math.isclose(
            trace.budget_after,
            plan.remaining_budget - plan.transition_costs[decision.selected_stage],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or trace.stop_reason != "none"
    ):
        raise ValueError("execution trace violates host authority or affordability")


def execute_aion_sequence(
    plan: AionExecutionPlan,
    decisions: Sequence[AionExecutionDecision],
    registry: HostActionRegistry,
) -> AionExecutionLedger:
    """Execute a bounded decision sequence, carrying budget and terminating at STOP."""

    plan = AionExecutionPlan.model_validate(plan.model_dump(mode="python"))
    requested = tuple(
        AionExecutionDecision.model_validate(decision.model_dump(mode="python"))
        for decision in decisions
    )
    if not requested or len(requested) > 256:
        raise ValueError("decision sequence must contain 1..256 requests")
    current_plan = plan
    traces: list[AionExecutionTrace] = []
    for decision in requested:
        trace = execute_aion_decision(current_plan, decision, registry)
        traces.append(trace)
        if trace.status == "stopped":
            break
        current_values = current_plan.model_dump(mode="python")
        current_values["remaining_budget"] = trace.budget_after
        current_plan = AionExecutionPlan.model_validate(current_values)
    return AionExecutionLedger(
        policy_id=plan.policy_id,
        initial_plan_sha256=plan.digest(),
        decisions_sha256=_decisions_digest(requested),
        registry_sha256=registry.digest(),
        requested_decisions=len(requested),
        consumed_decisions=len(traces),
        initial_budget=plan.remaining_budget,
        final_budget=traces[-1].budget_after,
        terminated_by_stop=traces[-1].status == "stopped",
        traces=tuple(traces),
    )


def verify_aion_ledger(
    plan: AionExecutionPlan,
    decisions: Sequence[AionExecutionDecision],
    ledger: AionExecutionLedger,
    *,
    registered_actions: set[str],
) -> None:
    """Verify a complete budget/STOP chain without invoking any callback."""

    plan = AionExecutionPlan.model_validate(plan.model_dump(mode="python"))
    requested = tuple(
        AionExecutionDecision.model_validate(decision.model_dump(mode="python"))
        for decision in decisions
    )
    if not requested or len(requested) > 256:
        raise ValueError("decision sequence must contain 1..256 requests")
    ledger = AionExecutionLedger.model_validate(ledger.model_dump(mode="python"))
    if ledger.policy_id != plan.policy_id or ledger.initial_plan_sha256 != plan.digest():
        raise ValueError("ledger does not bind the supplied initial plan")
    if ledger.decisions_sha256 != _decisions_digest(requested):
        raise ValueError("ledger does not bind the complete requested decision sequence")
    if ledger.registry_sha256 != _registry_digest(registered_actions):
        raise ValueError("ledger does not bind the supplied host registry")
    if ledger.requested_decisions != len(requested):
        raise ValueError("ledger requested-decision count differs from supplied requests")

    current_plan = plan
    stopped = False
    for index, trace in enumerate(ledger.traces):
        verify_aion_trace(
            current_plan,
            requested[index],
            trace,
            registered_actions=registered_actions,
        )
        stopped = trace.status == "stopped"
        if stopped:
            if index != len(ledger.traces) - 1:
                raise ValueError("STOP must be the terminal trace")
            break
        current_values = current_plan.model_dump(mode="python")
        current_values["remaining_budget"] = trace.budget_after
        current_plan = AionExecutionPlan.model_validate(current_values)
    if stopped != ledger.terminated_by_stop:
        raise ValueError("ledger STOP disposition differs from verified traces")
    if not stopped and len(ledger.traces) != len(requested):
        raise ValueError("non-stopped ledger omitted requested decisions")
