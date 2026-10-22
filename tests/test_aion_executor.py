from __future__ import annotations

from collections.abc import Callable
from math import inf, nan

import pytest

from olympus.evaluation.aion_executor import (
    AionExecutionDecision,
    AionExecutionLedger,
    AionExecutionPlan,
    HostActionRegistry,
    execute_aion_decision,
    execute_aion_sequence,
    verify_aion_ledger,
    verify_aion_trace,
)


def _plan(**changes: object) -> AionExecutionPlan:
    values: dict[str, object] = {
        "policy_id": "development-aion-policy-v1",
        "transitions": ["inspect", "write", "STOP"],
        "legal_transitions": [True, True, True],
        "transition_costs": [1.0, 3.0, 0.0],
        "remaining_budget": 2.0,
        "stop_stage": 2,
    }
    values.update(changes)
    return AionExecutionPlan.model_validate(values)


def test_executes_only_host_registered_legal_affordable_action_and_replays() -> None:
    events: list[str] = []
    plan = _plan()
    decision = AionExecutionDecision(selected_stage=0, forced_stop=False)

    def inspect() -> dict[str, int]:
        events.append("inspect")
        return {"records": 2}

    trace = execute_aion_decision(
        plan,
        decision,
        HostActionRegistry({"inspect": inspect}),
    )
    assert events == ["inspect"]
    assert trace.status == "completed"
    assert trace.action == "inspect"
    assert trace.budget_after == 1.0
    assert trace.result_sha256 is not None
    verify_aion_trace(plan, decision, trace, registered_actions={"inspect"})
    assert events == ["inspect"]


@pytest.mark.parametrize(
    ("decision", "plan_changes", "registered", "reason"),
    [
        (AionExecutionDecision(selected_stage=0, forced_stop=True), {}, {"inspect"}, "forced_stop"),
        (
            AionExecutionDecision(selected_stage=2, forced_stop=False),
            {},
            {"inspect"},
            "selected_stop",
        ),
        (
            AionExecutionDecision(selected_stage=1, forced_stop=False),
            {"legal_transitions": [True, False, True]},
            {"write"},
            "selected_transition_illegal",
        ),
        (
            AionExecutionDecision(selected_stage=1, forced_stop=False),
            {},
            {"write"},
            "selected_transition_unaffordable",
        ),
        (
            AionExecutionDecision(selected_stage=0, forced_stop=False),
            {},
            set(),
            "selected_transition_unregistered",
        ),
    ],
)
def test_stop_is_non_bypassable(
    decision: AionExecutionDecision,
    plan_changes: dict[str, object],
    registered: set[str],
    reason: str,
) -> None:
    events: list[str] = []
    plan = _plan(**plan_changes)

    def handler(action: str) -> dict[str, str]:
        events.append(action)
        return {"action": action}

    def handler_for(action: str) -> Callable[[], dict[str, str]]:
        def callback() -> dict[str, str]:
            return handler(action)

        return callback

    trace = execute_aion_decision(
        plan,
        decision,
        HostActionRegistry({action: handler_for(action) for action in registered}),
    )
    assert trace.status == "stopped"
    assert trace.stop_reason == reason
    assert trace.action == "STOP"
    assert events == []
    verify_aion_trace(plan, decision, trace, registered_actions=registered)


def test_handler_failure_is_retained_without_retry_or_budget_refund() -> None:
    plan = _plan()
    decision = AionExecutionDecision(selected_stage=0, forced_stop=False)

    def fail() -> None:
        raise RuntimeError("fixture failure")

    trace = execute_aion_decision(plan, decision, HostActionRegistry({"inspect": fail}))
    assert trace.status == "handler_failed"
    assert trace.action_executed is True
    assert trace.budget_after == 1.0
    assert trace.handler_error == "RuntimeError: fixture failure"
    verify_aion_trace(plan, decision, trace, registered_actions={"inspect"})


def test_trace_verifier_rejects_tampered_affordability_or_stop_bypass() -> None:
    plan = _plan()
    decision = AionExecutionDecision(selected_stage=0, forced_stop=True)
    trace = execute_aion_decision(plan, decision, HostActionRegistry({"inspect": lambda: {}}))
    tampered = trace.model_copy(
        update={
            "action_executed": True,
            "status": "completed",
            "stop_reason": "none",
            "result_sha256": "0" * 64,
        }
    )
    with pytest.raises(ValueError, match="STOP trace"):
        verify_aion_trace(plan, decision, tampered, registered_actions={"inspect"})

    affordable = execute_aion_decision(
        plan,
        AionExecutionDecision(selected_stage=0, forced_stop=False),
        HostActionRegistry({"inspect": lambda: {"ok": True}}),
    )
    cost_tampered = affordable.model_copy(update={"budget_after": 2.0})
    with pytest.raises(ValueError, match="budget delta|violates host authority"):
        verify_aion_trace(
            plan,
            AionExecutionDecision(selected_stage=0, forced_stop=False),
            cost_tampered,
            registered_actions={"inspect"},
        )


def test_trace_binds_forced_stop_decision_and_immutable_registry_snapshot() -> None:
    handlers: dict[str, Callable[[], object]] = {"inspect": lambda: {"ok": True}}
    registry = HostActionRegistry(handlers)
    handlers.clear()
    plan = _plan()
    executed = execute_aion_decision(
        plan,
        AionExecutionDecision(selected_stage=0, forced_stop=False),
        registry,
    )
    assert executed.status == "completed"
    with pytest.raises(ValueError, match="model decision"):
        verify_aion_trace(
            plan,
            AionExecutionDecision(selected_stage=0, forced_stop=True),
            executed,
            registered_actions={"inspect"},
        )
    with pytest.raises(ValueError, match="host registry"):
        verify_aion_trace(
            plan,
            AionExecutionDecision(selected_stage=0, forced_stop=False),
            executed,
            registered_actions={"inspect", "write"},
        )


@pytest.mark.parametrize("value", [nan, inf])
def test_plan_rejects_nonfinite_budget_and_costs(value: float) -> None:
    with pytest.raises(ValueError, match=r"remaining[_ ]budget"):
        _plan(remaining_budget=value)
    with pytest.raises(ValueError, match=r"transition[_ ]costs"):
        _plan(transition_costs=[value, 3.0, 0.0])


def test_plan_and_decision_reject_unsafe_boundaries() -> None:
    with pytest.raises(ValueError, match="named STOP"):
        _plan(transitions=["inspect", "write", "halt"])
    with pytest.raises(ValueError, match="STOP must be host-authorized"):
        _plan(legal_transitions=[True, True, False])
    with pytest.raises(ValueError, match="STOP must have zero cost"):
        _plan(transition_costs=[1.0, 3.0, 0.1])
    with pytest.raises(ValueError, match="selected stage"):
        execute_aion_decision(
            _plan(),
            AionExecutionDecision(selected_stage=4, forced_stop=False),
            HostActionRegistry({}),
        )
    plan = _plan()
    assert isinstance(plan.transitions, tuple)
    with pytest.raises(ValueError, match="frozen"):
        plan.remaining_budget = 10.0


def test_sequence_carries_budget_and_stop_is_terminal() -> None:
    events: list[str] = []
    plan = _plan(remaining_budget=4.0)
    decisions = (
        AionExecutionDecision(selected_stage=0, forced_stop=False),
        AionExecutionDecision(selected_stage=1, forced_stop=False),
        AionExecutionDecision(selected_stage=0, forced_stop=False),
    )
    ledger = execute_aion_sequence(
        plan,
        decisions,
        HostActionRegistry(
            {
                "inspect": lambda: events.append("inspect"),
                "write": lambda: events.append("write"),
            }
        ),
    )
    assert events == ["inspect", "write"]
    assert ledger.requested_decisions == 3
    assert ledger.consumed_decisions == 3
    assert ledger.terminated_by_stop is True
    assert [trace.status for trace in ledger.traces] == ["completed", "completed", "stopped"]
    assert ledger.final_budget == 0.0
    verify_aion_ledger(plan, decisions, ledger, registered_actions={"inspect", "write"})
    assert events == ["inspect", "write"]


def test_sequence_binds_ignored_post_stop_requests_and_rejects_chain_tampering() -> None:
    plan = _plan()
    decisions = (
        AionExecutionDecision(selected_stage=2, forced_stop=False),
        AionExecutionDecision(selected_stage=0, forced_stop=False),
    )
    ledger = execute_aion_sequence(plan, decisions, HostActionRegistry({"inspect": lambda: None}))
    assert ledger.consumed_decisions == 1
    assert ledger.requested_decisions == 2
    with pytest.raises(ValueError, match="complete requested decision sequence"):
        verify_aion_ledger(
            plan,
            decisions[:1],
            ledger,
            registered_actions={"inspect"},
        )

    tampered_trace = ledger.traces[0].model_copy(update={"budget_after": 1.0})
    tampered = ledger.model_copy(update={"final_budget": 1.0, "traces": (tampered_trace,)})
    with pytest.raises(ValueError, match="stopped traces cannot spend budget"):
        verify_aion_ledger(plan, decisions, tampered, registered_actions={"inspect"})


def test_ledger_rejects_nonterminal_stop_and_missing_unstopped_trace() -> None:
    plan = _plan()
    stopped = execute_aion_decision(
        plan,
        AionExecutionDecision(selected_stage=2, forced_stop=False),
        HostActionRegistry({"inspect": lambda: None}),
    )
    completed = execute_aion_decision(
        plan,
        AionExecutionDecision(selected_stage=0, forced_stop=False),
        HostActionRegistry({"inspect": lambda: None}),
    )
    values = {
        "policy_id": plan.policy_id,
        "initial_plan_sha256": plan.digest(),
        "decisions_sha256": "0" * 64,
        "registry_sha256": completed.registry_sha256,
        "requested_decisions": 2,
        "consumed_decisions": 2,
        "initial_budget": plan.remaining_budget,
        "final_budget": completed.budget_after,
        "terminated_by_stop": True,
        "traces": (stopped, completed),
    }
    with pytest.raises(ValueError, match="continuous chain|terminal STOP"):
        AionExecutionLedger.model_validate(values)
