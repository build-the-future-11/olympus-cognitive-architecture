from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from olympus.models.perseus import (
    ActionApproval,
    ActionEnvelope,
    ActionValidationError,
    ArgumentKind,
    ArgumentRule,
    AuthorizationError,
    CapabilityKernel,
    CompensatableExecutionError,
    DeterministicActionValidator,
    ExecutionContext,
    RecoveryClass,
    RetryableExecutionError,
    ToolDefinition,
    TrainableActionPolicy,
    TrainableRecoveryPolicy,
    TransactionManager,
    TransactionState,
    TransactionStateError,
    action_policy_loss,
    action_sha256,
    classify_execution_failure,
    recovery_policy_loss,
)


class RecordingSandbox:
    guarantees_idempotency = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def execute(
        self, tool_id: str, arguments: Mapping[str, Any], transaction_id: str
    ) -> Mapping[str, Any]:
        self.calls.append((tool_id, dict(arguments), transaction_id))
        return {"written": arguments["content"]}


def _write_tool() -> ToolDefinition:
    return ToolDefinition(
        tool_id="filesystem.write",
        schema_version="1",
        arguments={
            "path": ArgumentRule(kind=ArgumentKind.STRING),
            "content": ArgumentRule(kind=ArgumentKind.STRING),
        },
        required_capabilities={"filesystem.write"},
        required_preconditions={"workspace-clean"},
        material_effect=True,
        approval_required=True,
    )


def _write_action(*, key: str = "write-key-0001", content: str = "safe") -> ActionEnvelope:
    return ActionEnvelope(
        tool_id="filesystem.write",
        schema_version="1",
        arguments={"path": "/sandbox/result.txt", "content": content},
        preconditions=["workspace-clean"],
        idempotency_key=key,
    )


def _approval(action: ActionEnvelope, marker: str = "a") -> ActionApproval:
    return ActionApproval(
        approval_id=f"approval_{marker * 16}",
        action_sha256=action_sha256(action),
        issued_by="test-reviewer",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )


def _authorized_manager(
    *actions: ActionEnvelope,
) -> tuple[TransactionManager, ExecutionContext]:
    approvals = [_approval(action, chr(ord("a") + index)) for index, action in enumerate(actions)]
    kernel = CapabilityKernel([_write_tool()], trusted_approvals=approvals)
    context = ExecutionContext(
        capabilities={"filesystem.write"},
        satisfied_preconditions={"workspace-clean"},
        approval_ids={approval.approval_id for approval in approvals},
    )
    return TransactionManager(kernel), context


def test_action_and_recovery_policies_receive_gradient_updates() -> None:
    torch.manual_seed(11)
    action = TrainableActionPolicy(state_dim=5, tool_count=3, argument_dim=2)
    recovery = TrainableRecoveryPolicy(failure_dim=4, tool_count=3)
    optimizer = torch.optim.SGD([*action.parameters(), *recovery.parameters()], lr=0.1)
    action_state = torch.randn(4, 5)
    failure_state = torch.randn(4, 4)
    parameters = [*action.parameters(), *recovery.parameters()]
    before = [parameter.detach().clone() for parameter in parameters]

    optimizer.zero_grad()
    tool_logits, argument_values = action(action_state)
    class_logits, next_tool_logits = recovery(failure_state)
    loss = action_policy_loss(
        tool_logits,
        torch.tensor([0, 1, 2, 1]),
        argument_values,
        torch.zeros(4, 2),
    ) + recovery_policy_loss(
        class_logits,
        torch.tensor(
            [
                RecoveryClass.RETRYABLE,
                RecoveryClass.COMPENSATABLE,
                RecoveryClass.FATAL,
                RecoveryClass.AUTHORITY_REQUIRED,
            ]
        ),
        next_tool_logits,
        torch.tensor([1, 2, 0, 1]),
    )
    torch.autograd.backward(loss)
    optimizer.step()

    after = [*action.parameters(), *recovery.parameters()]
    assert all(parameter.grad is not None for parameter in after)
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after, strict=True))

    with pytest.raises(ValueError, match="integer tensor"):
        action_policy_loss(
            tool_logits,
            torch.tensor([0.0, 1.0, 2.0, 1.0]),
            argument_values,
            torch.zeros(4, 2),
        )


def test_validator_rejects_malformed_actions() -> None:
    validator = DeterministicActionValidator()
    definition = _write_tool()
    wrong_type = _write_action().model_copy(
        update={"arguments": {"path": 3, "content": "safe"}}
    )
    unknown_argument = _write_action().model_copy(
        update={"arguments": {"path": "/x", "content": "safe", "force": True}}
    )
    with pytest.raises(ActionValidationError, match="must be string"):
        validator.validate(wrong_type, definition)
    with pytest.raises(ActionValidationError, match="unknown action arguments"):
        validator.validate(unknown_argument, definition)

    with pytest.raises(ValidationError, match="keys must be strings"):
        ActionEnvelope(
            tool_id="nested.action",
            schema_version="1",
            arguments={"payload": {1: "ambiguous"}},
            idempotency_key="nested-key-0001",
        )
    with pytest.raises(ValidationError, match="exactly match"):
        ArgumentRule(kind=ArgumentKind.INTEGER, enum=[True])


def test_unauthorized_write_is_rejected_before_any_executor_call() -> None:
    manager = TransactionManager(CapabilityKernel([_write_tool()]))
    executor = RecordingSandbox()
    context = ExecutionContext(
        capabilities=set(),
        satisfied_preconditions={"workspace-clean"},
        approval_ids=set(),
    )
    with pytest.raises(AuthorizationError, match="missing capabilities"):
        manager.prepare(_write_action(), context)
    assert executor.calls == []


def test_prepare_commit_and_idempotent_replay_execute_once() -> None:
    action = _write_action()
    manager, context = _authorized_manager(action)
    executor = RecordingSandbox()
    prepared = manager.prepare(action, context)
    assert prepared.state == TransactionState.PREPARED
    assert executor.calls == []

    receipt = manager.commit(prepared.transaction_id, executor)
    assert receipt.output == {"written": "safe"}
    assert receipt.output is not None
    receipt.output["written"] = "forged-by-caller"
    replay = manager.commit(prepared.transaction_id, executor)
    assert replay.output == {"written": "safe"}
    assert replay.output is not None
    replay.output["written"] = "forged-replay"
    second_replay = manager.commit(prepared.transaction_id, executor)
    same_prepare = manager.prepare(action, context)

    assert receipt.state == TransactionState.COMMITTED
    assert replay.replayed is True
    assert second_replay.output == {"written": "safe"}
    assert same_prepare.transaction_id == prepared.transaction_id
    assert len(executor.calls) == 1


def test_material_approval_is_revalidated_at_commit() -> None:
    action = _write_action()
    current_time = [datetime(2026, 9, 6, tzinfo=UTC)]
    approval = ActionApproval(
        approval_id=f"approval_{'e' * 16}",
        action_sha256=action_sha256(action),
        issued_by="test-reviewer",
        expires_at=current_time[0] + timedelta(minutes=1),
    )
    manager = TransactionManager(
        CapabilityKernel(
            [_write_tool()],
            trusted_approvals=[approval],
            clock=lambda: current_time[0],
        )
    )
    context = ExecutionContext(
        capabilities={"filesystem.write"},
        satisfied_preconditions={"workspace-clean"},
        approval_ids={approval.approval_id},
    )
    prepared = manager.prepare(action, context)
    current_time[0] += timedelta(days=1)
    executor = RecordingSandbox()

    with pytest.raises(AuthorizationError, match="approval"):
        manager.commit(prepared.transaction_id, executor)
    assert executor.calls == []


def test_kernel_copies_trusted_approval_and_tool_definition() -> None:
    first = _write_action()
    second = _write_action(content="different")
    definition = _write_tool()
    approval = _approval(first)
    kernel = CapabilityKernel([definition], trusted_approvals=[approval])
    approval.action_sha256 = action_sha256(second)
    object.__setattr__(definition, "approval_required", False)

    with pytest.raises(AuthorizationError, match="approval"):
        kernel.authorize(
            second,
            ExecutionContext(
                capabilities={"filesystem.write"},
                satisfied_preconditions={"workspace-clean"},
                approval_ids={approval.approval_id},
            ),
        )

    returned_definition, _ = kernel.authorize(
        first,
        ExecutionContext(
            capabilities={"filesystem.write"},
            satisfied_preconditions={"workspace-clean"},
            approval_ids={approval.approval_id},
        ),
    )
    returned_definition.required_capabilities.clear()
    with pytest.raises(AuthorizationError, match="capabilities"):
        kernel.authorize(
            first,
            ExecutionContext(
                satisfied_preconditions={"workspace-clean"},
                approval_ids={approval.approval_id},
            ),
        )


def test_prepare_defensively_copies_action_before_commit() -> None:
    action = _write_action()
    manager, context = _authorized_manager(action)
    prepared = manager.prepare(action, context)
    action.tool_id = "mutated.danger"
    action.arguments["content"] = "mutated"
    executor = RecordingSandbox()

    receipt = manager.commit(prepared.transaction_id, executor)
    assert receipt.output == {"written": "safe"}
    assert executor.calls[0][0] == "filesystem.write"
    assert executor.calls[0][1]["content"] == "safe"


def test_idempotency_key_collision_is_rejected() -> None:
    first = _write_action()
    second = _write_action(content="different")
    manager, context = _authorized_manager(first, second)
    manager.prepare(first, context)
    with pytest.raises(ActionValidationError, match="different action"):
        manager.prepare(second, context)


def test_material_tools_preconditions_and_approvals_are_definition_bound() -> None:
    with pytest.raises(ValidationError, match="material-effect"):
        ToolDefinition(
            tool_id="unsafe.write",
            schema_version="1",
            arguments={},
            material_effect=True,
        )

    action = _write_action().model_copy(update={"preconditions": []})
    approval = _approval(action)
    kernel = CapabilityKernel([_write_tool()], trusted_approvals=[approval])
    with pytest.raises(ActionValidationError, match="workspace-clean"):
        kernel.authorize(
            action,
            ExecutionContext(
                capabilities={"filesystem.write"},
                approval_ids={approval.approval_id},
            ),
        )


def test_material_executor_must_guarantee_transaction_id_idempotency() -> None:
    action = _write_action()
    manager, context = _authorized_manager(action)
    prepared = manager.prepare(action, context)

    class UnsafeExecutor:
        guarantees_idempotency = False

        def execute(
            self, tool_id: str, arguments: Mapping[str, Any], transaction_id: str
        ) -> Mapping[str, Any]:
            raise AssertionError("unsafe executor must not be called")

    with pytest.raises(TransactionStateError, match="idempotency"):
        manager.commit(prepared.transaction_id, UnsafeExecutor())


def test_failed_material_effect_is_not_claimed_rolled_back_or_leaked() -> None:
    action = _write_action()
    manager, context = _authorized_manager(action)
    prepared = manager.prepare(action, context)

    class PartialExecutor:
        guarantees_idempotency = True

        def execute(
            self, tool_id: str, arguments: Mapping[str, Any], transaction_id: str
        ) -> Mapping[str, Any]:
            del tool_id, arguments, transaction_id
            raise CompensatableExecutionError("secret-token=leaked-123")

    receipt = manager.commit(prepared.transaction_id, PartialExecutor())
    assert receipt.state is TransactionState.FAILED
    assert receipt.error == "CompensatableExecutionError"
    assert "secret-token" not in (receipt.error or "")
    with pytest.raises(TransactionStateError, match="compensation action"):
        manager.rollback(prepared.transaction_id)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RetryableExecutionError("later"), RecoveryClass.RETRYABLE),
        (TimeoutError("slow"), RecoveryClass.RETRYABLE),
        (CompensatableExecutionError("partial"), RecoveryClass.COMPENSATABLE),
        (PermissionError("approval"), RecoveryClass.AUTHORITY_REQUIRED),
        (ValueError("bad"), RecoveryClass.FATAL),
    ],
)
def test_execution_failure_classes(error: BaseException, expected: RecoveryClass) -> None:
    assert classify_execution_failure(error) == expected
