from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Any, Literal, Protocol, Self

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn
from torch.nn import functional as F

from olympus.core.schemas import StrictModel


class ActionValidationError(ValueError):
    pass


class AuthorizationError(PermissionError):
    pass


class TransactionStateError(RuntimeError):
    pass


class RetryableExecutionError(RuntimeError):
    pass


class CompensatableExecutionError(RuntimeError):
    pass


class AuthorityRequiredExecutionError(RuntimeError):
    pass


class ArgumentKind(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


def _argument_kind_matches(kind: ArgumentKind, value: Any) -> bool:
    if kind is ArgumentKind.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind is ArgumentKind.NUMBER:
        return (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    expected: dict[ArgumentKind, type[Any]] = {
        ArgumentKind.STRING: str,
        ArgumentKind.BOOLEAN: bool,
        ArgumentKind.OBJECT: dict,
        ArgumentKind.ARRAY: list,
    }
    return isinstance(value, expected[kind])


def _json_identity(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class ArgumentRule(StrictModel):
    kind: ArgumentKind
    required: bool = True
    enum: list[str | int | float | bool] | None = None
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("minimum must be finite")
        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("maximum must be finite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum cannot exceed maximum")
        if (self.minimum is not None or self.maximum is not None) and self.kind not in {
            ArgumentKind.INTEGER,
            ArgumentKind.NUMBER,
        }:
            raise ValueError("numeric bounds require an integer or number argument")
        if self.enum is not None:
            if not self.enum:
                raise ValueError("argument enum cannot be empty")
            if any(not _argument_kind_matches(self.kind, value) for value in self.enum):
                raise ValueError("argument enum values must exactly match the argument kind")
        return self


class ToolDefinition(StrictModel):
    tool_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
    schema_version: str = Field(min_length=1, max_length=64)
    arguments: dict[str, ArgumentRule]
    required_capabilities: set[str] = Field(default_factory=set)
    required_preconditions: set[str] = Field(default_factory=set)
    material_effect: bool = False
    approval_required: bool = False

    @model_validator(mode="after")
    def material_effects_require_approval(self) -> Self:
        if self.material_effect and not self.approval_required:
            raise ValueError("material-effect tools must require explicit approval")
        return self


class ActionEnvelope(StrictModel):
    tool_id: str
    schema_version: str
    arguments: dict[str, Any]
    preconditions: list[str] = Field(default_factory=list)
    idempotency_key: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{7,127}$")

    @model_validator(mode="after")
    def arguments_are_finite_json(self) -> Self:
        _validate_json_tree(self.arguments, path="arguments")
        return self


def action_sha256(action: ActionEnvelope) -> str:
    payload = json.dumps(
        action.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scoped_action_sha256(
    action: ActionEnvelope, execution_scope_sha256: str | None = None
) -> str:
    """Derive the transaction fingerprint shared by manager and public audit."""

    if execution_scope_sha256 is None:
        return action_sha256(action)
    _validate_digest(execution_scope_sha256, "execution scope")
    payload = json.dumps(
        {
            "action": action.model_dump(mode="json"),
            "execution_scope_sha256": execution_scope_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ActionApproval(StrictModel):
    approval_id: str = Field(pattern=r"^approval_[0-9a-f]{16,64}$")
    action_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issued_by: str = Field(min_length=1, max_length=200)
    expires_at: datetime

    @model_validator(mode="after")
    def timezone_aware_expiry(self) -> Self:
        if self.expires_at.utcoffset() is None:
            raise ValueError("action approval expiry must be timezone-aware")
        return self


class ExecutionContext(StrictModel):
    capabilities: set[str] = Field(default_factory=set)
    satisfied_preconditions: set[str] = Field(default_factory=set)
    approval_ids: set[str] = Field(default_factory=set)


class DeterministicActionValidator:
    def validate(self, action: ActionEnvelope, definition: ToolDefinition) -> dict[str, Any]:
        if (
            action.tool_id != definition.tool_id
            or action.schema_version != definition.schema_version
        ):
            raise ActionValidationError("tool ID or schema version does not match the registry")
        unknown = action.arguments.keys() - definition.arguments.keys()
        if unknown:
            raise ActionValidationError(f"unknown action arguments: {sorted(unknown)}")
        missing = {
            name
            for name, rule in definition.arguments.items()
            if rule.required and name not in action.arguments
        }
        if missing:
            raise ActionValidationError(f"missing required action arguments: {sorted(missing)}")
        validated: dict[str, Any] = {}
        for name, value in action.arguments.items():
            rule = definition.arguments[name]
            self._validate_value(name, value, rule)
            validated[name] = value
        return validated

    @staticmethod
    def _validate_value(name: str, value: Any, rule: ArgumentRule) -> None:
        if not _argument_kind_matches(rule.kind, value):
            raise ActionValidationError(f"argument {name!r} must be {rule.kind.value}")
        if rule.enum is not None and _json_identity(value) not in {
            _json_identity(option) for option in rule.enum
        }:
            raise ActionValidationError(f"argument {name!r} is outside its allowed enum")
        if rule.kind in {ArgumentKind.INTEGER, ArgumentKind.NUMBER}:
            numeric = float(value)
            if rule.minimum is not None and numeric < rule.minimum:
                raise ActionValidationError(f"argument {name!r} is below its minimum")
            if rule.maximum is not None and numeric > rule.maximum:
                raise ActionValidationError(f"argument {name!r} exceeds its maximum")


class CapabilityKernel:
    """Deterministically resolves schemas and authority; it cannot execute tools."""

    def __init__(
        self,
        definitions: list[ToolDefinition],
        *,
        validator: DeterministicActionValidator | None = None,
        trusted_approvals: tuple[ActionApproval, ...] | list[ActionApproval] = (),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        keys = [(definition.tool_id, definition.schema_version) for definition in definitions]
        if len(keys) != len(set(keys)):
            raise ValueError("tool registry contains duplicate tool/version entries")
        self._definitions = {
            key: definition.model_copy(deep=True)
            for key, definition in zip(keys, definitions, strict=True)
        }
        self._validator = validator or DeterministicActionValidator()
        approval_ids = [approval.approval_id for approval in trusted_approvals]
        if len(approval_ids) != len(set(approval_ids)):
            raise ValueError("trusted action approval IDs must be unique")
        self._trusted_approvals = {
            approval.approval_id: approval.model_copy(deep=True)
            for approval in trusted_approvals
        }
        self._clock = clock or (lambda: datetime.now(UTC))

    def authorize(
        self, action: ActionEnvelope, context: ExecutionContext
    ) -> tuple[ToolDefinition, dict[str, Any]]:
        definition = self._definitions.get((action.tool_id, action.schema_version))
        if definition is None:
            raise ActionValidationError("unknown tool or schema version")
        arguments = self._validator.validate(action, definition)
        try:
            rewritten = _json_identity(arguments) != _json_identity(action.arguments)
        except (TypeError, ValueError) as error:
            raise ActionValidationError(
                "action validator returned non-JSON arguments"
            ) from error
        if rewritten:
            raise ActionValidationError("action validators may not rewrite arguments")
        unmet = (
            definition.required_preconditions | set(action.preconditions)
        ) - context.satisfied_preconditions
        if unmet:
            raise ActionValidationError(f"unsatisfied preconditions: {sorted(unmet)}")
        missing_capabilities = definition.required_capabilities - context.capabilities
        if missing_capabilities:
            raise AuthorizationError(f"missing capabilities: {sorted(missing_capabilities)}")
        if definition.approval_required:
            now = self._clock()
            if now.utcoffset() is None:
                raise ValueError("capability-kernel clock must be timezone-aware")
            expected_action = action_sha256(action)
            approved = any(
                approval_id in self._trusted_approvals
                and self._trusted_approvals[approval_id].action_sha256 == expected_action
                and self._trusted_approvals[approval_id].expires_at > now
                for approval_id in context.approval_ids
            )
            if not approved:
                raise AuthorizationError(
                    f"tool {definition.tool_id!r} requires a trusted action-bound approval"
                )
        return definition.model_copy(deep=True), _copy_json_object(arguments)

    def describe_tool(self, tool_id: str, schema_version: str) -> ToolDefinition:
        """Return a defensive copy for cross-runtime contract consistency checks."""

        definition = self._definitions.get((tool_id, schema_version))
        if definition is None:
            raise ActionValidationError("unknown tool or schema version")
        return definition.model_copy(deep=True)


class RecoveryClass(IntEnum):
    RETRYABLE = 0
    COMPENSATABLE = 1
    FATAL = 2
    AUTHORITY_REQUIRED = 3


class TrainableActionPolicy(nn.Module):
    def __init__(
        self, state_dim: int, tool_count: int, argument_dim: int, hidden_dim: int = 32
    ) -> None:
        super().__init__()
        if min(state_dim, tool_count, argument_dim, hidden_dim) < 1:
            raise ValueError("model dimensions must be positive")
        self.state_dim = state_dim
        self.encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.GELU())
        self.tool_head = nn.Linear(hidden_dim, tool_count)
        self.argument_head = nn.Linear(hidden_dim, argument_dim)

    def forward(self, state_features: Tensor) -> tuple[Tensor, Tensor]:
        if state_features.ndim != 2 or state_features.shape[1] != self.state_dim:
            raise ValueError("state_features must be a batch of state feature vectors")
        if state_features.shape[0] == 0 or not torch.isfinite(state_features).all():
            raise ValueError("state_features must be a non-empty finite batch")
        hidden = self.encoder(state_features)
        return self.tool_head(hidden), self.argument_head(hidden)


class TrainableRecoveryPolicy(nn.Module):
    def __init__(self, failure_dim: int, tool_count: int, hidden_dim: int = 32) -> None:
        super().__init__()
        if min(failure_dim, tool_count, hidden_dim) < 1:
            raise ValueError("model dimensions must be positive")
        self.failure_dim = failure_dim
        self.encoder = nn.Sequential(nn.Linear(failure_dim, hidden_dim), nn.GELU())
        self.class_head = nn.Linear(hidden_dim, len(RecoveryClass))
        self.next_tool_head = nn.Linear(hidden_dim, tool_count)

    def forward(self, failure_features: Tensor) -> tuple[Tensor, Tensor]:
        if failure_features.ndim != 2 or failure_features.shape[1] != self.failure_dim:
            raise ValueError("failure_features must be a batch of failure feature vectors")
        if failure_features.shape[0] == 0 or not torch.isfinite(failure_features).all():
            raise ValueError("failure_features must be a non-empty finite batch")
        hidden = self.encoder(failure_features)
        return self.class_head(hidden), self.next_tool_head(hidden)


def action_policy_loss(
    tool_logits: Tensor,
    target_tools: Tensor,
    argument_values: Tensor,
    target_arguments: Tensor,
    *,
    argument_weight: float = 1.0,
) -> Tensor:
    if not math.isfinite(argument_weight) or argument_weight < 0.0:
        raise ValueError("argument_weight must be finite and non-negative")
    if tool_logits.ndim != 2 or target_tools.shape != (tool_logits.shape[0],):
        raise ValueError("target_tools must contain one label per tool-logit row")
    if target_tools.dtype not in {torch.int32, torch.int64}:
        raise ValueError("target_tools must use an integer tensor dtype")
    if torch.any(target_tools < 0) or torch.any(target_tools >= tool_logits.shape[1]):
        raise ValueError("target_tools contain a class outside the tool logits")
    if (
        argument_values.shape != target_arguments.shape
        or argument_values.shape[0] != tool_logits.shape[0]
    ):
        raise ValueError("argument targets must match predicted argument values")
    if not torch.isfinite(tool_logits).all() or not torch.isfinite(argument_values).all():
        raise ValueError("action policy predictions must be finite")
    if not torch.isfinite(target_arguments).all():
        raise ValueError("argument targets must be finite")
    return F.cross_entropy(tool_logits, target_tools) + argument_weight * F.mse_loss(
        argument_values, target_arguments
    )


def recovery_policy_loss(
    class_logits: Tensor,
    target_classes: Tensor,
    next_tool_logits: Tensor,
    target_tools: Tensor,
) -> Tensor:
    if class_logits.ndim != 2 or class_logits.shape[1] != len(RecoveryClass):
        raise ValueError("class_logits have the wrong recovery-class shape")
    if next_tool_logits.ndim != 2 or next_tool_logits.shape[0] != class_logits.shape[0]:
        raise ValueError("next-tool logits must align with recovery-class logits")
    expected_shape = (class_logits.shape[0],)
    if target_classes.shape != expected_shape or target_tools.shape != expected_shape:
        raise ValueError("recovery targets must contain one label per batch row")
    if target_classes.dtype not in {torch.int32, torch.int64} or target_tools.dtype not in {
        torch.int32,
        torch.int64,
    }:
        raise ValueError("recovery targets must use integer tensor dtypes")
    if torch.any(target_classes < 0) or torch.any(target_classes >= class_logits.shape[1]):
        raise ValueError("recovery class targets are outside the class logits")
    if torch.any(target_tools < 0) or torch.any(target_tools >= next_tool_logits.shape[1]):
        raise ValueError("recovery tool targets are outside the tool logits")
    if not torch.isfinite(class_logits).all() or not torch.isfinite(next_tool_logits).all():
        raise ValueError("recovery policy predictions must be finite")
    return F.cross_entropy(class_logits, target_classes) + F.cross_entropy(
        next_tool_logits, target_tools
    )


class TransactionState(StrEnum):
    PREPARED = "prepared"
    EXECUTING = "executing"
    COMMITTED = "committed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


ExecutionFailureCode = Literal[
    "retryable_execution_failure",
    "compensatable_execution_failure",
    "fatal_execution_failure",
    "authority_required_execution_failure",
]


class ExecutionReceipt(StrictModel):
    transaction_id: str = Field(pattern=r"^tx_[0-9a-f]{64}$")
    idempotency_key: str
    state: TransactionState
    output: dict[str, Any] = Field(default_factory=dict)
    replayed: bool = False
    recovery_class: RecoveryClass | None = None
    error: ExecutionFailureCode | None = None

    @model_validator(mode="after")
    def require_terminal_consistency(self) -> Self:
        if self.state is TransactionState.COMMITTED:
            if self.recovery_class is not None or self.error is not None:
                raise ValueError("committed receipts cannot contain failure metadata")
        elif self.state is TransactionState.FAILED:
            if self.recovery_class is None or self.error is None or self.output:
                raise ValueError(
                    "failed receipts require closed failure metadata and no output"
                )
        else:
            raise ValueError("execution receipts must describe a terminal transaction")
        return self


class TransactionSnapshot(StrictModel):
    transaction_id: str = Field(pattern=r"^tx_[0-9a-f]{64}$")
    idempotency_key: str
    tool_id: str
    state: TransactionState
    material_effect: bool
    execution_scope_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class SandboxExecutor(Protocol):
    guarantees_idempotency: bool

    def execute(
        self, tool_id: str, arguments: Mapping[str, Any], transaction_id: str
    ) -> Mapping[str, Any]: ...


class _Transaction:
    def __init__(
        self,
        transaction_id: str,
        action: ActionEnvelope,
        definition: ToolDefinition,
        arguments: dict[str, Any],
        context: ExecutionContext,
        fingerprint: str,
        execution_scope_sha256: str | None,
        execution_scope_token: str | None,
    ) -> None:
        self.transaction_id = transaction_id
        self.action = action.model_copy(deep=True)
        self.definition = definition.model_copy(deep=True)
        self.arguments = _copy_json_object(arguments)
        self.context = context.model_copy(deep=True)
        self.fingerprint = fingerprint
        self.execution_scope_sha256 = execution_scope_sha256
        self.execution_scope_token = execution_scope_token
        self.state = TransactionState.PREPARED
        self.receipt: ExecutionReceipt | None = None
        self.lock = threading.RLock()

    def snapshot(self) -> TransactionSnapshot:
        return TransactionSnapshot(
            transaction_id=self.transaction_id,
            idempotency_key=self.action.idempotency_key,
            tool_id=self.action.tool_id,
            state=self.state,
            material_effect=self.definition.material_effect,
            execution_scope_sha256=self.execution_scope_sha256,
        )


class TransactionManager:
    """Prepare/commit manager with idempotent replay and no built-in side effects."""

    def __init__(self, kernel: CapabilityKernel) -> None:
        self._kernel = kernel
        self._transactions: dict[str, _Transaction] = {}
        self._by_idempotency_key: dict[tuple[str, str | None], str] = {}
        # A reservation is an in-process capability boundary.  It prevents a
        # caller that shares this manager from occupying or committing an
        # execution scope while an external authority is reviewing it.
        self._scope_reservations: dict[str, str] = {}
        self._lock = threading.RLock()

    def reserve_execution_scope(
        self, execution_scope_sha256: str, execution_scope_token: str
    ) -> None:
        """Atomically reserve a scoped transaction for its owning runtime."""

        _validate_digest(execution_scope_sha256, "execution scope")
        _validate_digest(execution_scope_token, "execution scope token")
        with self._lock:
            existing = self._scope_reservations.get(execution_scope_sha256)
            if existing is not None:
                if existing == execution_scope_token:
                    return
                raise TransactionStateError("execution scope is already reserved")
            if any(
                scope == execution_scope_sha256
                for _, scope in self._by_idempotency_key
            ):
                raise TransactionStateError(
                    "execution scope already contains a transaction"
                )
            self._scope_reservations[execution_scope_sha256] = execution_scope_token

    def release_execution_scope(
        self, execution_scope_sha256: str, execution_scope_token: str
    ) -> None:
        """Release an unused reservation after an aborted preparation path."""

        _validate_digest(execution_scope_sha256, "execution scope")
        _validate_digest(execution_scope_token, "execution scope token")
        with self._lock:
            if self._scope_reservations.get(execution_scope_sha256) != execution_scope_token:
                raise TransactionStateError("execution scope reservation is not owned")
            if any(
                scope == execution_scope_sha256
                for _, scope in self._by_idempotency_key
            ):
                raise TransactionStateError(
                    "execution scope with a transaction cannot be released"
                )
            del self._scope_reservations[execution_scope_sha256]

    def describe_tool(self, tool_id: str, schema_version: str) -> ToolDefinition:
        """Expose only a defensive description of the injected capability policy."""

        return self._kernel.describe_tool(tool_id, schema_version)

    def preflight(
        self, action: ActionEnvelope, context: ExecutionContext
    ) -> tuple[ToolDefinition, dict[str, Any]]:
        """Validate authority and arguments without creating a transaction."""

        definition, arguments = self._kernel.authorize(
            action.model_copy(deep=True), context.model_copy(deep=True)
        )
        return definition.model_copy(deep=True), _copy_json_object(arguments)

    def snapshot(self, transaction_id: str) -> TransactionSnapshot:
        """Return a defensive snapshot of the manager-owned transaction state."""

        with self._lock:
            transaction = self._transactions.get(transaction_id)
        if transaction is None:
            raise TransactionStateError("unknown transaction")
        with transaction.lock:
            return transaction.snapshot().model_copy(deep=True)

    def receipt(
        self, transaction_id: str, *, execution_scope_token: str | None = None
    ) -> ExecutionReceipt | None:
        """Return a defensive terminal receipt for process-local recovery."""

        with self._lock:
            transaction = self._transactions.get(transaction_id)
        if transaction is None:
            raise TransactionStateError("unknown transaction")
        with transaction.lock:
            if (
                transaction.execution_scope_token is not None
                and transaction.execution_scope_token != execution_scope_token
            ):
                raise AuthorizationError(
                    "scoped transaction requires its reservation token"
                )
            if transaction.receipt is None:
                return None
            return transaction.receipt.model_copy(deep=True)

    @staticmethod
    def _fingerprint(
        action: ActionEnvelope, execution_scope_sha256: str | None = None
    ) -> str:
        return scoped_action_sha256(action, execution_scope_sha256)

    def prepare(
        self,
        action: ActionEnvelope,
        context: ExecutionContext,
        *,
        execution_scope_sha256: str | None = None,
        execution_scope_token: str | None = None,
        require_new: bool = False,
    ) -> TransactionSnapshot:
        if execution_scope_sha256 is not None:
            _validate_digest(execution_scope_sha256, "execution scope")
        if execution_scope_token is not None:
            _validate_digest(execution_scope_token, "execution scope token")
        if execution_scope_token is not None and execution_scope_sha256 is None:
            raise ValueError("an execution scope token requires an execution scope")
        action_snapshot = action.model_copy(deep=True)
        definition, arguments = self._kernel.authorize(action_snapshot, context)
        fingerprint = self._fingerprint(action_snapshot, execution_scope_sha256)
        idempotency_scope = (
            action_snapshot.idempotency_key,
            execution_scope_sha256,
        )
        with self._lock:
            reservation = (
                self._scope_reservations.get(execution_scope_sha256)
                if execution_scope_sha256 is not None
                else None
            )
            if reservation is not None and reservation != execution_scope_token:
                raise AuthorizationError("execution scope requires its reservation token")
            if reservation is None and execution_scope_token is not None:
                raise AuthorizationError("execution scope token has no reservation")
            existing_id = self._by_idempotency_key.get(idempotency_scope)
            if existing_id is not None:
                existing = self._transactions[existing_id]
                if existing.fingerprint != fingerprint:
                    raise ActionValidationError("idempotency key was reused for a different action")
                if require_new:
                    raise TransactionStateError(
                        "execution scope already contains a transaction"
                    )
                return existing.snapshot()
            transaction_id = f"tx_{fingerprint}"
            collision = self._transactions.get(transaction_id)
            if collision is not None and collision.fingerprint != fingerprint:
                raise TransactionStateError("transaction ID collision detected")
            transaction = _Transaction(
                transaction_id,
                action_snapshot,
                definition,
                arguments,
                context,
                fingerprint,
                execution_scope_sha256,
                execution_scope_token,
            )
            self._transactions[transaction_id] = transaction
            self._by_idempotency_key[idempotency_scope] = transaction_id
            return transaction.snapshot()

    def commit(
        self,
        transaction_id: str,
        executor: SandboxExecutor,
        *,
        execution_scope_token: str | None = None,
    ) -> ExecutionReceipt:
        with self._lock:
            transaction = self._transactions.get(transaction_id)
        if transaction is None:
            raise TransactionStateError("unknown transaction")
        with transaction.lock:
            if (
                transaction.execution_scope_token is not None
                and transaction.execution_scope_token != execution_scope_token
            ):
                raise AuthorizationError(
                    "scoped transaction requires its reservation token"
                )
            if transaction.state == TransactionState.COMMITTED:
                if transaction.receipt is None:
                    raise TransactionStateError("committed transaction is missing its receipt")
                return transaction.receipt.model_copy(
                    update={"replayed": True}, deep=True
                )
            if transaction.state == TransactionState.ROLLED_BACK:
                raise TransactionStateError("rolled-back transactions cannot be committed")
            if transaction.state == TransactionState.EXECUTING:
                raise TransactionStateError("transaction is already executing")
            if transaction.state == TransactionState.FAILED:
                if (
                    transaction.receipt is None
                    or transaction.receipt.recovery_class != RecoveryClass.RETRYABLE
                ):
                    raise TransactionStateError("only retryable failures may be committed again")
                transaction.state = TransactionState.PREPARED
            if (
                self._fingerprint(
                    transaction.action, transaction.execution_scope_sha256
                )
                != transaction.fingerprint
            ):
                raise TransactionStateError("prepared action no longer matches its fingerprint")
            definition, arguments = self._kernel.authorize(
                transaction.action, transaction.context
            )
            if definition != transaction.definition or arguments != transaction.arguments:
                raise TransactionStateError(
                    "prepared action no longer matches its registry decision"
                )
            if definition.material_effect and not bool(
                getattr(executor, "guarantees_idempotency", False)
            ):
                raise TransactionStateError(
                    "material effects require an executor with transaction-id idempotency"
                )
            transaction.state = TransactionState.EXECUTING
            try:
                raw_output = dict(
                    executor.execute(
                        transaction.action.tool_id,
                        _copy_json_object(arguments),
                        transaction.transaction_id,
                    )
                )
                output = _safe_json_object(raw_output)
            except Exception as error:
                recovery_class = classify_execution_failure(error)
                transaction.state = TransactionState.FAILED
                transaction.receipt = ExecutionReceipt(
                    transaction_id=transaction.transaction_id,
                    idempotency_key=transaction.action.idempotency_key,
                    state=TransactionState.FAILED,
                    recovery_class=recovery_class,
                    # Neither tool-controlled exception text nor dynamic class
                    # names belong in a public receipt: both can carry secrets.
                    error=_public_failure_code(recovery_class),
                )
                return transaction.receipt.model_copy(deep=True)
            transaction.state = TransactionState.COMMITTED
            transaction.receipt = ExecutionReceipt(
                transaction_id=transaction.transaction_id,
                idempotency_key=transaction.action.idempotency_key,
                state=TransactionState.COMMITTED,
                output=output,
            )
            return transaction.receipt.model_copy(deep=True)

    def rollback(
        self, transaction_id: str, *, execution_scope_token: str | None = None
    ) -> TransactionSnapshot:
        with self._lock:
            transaction = self._transactions.get(transaction_id)
        if transaction is None:
            raise TransactionStateError("unknown transaction")
        with transaction.lock:
            if (
                transaction.execution_scope_token is not None
                and transaction.execution_scope_token != execution_scope_token
            ):
                raise AuthorizationError(
                    "scoped transaction requires its reservation token"
                )
            if transaction.state == TransactionState.COMMITTED:
                raise TransactionStateError(
                    "committed effects require an explicit compensation action"
                )
            if (
                transaction.state == TransactionState.FAILED
                and transaction.definition.material_effect
            ):
                raise TransactionStateError(
                    "failed material effects require an explicit compensation action"
                )
            transaction.state = TransactionState.ROLLED_BACK
            return transaction.snapshot()


def classify_execution_failure(error: BaseException) -> RecoveryClass:
    if isinstance(error, RetryableExecutionError | TimeoutError | ConnectionError):
        return RecoveryClass.RETRYABLE
    if isinstance(error, CompensatableExecutionError):
        return RecoveryClass.COMPENSATABLE
    if isinstance(error, AuthorityRequiredExecutionError | PermissionError):
        return RecoveryClass.AUTHORITY_REQUIRED
    return RecoveryClass.FATAL


def _public_failure_code(recovery_class: RecoveryClass) -> ExecutionFailureCode:
    failure_codes: dict[RecoveryClass, ExecutionFailureCode] = {
        RecoveryClass.RETRYABLE: "retryable_execution_failure",
        RecoveryClass.COMPENSATABLE: "compensatable_execution_failure",
        RecoveryClass.FATAL: "fatal_execution_failure",
        RecoveryClass.AUTHORITY_REQUIRED: "authority_required_execution_failure",
    }
    return failure_codes[recovery_class]


def _safe_json_object(value: dict[str, Any]) -> dict[str, Any]:
    _validate_json_tree(value, path="executor output")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("executor output must contain finite JSON values") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("executor output must be an object")
    return decoded


def _copy_json_object(value: dict[str, Any]) -> dict[str, Any]:
    _validate_json_tree(value, path="JSON object")
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ValueError("JSON object copy did not produce an object")
    return decoded


def _validate_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 value")


def _validate_json_tree(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} JSON object keys must be strings")
            _validate_json_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    raise ValueError(f"{path} must contain only finite JSON values")
