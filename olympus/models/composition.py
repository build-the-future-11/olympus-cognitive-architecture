"""Two-phase composition of the six Olympus reference model roles.

The implementation proves a narrow interoperability claim.  It is not an
autonomous research agent, a durable workflow engine, or a sandbox.  It pauses
before execution for a host-registered Aion approval bound to the exact action,
tool policies, and host-asserted executor profile.  It refuses tools declared
material and never exposes the trusted workspace or raw tool output in public
pending/result artifacts. Retrieval and final-response requests are explicit
public artifact fields; callers must not place secrets in them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, Self, cast

import torch
from pydantic import Field, model_validator

from olympus.core.schemas import StrictModel
from olympus.models.aion import (
    AionController,
    AionRunState,
    Approval,
    AuthorityKind,
    FrozenProtocol,
    ResearchStage,
)
from olympus.models.atlas import AuthorizedQuery, EvidenceSet, OlympusAtlasService
from olympus.models.hermes import HermesWorkspaceRuntime
from olympus.models.kronos import (
    EventKind,
    KronosTemporalPlanner,
    TemporalEvent,
    tensorize_event_stream,
)
from olympus.models.perseus import (
    ActionEnvelope,
    ActionValidationError,
    ExecutionContext,
    ExecutionFailureCode,
    ExecutionReceipt,
    RecoveryClass,
    SandboxExecutor,
    ToolDefinition,
    TransactionManager,
    TransactionSnapshot,
    TransactionState,
    action_sha256,
    scoped_action_sha256,
)
from olympus.models.prometheus import (
    BranchAssessment,
    HypothesisBranch,
    PrometheusSelector,
)
from olympus.models.substrate import (
    AbstainOutput,
    ActionOutput,
    AuthorizedWorkspaceView,
    Claim,
    ClaimOutput,
    ClaimStatus,
    EvidenceItem,
    TextOutput,
    ToolInvocation,
    ToolSpec,
    WorkspaceState,
    append_workspace_event,
    authorized_workspace_view,
    canonical_json_bytes,
    canonical_sha256,
    ensure_finite_json,
    tool_invocation_sha256,
    validate_envelope_against_workspace,
    validate_workspace_snapshot,
)

FamilyName = Literal[
    "Hermes", "Prometheus", "Perseus", "Olympus-Atlas", "Kronos", "Aion"
]
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SUCCESS_TRACE_ORDER: tuple[FamilyName, ...] = (
    "Olympus-Atlas",
    "Prometheus",
    "Aion",
    "Perseus",
    "Kronos",
    "Hermes",
    "Aion",
)


class ProfiledSandboxExecutor(SandboxExecutor, Protocol):
    """Executor contract used by the reference composition.

    The profile is a host assertion included in the reviewed execution
    manifest. The reference runtime does not authenticate it or provide a
    sandbox.
    """

    execution_profile_sha256: str


class FamilyTrace(StrictModel):
    family: FamilyName
    operation: str = Field(min_length=1, max_length=200)
    status: Literal["completed", "prepared", "abstained", "failed"]
    input_sha256: str = Field(pattern=_SHA256_PATTERN)
    output_sha256: str = Field(pattern=_SHA256_PATTERN)


class ExecutionManifest(StrictModel):
    """Public review material for one proposed non-material execution."""

    schema_version: Literal[1] = 1
    action_sha256: str = Field(pattern=_SHA256_PATTERN)
    shared_tool: ToolSpec
    perseus_tool: ToolDefinition
    executor_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_scope_sha256: str = Field(pattern=_SHA256_PATTERN)
    declared_material_effect: Literal[False] = False

    @model_validator(mode="after")
    def validate_policy_alignment(self) -> Self:
        shared = self.shared_tool
        native = self.perseus_tool
        if (shared.tool_id, shared.schema_version) != (
            native.tool_id,
            native.schema_version,
        ):
            raise ValueError("execution manifest tool identities disagree")
        if (
            shared.required_capabilities != native.required_capabilities
            or shared.material_effect != native.material_effect
            or shared.approval_required != native.approval_required
        ):
            raise ValueError("shared and Perseus tool authority policies disagree")
        if shared.material_effect or native.material_effect:
            raise ValueError("reference composition refuses tools declared material")
        return self

    @property
    def sha256(self) -> str:
        return canonical_sha256(self)


class ExecutionApprovalRequirement(StrictModel):
    scope: Literal["execute"] = "execute"
    run_id: str = Field(pattern=r"^aion_[0-9a-f]{16,64}$")
    protocol_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_head_sha256: str = Field(pattern=_SHA256_PATTERN)
    action_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_scope_sha256: str = Field(pattern=_SHA256_PATTERN)
    non_model_authority_required: Literal[True] = True


class TemporalObservationFeatures(StrictModel):
    event_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    occurred_at: datetime
    features: list[float] = Field(min_length=1, max_length=4_096)
    environment_version: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.occurred_at.utcoffset() is None:
            raise ValueError("temporal observation time must be timezone-aware")
        if not all(math.isfinite(value) for value in self.features):
            raise ValueError("temporal observation features must be finite")
        return self


class KronosObservation(StrictModel):
    """Public, explicitly unauthenticated summary of a host-supplied forecast."""

    status: Literal["completed", "failed"] = "completed"
    failure_code: Literal[
        "component_runtime_failure", "model_call_budget_exhausted"
    ] | None = None
    input_reference_sha256: str = Field(pattern=_SHA256_PATTERN)
    temporal_event_id: str
    environment_version: str
    model_state_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    forecast_sha256: str = Field(pattern=_SHA256_PATTERN)
    forecast_mean: list[list[float]]
    forecast_log_variance: list[list[float]]
    proposed_action_indices: list[int]
    source_kind: Literal["host_supplied_pre_execution_features"] = (
        "host_supplied_pre_execution_features"
    )
    source_authenticated: Literal[False] = False
    feature_values_disclosed: Literal[False] = False
    authoritative: Literal[False] = False
    used_for_action_selection: Literal[False] = False

    @model_validator(mode="after")
    def validate_outputs(self) -> Self:
        expected_input = canonical_sha256(
            {
                "event_id": self.temporal_event_id,
                "environment_version": self.environment_version,
                "source_kind": self.source_kind,
            }
        )
        if self.input_reference_sha256 != expected_input:
            raise ValueError("Kronos input reference is not bound to its public metadata")
        if self.status == "failed":
            if self.failure_code is None:
                raise ValueError("failed Kronos observations require a failure code")
            if (
                self.forecast_mean
                or self.forecast_log_variance
                or self.proposed_action_indices
            ):
                raise ValueError("failed Kronos observations cannot contain forecasts")
            if self.model_state_sha256 is not None:
                raise ValueError("failed Kronos observations cannot claim a model-state hash")
            if self.forecast_sha256 != canonical_sha256(
                {"failure_code": self.failure_code}
            ):
                raise ValueError("Kronos failure record hash is invalid")
            return self
        if self.failure_code is not None:
            raise ValueError("completed Kronos observations cannot contain a failure code")
        if self.model_state_sha256 is None:
            raise ValueError("completed Kronos observations require a model-state hash")
        if not self.forecast_mean or not self.forecast_log_variance:
            raise ValueError("completed Kronos forecasts require non-empty matrices")
        mean_width = len(self.forecast_mean[0])
        variance_width = len(self.forecast_log_variance[0])
        if (
            mean_width == 0
            or variance_width == 0
            or any(len(row) != mean_width for row in self.forecast_mean)
            or any(
                len(row) != variance_width for row in self.forecast_log_variance
            )
            or len(self.forecast_mean) != len(self.forecast_log_variance)
            or mean_width != variance_width
        ):
            raise ValueError(
                "Kronos mean and log-variance must be equal rectangular matrices"
            )
        if not self.proposed_action_indices or any(
            index < 0 for index in self.proposed_action_indices
        ):
            raise ValueError("completed Kronos observations require non-negative actions")
        values = [
            value
            for horizon in [*self.forecast_mean, *self.forecast_log_variance]
            for value in horizon
        ]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("Kronos observation outputs must be non-empty and finite")
        payload = {
            "mean": self.forecast_mean,
            "log_variance": self.forecast_log_variance,
            "proposed_action_indices": self.proposed_action_indices,
        }
        if canonical_sha256(payload) != self.forecast_sha256:
            raise ValueError("Kronos forecast hash is invalid")
        return self


class ExecutionSummary(StrictModel):
    transaction_id: str = Field(pattern=r"^tx_[0-9a-f]{64}$")
    idempotency_key: str
    state: TransactionState
    summary_sha256: str = Field(pattern=_SHA256_PATTERN)
    replayed: bool
    recovery_class: RecoveryClass | None = None
    error_code: ExecutionFailureCode | None = None
    raw_output_recorded: Literal[False] = False

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        if self.state is TransactionState.COMMITTED:
            if self.recovery_class is not None or self.error_code is not None:
                raise ValueError("committed summaries cannot contain failure metadata")
        elif self.state is TransactionState.FAILED:
            if self.recovery_class is None or not self.error_code:
                raise ValueError("failed summaries require sanitized failure metadata")
            expected_code: dict[RecoveryClass, ExecutionFailureCode] = {
                RecoveryClass.RETRYABLE: "retryable_execution_failure",
                RecoveryClass.COMPENSATABLE: "compensatable_execution_failure",
                RecoveryClass.FATAL: "fatal_execution_failure",
                RecoveryClass.AUTHORITY_REQUIRED: (
                    "authority_required_execution_failure"
                ),
            }
            if self.error_code != expected_code[self.recovery_class]:
                raise ValueError("failure code does not match its recovery class")
        else:
            raise ValueError("public execution summaries must be terminal")
        if self.summary_sha256 != canonical_sha256(_execution_public_payload(self)):
            raise ValueError("execution summary hash is invalid")
        return self


class PublicAnswer(StrictModel):
    kind: Literal["text", "abstain"]
    producer: Literal["hermes", "runtime", "runtime_fail_closed"] = "hermes"
    text: str | None = Field(default=None, max_length=100_000)
    citation_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = Field(default=None, max_length=10_000)

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        closed_runtime_reasons = {
            "hermes:component_runtime_failure",
            "hermes:model_call_budget_exhausted",
        }
        if self.kind == "text":
            if not self.text or self.reason is not None or not self.citation_evidence_ids:
                raise ValueError(
                    "public text answers require text and citations and forbid a reason"
                )
        elif self.text is not None or self.citation_evidence_ids or not self.reason:
            raise ValueError("public abstentions require only an abstention reason")
        if self.producer == "runtime_fail_closed" and (
            self.kind != "abstain"
            or self.confidence != 0.0
            or self.reason not in closed_runtime_reasons
        ):
            raise ValueError(
                "runtime fail-closed answers require one closed Hermes failure reason"
            )
        if self.producer == "runtime" and self.kind != "abstain":
            raise ValueError("runtime-generated public answers must abstain")
        if (
            self.producer != "runtime_fail_closed"
            and self.reason in closed_runtime_reasons
        ):
            raise ValueError("closed runtime failure reasons require the runtime producer")
        return self


class AionSummary(StrictModel):
    run_id: str = Field(pattern=r"^aion_[0-9a-f]{16,64}$")
    protocol_sha256: str = Field(pattern=_SHA256_PATTERN)
    stage: ResearchStage
    steps_used: int = Field(ge=0)
    tool_calls_used: int = Field(ge=0)
    final_head_sha256: str = Field(pattern=_SHA256_PATTERN)


class CompositionBudgetUsage(StrictModel):
    minimum_model_calls: Literal[4] = 4
    model_call_budget: int = Field(ge=4)
    model_calls_used: int = Field(ge=4)
    tool_call_budget: int = Field(ge=1)
    tool_calls_used: Literal[1] = 1

    @model_validator(mode="after")
    def usage_fits_admitted_budget(self) -> Self:
        if self.model_calls_used > self.model_call_budget:
            raise ValueError("model-call usage exceeds the admitted workspace budget")
        if self.tool_calls_used > self.tool_call_budget:
            raise ValueError("tool-call usage exceeds the admitted workspace budget")
        return self


class PendingReferenceReplay(StrictModel):
    schema_version: Literal[1] = 1
    pending_id: str = Field(pattern=r"^pending_[0-9a-f]{32}$")
    run_id: str = Field(pattern=r"^aion_[0-9a-f]{16,64}$")
    retrieval_query: str = Field(min_length=1, max_length=20_000)
    retrieval_input_view_sha256: str = Field(pattern=_SHA256_PATTERN)
    workspace_view: AuthorizedWorkspaceView
    workspace_view_sha256: str = Field(pattern=_SHA256_PATTERN)
    proposed_claim: Claim
    action: ToolInvocation
    execution_manifest: ExecutionManifest
    execution_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    approval_requirement: ExecutionApprovalRequirement
    traces: list[FamilyTrace] = Field(min_length=3, max_length=3)
    scientific_claim: Literal["contract_composition_only"] = "contract_composition_only"
    promotion_authorized: Literal[False] = False
    qualifying_result: Literal[False] = False
    tool_execution_attempted: Literal[False] = False

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        if self.workspace_view.sha256 != self.workspace_view_sha256:
            raise ValueError("pending authorized workspace view hash is stale")
        if self.proposed_claim.status is not ClaimStatus.PROPOSED:
            raise ValueError("pending Prometheus claim must remain proposed")
        authorized_ids = {item.evidence_id for item in self.workspace_view.evidence}
        claim_evidence_ids = set(self.proposed_claim.support_evidence_ids) | set(
            self.proposed_claim.refute_evidence_ids
        )
        if not claim_evidence_ids.issubset(authorized_ids):
            raise ValueError("pending claim references evidence outside its authorized view")
        action_digest = tool_invocation_sha256(self.action)
        if self.execution_manifest.action_sha256 != action_digest:
            raise ValueError("pending action does not match its execution manifest")
        if self.execution_manifest.sha256 != self.execution_manifest_sha256:
            raise ValueError("pending execution manifest hash is stale")
        if self.execution_manifest.shared_tool not in self.workspace_view.tools:
            raise ValueError("pending selected tool is absent from its authorized view")
        requirement = self.approval_requirement
        if (
            requirement.run_id != self.run_id
            or requirement.action_sha256 != action_digest
            or requirement.execution_manifest_sha256
            != self.execution_manifest_sha256
            or requirement.execution_scope_sha256
            != self.execution_manifest.execution_scope_sha256
        ):
            raise ValueError("approval requirement is not bound to the pending execution")
        if tuple(trace.family for trace in self.traces) != _SUCCESS_TRACE_ORDER[:3]:
            raise ValueError("pending replay has an invalid family trace order")
        if self.traces[1].output_sha256 != canonical_sha256(self.proposed_claim):
            raise ValueError("pending claim does not match its Prometheus trace")
        if self.traces[1].input_sha256 != self.workspace_view_sha256:
            raise ValueError("pending Prometheus trace does not match its authorized view")
        if self.traces[2].output_sha256 != requirement.expected_head_sha256:
            raise ValueError("pending Aion trace does not match its approval head")
        expected_trace = (
            ("Olympus-Atlas", "authorized_retrieval", "completed"),
            ("Prometheus", "trusted_process_selection", "completed"),
            ("Aion", "preregistered_execution_gate", "prepared"),
        )
        observed_trace = tuple(
            (trace.family, trace.operation, trace.status) for trace in self.traces
        )
        if observed_trace != expected_trace:
            raise ValueError("pending replay trace semantics are inconsistent")
        if self.traces[2].input_sha256 != requirement.protocol_sha256:
            raise ValueError("pending approval protocol does not match its Aion trace")
        if self.traces[0].input_sha256 != _request_input_sha256(
            self.retrieval_input_view_sha256, self.retrieval_query
        ):
            raise ValueError("pending Atlas trace does not bind its public query")
        return self


class ReferenceReplayHalt(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^aion_[0-9a-f]{16,64}$")
    retrieval_query: str = Field(min_length=1, max_length=20_000)
    retrieval_input_view_sha256: str = Field(pattern=_SHA256_PATTERN)
    reason: str = Field(min_length=1, max_length=1_000)
    workspace_view: AuthorizedWorkspaceView
    workspace_view_sha256: str = Field(pattern=_SHA256_PATTERN)
    aion: AionSummary
    abstention: PublicAnswer
    traces: list[FamilyTrace] = Field(min_length=2, max_length=3)
    scientific_claim: Literal["contract_composition_only"] = "contract_composition_only"
    promotion_authorized: Literal[False] = False
    qualifying_result: Literal[False] = False
    tool_execution_attempted: Literal[False] = False

    @model_validator(mode="after")
    def validate_halt(self) -> Self:
        if self.workspace_view.sha256 != self.workspace_view_sha256:
            raise ValueError("halt authorized workspace view hash is stale")
        if self.aion.stage is not ResearchStage.STOP or self.aion.run_id != self.run_id:
            raise ValueError("halt must expose the matching terminal Aion summary")
        if self.abstention.kind != "abstain":
            raise ValueError("halt result must expose an abstention")
        if self.abstention.producer != "runtime":
            raise ValueError("pre-execution halt abstention must be runtime-generated")
        if self.traces[-1].family != "Aion":
            raise ValueError("halt trace must terminate at Aion")
        if self.traces[-1].output_sha256 != self.aion.final_head_sha256:
            raise ValueError("halt Aion trace does not match its terminal head")
        if self.aion.steps_used != 1 or self.aion.tool_calls_used != 0:
            raise ValueError("pre-execution halt must use one Aion STOP transition")
        if self.traces[-1].input_sha256 != self.aion.protocol_sha256:
            raise ValueError("halt Aion trace does not match its protocol")
        expected_prefix = (
            (("Olympus-Atlas", "authorized_retrieval", "abstained"),)
            if len(self.traces) == 2
            else (
                ("Olympus-Atlas", "authorized_retrieval", "completed"),
                ("Prometheus", "trusted_process_selection", "abstained"),
            )
        )
        observed_prefix = tuple(
            (trace.family, trace.operation, trace.status) for trace in self.traces[:-1]
        )
        if observed_prefix != expected_prefix or (
            self.traces[-1].family,
            self.traces[-1].operation,
            self.traces[-1].status,
        ) != ("Aion", "fail_closed_stop", "abstained"):
            raise ValueError("halt replay trace semantics are inconsistent")
        if self.abstention.reason != self.reason:
            raise ValueError("halt reason and abstention reason disagree")
        if self.traces[0].input_sha256 != _request_input_sha256(
            self.retrieval_input_view_sha256, self.retrieval_query
        ):
            raise ValueError("halt Atlas trace does not bind its public query")
        return self


class ReferenceReplayResult(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^aion_[0-9a-f]{16,64}$")
    retrieval_query: str = Field(min_length=1, max_length=20_000)
    retrieval_input_view_sha256: str = Field(pattern=_SHA256_PATTERN)
    final_request: str = Field(min_length=1, max_length=20_000)
    hermes_input_view_sha256: str = Field(pattern=_SHA256_PATTERN)
    hermes_input_evidence_ids: list[str]
    workspace_view: AuthorizedWorkspaceView
    workspace_view_sha256: str = Field(pattern=_SHA256_PATTERN)
    aion: AionSummary
    proposed_claim: Claim
    action: ToolInvocation
    execution_manifest: ExecutionManifest
    execution_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    approval: Approval
    transaction: TransactionSnapshot
    execution: ExecutionSummary
    kronos_observation: KronosObservation
    answer: PublicAnswer
    budget_usage: CompositionBudgetUsage
    traces: list[FamilyTrace] = Field(min_length=7, max_length=7)
    execution_succeeded: bool
    declared_material_effect: Literal[False] = False
    scientific_claim: Literal["contract_composition_only"] = "contract_composition_only"
    promotion_authorized: Literal[False] = False
    qualifying_result: Literal[False] = False

    @model_validator(mode="after")
    def validate_boundary(self) -> Self:
        if self.workspace_view.sha256 != self.workspace_view_sha256:
            raise ValueError("result authorized workspace view hash is stale")
        if (
            self.budget_usage.model_call_budget
            != self.workspace_view.budget.model_call_budget
            or self.budget_usage.tool_call_budget
            != self.workspace_view.budget.tool_budget
        ):
            raise ValueError("result budget usage does not match its admitted workspace budget")
        if self.aion.run_id != self.run_id or self.aion.stage is not ResearchStage.STOP:
            raise ValueError("finished replay must expose the matching Aion STOP summary")
        if self.aion.steps_used != 6 or self.aion.tool_calls_used != 1:
            raise ValueError("finished replay must use the fixed six-step/one-tool path")
        if tuple(trace.family for trace in self.traces) != _SUCCESS_TRACE_ORDER:
            raise ValueError("finished replay has an invalid family trace order")
        if self.execution_manifest.action_sha256 != tool_invocation_sha256(self.action):
            raise ValueError("result action does not match its execution manifest")
        if self.execution_manifest.sha256 != self.execution_manifest_sha256:
            raise ValueError("result execution manifest hash is stale")
        if self.execution_manifest.shared_tool not in self.workspace_view.tools:
            raise ValueError("result selected tool is absent from its authorized view")
        if self.proposed_claim.status is not ClaimStatus.PROPOSED:
            raise ValueError("result Prometheus claim must remain proposed")
        expected_success = self.execution.state is TransactionState.COMMITTED
        if self.execution_succeeded is not expected_success:
            raise ValueError("execution success does not match its summary")
        if not self.execution_succeeded and self.answer.kind != "abstain":
            raise ValueError("failed execution requires a Hermes abstention")
        if not self.execution_succeeded and self.answer.producer == "hermes" and (
            self.answer.reason
            != f"upstream_execution_failed:{self.execution.error_code}"
        ):
            raise ValueError("failed execution requires the exact Hermes failure reason")
        if (
            self.execution_succeeded
            and self.kronos_observation.status == "failed"
            and (
                self.answer.kind != "abstain"
                or (
                    self.answer.producer == "hermes"
                    and self.answer.reason != "upstream_component_failed:kronos"
                )
            )
        ):
            raise ValueError("failed Kronos observation requires a Hermes abstention")
        if self.answer.producer == "runtime":
            raise ValueError(
                "finished replay answers cannot use the pre-execution runtime producer"
            )
        if (
            self.transaction.state is not self.execution.state
            or self.transaction.transaction_id != self.execution.transaction_id
            or self.transaction.idempotency_key != self.execution.idempotency_key
            or self.transaction.tool_id != self.action.tool_id
            or self.action.idempotency_key != self.execution.idempotency_key
            or self.transaction.material_effect
            or self.transaction.execution_scope_sha256
            != self.execution_manifest.execution_scope_sha256
        ):
            raise ValueError("transaction, action, and execution summary are inconsistent")
        native_action = ActionEnvelope(
            tool_id=self.action.tool_id,
            schema_version=self.action.schema_version,
            arguments=self.action.arguments,
            preconditions=self.action.preconditions,
            idempotency_key=self.action.idempotency_key,
        )
        expected_transaction_id = "tx_" + scoped_action_sha256(
            native_action, self.execution_manifest.execution_scope_sha256
        )
        if self.transaction.transaction_id != expected_transaction_id:
            raise ValueError("transaction ID is not bound to the approved action and scope")
        approval = self.approval
        if (
            approval.authority is AuthorityKind.MODEL
            or approval.scope != "execute"
            or approval.run_id != self.run_id
            or approval.protocol_sha256 != self.aion.protocol_sha256
            or approval.expected_head_sha256 != self.traces[2].output_sha256
            or approval.subject_sha256 != self.execution_manifest_sha256
        ):
            raise ValueError("result approval is not bound to its execution boundary")
        authorized_ids = {item.evidence_id for item in self.workspace_view.evidence}
        if (
            len(self.hermes_input_evidence_ids)
            != len(set(self.hermes_input_evidence_ids))
            or not set(self.hermes_input_evidence_ids).issubset(authorized_ids)
            or _protocol_workspace_view(
                self.workspace_view, self.hermes_input_evidence_ids
            ).sha256
            != self.hermes_input_view_sha256
        ):
            raise ValueError("Hermes input view is not bound to its disclosed evidence IDs")
        claim_evidence_ids = set(self.proposed_claim.support_evidence_ids) | set(
            self.proposed_claim.refute_evidence_ids
        )
        if not claim_evidence_ids.issubset(authorized_ids):
            raise ValueError("result claim references evidence outside its authorized view")
        if not set(self.answer.citation_evidence_ids).issubset(authorized_ids):
            raise ValueError("result answer cites evidence outside its authorized view")
        if not set(self.answer.citation_evidence_ids).issubset(
            self.hermes_input_evidence_ids
        ):
            raise ValueError("result answer cites evidence outside its Hermes input view")
        if not _answer_is_extractively_supported(self.answer, self.workspace_view):
            raise ValueError("Hermes text is not an extract from every cited evidence item")
        if self.traces[1].output_sha256 != canonical_sha256(self.proposed_claim):
            raise ValueError("result claim does not match its Prometheus trace")
        if self.traces[1].input_sha256 != self.workspace_view_sha256:
            raise ValueError("result Prometheus trace does not match its authorized view")
        if self.traces[3].input_sha256 != tool_invocation_sha256(self.action):
            raise ValueError("result action does not match its Perseus trace")
        if self.traces[3].output_sha256 != self.execution.summary_sha256:
            raise ValueError("execution summary does not match its Perseus trace")
        if self.traces[4].input_sha256 != self.kronos_observation.input_reference_sha256:
            raise ValueError("Kronos input reference does not match its trace")
        if self.traces[4].output_sha256 != canonical_sha256(self.kronos_observation):
            raise ValueError("Kronos observation does not match its trace")
        if self.traces[5].output_sha256 != canonical_sha256(self.answer):
            raise ValueError("public answer does not match its Hermes trace")
        if self.traces[2].input_sha256 != self.aion.protocol_sha256:
            raise ValueError("prepared Aion trace does not match the protocol")
        if self.traces[6].input_sha256 != self.traces[2].output_sha256:
            raise ValueError("terminal Aion trace does not continue from the approval head")
        if self.traces[6].output_sha256 != self.aion.final_head_sha256:
            raise ValueError("Aion summary does not match its terminal trace")
        expected_trace = (
            ("Olympus-Atlas", "authorized_retrieval", "completed"),
            ("Prometheus", "trusted_process_selection", "completed"),
            ("Aion", "preregistered_execution_gate", "prepared"),
            (
                "Perseus",
                "declared_non_material_transaction",
                "completed" if self.execution_succeeded else "failed",
            ),
            (
                "Kronos",
                "post_execution_forecast_from_host_pre_execution_features",
                self.kronos_observation.status,
            ),
            (
                "Hermes",
                "post_execution_grounded_response",
                "failed"
                if self.answer.producer == "runtime_fail_closed"
                else "abstained"
                if self.answer.kind == "abstain"
                else "completed",
            ),
            ("Aion", "record_outcome_and_stop", "completed"),
        )
        observed_trace = tuple(
            (trace.family, trace.operation, trace.status) for trace in self.traces
        )
        if observed_trace != expected_trace:
            raise ValueError("finished replay trace semantics are inconsistent")
        if self.traces[5].input_sha256 != _request_input_sha256(
            self.hermes_input_view_sha256, self.final_request
        ):
            raise ValueError("Hermes trace does not bind its public request")
        if self.traces[0].input_sha256 != _request_input_sha256(
            self.retrieval_input_view_sha256, self.retrieval_query
        ):
            raise ValueError("Atlas trace does not bind its public query")
        return self


PrepareReferenceReplayResult = PendingReferenceReplay | ReferenceReplayHalt


class _KronosDraft(StrictModel):
    status: Literal["completed", "failed"] = "completed"
    failure_code: Literal[
        "component_runtime_failure", "model_call_budget_exhausted"
    ] | None = None
    private_observation_sha256: str = Field(pattern=_SHA256_PATTERN)
    public_input_reference_sha256: str = Field(pattern=_SHA256_PATTERN)
    temporal_event_id: str
    environment_version: str
    model_state_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    forecast_sha256: str = Field(pattern=_SHA256_PATTERN)
    forecast_mean: list[list[float]]
    forecast_log_variance: list[list[float]]
    proposed_action_indices: list[int]


@dataclass(slots=True)
class _PendingRecord:
    public_sha256: str
    workspace: WorkspaceState
    aion_state: AionRunState
    action_output: ActionOutput
    execution_scope_token: str
    model_calls_used: int = 2
    resume_binding_sha256: str | None = None
    transaction_id: str | None = None
    kronos_observation: KronosObservation | None = None
    kronos_private_observation_sha256: str | None = None
    answer: PublicAnswer | None = None


class _ModelCallBudgetExhausted(RuntimeError):
    """Internal control signal; never serialize its text or type."""


@dataclass(slots=True)
class _CompletedRecord:
    workspace: WorkspaceState
    aion_state: AionRunState


class OlympusReferenceReplay:
    """Process-local, two-phase interoperability facade for all six roles."""

    def __init__(
        self,
        *,
        atlas: OlympusAtlasService,
        prometheus: PrometheusSelector,
        perseus: TransactionManager,
        kronos: KronosTemporalPlanner,
        aion: AionController,
        hermes: HermesWorkspaceRuntime,
        actor: str = "olympus-reference-runtime",
        runtime_secret: bytes | None = None,
    ) -> None:
        if not actor or len(actor) > 200:
            raise ValueError("composition actor must contain 1 to 200 characters")
        if runtime_secret is not None and len(runtime_secret) < 32:
            raise ValueError("composition runtime secret must contain at least 32 bytes")
        self.atlas = atlas
        self.prometheus = prometheus
        self.perseus = perseus
        self.kronos = kronos
        self.aion = aion
        self.hermes = hermes
        self.actor = actor
        self._runtime_secret = (
            bytes(runtime_secret) if runtime_secret is not None else secrets.token_bytes(32)
        )
        self._pending: dict[str, _PendingRecord] = {}
        self._completed: dict[str, _CompletedRecord] = {}
        self._lock = threading.RLock()

    def prepare(
        self,
        workspace: WorkspaceState,
        *,
        run_id: str,
        protocol: FrozenProtocol,
        query: str,
        corpus_version: str,
        assessments: Sequence[BranchAssessment],
        action: ToolInvocation,
        executor_profile_sha256: str,
        occurred_at: datetime,
    ) -> PrepareReferenceReplayResult:
        """Retrieve, select, and expose the exact execution for approval."""

        with self._lock:
            if run_id in self._completed or any(
                record.aion_state.run_id == run_id for record in self._pending.values()
            ):
                raise ValueError(f"reference replay is already registered: {run_id}")

        workspace = validate_workspace_snapshot(workspace)
        protocol = FrozenProtocol.model_validate(protocol.model_dump(mode="python"))
        action = ToolInvocation.model_validate(action.model_dump(mode="python"))
        if not query.strip():
            raise ValueError("retrieval query must contain groundable text")
        normalized_assessments = [
            BranchAssessment.model_validate(item.model_dump(mode="python"))
            for item in assessments
        ]
        _validate_sha256(executor_profile_sha256, "executor profile")
        _preflight_composition(
            workspace,
            protocol=protocol,
            assessments=normalized_assessments,
            occurred_at=occurred_at,
        )

        initial_view = authorized_workspace_view(workspace)
        retrieval = self.atlas.retrieve(
            AuthorizedQuery(
                text=query,
                granted_acl=frozenset(workspace.permissions.granted_acl_labels),
            ),
            corpus_version=corpus_version,
        )
        if not isinstance(retrieval, EvidenceSet):
            reason = retrieval.reason
            result_sha256 = canonical_sha256({"reason": reason})
            workspace = _append_family_event(
                workspace,
                family_key="atlas",
                kind="atlas_abstention",
                actor=self.actor,
                occurred_at=occurred_at,
                payload={"reason": reason, "result_sha256": result_sha256},
            )
            trace = FamilyTrace(
                family="Olympus-Atlas",
                operation="authorized_retrieval",
                status="abstained",
                input_sha256=_request_input_sha256(initial_view, query),
                output_sha256=result_sha256,
            )
            return self._halt(
                workspace,
                run_id=run_id,
                protocol=protocol,
                retrieval_query=query,
                retrieval_input_view_sha256=initial_view.sha256,
                reason=f"atlas:{reason}",
                occurred_at=occurred_at,
                traces=[trace],
            )

        retrieved_evidence, source_spans = _atlas_evidence(retrieval, workspace)
        hit_ids = [hit.passage.passage_id for hit in retrieval.hits]
        undeclared_hits = set(hit_ids) - set(protocol.evidence_ids)
        if undeclared_hits:
            raise ValueError(
                "Atlas returned evidence outside the frozen protocol: "
                f"{sorted(undeclared_hits)}"
            )
        atlas_result_sha256 = canonical_sha256(
            {
                "query_sha256": retrieval.query_hash,
                "evidence": [
                    {
                        "evidence_id": hit.passage.passage_id,
                        "content_sha256": hit.passage.passage_hash,
                    }
                    for hit in retrieval.hits
                ],
            }
        )
        workspace = _append_family_event(
            workspace,
            family_key="atlas",
            kind="atlas_retrieval",
            actor=self.actor,
            occurred_at=occurred_at,
            payload={
                "query_sha256": retrieval.query_hash,
                "source_spans": source_spans,
                "result_sha256": atlas_result_sha256,
            },
            evidence=retrieved_evidence,
        )
        atlas_trace = FamilyTrace(
            family="Olympus-Atlas",
            operation="authorized_retrieval",
            status="completed",
            input_sha256=_request_input_sha256(initial_view, query),
            output_sha256=atlas_result_sha256,
        )

        prometheus_view = authorized_workspace_view(workspace)
        selection = self.prometheus.select(normalized_assessments, workspace)
        if selection.decision == "abstained":
            selection_sha256 = canonical_sha256(selection)
            workspace = _append_family_event(
                workspace,
                family_key="prometheus",
                kind="prometheus_abstention",
                actor=self.actor,
                occurred_at=occurred_at,
                payload={
                    "reason": selection.reason,
                    "selection_sha256": selection_sha256,
                },
            )
            prometheus_trace = FamilyTrace(
                family="Prometheus",
                operation="trusted_process_selection",
                status="abstained",
                input_sha256=prometheus_view.sha256,
                output_sha256=selection_sha256,
            )
            return self._halt(
                workspace,
                run_id=run_id,
                protocol=protocol,
                retrieval_query=query,
                retrieval_input_view_sha256=initial_view.sha256,
                reason="prometheus:no_eligible_branch",
                occurred_at=occurred_at,
                traces=[atlas_trace, prometheus_trace],
            )

        selected = _selected_branch(normalized_assessments, selection.branch_id)
        support_ids = sorted(
            {
                evidence_id
                for transition in selected.transitions
                for evidence_id in transition.evidence_ids
            }
        )
        if not set(support_ids).issubset(protocol.evidence_ids):
            raise ValueError("selected branch evidence is outside the frozen protocol")
        claim = Claim(
            claim_id=f"claim:{hashlib.sha256(selected.claim.encode('utf-8')).hexdigest()}",
            text=selected.claim,
            status=ClaimStatus.PROPOSED,
            support_evidence_ids=support_ids,
            confidence=selection.score,
        )
        claim_output = ClaimOutput(
            workspace_sha256=workspace.sha256,
            model_identity_sha256=canonical_sha256(workspace.model_identity),
            claims=[claim],
        )
        validate_envelope_against_workspace(claim_output, workspace)
        selection_record_sha256 = canonical_sha256(selection)
        workspace = _append_family_event(
            workspace,
            family_key="prometheus",
            kind="prometheus_proposal",
            actor=self.actor,
            occurred_at=occurred_at,
            payload={
                "branch_id": selected.branch_id,
                "claim_sha256": canonical_sha256(claim),
                "selection_record_sha256": selection_record_sha256,
            },
            claims=[claim],
        )
        prometheus_trace = FamilyTrace(
            family="Prometheus",
            operation="trusted_process_selection",
            status="completed",
            input_sha256=prometheus_view.sha256,
            output_sha256=canonical_sha256(claim),
        )

        action_output = _bound_action(action, workspace)
        shared_tool, native_tool, native_action, context = self._preflight_action(
            action_output, workspace
        )
        # The manager validates its independent argument/precondition policy
        # without creating a transaction or invoking an executor.
        authorized_tool, normalized_arguments = self.perseus.preflight(
            native_action, context
        )
        if (
            authorized_tool != native_tool
            or canonical_sha256(normalized_arguments)
            != canonical_sha256(native_action.arguments)
        ):
            raise ActionValidationError("Perseus policy changed during action preflight")
        pending_seed = {
            "run_id": run_id,
            "protocol_sha256": protocol.sha256,
            # Only the redacted projection is committed here.  A public
            # pending identifier must not become an oracle over private state.
            "workspace_view_sha256": authorized_workspace_view(workspace).sha256,
            "retrieval_query": query,
            "action_sha256": tool_invocation_sha256(action),
            "executor_profile_sha256": executor_profile_sha256,
        }
        pending_id = "pending_" + self._runtime_digest("pending", pending_seed)[:32]
        execution_scope_sha256 = self._runtime_digest(
            "execution-scope", pending_seed
        )
        execution_scope_token = self._runtime_digest(
            "execution-scope-token", {"scope_sha256": execution_scope_sha256}
        )
        manifest = ExecutionManifest(
            action_sha256=tool_invocation_sha256(action),
            shared_tool=shared_tool,
            perseus_tool=native_tool,
            executor_profile_sha256=executor_profile_sha256,
            execution_scope_sha256=execution_scope_sha256,
        )

        # All model/service and action validation above occurs before Aion is
        # registered, so an ordinary validation failure cannot strand a run.
        aion_state = self.aion.create_run(run_id, protocol)
        aion_state = self.aion.transition(
            aion_state,
            ResearchStage.EVIDENCE_MAP,
            actor=self.actor,
            authority=AuthorityKind.SYSTEM,
            evidence_ids=hit_ids,
            metadata={"authorized_view_sha256": authorized_workspace_view(workspace).sha256},
        )
        assessment_evidence_ids = _assessment_evidence_ids(normalized_assessments)
        aion_state = self.aion.transition(
            aion_state,
            ResearchStage.HYPOTHESES,
            actor=self.actor,
            authority=AuthorityKind.SYSTEM,
            evidence_ids=assessment_evidence_ids,
            metadata={"selection_record_sha256": selection_record_sha256},
        )
        aion_state = self.aion.transition(
            aion_state,
            ResearchStage.PREREGISTERED_PROTOCOL,
            actor=self.actor,
            authority=AuthorityKind.SYSTEM,
            evidence_ids=support_ids,
            metadata={
                "claim_sha256": canonical_sha256(claim),
                "execution_manifest_sha256": manifest.sha256,
            },
        )
        aion_head = aion_state.receipts[-1].receipt_sha256
        workspace = _append_family_event(
            workspace,
            family_key="aion",
            kind="aion_preregistered",
            actor=self.actor,
            occurred_at=occurred_at,
            payload={
                "run_id": run_id,
                "protocol_sha256": protocol.sha256,
                "aion_head_sha256": aion_head,
                "execution_manifest_sha256": manifest.sha256,
            },
        )
        # Rebind after the Aion append; the private envelope is the object later
        # passed through the shared validator at resume.
        action_output = _bound_action(action, workspace)
        aion_trace = FamilyTrace(
            family="Aion",
            operation="preregistered_execution_gate",
            status="prepared",
            input_sha256=protocol.sha256,
            output_sha256=aion_head,
        )
        requirement = ExecutionApprovalRequirement(
            run_id=run_id,
            protocol_sha256=protocol.sha256,
            expected_head_sha256=aion_head,
            action_sha256=tool_invocation_sha256(action),
            execution_manifest_sha256=manifest.sha256,
            execution_scope_sha256=execution_scope_sha256,
        )
        view = authorized_workspace_view(workspace)
        pending = PendingReferenceReplay(
            pending_id=pending_id,
            run_id=run_id,
            retrieval_query=query,
            retrieval_input_view_sha256=initial_view.sha256,
            workspace_view=view,
            workspace_view_sha256=view.sha256,
            proposed_claim=claim,
            action=action,
            execution_manifest=manifest,
            execution_manifest_sha256=manifest.sha256,
            approval_requirement=requirement,
            traces=[atlas_trace, prometheus_trace, aion_trace],
        )
        # Reserve the now-public execution scope before returning it.  The
        # secret capability stays only in trusted runtime state, so a caller
        # sharing the manager cannot pre-create or commit this transaction.
        self.perseus.reserve_execution_scope(
            execution_scope_sha256, execution_scope_token
        )
        try:
            with self._lock:
                if run_id in self._completed or any(
                    record.aion_state.run_id == run_id
                    for record in self._pending.values()
                ):
                    raise ValueError(
                        f"reference replay is already registered: {run_id}"
                    )
                self._pending[pending.pending_id] = _PendingRecord(
                    public_sha256=canonical_sha256(pending),
                    workspace=workspace.model_copy(deep=True),
                    aion_state=aion_state.model_copy(deep=True),
                    action_output=action_output.model_copy(deep=True),
                    execution_scope_token=execution_scope_token,
                )
        except Exception:
            self.perseus.release_execution_scope(
                execution_scope_sha256, execution_scope_token
            )
            raise
        return pending.model_copy(deep=True)

    def resume(
        self,
        pending: PendingReferenceReplay,
        *,
        execution_approval: Approval,
        executor: ProfiledSandboxExecutor,
        temporal_observation: TemporalObservationFeatures,
        final_request: str,
    ) -> ReferenceReplayResult:
        """Execute the approved action and resumably close the Aion path at STOP.

        Recovery is process-local: if an exception occurs after the Aion gate,
        the manager-owned transaction, cached model outputs, and current Aion
        head are reused on the next call with byte-identical resume inputs. No
        claim of crash-safe durability is made.
        """

        pending = PendingReferenceReplay.model_validate(pending.model_dump(mode="python"))
        execution_approval = Approval.model_validate(
            execution_approval.model_dump(mode="python")
        )
        temporal_observation = TemporalObservationFeatures.model_validate(
            temporal_observation.model_dump(mode="python")
        )
        if not final_request.strip():
            raise ValueError("final_request must contain groundable text")
        if len(temporal_observation.features) != self.kronos.config.feature_dim:
            raise ValueError("temporal feature width does not match Kronos configuration")

        with self._lock:
            record = self._pending.get(pending.pending_id)
            if record is None or record.public_sha256 != canonical_sha256(pending):
                raise ValueError("pending replay is stale, mutated, consumed, or foreign")
            workspace = validate_workspace_snapshot(record.workspace)
            aion_state = AionRunState.model_validate(
                record.aion_state.model_dump(mode="python")
            )
            action_output = ActionOutput.model_validate(
                record.action_output.model_dump(mode="python")
            )
            validate_envelope_against_workspace(action_output, workspace)
            if (
                workspace.events
                and temporal_observation.occurred_at
                < workspace.events[-1].occurred_at
            ):
                raise ValueError("resume observation precedes the pending workspace")

            executor_profile = getattr(executor, "execution_profile_sha256", None)
            if executor_profile != pending.execution_manifest.executor_profile_sha256:
                raise PermissionError("executor profile does not match the approved manifest")
            _validate_sha256(cast(str, executor_profile), "executor profile")
            requirement = pending.approval_requirement
            if not execution_approval.valid_for(
                requirement.scope,
                requirement.run_id,
                requirement.protocol_sha256,
                requirement.expected_head_sha256,
                temporal_observation.occurred_at,
                requirement.execution_manifest_sha256,
            ):
                raise PermissionError("execution approval does not satisfy the pending requirement")

            resume_binding_sha256 = canonical_sha256(
                {
                    "approval": execution_approval,
                    "temporal_observation": temporal_observation,
                    "final_request": final_request,
                    "executor_profile_sha256": executor_profile,
                }
            )
            if (
                record.resume_binding_sha256 is not None
                and record.resume_binding_sha256 != resume_binding_sha256
            ):
                raise PermissionError(
                    "post-gate recovery requires the original approved resume inputs"
                )

            if aion_state.stage is ResearchStage.PREREGISTERED_PROTOCOL:
                # Reject structurally valid but host-unregistered authority
                # before a transaction or any additional model call is made.
                self.aion.preflight_approval(
                    aion_state,
                    ResearchStage.AUTHORIZED_EXECUTION,
                    approval=execution_approval,
                    approval_subject_sha256=requirement.execution_manifest_sha256,
                )
            elif aion_state.stage not in {
                ResearchStage.AUTHORIZED_EXECUTION,
                ResearchStage.AUDIT,
                ResearchStage.STOP,
            }:
                raise RuntimeError("pending replay has an unrecoverable Aion stage")
            elif not self.aion.verify_chain(aion_state):
                raise RuntimeError("pending replay has an invalid recovery chain")

            shared_tool, native_tool, native_action, context = self._preflight_action(
                action_output, workspace
            )
            expected_manifest = ExecutionManifest(
                action_sha256=tool_invocation_sha256(action_output.invocation),
                shared_tool=shared_tool,
                perseus_tool=native_tool,
                executor_profile_sha256=cast(str, executor_profile),
                execution_scope_sha256=(
                    pending.execution_manifest.execution_scope_sha256
                ),
            )
            if expected_manifest != pending.execution_manifest:
                raise PermissionError("current execution policy differs from the approved manifest")
            authorized_tool, normalized_arguments = self.perseus.preflight(
                native_action, context
            )
            if (
                authorized_tool != native_tool
                or canonical_sha256(normalized_arguments)
                != canonical_sha256(native_action.arguments)
            ):
                raise ActionValidationError("Perseus policy changed after approval")

            if record.resume_binding_sha256 is None:
                record.resume_binding_sha256 = resume_binding_sha256

            if aion_state.stage is ResearchStage.PREREGISTERED_PROTOCOL:
                aion_state = self.aion.transition(
                    aion_state,
                    ResearchStage.AUTHORIZED_EXECUTION,
                    actor=execution_approval.actor,
                    authority=execution_approval.authority,
                    approval=execution_approval,
                    approval_subject_sha256=requirement.execution_manifest_sha256,
                    metadata={
                        "action_sha256": requirement.action_sha256,
                        "execution_manifest_sha256": (
                            requirement.execution_manifest_sha256
                        ),
                        "execution_scope_sha256": (
                            requirement.execution_scope_sha256
                        ),
                    },
                    tool_calls=1,
                )
                record.aion_state = aion_state.model_copy(deep=True)

            if record.transaction_id is None:
                transaction = self.perseus.prepare(
                    native_action,
                    context,
                    execution_scope_sha256=requirement.execution_scope_sha256,
                    execution_scope_token=record.execution_scope_token,
                    require_new=True,
                )
                if transaction.state is not TransactionState.PREPARED:
                    raise RuntimeError("new scoped transaction was not prepared")
                record.transaction_id = transaction.transaction_id
            else:
                transaction = self.perseus.snapshot(record.transaction_id)
            if (
                transaction.tool_id != native_action.tool_id
                or transaction.idempotency_key != native_action.idempotency_key
                or transaction.execution_scope_sha256
                != requirement.execution_scope_sha256
            ):
                raise RuntimeError("recovery transaction does not match the approved scope")

            prior_receipt = self.perseus.receipt(
                transaction.transaction_id,
                execution_scope_token=record.execution_scope_token,
            )
            if prior_receipt is None:
                receipt = self.perseus.commit(
                    transaction.transaction_id,
                    executor,
                    execution_scope_token=record.execution_scope_token,
                )
            else:
                receipt = prior_receipt.model_copy(update={"replayed": True}, deep=True)
            execution = _execution_summary(receipt)
            terminal_transaction = self.perseus.snapshot(transaction.transaction_id)
            if terminal_transaction.state is not receipt.state:
                raise RuntimeError("Perseus terminal snapshot disagrees with its receipt")
            workspace = _append_family_event(
                workspace,
                family_key="perseus",
                kind="perseus_execution",
                actor=execution_approval.actor,
                occurred_at=temporal_observation.occurred_at,
                payload={
                    "transaction_id": receipt.transaction_id,
                    "state": receipt.state.value,
                    "execution_summary_sha256": execution.summary_sha256,
                    "recovery_class": (
                        int(receipt.recovery_class)
                        if receipt.recovery_class is not None
                        else None
                    ),
                    "error_code": receipt.error,
                    "raw_output_recorded": False,
                },
            )
            perseus_trace = FamilyTrace(
                family="Perseus",
                operation="declared_non_material_transaction",
                status=(
                    "completed"
                    if receipt.state is TransactionState.COMMITTED
                    else "failed"
                ),
                input_sha256=tool_invocation_sha256(action_output.invocation),
                output_sha256=execution.summary_sha256,
            )

            if record.kronos_observation is None:
                try:
                    _reserve_model_call(record)
                except _ModelCallBudgetExhausted:
                    kronos_draft = self._failed_forecast_draft(
                        temporal_observation, "model_call_budget_exhausted"
                    )
                    validated_kronos = _public_kronos_observation(kronos_draft)
                else:
                    try:
                        kronos_draft = self._forecast_draft(temporal_observation)
                        validated_kronos = _public_kronos_observation(kronos_draft)
                    except Exception:
                        # Component-controlled exception text/type is neither
                        # public provenance nor a reason to strand a gated run.
                        kronos_draft = self._failed_forecast_draft(
                            temporal_observation, "component_runtime_failure"
                        )
                        validated_kronos = _public_kronos_observation(kronos_draft)
                # Validate the complete public object before caching it.  A
                # malformed component return cannot poison every later retry.
                record.kronos_observation = validated_kronos
                record.kronos_private_observation_sha256 = (
                    kronos_draft.private_observation_sha256
                )
            kronos_observation = record.kronos_observation.model_copy(deep=True)
            private_observation_sha256 = record.kronos_private_observation_sha256
            if private_observation_sha256 is None:
                raise RuntimeError("trusted Kronos observation binding is missing")
            workspace = _append_family_event(
                workspace,
                family_key="kronos",
                kind="kronos_host_observation",
                actor=self.actor,
                occurred_at=temporal_observation.occurred_at,
                payload={
                    "private_observation_sha256": private_observation_sha256,
                    "forecast_sha256": kronos_observation.forecast_sha256,
                    "model_state_sha256": kronos_observation.model_state_sha256,
                    "status": kronos_observation.status,
                    "failure_code": kronos_observation.failure_code,
                    "source_authenticated": False,
                    "authoritative": False,
                    "used_for_action_selection": False,
                },
            )
            kronos_trace = FamilyTrace(
                family="Kronos",
                operation=(
                    "post_execution_forecast_from_host_pre_execution_features"
                ),
                status=kronos_observation.status,
                input_sha256=kronos_observation.input_reference_sha256,
                output_sha256=canonical_sha256(kronos_observation),
            )

            hermes_view = _protocol_workspace_view(
                authorized_workspace_view(workspace), aion_state.protocol.evidence_ids
            )
            if record.answer is None:
                try:
                    _reserve_model_call(record)
                except _ModelCallBudgetExhausted:
                    answer = PublicAnswer(
                        kind="abstain",
                        producer="runtime_fail_closed",
                        reason="hermes:model_call_budget_exhausted",
                    )
                else:
                    try:
                        answer_envelope = self.hermes.respond_after_execution(
                            workspace,
                            final_request,
                            execution_succeeded=(
                                receipt.state is TransactionState.COMMITTED
                            ),
                            failure_code=execution.error_code,
                            upstream_component_failure=(
                                "kronos"
                                if kronos_observation.status == "failed"
                                else None
                            ),
                            allowed_evidence_ids=aion_state.protocol.evidence_ids,
                        )
                        answer = _public_answer(answer_envelope)
                        authorized_ids = {
                            item.evidence_id
                            for item in authorized_workspace_view(workspace).evidence
                        }
                        if not set(answer.citation_evidence_ids).issubset(
                            authorized_ids
                        ):
                            raise RuntimeError(
                                "Hermes returned a citation outside the authorized view"
                            )
                        if not set(answer.citation_evidence_ids).issubset(
                            aion_state.protocol.evidence_ids
                        ):
                            raise RuntimeError(
                                "Hermes returned a citation outside the frozen protocol"
                            )
                        if not _answer_is_extractively_supported(answer, hermes_view):
                            raise RuntimeError(
                                "Hermes returned text unsupported by its cited evidence"
                            )
                    except Exception:
                        answer = PublicAnswer(
                            kind="abstain",
                            producer="runtime_fail_closed",
                            reason="hermes:component_runtime_failure",
                        )
                record.answer = answer.model_copy(deep=True)
            answer = record.answer.model_copy(deep=True)
            audit_evidence = list(answer.citation_evidence_ids)
            public_answer_sha256 = canonical_sha256(answer)
            workspace = _append_family_event(
                workspace,
                family_key="hermes",
                kind="hermes_response",
                actor=self.actor,
                occurred_at=temporal_observation.occurred_at,
                payload={
                    "answer_kind": answer.kind,
                    "public_answer_sha256": public_answer_sha256,
                    "citation_evidence_ids": answer.citation_evidence_ids,
                },
            )
            hermes_trace = FamilyTrace(
                family="Hermes",
                operation="post_execution_grounded_response",
                status=(
                    "failed"
                    if answer.producer == "runtime_fail_closed"
                    else "abstained"
                    if answer.kind == "abstain"
                    else "completed"
                ),
                input_sha256=_request_input_sha256(hermes_view, final_request),
                output_sha256=public_answer_sha256,
            )

            if aion_state.stage is ResearchStage.AUTHORIZED_EXECUTION:
                aion_state = self.aion.transition(
                    aion_state,
                    ResearchStage.AUDIT,
                    actor=self.actor,
                    authority=AuthorityKind.SYSTEM,
                    evidence_ids=audit_evidence,
                    metadata={
                        "transaction_id": transaction.transaction_id,
                        "execution_summary_sha256": execution.summary_sha256,
                        "kronos_observation_sha256": canonical_sha256(
                            kronos_observation
                        ),
                        "public_answer_sha256": public_answer_sha256,
                        "promotion_attempted": False,
                    },
                )
                record.aion_state = aion_state.model_copy(deep=True)
            if aion_state.stage is ResearchStage.AUDIT:
                aion_state = self.aion.transition(
                    aion_state,
                    ResearchStage.STOP,
                    actor=self.actor,
                    authority=AuthorityKind.SYSTEM,
                    metadata={"reason": "reference_replay_complete_without_promotion"},
                )
                record.aion_state = aion_state.model_copy(deep=True)
            if aion_state.stage is not ResearchStage.STOP:
                raise RuntimeError("reference replay did not reach its terminal Aion stage")
            final_head = aion_state.receipts[-1].receipt_sha256
            workspace = _append_family_event(
                workspace,
                family_key="aion",
                kind="aion_stopped",
                actor=self.actor,
                occurred_at=temporal_observation.occurred_at,
                payload={
                    "run_id": pending.run_id,
                    "aion_head_sha256": final_head,
                    "stage": ResearchStage.STOP.value,
                    "promotion_attempted": False,
                },
            )
            if not self.aion.verify_chain(aion_state):
                raise RuntimeError("Aion rejected its completed reference replay chain")
            aion_trace = FamilyTrace(
                family="Aion",
                operation="record_outcome_and_stop",
                status="completed",
                input_sha256=requirement.expected_head_sha256,
                output_sha256=final_head,
            )
            view = authorized_workspace_view(workspace)
            result = ReferenceReplayResult(
                run_id=pending.run_id,
                retrieval_query=pending.retrieval_query,
                retrieval_input_view_sha256=(
                    pending.retrieval_input_view_sha256
                ),
                final_request=final_request,
                hermes_input_view_sha256=hermes_view.sha256,
                hermes_input_evidence_ids=[
                    item.evidence_id for item in hermes_view.evidence
                ],
                workspace_view=view,
                workspace_view_sha256=view.sha256,
                aion=_aion_summary(aion_state),
                proposed_claim=pending.proposed_claim,
                action=pending.action,
                execution_manifest=pending.execution_manifest,
                execution_manifest_sha256=pending.execution_manifest_sha256,
                approval=execution_approval,
                transaction=terminal_transaction,
                execution=execution,
                kronos_observation=kronos_observation,
                answer=answer,
                budget_usage=CompositionBudgetUsage(
                    model_call_budget=record.workspace.budget.model_call_budget,
                    model_calls_used=record.model_calls_used,
                    tool_call_budget=record.workspace.budget.tool_budget,
                ),
                traces=[
                    *pending.traces,
                    perseus_trace,
                    kronos_trace,
                    hermes_trace,
                    aion_trace,
                ],
                execution_succeeded=receipt.state is TransactionState.COMMITTED,
            )
            self._completed[pending.run_id] = _CompletedRecord(
                workspace=workspace.model_copy(deep=True),
                aion_state=aion_state.model_copy(deep=True),
            )
            del self._pending[pending.pending_id]
            return result.model_copy(deep=True)

    def trusted_workspace(self, run_id: str) -> WorkspaceState:
        """Return a defensive full snapshot to trusted in-process operators only."""

        with self._lock:
            record = self._completed.get(run_id)
            if record is None:
                raise KeyError(f"completed reference replay not found: {run_id}")
            return validate_workspace_snapshot(record.workspace)

    def trusted_aion_state(self, run_id: str) -> AionRunState:
        """Return a defensive full Aion record to trusted in-process operators only."""

        with self._lock:
            record = self._completed.get(run_id)
            if record is None:
                raise KeyError(f"completed reference replay not found: {run_id}")
            return AionRunState.model_validate(record.aion_state.model_dump(mode="python"))

    def _runtime_digest(self, domain: str, payload: object) -> str:
        """Create a domain-separated, runtime-private deterministic capability."""

        message = domain.encode("utf-8") + b"\0" + canonical_json_bytes(payload)
        return hmac.new(self._runtime_secret, message, hashlib.sha256).hexdigest()

    def _preflight_action(
        self, action: ActionOutput, workspace: WorkspaceState
    ) -> tuple[ToolSpec, ToolDefinition, ActionEnvelope, ExecutionContext]:
        validate_envelope_against_workspace(action, workspace)
        invocation = action.invocation
        shared_tool = next(
            (
                tool
                for tool in workspace.tools
                if tool.tool_id == invocation.tool_id
                and tool.schema_version == invocation.schema_version
            ),
            None,
        )
        if shared_tool is None:
            raise ActionValidationError("action tool disappeared from the workspace")
        native_tool = self.perseus.describe_tool(
            invocation.tool_id, invocation.schema_version
        )
        manifest_probe = ExecutionManifest(
            action_sha256=tool_invocation_sha256(invocation),
            shared_tool=shared_tool,
            perseus_tool=native_tool,
            executor_profile_sha256="0" * 64,
            execution_scope_sha256="0" * 64,
        )
        del manifest_probe
        native_action = ActionEnvelope(
            tool_id=invocation.tool_id,
            schema_version=invocation.schema_version,
            arguments=invocation.arguments,
            preconditions=invocation.preconditions,
            idempotency_key=invocation.idempotency_key,
        )
        if action_sha256(native_action) != tool_invocation_sha256(invocation):
            raise ActionValidationError("shared-to-Perseus action translation changed intent")
        context = _execution_context(workspace, invocation)
        return (
            shared_tool.model_copy(deep=True),
            native_tool.model_copy(deep=True),
            native_action,
            context,
        )

    def _forecast_draft(self, observation: TemporalObservationFeatures) -> _KronosDraft:
        if len(observation.features) != self.kronos.config.feature_dim:
            raise ValueError("temporal feature width does not match Kronos configuration")
        private_observation_sha256 = canonical_sha256(observation)
        public_reference = canonical_sha256(
            {
                "event_id": observation.event_id,
                "environment_version": observation.environment_version,
                "source_kind": "host_supplied_pre_execution_features",
            }
        )
        event = TemporalEvent(
            event_id=observation.event_id,
            occurred_at=observation.occurred_at,
            kind=EventKind.OBSERVATION,
            features=observation.features,
            environment_version=observation.environment_version,
            source_sha256=private_observation_sha256,
            outcome_observed=False,
        )
        features, deltas, mask, _ = tensorize_event_stream(
            [[event]], feature_dim=self.kronos.config.feature_dim
        )
        was_training = self.kronos.training
        try:
            self.kronos.eval()
            with torch.no_grad():
                output = self.kronos(features, deltas, mask)
        finally:
            self.kronos.train(was_training)
        forecast_mean = cast(list[list[float]], output.forecast_mean[0].cpu().tolist())
        forecast_log_variance = cast(
            list[list[float]], output.forecast_log_variance[0].cpu().tolist()
        )
        proposed_actions = cast(
            list[int], output.action_logits[0].argmax(dim=-1).cpu().tolist()
        )
        forecast_payload = {
            "mean": forecast_mean,
            "log_variance": forecast_log_variance,
            "proposed_action_indices": proposed_actions,
        }
        ensure_finite_json(forecast_payload, label="Kronos forecast")
        return _KronosDraft(
            private_observation_sha256=private_observation_sha256,
            public_input_reference_sha256=public_reference,
            temporal_event_id=observation.event_id,
            environment_version=observation.environment_version,
            model_state_sha256=_torch_model_sha256(self.kronos),
            forecast_sha256=canonical_sha256(forecast_payload),
            forecast_mean=forecast_mean,
            forecast_log_variance=forecast_log_variance,
            proposed_action_indices=proposed_actions,
        )

    def _failed_forecast_draft(
        self,
        observation: TemporalObservationFeatures,
        failure_code: Literal[
            "component_runtime_failure", "model_call_budget_exhausted"
        ],
    ) -> _KronosDraft:
        """Create a closed, detail-free terminal record for a failed Kronos call."""

        public_reference = canonical_sha256(
            {
                "event_id": observation.event_id,
                "environment_version": observation.environment_version,
                "source_kind": "host_supplied_pre_execution_features",
            }
        )
        return _KronosDraft(
            status="failed",
            failure_code=failure_code,
            private_observation_sha256=canonical_sha256(observation),
            public_input_reference_sha256=public_reference,
            temporal_event_id=observation.event_id,
            environment_version=observation.environment_version,
            model_state_sha256=None,
            forecast_sha256=canonical_sha256({"failure_code": failure_code}),
            forecast_mean=[],
            forecast_log_variance=[],
            proposed_action_indices=[],
        )

    def _halt(
        self,
        workspace: WorkspaceState,
        *,
        run_id: str,
        protocol: FrozenProtocol,
        retrieval_query: str,
        retrieval_input_view_sha256: str,
        reason: str,
        occurred_at: datetime,
        traces: list[FamilyTrace],
    ) -> ReferenceReplayHalt:
        state = self.aion.create_run(run_id, protocol)
        state = self.aion.transition(
            state,
            ResearchStage.STOP,
            actor=self.actor,
            authority=AuthorityKind.SYSTEM,
            metadata={"reason": reason},
        )
        head = state.receipts[-1].receipt_sha256
        workspace = _append_family_event(
            workspace,
            family_key="aion",
            kind="aion_halted",
            actor=self.actor,
            occurred_at=occurred_at,
            payload={"run_id": run_id, "aion_head_sha256": head, "reason": reason},
        )
        trace = FamilyTrace(
            family="Aion",
            operation="fail_closed_stop",
            status="abstained",
            input_sha256=protocol.sha256,
            output_sha256=head,
        )
        abstention_envelope = AbstainOutput(
            workspace_sha256=workspace.sha256,
            model_identity_sha256=canonical_sha256(workspace.model_identity),
            reason=reason,
            unresolved=[workspace.objective],
        )
        validate_envelope_against_workspace(abstention_envelope, workspace)
        view = authorized_workspace_view(workspace)
        result = ReferenceReplayHalt(
            run_id=run_id,
            retrieval_query=retrieval_query,
            retrieval_input_view_sha256=retrieval_input_view_sha256,
            reason=reason,
            workspace_view=view,
            workspace_view_sha256=view.sha256,
            aion=_aion_summary(state),
            abstention=_public_answer(abstention_envelope, producer="runtime"),
            traces=[*traces, trace],
        )
        with self._lock:
            self._completed[run_id] = _CompletedRecord(
                workspace=workspace.model_copy(deep=True),
                aion_state=state.model_copy(deep=True),
            )
        return result.model_copy(deep=True)


def _preflight_composition(
    workspace: WorkspaceState,
    *,
    protocol: FrozenProtocol,
    assessments: Sequence[BranchAssessment],
    occurred_at: datetime,
) -> None:
    if occurred_at.utcoffset() is None:
        raise ValueError("composition timestamps must be timezone-aware")
    if workspace.events and occurred_at < workspace.events[-1].occurred_at:
        raise ValueError("composition timestamp precedes the active workspace")
    if protocol.question != workspace.objective:
        raise ValueError("frozen protocol question must equal the workspace objective")
    if protocol.maximum_steps < 6:
        raise ValueError("reference replay requires a six-step Aion protocol budget")
    if protocol.maximum_tool_calls < 1:
        raise ValueError("reference replay requires one Aion tool-call budget")
    if workspace.budget.model_call_budget < 4:
        raise ValueError("reference replay requires a workspace model-call budget of four")
    if workspace.budget.tool_budget < 1:
        raise ValueError("reference replay requires a workspace tool budget of one")
    if len(assessments) > workspace.budget.branch_budget:
        raise ValueError("reference replay assessments exceed the workspace branch budget")
    branch_ids = [item.branch.branch_id for item in assessments]
    if len(branch_ids) != len(set(branch_ids)):
        raise ValueError("Prometheus assessment branch IDs must be unique")
    if not set(branch_ids).issubset(protocol.hypothesis_ids):
        raise ValueError("assessed branches must be declared by the frozen protocol")
    assessment_evidence = set(_assessment_evidence_ids(assessments))
    if not assessment_evidence.issubset(protocol.evidence_ids):
        raise ValueError("assessment evidence must be declared by the frozen protocol")


def _reserve_model_call(record: _PendingRecord) -> None:
    """Account for an attempted family/model call before invoking it."""

    if record.model_calls_used >= record.workspace.budget.model_call_budget:
        raise _ModelCallBudgetExhausted
    record.model_calls_used += 1


def _request_input_sha256(
    view: AuthorizedWorkspaceView | str, request: str
) -> str:
    """Bind a public request to the exact authorized-view digest."""

    view_sha256 = view if isinstance(view, str) else view.sha256
    _validate_sha256(view_sha256, "authorized workspace view")
    return canonical_sha256(
        {"workspace_view_sha256": view_sha256, "request": request}
    )


def _protocol_workspace_view(
    view: AuthorizedWorkspaceView, evidence_ids: Sequence[str]
) -> AuthorizedWorkspaceView:
    """Restrict an already-authorized view to preregistered evidence."""

    allowed = set(evidence_ids)
    return AuthorizedWorkspaceView.model_validate(
        {
            **view.model_dump(mode="python"),
            "evidence": [
                item.model_copy(deep=True)
                for item in view.evidence
                if item.evidence_id in allowed
            ],
        }
    )


def _answer_is_extractively_supported(
    answer: PublicAnswer, view: AuthorizedWorkspaceView
) -> bool:
    """Check the concrete extractive contract implemented by Hermes today."""

    if answer.kind != "text" or answer.text is None:
        return True
    evidence_by_id = {item.evidence_id: item for item in view.evidence}
    cited = [evidence_by_id.get(item_id) for item_id in answer.citation_evidence_ids]
    return bool(cited) and all(
        item is not None and item.text is not None and answer.text in item.text
        for item in cited
    )


def _execution_context(
    workspace: WorkspaceState, invocation: ToolInvocation
) -> ExecutionContext:
    completed_postconditions = {
        postcondition
        for step in workspace.plan
        if step.status.value == "completed"
        for postcondition in step.postconditions
    }
    action_digest = tool_invocation_sha256(invocation)
    return ExecutionContext(
        capabilities=set(workspace.permissions.capabilities),
        satisfied_preconditions=completed_postconditions,
        approval_ids={
            approval_id
            for approval_id, approved in workspace.permissions.approval_bindings.items()
            if approved == action_digest
        },
    )


def _bound_action(invocation: ToolInvocation, workspace: WorkspaceState) -> ActionOutput:
    action = ActionOutput(
        workspace_sha256=workspace.sha256,
        model_identity_sha256=canonical_sha256(workspace.model_identity),
        invocation=invocation.model_copy(deep=True),
    )
    validate_envelope_against_workspace(action, workspace)
    return action


def _atlas_evidence(
    result: EvidenceSet, workspace: WorkspaceState
) -> tuple[list[EvidenceItem], list[dict[str, object]]]:
    existing = {item.evidence_id: item for item in workspace.evidence}
    additions: list[EvidenceItem] = []
    source_spans: list[dict[str, object]] = []
    for hit in result.hits:
        passage = hit.passage
        acquired_at = _parse_timestamp(passage.acquired_at)
        item = EvidenceItem(
            evidence_id=passage.passage_id,
            source_uri=passage.source_uri,
            content_sha256=passage.passage_hash,
            acquired_at=acquired_at,
            license_id=passage.license_id,
            acl_labels=set(passage.acl_labels),
            span_start=0,
            span_end=len(passage.text),
            text=passage.text,
        )
        prior = existing.get(item.evidence_id)
        if prior is not None and prior != item:
            raise ValueError(
                f"Atlas evidence ID collides with workspace evidence: {item.evidence_id}"
            )
        if prior is None:
            additions.append(item)
        source_spans.append(
            {
                "evidence_id": item.evidence_id,
                "document_sha256": passage.document_hash,
                "source_span_start": passage.span_start,
                "source_span_end": passage.span_end,
            }
        )
    return additions, source_spans


def _parse_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("Atlas acquired_at must be an ISO-8601 timestamp") from error
    if timestamp.utcoffset() is None:
        raise ValueError("Atlas acquired_at must be timezone-aware")
    return timestamp


def _assessment_evidence_ids(assessments: Sequence[BranchAssessment]) -> list[str]:
    return sorted(
        {
            evidence_id
            for assessment in assessments
            for transition in assessment.branch.transitions
            for evidence_id in transition.evidence_ids
        }
    )


def _selected_branch(
    assessments: Sequence[BranchAssessment], branch_id: str | None
) -> HypothesisBranch:
    matches = [item.branch for item in assessments if item.branch.branch_id == branch_id]
    if len(matches) != 1:
        raise RuntimeError("Prometheus selected a missing or ambiguous branch")
    return matches[0]


def _append_family_event(
    workspace: WorkspaceState,
    *,
    family_key: str,
    kind: str,
    actor: str,
    occurred_at: datetime,
    payload: dict[str, object],
    evidence: Sequence[EvidenceItem] = (),
    claims: Sequence[Claim] = (),
) -> WorkspaceState:
    event_identity = {
        "workspace_sha256": workspace.sha256,
        "sequence": workspace.event_sequence + 1,
        "kind": kind,
        "payload": payload,
    }
    event_id = (
        f"{family_key}:{workspace.event_sequence + 1}:"
        f"{canonical_sha256(event_identity)[:16]}"
    )
    return append_workspace_event(
        workspace,
        event_id=event_id,
        kind=kind,
        actor=actor,
        occurred_at=occurred_at,
        payload=dict(payload),
        evidence=evidence,
        claims=claims,
    )


def _execution_public_payload(summary: ExecutionSummary) -> dict[str, object]:
    return {
        "transaction_id": summary.transaction_id,
        "idempotency_key": summary.idempotency_key,
        "state": summary.state.value,
        "replayed": summary.replayed,
        "recovery_class": (
            int(summary.recovery_class) if summary.recovery_class is not None else None
        ),
        "error_code": summary.error_code,
        "raw_output_recorded": False,
    }


def _execution_summary(receipt: ExecutionReceipt) -> ExecutionSummary:
    probe = ExecutionSummary.model_construct(
        transaction_id=receipt.transaction_id,
        idempotency_key=receipt.idempotency_key,
        state=receipt.state,
        summary_sha256="0" * 64,
        replayed=receipt.replayed,
        recovery_class=receipt.recovery_class,
        error_code=receipt.error,
        raw_output_recorded=False,
    )
    return ExecutionSummary(
        transaction_id=receipt.transaction_id,
        idempotency_key=receipt.idempotency_key,
        state=receipt.state,
        summary_sha256=canonical_sha256(_execution_public_payload(probe)),
        replayed=receipt.replayed,
        recovery_class=receipt.recovery_class,
        error_code=receipt.error,
        raw_output_recorded=False,
    )


def _public_kronos_observation(draft: _KronosDraft) -> KronosObservation:
    return KronosObservation(
        status=draft.status,
        failure_code=draft.failure_code,
        input_reference_sha256=draft.public_input_reference_sha256,
        temporal_event_id=draft.temporal_event_id,
        environment_version=draft.environment_version,
        model_state_sha256=draft.model_state_sha256,
        forecast_sha256=draft.forecast_sha256,
        forecast_mean=draft.forecast_mean,
        forecast_log_variance=draft.forecast_log_variance,
        proposed_action_indices=draft.proposed_action_indices,
    )


def _public_answer(
    answer: TextOutput | AbstainOutput,
    *,
    producer: Literal["hermes", "runtime", "runtime_fail_closed"] = "hermes",
) -> PublicAnswer:
    if isinstance(answer, TextOutput):
        return PublicAnswer(
            kind="text",
            producer=producer,
            text=answer.text,
            citation_evidence_ids=list(answer.citation_evidence_ids),
            confidence=answer.confidence,
        )
    return PublicAnswer(kind="abstain", producer=producer, reason=answer.reason)


def _aion_summary(state: AionRunState) -> AionSummary:
    head = state.receipts[-1].receipt_sha256 if state.receipts else "0" * 64
    return AionSummary(
        run_id=state.run_id,
        protocol_sha256=state.protocol.sha256,
        stage=state.stage,
        steps_used=state.steps_used,
        tool_calls_used=state.tool_calls_used,
        final_head_sha256=head,
    )


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 value")


def _torch_model_sha256(model: KronosTemporalPlanner) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            model.config.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
