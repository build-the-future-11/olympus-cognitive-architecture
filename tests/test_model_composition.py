from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from olympus.core.schemas import ComputeBudget
from olympus.models.aion import (
    AionController,
    Approval,
    AuthorityKind,
    FrozenProtocol,
    ResearchStage,
)
from olympus.models.atlas import AtlasDocument, OlympusAtlasService
from olympus.models.composition import (
    KronosObservation,
    OlympusReferenceReplay,
    PendingReferenceReplay,
    PublicAnswer,
    ReferenceReplayHalt,
    ReferenceReplayResult,
    TemporalObservationFeatures,
)
from olympus.models.hermes import HermesGroundedService, HermesWorkspaceRuntime
from olympus.models.kronos import KronosConfig, KronosTemporalPlanner
from olympus.models.perseus import (
    ActionApproval,
    ActionEnvelope,
    ArgumentKind,
    ArgumentRule,
    AuthorizationError,
    CapabilityKernel,
    CompensatableExecutionError,
    ExecutionContext,
    ToolDefinition,
    TransactionManager,
    TransactionState,
    action_sha256,
)
from olympus.models.prometheus import (
    BranchAssessment,
    HypothesisBranch,
    PrometheusSelector,
    TransitionReceipt,
    transition_from_receipt,
)
from olympus.models.substrate import (
    ClaimStatus,
    EvidenceItem,
    ModelIdentity,
    ModelLifecycle,
    PermissionSet,
    PlanStatus,
    PlanStep,
    ToolInvocation,
    ToolSpec,
    WorkspaceState,
    canonical_sha256,
    tool_invocation_sha256,
    validate_workspace_snapshot,
)

NOW = datetime(2026, 9, 6, 12, tzinfo=UTC)
QUESTION = "What evidence shows the Olympus runtime integration works?"
CLAIM = "The controlled workflow reached its declared pre-execution boundary."
RUN_ID = f"aion_{'c' * 16}"
EXECUTOR_PROFILE = "f" * 64


class RecordingExecutor:
    guarantees_idempotency = True
    execution_profile_sha256 = EXECUTOR_PROFILE

    def __init__(self, *, fail: bool = False, result_marker: str = "hidden") -> None:
        self.fail = fail
        self.result_marker = result_marker
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def execute(
        self, tool_id: str, arguments: Mapping[str, Any], transaction_id: str
    ) -> Mapping[str, Any]:
        copied = dict(arguments)
        self.calls.append((tool_id, copied, transaction_id))
        if self.fail:
            raise CompensatableExecutionError("secret-token-must-not-escape")
        return {
            "sum": copied["left"] + copied["right"],
            "private_result": self.result_marker,
        }


@dataclass(slots=True)
class ReplayFixture:
    runtime: OlympusReferenceReplay
    controller: AionController
    manager: TransactionManager
    workspace: WorkspaceState
    protocol: FrozenProtocol
    assessment: BranchAssessment
    corpus_version: str
    passage_id: str

    def prepare(
        self,
        *,
        action: ToolInvocation | None = None,
        executor_profile_sha256: str = EXECUTOR_PROFILE,
    ) -> PendingReferenceReplay | ReferenceReplayHalt:
        return self.runtime.prepare(
            self.workspace,
            run_id=RUN_ID,
            protocol=self.protocol,
            query=QUESTION,
            corpus_version=self.corpus_version,
            assessments=[self.assessment],
            action=action or _invocation(),
            executor_profile_sha256=executor_profile_sha256,
            occurred_at=NOW,
        )


def _identity() -> ModelIdentity:
    return ModelIdentity(
        model_id="olympus.reference.replay",
        family="Olympus",
        version="0.1.0",
        base_model_id="repository://reference-components",
        base_revision="d77d49d",
        base_sha256="a" * 64,
        lifecycle=ModelLifecycle.EXPERIMENTAL,
    )


def _document(
    identifier: str,
    text: str,
    *,
    acl_labels: tuple[str, ...] = (),
) -> AtlasDocument:
    return AtlasDocument(
        document_id=identifier,
        text=text,
        source_uri=f"memory://{identifier}",
        source_version="1",
        acquired_at="2026-09-06T10:00:00Z",
        license_id="test-fixture-only",
        acl_labels=acl_labels,
    )


def _private_workspace_evidence() -> EvidenceItem:
    text = "PRIVATE-ORCHID must never enter a public replay artifact."
    return EvidenceItem(
        evidence_id="workspace:private",
        source_uri="memory://workspace/private",
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        acquired_at=NOW - timedelta(hours=2),
        license_id="test-fixture-only",
        acl_labels={"secret-workspace"},
        text=text,
    )


def _build_fixture(
    *,
    trust_receipt: bool = True,
    private_only: bool = False,
    grant_capability: bool = True,
) -> ReplayFixture:
    identity = _identity()
    receipt = TransitionReceipt(
        workspace_id="workspace:composition",
        model_identity_sha256=canonical_sha256(identity),
        branch_id="branch-a",
        claim_sha256=hashlib.sha256(CLAIM.encode("utf-8")).hexdigest(),
        operation="verify deterministic integration fixture",
        check="boolean",
        expected=True,
        observed=True,
        issued_by="host-test-fixture",
    )
    receipt_text = receipt.canonical_text()
    receipt_evidence = EvidenceItem(
        evidence_id="receipt:branch-a:0",
        source_uri="memory://trusted-receipt",
        content_sha256=hashlib.sha256(receipt_text.encode("utf-8")).hexdigest(),
        acquired_at=NOW - timedelta(hours=1),
        license_id="test-fixture-only",
        text=receipt_text,
    )
    branch = HypothesisBranch(
        branch_id="branch-a",
        claim=CLAIM,
        transitions=[transition_from_receipt(step=0, evidence=receipt_evidence)],
    )
    assessment = BranchAssessment(
        branch=branch,
        proposer_score=0.9,
        verifier_scores=[0.95],
    )
    tool = ToolSpec(
        tool_id="calculate.sum",
        schema_version="1",
        argument_schema={
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
        required_capabilities={"calculation"},
    )
    workspace = WorkspaceState(
        workspace_id="workspace:composition",
        objective=QUESTION,
        evidence=[receipt_evidence, _private_workspace_evidence()],
        plan=[
            PlanStep(
                step_id="review-inputs",
                description="Host reviews the deterministic fixture inputs.",
                status=PlanStatus.COMPLETED,
                tool_id="calculate.sum",
                postconditions=["inputs-reviewed"],
            )
        ],
        tools=[tool],
        permissions=PermissionSet(
            allowed_tool_ids={tool.tool_id},
            capabilities={"calculation"} if grant_capability else set(),
        ),
        model_identity=identity,
    )
    atlas = OlympusAtlasService(dense_dimensions=16)
    public = _document(
        "public-integration",
        "Evidence shows the Olympus runtime integration works in a controlled synthetic test.",
    )
    private = _document(
        "private-integration",
        "Evidence shows the Olympus runtime integration works with PRIVATE-ZEPHYR telemetry.",
        acl_labels=("secret-flight",),
    )
    corpus_version = atlas.build_index([private] if private_only else [public, private])
    index = atlas.get_index(
        corpus_version,
        granted_acl=frozenset({"secret-flight"}) if private_only else frozenset(),
    )
    assert index is not None
    passage_id = next(
        passage.passage_id
        for passage in index.passages
        if passage.document_id
        == ("private-integration" if private_only else "public-integration")
    )
    protocol = FrozenProtocol(
        protocol_id="composition-protocol",
        version=1,
        question=QUESTION,
        hypothesis_ids=[branch.branch_id],
        evidence_ids=[receipt_evidence.evidence_id, passage_id],
        procedure=["retrieve", "select", "authorize", "execute", "audit", "stop"],
        falsification_criterion=(
            "Any unapproved execution or promoted output falsifies the fixture."
        ),
        maximum_steps=8,
        maximum_tool_calls=1,
        frozen_at=NOW - timedelta(minutes=1),
    )
    definition = ToolDefinition(
        tool_id=tool.tool_id,
        schema_version=tool.schema_version,
        arguments={
            "left": ArgumentRule(kind=ArgumentKind.INTEGER),
            "right": ArgumentRule(kind=ArgumentKind.INTEGER),
        },
        required_capabilities={"calculation"},
        required_preconditions={"inputs-reviewed"},
    )
    manager = TransactionManager(CapabilityKernel([definition], clock=lambda: NOW))
    controller = AionController(clock=lambda: NOW)
    torch.manual_seed(11)
    planner = KronosTemporalPlanner(
        KronosConfig(
            feature_dim=2,
            hidden_dim=8,
            output_dim=1,
            horizons=2,
            plan_steps=2,
            action_count=3,
            attention_heads=2,
        )
    )
    selector = PrometheusSelector(
        minimum_score=0.5,
        minimum_evidence_coverage=1.0,
        trusted_receipt_sha256=(
            {receipt_evidence.content_sha256} if trust_receipt else set()
        ),
    )
    runtime = OlympusReferenceReplay(
        atlas=atlas,
        prometheus=selector,
        perseus=manager,
        kronos=planner,
        aion=controller,
        hermes=HermesWorkspaceRuntime(HermesGroundedService(minimum_support=0.5)),
        runtime_secret=hashlib.sha256(b"composition-test-runtime-secret").digest(),
    )
    return ReplayFixture(
        runtime=runtime,
        controller=controller,
        manager=manager,
        workspace=workspace,
        protocol=protocol,
        assessment=assessment,
        corpus_version=corpus_version,
        passage_id=passage_id,
    )


def _invocation(*, left: int = 2, right: int = 3) -> ToolInvocation:
    return ToolInvocation(
        tool_id="calculate.sum",
        schema_version="1",
        arguments={"left": left, "right": right},
        preconditions=["inputs-reviewed"],
        idempotency_key=f"composition-action-{left}-{right}",
    )


def _approval(
    pending: PendingReferenceReplay,
    *,
    head: str | None = None,
    subject: str | None = None,
) -> Approval:
    requirement = pending.approval_requirement
    return Approval(
        approval_id=f"approval_{'d' * 16}",
        authority=AuthorityKind.HUMAN,
        actor="integration-reviewer",
        scope=requirement.scope,
        run_id=requirement.run_id,
        protocol_sha256=requirement.protocol_sha256,
        expected_head_sha256=head or requirement.expected_head_sha256,
        subject_sha256=(
            requirement.execution_manifest_sha256 if subject is None else subject
        ),
        expires_at=NOW + timedelta(hours=1),
    )


def _observation() -> TemporalObservationFeatures:
    return TemporalObservationFeatures(
        event_id="temporal:composition:1",
        occurred_at=NOW,
        features=[1.0, 0.0],
        environment_version="fixture-v1",
    )


def _resume(
    fixture: ReplayFixture,
    pending: PendingReferenceReplay,
    executor: RecordingExecutor,
) -> ReferenceReplayResult:
    approval = _approval(pending)
    fixture.controller.register_approval(approval)
    return fixture.runtime.resume(
        pending,
        execution_approval=approval,
        executor=executor,
        temporal_observation=_observation(),
        final_request=QUESTION,
    )


def test_reference_replay_composes_six_roles_and_preserves_boundaries() -> None:
    fixture = _build_fixture()
    prepared = fixture.prepare()
    assert isinstance(prepared, PendingReferenceReplay)
    assert "PRIVATE-ORCHID" not in prepared.model_dump_json()
    executor = RecordingExecutor()

    result = _resume(fixture, prepared, executor)
    trusted_state = fixture.runtime.trusted_aion_state(result.run_id)
    trusted_workspace = fixture.runtime.trusted_workspace(result.run_id)

    assert result.aion.stage is ResearchStage.STOP
    assert fixture.controller.verify_chain(trusted_state)
    assert result.execution_succeeded is True
    assert result.execution.state is TransactionState.COMMITTED
    assert result.transaction.state is TransactionState.COMMITTED
    assert result.declared_material_effect is False
    assert result.promotion_authorized is False
    assert result.qualifying_result is False
    assert result.scientific_claim == "contract_composition_only"
    assert {trace.family for trace in result.traces} == {
        "Hermes",
        "Prometheus",
        "Perseus",
        "Olympus-Atlas",
        "Kronos",
        "Aion",
    }
    assert result.proposed_claim.status is ClaimStatus.PROPOSED
    assert result.answer.kind == "text"
    assert result.answer.citation_evidence_ids == [fixture.passage_id]
    assert result.kronos_observation.authoritative is False
    assert result.kronos_observation.used_for_action_selection is False
    assert result.kronos_observation.source_authenticated is False
    assert result.budget_usage.model_calls_used == 4
    assert result.budget_usage.tool_calls_used == 1
    assert len(executor.calls) == 1
    assert executor.calls[0][0] == "calculate.sum"
    assert executor.calls[0][1] == {"left": 2, "right": 3}
    assert validate_workspace_snapshot(trusted_workspace) == trusted_workspace
    assert trusted_workspace.event_sequence == len(trusted_workspace.events)
    assert any(item.text and "PRIVATE-ORCHID" in item.text for item in trusted_workspace.evidence)
    serialized = result.model_dump_json()
    assert "PRIVATE-ORCHID" not in serialized
    assert "PRIVATE-ZEPHYR" not in serialized
    assert "private_result" not in serialized
    assert "hidden" not in serialized


def test_prepare_halts_without_serializing_private_state() -> None:
    atlas_fixture = _build_fixture(private_only=True)
    atlas_halt = atlas_fixture.prepare()
    assert isinstance(atlas_halt, ReferenceReplayHalt)
    assert atlas_halt.reason == "atlas:no_authorized_evidence"
    assert atlas_halt.aion.stage is ResearchStage.STOP
    assert atlas_halt.abstention.producer == "runtime"
    assert {trace.family for trace in atlas_halt.traces} == {"Olympus-Atlas", "Aion"}
    assert "PRIVATE-ORCHID" not in atlas_halt.model_dump_json()

    prometheus_fixture = _build_fixture(trust_receipt=False)
    prometheus_halt = prometheus_fixture.prepare()
    assert isinstance(prometheus_halt, ReferenceReplayHalt)
    assert prometheus_halt.reason == "prometheus:no_eligible_branch"
    assert prometheus_halt.aion.stage is ResearchStage.STOP
    assert {trace.family for trace in prometheus_halt.traces} == {
        "Olympus-Atlas",
        "Prometheus",
        "Aion",
    }


def test_resume_rejects_wrong_untrusted_or_unbound_approval_before_executor() -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    executor = RecordingExecutor()

    for bad in (
        _approval(pending, head="0" * 64),
        _approval(pending, subject="0" * 64),
    ):
        with pytest.raises(PermissionError, match="pending requirement"):
            fixture.runtime.resume(
                pending,
                execution_approval=bad,
                executor=executor,
                temporal_observation=_observation(),
                final_request=QUESTION,
            )
    assert executor.calls == []


def test_untrusted_approval_cannot_create_transaction_or_spend_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    executor = RecordingExecutor()
    prepare_calls = 0
    forecast_calls = 0
    hermes_calls = 0
    real_prepare = fixture.manager.prepare
    real_forecast = fixture.runtime._forecast_draft
    real_respond = fixture.runtime.hermes.respond_after_execution

    def counted_prepare(*args: Any, **kwargs: Any) -> Any:
        nonlocal prepare_calls
        prepare_calls += 1
        return real_prepare(*args, **kwargs)

    def counted_forecast(*args: Any, **kwargs: Any) -> Any:
        nonlocal forecast_calls
        forecast_calls += 1
        return real_forecast(*args, **kwargs)

    def counted_respond(*args: Any, **kwargs: Any) -> Any:
        nonlocal hermes_calls
        hermes_calls += 1
        return real_respond(*args, **kwargs)

    monkeypatch.setattr(fixture.manager, "prepare", counted_prepare)
    monkeypatch.setattr(fixture.runtime, "_forecast_draft", counted_forecast)
    monkeypatch.setattr(
        fixture.runtime.hermes, "respond_after_execution", counted_respond
    )

    with pytest.raises(PermissionError, match="trusted approval"):
        fixture.runtime.resume(
            pending,
            execution_approval=_approval(pending),
            executor=executor,
            temporal_observation=_observation(),
            final_request=QUESTION,
        )

    assert (prepare_calls, forecast_calls, hermes_calls) == (0, 0, 0)
    assert executor.calls == []

    untrusted = _approval(pending)
    with pytest.raises(PermissionError, match="trusted approval"):
        fixture.runtime.resume(
            pending,
            execution_approval=untrusted,
            executor=executor,
            temporal_observation=_observation(),
            final_request=QUESTION,
        )
    assert executor.calls == []


def test_resume_rejects_mutated_pending_and_executor_profile() -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    pending.action.arguments["left"] = 900

    with pytest.raises(ValidationError, match="execution manifest"):
        fixture.runtime.resume(
            pending,
            execution_approval=_approval(pending),
            executor=RecordingExecutor(),
            temporal_observation=_observation(),
            final_request=QUESTION,
        )

    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    approval = _approval(pending)
    fixture.controller.register_approval(approval)
    executor = RecordingExecutor()
    executor.execution_profile_sha256 = "e" * 64
    with pytest.raises(PermissionError, match="executor profile"):
        fixture.runtime.resume(
            pending,
            execution_approval=approval,
            executor=executor,
            temporal_observation=_observation(),
            final_request=QUESTION,
        )
    assert executor.calls == []


def test_prepare_preflights_shared_and_aion_budgets_and_capability() -> None:
    fixture = _build_fixture()
    fixture.workspace = WorkspaceState.model_validate(
        {
            **fixture.workspace.model_dump(mode="python"),
            "budget": ComputeBudget(model_call_budget=3, tool_budget=1),
        }
    )
    with pytest.raises(ValueError, match="model-call budget"):
        fixture.prepare()

    fixture = _build_fixture()
    fixture.workspace = WorkspaceState.model_validate(
        {
            **fixture.workspace.model_dump(mode="python"),
            "budget": ComputeBudget(model_call_budget=4, tool_budget=0),
        }
    )
    with pytest.raises(ValueError, match="tool budget"):
        fixture.prepare()

    fixture = _build_fixture()
    fixture.protocol = FrozenProtocol.model_validate(
        {**fixture.protocol.model_dump(mode="python"), "maximum_steps": 5}
    )
    with pytest.raises(ValueError, match="six-step"):
        fixture.prepare()

    missing = _build_fixture(grant_capability=False)
    with pytest.raises(PermissionError, match="missing tool capabilities"):
        missing.prepare()
    # The failed action preflight happened before Aion registration: correcting
    # the trusted workspace can reuse the same run ID successfully.
    missing.workspace = WorkspaceState.model_validate(
        {
            **missing.workspace.model_dump(mode="python"),
            "permissions": PermissionSet(
                allowed_tool_ids={"calculate.sum"}, capabilities={"calculation"}
            ),
        }
    )
    assert isinstance(missing.prepare(), PendingReferenceReplay)


def test_approval_is_bound_to_exact_action_and_tool_policy() -> None:
    fixture = _build_fixture()
    first = fixture.prepare(action=_invocation(left=2, right=3))
    assert isinstance(first, PendingReferenceReplay)

    other_fixture = _build_fixture()
    second = other_fixture.prepare(action=_invocation(left=900, right=100))
    assert isinstance(second, PendingReferenceReplay)
    assert first.execution_manifest_sha256 != second.execution_manifest_sha256

    wrong_subject = _approval(second, subject=first.execution_manifest_sha256)
    other_fixture.controller.register_approval(wrong_subject)
    executor = RecordingExecutor()
    with pytest.raises(PermissionError, match="pending requirement"):
        other_fixture.runtime.resume(
            second,
            execution_approval=wrong_subject,
            executor=executor,
            temporal_observation=_observation(),
            final_request=QUESTION,
        )
    assert executor.calls == []


def test_post_execution_hermes_failure_terminalizes_without_stranding() -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    approval = _approval(pending)
    fixture.controller.register_approval(approval)
    executor = RecordingExecutor()
    hermes_attempts = 0
    class FailingHermes:
        def respond_after_execution(
            self,
            workspace: WorkspaceState,
            request: str,
            *,
            execution_succeeded: bool,
            failure_code: str | None = None,
            upstream_component_failure: str | None = None,
            allowed_evidence_ids: Any = None,
        ) -> None:
            nonlocal hermes_attempts
            hermes_attempts += 1
            del (
                workspace,
                request,
                execution_succeeded,
                failure_code,
                upstream_component_failure,
                allowed_evidence_ids,
            )
            raise RuntimeError("injected post-execution failure")

    fixture.runtime.hermes = FailingHermes()  # type: ignore[assignment]
    result = fixture.runtime.resume(
        pending,
        execution_approval=approval,
        executor=executor,
        temporal_observation=_observation(),
        final_request=QUESTION,
    )
    assert result.execution_succeeded
    assert len(executor.calls) == 1
    assert result.execution.replayed is False
    assert result.answer.kind == "abstain"
    assert result.answer.reason == "hermes:component_runtime_failure"
    assert hermes_attempts == 1
    assert result.aion.stage is ResearchStage.STOP
    assert result.budget_usage.model_calls_used == 4


def test_post_execution_kronos_failure_terminalizes_without_stranding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)

    def fail_forecast(observation: TemporalObservationFeatures) -> None:
        del observation
        raise RuntimeError("PRIVATE-KRONOS-DETAIL-MUST-NOT-ESCAPE")

    monkeypatch.setattr(fixture.runtime, "_forecast_draft", fail_forecast)
    result = _resume(fixture, pending, RecordingExecutor())

    assert result.aion.stage is ResearchStage.STOP
    assert result.kronos_observation.status == "failed"
    assert result.kronos_observation.failure_code == "component_runtime_failure"
    assert result.kronos_observation.forecast_mean == []
    assert result.traces[4].status == "failed"
    assert result.budget_usage.model_calls_used == 4
    assert "PRIVATE-KRONOS" not in result.model_dump_json()


def test_malformed_kronos_return_cannot_poison_recovery_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)

    monkeypatch.setattr(fixture.runtime, "_forecast_draft", lambda observation: object())
    result = _resume(fixture, pending, RecordingExecutor())

    assert result.kronos_observation.status == "failed"
    assert result.kronos_observation.failure_code == "component_runtime_failure"
    assert result.aion.stage is ResearchStage.STOP


def test_exhausted_post_gate_model_budget_terminalizes_with_closed_records() -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    admitted_budget = fixture.workspace.budget.model_call_budget
    fixture.runtime._pending[pending.pending_id].model_calls_used = admitted_budget

    result = _resume(fixture, pending, RecordingExecutor())

    assert result.kronos_observation.status == "failed"
    assert result.kronos_observation.failure_code == "model_call_budget_exhausted"
    assert result.answer.producer == "runtime_fail_closed"
    assert result.answer.reason == "hermes:model_call_budget_exhausted"
    assert result.budget_usage.model_calls_used == admitted_budget
    assert result.aion.stage is ResearchStage.STOP


def test_hermes_is_restricted_to_frozen_protocol_evidence() -> None:
    fixture = _build_fixture()
    extra_text = "UNFROZEN-QUASAR uniquely answers the post-execution request."
    extra = EvidenceItem(
        evidence_id="workspace:authorized-but-unfrozen",
        source_uri="memory://workspace/unfrozen",
        content_sha256=hashlib.sha256(extra_text.encode()).hexdigest(),
        acquired_at=NOW - timedelta(hours=2),
        license_id="test-fixture-only",
        text=extra_text,
    )
    fixture.workspace = WorkspaceState.model_validate(
        {
            **fixture.workspace.model_dump(mode="python"),
            "evidence": [*fixture.workspace.evidence, extra],
        }
    )
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    approval = _approval(pending)
    fixture.controller.register_approval(approval)
    executor = RecordingExecutor()
    result = fixture.runtime.resume(
        pending,
        execution_approval=approval,
        executor=executor,
        temporal_observation=_observation(),
        final_request="What is UNFROZEN-QUASAR?",
    )

    assert result.aion.stage is ResearchStage.STOP
    assert result.answer.kind == "abstain"
    assert extra.evidence_id not in result.answer.citation_evidence_ids
    assert len(executor.calls) == 1


def test_post_gate_commit_exception_is_recoverable_without_duplicate_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    approval = _approval(pending)
    fixture.controller.register_approval(approval)
    executor = RecordingExecutor()
    real_commit = fixture.manager.commit
    attempts = 0

    def fail_once(
        transaction_id: str,
        selected_executor: RecordingExecutor,
        **kwargs: Any,
    ) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected post-gate commit failure")
        return real_commit(transaction_id, selected_executor, **kwargs)

    monkeypatch.setattr(fixture.manager, "commit", fail_once)
    with pytest.raises(RuntimeError, match="post-gate commit failure"):
        fixture.runtime.resume(
            pending,
            execution_approval=approval,
            executor=executor,
            temporal_observation=_observation(),
            final_request=QUESTION,
        )

    result = fixture.runtime.resume(
        pending,
        execution_approval=approval,
        executor=executor,
        temporal_observation=_observation(),
        final_request=QUESTION,
    )
    assert result.aion.stage is ResearchStage.STOP
    assert result.execution_succeeded
    assert attempts == 2
    assert len(executor.calls) == 1
    assert result.budget_usage.model_calls_used == 4


def test_prior_unscoped_transaction_cannot_satisfy_fresh_aion_gate() -> None:
    fixture = _build_fixture()
    invocation = _invocation()
    native = ActionEnvelope(
        tool_id=invocation.tool_id,
        schema_version=invocation.schema_version,
        arguments=invocation.arguments,
        preconditions=invocation.preconditions,
        idempotency_key=invocation.idempotency_key,
    )
    prior = fixture.manager.prepare(
        native,
        ExecutionContext(
            capabilities={"calculation"},
            satisfied_preconditions={"inputs-reviewed"},
        ),
    )
    prior_executor = RecordingExecutor()
    fixture.manager.commit(prior.transaction_id, prior_executor)
    assert len(prior_executor.calls) == 1

    pending = fixture.prepare(action=invocation)
    assert isinstance(pending, PendingReferenceReplay)
    executor = RecordingExecutor()
    result = _resume(fixture, pending, executor)

    assert len(executor.calls) == 1
    assert result.execution.replayed is False
    assert result.transaction.transaction_id != prior.transaction_id
    assert (
        result.transaction.execution_scope_sha256
        == pending.approval_requirement.execution_scope_sha256
    )


def test_public_pending_identity_ignores_unauthorized_workspace_plaintext() -> None:
    first_fixture = _build_fixture()
    first = first_fixture.prepare()
    assert isinstance(first, PendingReferenceReplay)

    second_fixture = _build_fixture()
    replacement = "A different low-entropy private value: PRIVATE-VIOLET."
    private = second_fixture.workspace.evidence[1].model_copy(
        update={
            "text": replacement,
            "content_sha256": hashlib.sha256(replacement.encode()).hexdigest(),
        },
        deep=True,
    )
    second_fixture.workspace = WorkspaceState.model_validate(
        {
            **second_fixture.workspace.model_dump(mode="python"),
            "evidence": [second_fixture.workspace.evidence[0], private],
        }
    )
    second = second_fixture.prepare()
    assert isinstance(second, PendingReferenceReplay)

    assert first.pending_id == second.pending_id
    assert (
        first.approval_requirement.execution_scope_sha256
        == second.approval_requirement.execution_scope_sha256
    )
    assert "PRIVATE-ORCHID" not in first.model_dump_json()
    assert "PRIVATE-VIOLET" not in second.model_dump_json()


def test_public_scope_cannot_be_precreated_or_committed_without_runtime_capability() -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    native = ActionEnvelope(
        tool_id=pending.action.tool_id,
        schema_version=pending.action.schema_version,
        arguments=pending.action.arguments,
        preconditions=pending.action.preconditions,
        idempotency_key=pending.action.idempotency_key,
    )
    context = ExecutionContext(
        capabilities={"calculation"},
        satisfied_preconditions={"inputs-reviewed"},
    )
    with pytest.raises(AuthorizationError, match="reservation token"):
        fixture.manager.prepare(
            native,
            context,
            execution_scope_sha256=(
                pending.approval_requirement.execution_scope_sha256
            ),
            require_new=True,
        )

    result = _resume(fixture, pending, RecordingExecutor())
    with pytest.raises(AuthorizationError, match="reservation token"):
        fixture.manager.commit(result.transaction.transaction_id, RecordingExecutor())


def test_scoped_transaction_executes_once_and_kronos_cannot_choose_action() -> None:
    fixture = _build_fixture()
    with torch.no_grad():
        fixture.runtime.kronos.action_head.weight.zero_()
        fixture.runtime.kronos.action_head.bias.copy_(torch.tensor([-10.0, -10.0, 10.0]))
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    native = ActionEnvelope(
        tool_id=pending.action.tool_id,
        schema_version=pending.action.schema_version,
        arguments=pending.action.arguments,
        preconditions=pending.action.preconditions,
        idempotency_key=pending.action.idempotency_key,
    )
    assert action_sha256(native) == tool_invocation_sha256(pending.action)
    executor = RecordingExecutor()

    result = _resume(fixture, pending, executor)
    with pytest.raises(AuthorizationError, match="reservation token"):
        fixture.manager.receipt(result.transaction.transaction_id)

    assert result.kronos_observation.proposed_action_indices == [2, 2]
    assert executor.calls[0][0] == "calculate.sum"
    assert result.execution.replayed is False
    assert len(executor.calls) == 1


def test_tool_failure_is_sanitized_audited_and_not_claimed_rolled_back() -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)

    result = _resume(fixture, pending, RecordingExecutor(fail=True))
    trusted_state = fixture.runtime.trusted_aion_state(result.run_id)

    assert result.execution_succeeded is False
    assert result.execution.state is TransactionState.FAILED
    assert result.transaction.state is TransactionState.FAILED
    assert result.execution.error_code == "compensatable_execution_failure"
    assert result.answer.kind == "abstain"
    assert result.aion.stage is ResearchStage.STOP
    assert fixture.controller.verify_chain(trusted_state)
    serialized = result.model_dump_json()
    assert "secret-token-must-not-escape" not in serialized
    assert "rolled_back" not in serialized

    forged = result.model_dump(mode="python")
    forged_answer = {
        "kind": "text",
        "text": "Forged success despite the failed execution.",
        "citation_evidence_ids": [result.proposed_claim.support_evidence_ids[0]],
        "confidence": 0.9,
        "reason": None,
    }
    forged["answer"] = forged_answer
    forged["traces"][5]["status"] = "completed"
    forged["traces"][5]["output_sha256"] = canonical_sha256(forged_answer)
    with pytest.raises(ValidationError, match="failed execution requires"):
        ReferenceReplayResult.model_validate(forged)


def test_public_replay_does_not_commit_to_raw_executor_output() -> None:
    first_fixture = _build_fixture()
    first_pending = first_fixture.prepare()
    assert isinstance(first_pending, PendingReferenceReplay)
    first = _resume(
        first_fixture,
        first_pending,
        RecordingExecutor(result_marker="low-entropy-secret-a"),
    )

    second_fixture = _build_fixture()
    second_pending = second_fixture.prepare()
    assert isinstance(second_pending, PendingReferenceReplay)
    second = _resume(
        second_fixture,
        second_pending,
        RecordingExecutor(result_marker="low-entropy-secret-b"),
    )

    assert first == second
    assert "low-entropy-secret" not in first.model_dump_json()


def test_reference_replay_refuses_material_tools_during_prepare() -> None:
    fixture = _build_fixture()
    approval_id = f"approval_{'e' * 16}"
    invocation = _invocation().model_copy(update={"approval_id": approval_id})
    invocation_sha256 = tool_invocation_sha256(invocation)
    original_tool = fixture.workspace.tools[0]
    material_tool = ToolSpec(
        tool_id=original_tool.tool_id,
        schema_version=original_tool.schema_version,
        argument_schema=original_tool.argument_schema,
        required_capabilities=original_tool.required_capabilities,
        material_effect=True,
        approval_required=True,
    )
    fixture.workspace = WorkspaceState.model_validate(
        {
            **fixture.workspace.model_dump(mode="python"),
            "tools": [material_tool],
            "permissions": PermissionSet(
                allowed_tool_ids={material_tool.tool_id},
                capabilities={"calculation"},
                approval_bindings={approval_id: invocation_sha256},
            ),
        }
    )
    native_definition = ToolDefinition(
        tool_id="calculate.sum",
        schema_version="1",
        arguments={
            "left": ArgumentRule(kind=ArgumentKind.INTEGER),
            "right": ArgumentRule(kind=ArgumentKind.INTEGER),
        },
        required_capabilities={"calculation"},
        required_preconditions={"inputs-reviewed"},
        material_effect=True,
        approval_required=True,
    )
    action_approval = ActionApproval(
        approval_id=approval_id,
        action_sha256=invocation_sha256,
        issued_by="host-test-fixture",
        expires_at=NOW + timedelta(hours=1),
    )
    fixture.manager = TransactionManager(
        CapabilityKernel(
            [native_definition],
            trusted_approvals=[action_approval],
            clock=lambda: NOW,
        )
    )
    fixture.runtime.perseus = fixture.manager

    with pytest.raises(ValidationError, match="refuses tools declared material"):
        fixture.prepare(action=invocation)


def test_result_validator_rejects_forged_public_answer() -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    result = _resume(fixture, pending, RecordingExecutor())
    forged = result.model_copy(deep=True)
    forged.answer.text = "Unsupported forged conclusion."

    with pytest.raises(ValidationError, match="Hermes text is not an extract"):
        ReferenceReplayResult.model_validate(forged.model_dump(mode="python"))


def test_public_validators_reject_semantically_contradictory_traces() -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)

    malformed_pending = pending.model_copy(deep=True)
    malformed_pending.traces[2].input_sha256 = "0" * 64
    with pytest.raises(ValidationError, match="approval protocol"):
        PendingReferenceReplay.model_validate(
            malformed_pending.model_dump(mode="python")
        )

    malformed_pending = pending.model_copy(deep=True)
    malformed_pending.traces[1].input_sha256 = "9" * 64
    with pytest.raises(ValidationError, match="Prometheus trace"):
        PendingReferenceReplay.model_validate(
            malformed_pending.model_dump(mode="python")
        )

    result = _resume(fixture, pending, RecordingExecutor())
    malformed_result = result.model_copy(deep=True)
    malformed_result.traces[3].status = "failed"
    with pytest.raises(ValidationError, match="trace semantics"):
        ReferenceReplayResult.model_validate(malformed_result.model_dump(mode="python"))

    malformed_claim_result = result.model_dump(mode="python")
    malformed_claim_result["proposed_claim"]["status"] = "supported"
    malformed_claim_result["traces"][1]["output_sha256"] = canonical_sha256(
        malformed_claim_result["proposed_claim"]
    )
    with pytest.raises(ValidationError, match="claim must remain proposed"):
        ReferenceReplayResult.model_validate(malformed_claim_result)

    malformed_kronos_result = result.model_dump(mode="python")
    malformed_kronos_result["kronos_observation"][
        "temporal_event_id"
    ] = "forged:event"
    with pytest.raises(ValidationError, match="input reference"):
        ReferenceReplayResult.model_validate(malformed_kronos_result)

    ragged_result = result.model_dump(mode="python")
    ragged_observation = ragged_result["kronos_observation"]
    ragged_observation["forecast_mean"] = [[1.0], [2.0]]
    ragged_observation["forecast_log_variance"] = [[0.0]]
    ragged_observation["proposed_action_indices"] = []
    ragged_observation["forecast_sha256"] = canonical_sha256(
        {
            "mean": ragged_observation["forecast_mean"],
            "log_variance": ragged_observation["forecast_log_variance"],
            "proposed_action_indices": [],
        }
    )
    ragged_result["traces"][4]["output_sha256"] = canonical_sha256(
        ragged_observation
    )
    with pytest.raises(ValidationError, match="rectangular|non-negative actions"):
        ReferenceReplayResult.model_validate(ragged_result)

    uncited_result = result.model_dump(mode="python")
    uncited_result["answer"]["citation_evidence_ids"] = []
    uncited_result["traces"][5]["output_sha256"] = canonical_sha256(
        uncited_result["answer"]
    )
    with pytest.raises(ValidationError, match="text and citations"):
        ReferenceReplayResult.model_validate(uncited_result)

    forged_budget_result = result.model_dump(mode="python")
    forged_budget_result["budget_usage"]["model_call_budget"] = 999
    forged_budget_result["budget_usage"]["model_calls_used"] = 999
    with pytest.raises(ValidationError, match="admitted workspace budget"):
        ReferenceReplayResult.model_validate(forged_budget_result)

    forged_transaction_result = result.model_dump(mode="python")
    forged_transaction_id = f"tx_{'0' * 64}"
    forged_transaction_result["transaction"][
        "transaction_id"
    ] = forged_transaction_id
    forged_transaction_result["execution"]["transaction_id"] = forged_transaction_id
    forged_execution = forged_transaction_result["execution"]
    forged_execution["summary_sha256"] = canonical_sha256(
        {
            "transaction_id": forged_transaction_id,
            "idempotency_key": forged_execution["idempotency_key"],
            "state": forged_execution["state"],
            "replayed": forged_execution["replayed"],
            "recovery_class": forged_execution["recovery_class"],
            "error_code": forged_execution["error_code"],
            "raw_output_recorded": False,
        }
    )
    forged_transaction_result["traces"][3]["output_sha256"] = forged_execution[
        "summary_sha256"
    ]
    with pytest.raises(ValidationError, match="transaction ID is not bound"):
        ReferenceReplayResult.model_validate(forged_transaction_result)

    halt_fixture = _build_fixture(private_only=True)
    halt = halt_fixture.prepare()
    assert isinstance(halt, ReferenceReplayHalt)
    malformed_halt = halt.model_copy(deep=True)
    malformed_halt.traces[0].family = "Hermes"
    with pytest.raises(ValidationError, match="trace semantics"):
        ReferenceReplayHalt.model_validate(malformed_halt.model_dump(mode="python"))


def test_public_answer_fail_closed_producer_is_a_closed_abstention_contract() -> None:
    with pytest.raises(ValidationError, match="runtime fail-closed"):
        PublicAnswer(
            kind="text",
            producer="runtime_fail_closed",
            text="forged",
            citation_evidence_ids=["forged:evidence"],
            confidence=0.5,
            reason=None,
        )
    with pytest.raises(ValidationError, match="runtime fail-closed"):
        PublicAnswer(
            kind="abstain",
            producer="runtime_fail_closed",
            confidence=0.1,
            reason="hermes:component_runtime_failure",
        )
    with pytest.raises(ValidationError, match="closed runtime failure"):
        PublicAnswer(
            kind="abstain",
            producer="hermes",
            reason="hermes:component_runtime_failure",
        )


def test_kronos_public_observation_rejects_ragged_or_empty_completed_outputs() -> None:
    input_reference = canonical_sha256(
        {
            "event_id": "event:shape-test",
            "environment_version": "shape-v1",
            "source_kind": "host_supplied_pre_execution_features",
        }
    )
    mean = [[1.0], [2.0]]
    log_variance = [[0.0]]
    proposed_action_indices: list[int] = []
    payload = {
        "mean": mean,
        "log_variance": log_variance,
        "proposed_action_indices": proposed_action_indices,
    }
    with pytest.raises(ValidationError, match="rectangular|non-negative actions"):
        KronosObservation(
            input_reference_sha256=input_reference,
            temporal_event_id="event:shape-test",
            environment_version="shape-v1",
            model_state_sha256="a" * 64,
            forecast_sha256=canonical_sha256(payload),
            forecast_mean=mean,
            forecast_log_variance=log_variance,
            proposed_action_indices=proposed_action_indices,
        )


def test_completed_pending_artifact_cannot_be_resumed_again() -> None:
    fixture = _build_fixture()
    pending = fixture.prepare()
    assert isinstance(pending, PendingReferenceReplay)
    _resume(fixture, pending, RecordingExecutor())

    with pytest.raises(ValueError, match="consumed"):
        fixture.runtime.resume(
            pending,
            execution_approval=_approval(pending),
            executor=RecordingExecutor(),
            temporal_observation=_observation(),
            final_request=QUESTION,
        )
