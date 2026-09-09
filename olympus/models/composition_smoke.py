"""Reproducible synthetic smoke for the two-phase six-role reference replay."""

from __future__ import annotations

import hashlib
import os
import platform
import secrets
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from importlib.resources import files as package_files
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Self

import torch
from pydantic import Field, model_validator

from olympus.core.schemas import ComputeBudget, StrictModel
from olympus.models.aion import (
    AionController,
    Approval,
    AuthorityKind,
    FrozenProtocol,
    ResearchStage,
)
from olympus.models.atlas import AtlasDocument, OlympusAtlasService
from olympus.models.composition import (
    OlympusReferenceReplay,
    PendingReferenceReplay,
    ReferenceReplayResult,
    TemporalObservationFeatures,
)
from olympus.models.hermes import HermesGroundedService, HermesWorkspaceRuntime
from olympus.models.kronos import KronosConfig, KronosTemporalPlanner
from olympus.models.perseus import (
    ArgumentKind,
    ArgumentRule,
    CapabilityKernel,
    ToolDefinition,
    TransactionManager,
    TransactionSnapshot,
    TransactionState,
)
from olympus.models.prometheus import (
    BranchAssessment,
    HypothesisBranch,
    PrometheusSelector,
    TransitionReceipt,
    transition_from_receipt,
)
from olympus.models.substrate import (
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
)

_FIXED_TIME = datetime(2026, 9, 6, 12, tzinfo=UTC)
_QUESTION = "What evidence shows the Olympus runtime integration works?"
_CLAIM = "The controlled workflow reached its declared pre-execution boundary."
_EXECUTOR_PROFILE_SHA256 = hashlib.sha256(
    b"olympus-composition-smoke-in-memory-executor-v1"
).hexdigest()
_RAW_EXECUTOR_SENTINEL = "RAW-CALCULATION-RESULT-MUST-NOT-PERSIST"
_PACKAGE_SOURCE_PATHS = (
    "core/schemas.py",
    "models/__init__.py",
    "models/registry.py",
    "models/substrate.py",
    "models/hermes.py",
    "models/prometheus.py",
    "models/perseus.py",
    "models/atlas.py",
    "models/kronos.py",
    "models/aion.py",
    "models/composition.py",
    "models/composition_smoke.py",
)


class ReferenceReplaySmokeManifest(StrictModel):
    schema_version: Literal[1] = 1
    execution_id: str = Field(pattern=r"^composition-[0-9TZ-]+-[0-9a-f]{16}$")
    protocol_id: str = Field(pattern=r"^composition-[0-9a-f]{12}-s[0-9]+$")
    created_at: datetime
    seed: int
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_paths: list[str] = Field(min_length=1)
    python_version: str
    torch_version: str
    platform: str
    command: str
    data_scope: Literal["deterministic_synthetic_contract_fixture"] = (
        "deterministic_synthetic_contract_fixture"
    )
    result: ReferenceReplayResult
    deterministic_checks: dict[str, bool] = Field(min_length=1)
    passed: bool
    scientific_claim: Literal["contract_composition_only"] = "contract_composition_only"
    promotion_authorized: Literal[False] = False
    qualifying_result: Literal[False] = False
    limitations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def summary_matches_checks(self) -> Self:
        if self.passed != all(self.deterministic_checks.values()):
            raise ValueError("composition smoke pass flag does not match its checks")
        if self.result.promotion_authorized or self.result.qualifying_result:
            raise ValueError("composition smoke cannot contain a promoted or qualifying result")
        return self


class _SmokeExecutor:
    guarantees_idempotency = True
    execution_profile_sha256 = _EXECUTOR_PROFILE_SHA256

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def execute(
        self, tool_id: str, arguments: Mapping[str, Any], transaction_id: str
    ) -> Mapping[str, Any]:
        copied = dict(arguments)
        self.calls.append((tool_id, copied, transaction_id))
        return {
            "sum": copied["left"] + copied["right"],
            "private_result": _RAW_EXECUTOR_SENTINEL,
        }


def reference_replay_source_sha256() -> str:
    """Hash the shipped package sources that define this smoke's semantics."""

    root = package_files("olympus")
    digest = hashlib.sha256()
    for relative in _PACKAGE_SOURCE_PATHS:
        resource = root.joinpath(*PurePosixPath(relative).parts)
        label = f"olympus/{relative}"
        if not resource.is_file():
            raise RuntimeError(f"composition source resource is missing: {label}")
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resource.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_reference_replay_smoke(
    output_dir: Path, *, seed: int = 20_260_906
) -> ReferenceReplaySmokeManifest:
    """Run one fixed replay and atomically persist its sanitized public record."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("composition smoke seed must be a non-negative integer")
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source_sha256 = reference_replay_source_sha256()
    protocol_id = f"composition-{source_sha256[:12]}-s{seed}"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        runtime, controller, workspace, protocol, assessment, corpus_version = (
            _build_runtime_fixture(source_sha256, protocol_id)
        )
        action = ToolInvocation(
            tool_id="calculate.sum",
            schema_version="1",
            arguments={"left": 2, "right": 3},
            preconditions=["inputs-reviewed"],
            idempotency_key="composition-smoke-action-0001",
        )
        executor = _SmokeExecutor()
        prepared = runtime.prepare(
            workspace,
            run_id=f"aion_{hashlib.sha256(protocol_id.encode()).hexdigest()[:16]}",
            protocol=protocol,
            query=_QUESTION,
            corpus_version=corpus_version,
            assessments=[assessment],
            action=action,
            executor_profile_sha256=executor.execution_profile_sha256,
            occurred_at=_FIXED_TIME,
        )
        if not isinstance(prepared, PendingReferenceReplay):
            raise RuntimeError(f"composition smoke halted during prepare: {prepared.reason}")
        approval = Approval(
            approval_id=(
                f"approval_{hashlib.sha256((protocol_id + ':approval').encode()).hexdigest()[:16]}"
            ),
            authority=AuthorityKind.HUMAN,
            actor="composition-smoke-reviewer",
            scope=prepared.approval_requirement.scope,
            run_id=prepared.run_id,
            protocol_sha256=prepared.approval_requirement.protocol_sha256,
            expected_head_sha256=prepared.approval_requirement.expected_head_sha256,
            subject_sha256=prepared.approval_requirement.execution_manifest_sha256,
            expires_at=_FIXED_TIME + timedelta(hours=1),
        )
        controller.register_approval(approval)
        result = runtime.resume(
            prepared,
            execution_approval=approval,
            executor=executor,
            temporal_observation=TemporalObservationFeatures(
                event_id="temporal:composition-smoke:1",
                occurred_at=_FIXED_TIME,
                features=[1.0, 0.0],
                environment_version="composition-smoke-v1",
            ),
            final_request=_QUESTION,
        )

    trusted_aion_state = runtime.trusted_aion_state(result.run_id)
    trusted_workspace = runtime.trusted_workspace(result.run_id)
    serialized = result.model_dump_json()
    public_result_keys = set(result.model_dump(mode="python"))
    public_evidence_by_id = {
        item.evidence_id: item for item in result.workspace_view.evidence
    }
    cited_evidence = [
        public_evidence_by_id.get(evidence_id)
        for evidence_id in result.answer.citation_evidence_ids
    ]
    answer_text = result.answer.text or ""
    checks = {
        "manifest_protocol_id_matches_frozen_protocol": protocol.protocol_id
        == protocol_id,
        "all_six_roles_exercised": {trace.family for trace in result.traces}
        == {"Hermes", "Prometheus", "Perseus", "Olympus-Atlas", "Kronos", "Aion"},
        "expected_role_statuses_completed": [trace.status for trace in result.traces]
        == [
            "completed",
            "completed",
            "prepared",
            "completed",
            "completed",
            "completed",
            "completed",
        ],
        "aion_chain_verified_and_stopped": (
            controller.verify_chain(trusted_aion_state)
            and trusted_aion_state.stage is ResearchStage.STOP
            and result.aion.stage is ResearchStage.STOP
        ),
        "non_material_execution_committed_once": (
            result.execution_succeeded
            and not result.declared_material_effect
            and len(executor.calls) == 1
        ),
        "workspace_event_chain_revalidated": (
            trusted_workspace.event_sequence == len(trusted_workspace.events)
            and WorkspaceState.model_validate(
                trusted_workspace.model_dump(mode="python")
            )
            == trusted_workspace
        ),
        "prometheus_claim_remains_proposed": (
            result.proposed_claim.status.value == "proposed"
        ),
        "approval_binds_action_policy_and_executor_profile": (
            approval.subject_sha256 == prepared.execution_manifest_sha256
            and prepared.approval_requirement.action_sha256
            == tool_invocation_sha256(action)
            and prepared.action == action
            and result.action == action
            and result.execution_manifest_sha256
            == prepared.execution_manifest_sha256
            and result.execution_manifest.executor_profile_sha256
            == executor.execution_profile_sha256
            and prepared.approval_requirement.execution_scope_sha256
            == result.transaction.execution_scope_sha256
        ),
        "budget_accounting_is_exact": (
            result.budget_usage.minimum_model_calls == 4
            and result.budget_usage.model_call_budget == 4
            and result.budget_usage.model_calls_used == 4
            and result.budget_usage.tool_call_budget == 1
            and result.budget_usage.tool_calls_used == 1
        ),
        "transaction_snapshot_is_terminal": (
            isinstance(result.transaction, TransactionSnapshot)
            and result.transaction.state is TransactionState.COMMITTED
            and result.transaction.state is result.execution.state
            and result.transaction.transaction_id == result.execution.transaction_id
            and result.transaction.execution_scope_sha256
            == prepared.approval_requirement.execution_scope_sha256
        ),
        "kronos_is_host_supplied_and_non_authoritative": (
            result.kronos_observation.status == "completed"
            and result.kronos_observation.failure_code is None
            and bool(result.kronos_observation.forecast_mean)
            and
            result.kronos_observation.source_kind
            == "host_supplied_pre_execution_features"
            and not result.kronos_observation.source_authenticated
            and not result.kronos_observation.authoritative
            and not result.kronos_observation.used_for_action_selection
            and result.traces[4].operation
            == "post_execution_forecast_from_host_pre_execution_features"
        ),
        "hermes_returned_grounded_text": (
            result.answer.producer == "hermes"
            and result.answer.kind == "text"
            and bool(answer_text)
            and bool(result.answer.citation_evidence_ids)
            and all(
                item is not None
                and item.text is not None
                and answer_text in item.text
                for item in cited_evidence
            )
        ),
        "public_artifact_excludes_trusted_state": (
            not {"workspace", "aion_state", "claim_output"} & public_result_keys
        ),
        "protected_and_raw_executor_output_absent": (
            "PRIVATE-ZEPHYR" not in serialized
            and _RAW_EXECUTOR_SENTINEL not in serialized
            and '"private_result"' not in serialized
            and '"sum"' not in serialized
        ),
        "promotion_and_qualification_denied": (
            not result.promotion_authorized and not result.qualifying_result
        ),
    }
    created_at = datetime.now(UTC)
    manifest = ReferenceReplaySmokeManifest(
        execution_id=(
            f"composition-{created_at.strftime('%Y-%m-%dT%H%M%SZ')}-"
            f"{secrets.token_hex(8)}"
        ),
        protocol_id=protocol_id,
        created_at=created_at,
        seed=seed,
        source_sha256=source_sha256,
        source_paths=[f"olympus/{relative}" for relative in _PACKAGE_SOURCE_PATHS],
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        platform=platform.platform(),
        command=f"olympus models composition-smoke --seed {seed}",
        result=result,
        deterministic_checks=checks,
        passed=all(checks.values()),
        limitations=[
            "The fixture uses one synthetic public document, one trusted synthetic receipt, "
            "one non-material in-memory tool, and a randomly initialized Kronos component.",
            "The replay proves contract composition and fail-closed gates only; it does not "
            "establish model quality, sandboxing, durability, scientific validity, or promotion.",
        ],
    )
    _write_manifest(destination / "reference_replay_manifest.json", manifest)
    return manifest


def _build_runtime_fixture(
    source_sha256: str,
    protocol_id: str,
) -> tuple[
    OlympusReferenceReplay,
    AionController,
    WorkspaceState,
    FrozenProtocol,
    BranchAssessment,
    str,
]:
    identity = ModelIdentity(
        model_id="olympus.reference.replay",
        family="Olympus",
        version="0.1.0",
        base_model_id="repository://reference-components",
        base_revision=source_sha256[:12],
        base_sha256=source_sha256,
        lifecycle=ModelLifecycle.EXPERIMENTAL,
    )
    receipt = TransitionReceipt(
        workspace_id="workspace:composition-smoke",
        model_identity_sha256=canonical_sha256(identity),
        branch_id="branch-a",
        claim_sha256=hashlib.sha256(_CLAIM.encode("utf-8")).hexdigest(),
        operation="verify deterministic integration fixture",
        check="boolean",
        expected=True,
        observed=True,
        issued_by="host-composition-smoke",
    )
    receipt_text = receipt.canonical_text()
    receipt_evidence = EvidenceItem(
        evidence_id="receipt:composition-smoke:0",
        source_uri="memory://composition-smoke/receipt",
        content_sha256=hashlib.sha256(receipt_text.encode("utf-8")).hexdigest(),
        acquired_at=_FIXED_TIME - timedelta(hours=1),
        license_id="synthetic-test-only",
        text=receipt_text,
    )
    branch = HypothesisBranch(
        branch_id="branch-a",
        claim=_CLAIM,
        transitions=[transition_from_receipt(step=0, evidence=receipt_evidence)],
    )
    assessment = BranchAssessment(
        branch=branch,
        proposer_score=0.9,
        verifier_scores=[0.95],
    )
    shared_tool = ToolSpec(
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
        workspace_id="workspace:composition-smoke",
        objective=_QUESTION,
        budget=ComputeBudget(model_call_budget=4, tool_budget=1),
        evidence=[receipt_evidence],
        plan=[
            PlanStep(
                step_id="review-inputs",
                description="Host reviewed the deterministic smoke inputs.",
                status=PlanStatus.COMPLETED,
                tool_id=shared_tool.tool_id,
                postconditions=["inputs-reviewed"],
            )
        ],
        tools=[shared_tool],
        permissions=PermissionSet(
            allowed_tool_ids={shared_tool.tool_id},
            capabilities={"calculation"},
        ),
        model_identity=identity,
    )
    atlas = OlympusAtlasService(dense_dimensions=16)
    corpus_version = atlas.build_index(
        [
            AtlasDocument(
                document_id="public-integration",
                text=(
                    "Evidence shows the Olympus runtime integration works in a controlled "
                    "synthetic test."
                ),
                source_uri="memory://composition-smoke/public",
                source_version="1",
                acquired_at="2026-09-06T10:00:00Z",
                license_id="synthetic-test-only",
            ),
            AtlasDocument(
                document_id="private-integration",
                text=(
                    "Evidence shows the Olympus runtime integration works with PRIVATE-ZEPHYR "
                    "telemetry."
                ),
                source_uri="memory://composition-smoke/private",
                source_version="1",
                acquired_at="2026-09-06T10:00:00Z",
                license_id="synthetic-test-only",
                acl_labels=("secret-flight",),
            ),
        ]
    )
    public_index = atlas.get_index(corpus_version)
    if public_index is None:
        raise RuntimeError("composition smoke failed to build its Atlas index")
    passage_id = next(
        passage.passage_id
        for passage in public_index.passages
        if passage.document_id == "public-integration"
    )
    protocol = FrozenProtocol(
        protocol_id=protocol_id,
        version=1,
        question=_QUESTION,
        hypothesis_ids=[branch.branch_id],
        evidence_ids=[receipt_evidence.evidence_id, passage_id],
        procedure=["retrieve", "select", "authorize", "execute", "audit", "stop"],
        falsification_criterion=(
            "Any unauthorized action, private-text leak, or promoted result falsifies the smoke."
        ),
        maximum_steps=8,
        maximum_tool_calls=1,
        frozen_at=_FIXED_TIME - timedelta(minutes=1),
    )
    definition = ToolDefinition(
        tool_id=shared_tool.tool_id,
        schema_version=shared_tool.schema_version,
        arguments={
            "left": ArgumentRule(kind=ArgumentKind.INTEGER),
            "right": ArgumentRule(kind=ArgumentKind.INTEGER),
        },
        required_capabilities={"calculation"},
        required_preconditions={"inputs-reviewed"},
    )
    manager = TransactionManager(CapabilityKernel([definition], clock=lambda: _FIXED_TIME))
    controller = AionController(clock=lambda: _FIXED_TIME)
    runtime = OlympusReferenceReplay(
        atlas=atlas,
        prometheus=PrometheusSelector(
            minimum_score=0.5,
            minimum_evidence_coverage=1.0,
            trusted_receipt_sha256={receipt_evidence.content_sha256},
        ),
        perseus=manager,
        kronos=KronosTemporalPlanner(
            KronosConfig(
                feature_dim=2,
                hidden_dim=8,
                output_dim=1,
                horizons=2,
                plan_steps=2,
                action_count=3,
                attention_heads=2,
            )
        ),
        aion=controller,
        hermes=HermesWorkspaceRuntime(HermesGroundedService(minimum_support=0.5)),
        runtime_secret=hashlib.sha256(
            f"{protocol_id}:synthetic-runtime-secret".encode()
        ).digest(),
    )
    return runtime, controller, workspace, protocol, assessment, corpus_version


def _write_manifest(path: Path, manifest: ReferenceReplaySmokeManifest) -> None:
    encoded = manifest.model_dump_json(indent=2).encode("utf-8") + b"\n"
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
