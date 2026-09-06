"""Bounded, reproducible execution smoke for the Olympus model families.

The runner proves that each reference implementation has a differentiable
training path and that its deterministic safety boundary executes. It uses
small synthetic contract fixtures, so its checkpoints are never qualifying or
promoted family models.
"""

from __future__ import annotations

import hashlib
import os
import platform
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Literal, Self, cast

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn

from olympus.core.schemas import StrictModel
from olympus.models.aion import (
    AionController,
    AionRouter,
    AionRouterConfig,
    Approval,
    AuthorityKind,
    FrozenProtocol,
    ResearchStage,
    aion_router_loss,
)
from olympus.models.atlas import (
    AtlasDocument,
    AuthorizedQuery,
    EvidenceSet,
    OlympusAtlasRetriever,
    OlympusAtlasService,
)
from olympus.models.hermes import (
    HermesEvidence,
    HermesGroundedService,
    HermesGroundingModule,
)
from olympus.models.kronos import (
    AdaptationMetrics,
    KronosConfig,
    KronosTemporalPlanner,
    TemporalAdapterManager,
    kronos_loss,
)
from olympus.models.perseus import (
    ActionApproval,
    ActionEnvelope,
    ArgumentKind,
    ArgumentRule,
    AuthorizationError,
    CapabilityKernel,
    ExecutionContext,
    RecoveryClass,
    ToolDefinition,
    TrainableActionPolicy,
    TrainableRecoveryPolicy,
    TransactionManager,
    action_policy_loss,
    action_sha256,
    recovery_policy_loss,
)
from olympus.models.prometheus import (
    BranchAssessment,
    CheckableTransition,
    HypothesisBranch,
    PrometheusSelector,
    TrainableBranchProposer,
    TrainableProcessVerifier,
    TransitionReceipt,
    prometheus_training_loss,
)
from olympus.models.substrate import (
    AdapterCausalDecoder,
    DecoderConfig,
    EvidenceItem,
    ModelIdentity,
    WorkspaceState,
    canonical_sha256,
    causal_language_model_loss,
)

_SMOKE_STATUS: Literal["EXPERIMENTAL_SMOKE_NOT_PROMOTED"] = (
    "EXPERIMENTAL_SMOKE_NOT_PROMOTED"
)
_MODEL_SOURCE_NAMES = (
    "substrate.py",
    "hermes.py",
    "prometheus.py",
    "perseus.py",
    "atlas.py",
    "kronos.py",
    "aion.py",
    "smoke.py",
)


class TrainingMetric(StrictModel):
    initial_loss: float
    final_loss: float
    learned_parameters: int = Field(gt=0)
    gradient_update_observed: bool
    loss_decreased: bool
    finite: bool

    @property
    def passed(self) -> bool:
        return self.gradient_update_observed and self.loss_decreased and self.finite


class CheckpointArtifact(StrictModel):
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(gt=0)
    qualifying_checkpoint: Literal[False] = False
    promoted: Literal[False] = False


class FamilySmokeResult(StrictModel):
    family: str
    implementation_classes: list[str] = Field(min_length=1)
    training: TrainingMetric
    deterministic_checks: dict[str, bool] = Field(min_length=1)
    checkpoint: CheckpointArtifact
    status: Literal["EXPERIMENTAL_SMOKE_NOT_PROMOTED"] = _SMOKE_STATUS
    passed: bool
    limitations: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def result_matches_checks(self) -> Self:
        expected = self.training.passed and all(self.deterministic_checks.values())
        if self.passed != expected:
            raise ValueError("family smoke pass flag does not match recorded checks")
        return self


class FamilySmokeManifest(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str
    created_at: datetime
    seed: int
    optimization_steps: int = Field(ge=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_version: str
    torch_version: str
    platform: str
    data_scope: Literal["deterministic_synthetic_contract_fixtures"] = (
        "deterministic_synthetic_contract_fixtures"
    )
    substrate: TrainingMetric
    substrate_checkpoint: CheckpointArtifact
    families: list[FamilySmokeResult] = Field(min_length=6, max_length=6)
    passed: bool
    promotion_authorized: Literal[False] = False
    scientific_claim: Literal["execution_smoke_only"] = "execution_smoke_only"

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        names = [result.family for result in self.families]
        if len(names) != len(set(names)):
            raise ValueError("smoke manifest family names must be unique")
        expected = self.substrate.passed and all(result.passed for result in self.families)
        if self.passed != expected:
            raise ValueError("manifest pass flag does not match component results")
        return self


def run_family_smoke(
    output_dir: Path, *, seed: int = 20_260_906, steps: int = 24
) -> FamilySmokeManifest:
    """Train every component on tiny fixtures and persist auditable artifacts."""

    if steps < 1:
        raise ValueError("steps must be at least one")
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source_sha256 = _source_digest()
    run_id = f"models-smoke-{source_sha256[:12]}-s{seed}-n{steps}"

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        substrate_metric, substrate_state = _run_substrate(steps)
        substrate_artifact = _save_checkpoint(
            destination / "shared-substrate.pt",
            family="SharedSubstrate",
            implementation_classes=["AdapterCausalDecoder"],
            state_dicts=substrate_state,
            seed=seed,
            steps=steps,
            source_sha256=source_sha256,
        )
        runners: tuple[
            Callable[[int], tuple[TrainingMetric, dict[str, bool], dict[str, Any]]], ...
        ] = (
            _run_hermes,
            _run_prometheus,
            _run_perseus,
            _run_atlas,
            _run_kronos,
            _run_aion,
        )
        specifications = (
            (
                "Hermes",
                [
                    "HermesGroundingModule",
                    "HermesGroundedService",
                    "HermesWorkspaceRuntime",
                ],
                [
                    "Synthetic token IDs do not establish grounded-response quality, "
                    "and the learned heads are not yet wired into the deterministic service."
                ],
            ),
            (
                "Prometheus",
                ["TrainableBranchProposer", "TrainableProcessVerifier", "PrometheusSelector"],
                [
                    "Synthetic branches and host-trusted receipts do not establish "
                    "scientific reasoning accuracy."
                ],
            ),
            (
                "Perseus",
                ["TrainableActionPolicy", "TrainableRecoveryPolicy", "TransactionManager"],
                [
                    "The injected executor protocol and process-local transactions do not "
                    "establish sandboxing, durable recovery, or real-environment success."
                ],
            ),
            (
                "Olympus-Atlas",
                ["OlympusAtlasRetriever", "OlympusAtlasService"],
                [
                    "A two-document, process-local index does not establish retrieval "
                    "benchmark quality or durable serving behavior."
                ],
            ),
            (
                "Kronos",
                ["KronosTemporalPlanner", "TemporalAdapterManager"],
                [
                    "Synthetic sequences and a local full-planner checkpoint gate do not "
                    "establish forecasting or continual-learning value."
                ],
            ),
            (
                "Aion",
                ["AionRouter", "AionController"],
                [
                    "Process-local, unsigned protocol execution does not establish "
                    "autonomous-research value or durable governance."
                ],
            ),
        )
        results: list[FamilySmokeResult] = []
        for runner, (family, classes, limitations) in zip(
            runners, specifications, strict=True
        ):
            metric, checks, state = runner(steps)
            filename = family.lower().replace("olympus-", "").replace("-", "_") + ".pt"
            artifact = _save_checkpoint(
                destination / filename,
                family=family,
                implementation_classes=classes,
                state_dicts=state,
                seed=seed,
                steps=steps,
                source_sha256=source_sha256,
            )
            results.append(
                FamilySmokeResult(
                    family=family,
                    implementation_classes=classes,
                    training=metric,
                    deterministic_checks=checks,
                    checkpoint=artifact,
                    passed=metric.passed and all(checks.values()),
                    limitations=limitations,
                )
            )

    manifest = FamilySmokeManifest(
        run_id=run_id,
        created_at=datetime.now(UTC),
        seed=seed,
        optimization_steps=steps,
        source_sha256=source_sha256,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        platform=platform.platform(),
        substrate=substrate_metric,
        substrate_checkpoint=substrate_artifact,
        families=results,
        passed=substrate_metric.passed and all(result.passed for result in results),
    )
    _write_json_atomic(destination / "manifest.json", manifest)
    return manifest


def _run_substrate(steps: int) -> tuple[TrainingMetric, dict[str, Any]]:
    config = DecoderConfig(
        vocab_size=32,
        width=16,
        layers=1,
        heads=4,
        max_sequence_tokens=8,
        adapter_rank=4,
    )
    model = AdapterCausalDecoder(config)
    adapter_names = (
        "hermes",
        "prometheus",
        "perseus",
        "olympus_atlas",
        "kronos",
        "aion",
    )
    for name in adapter_names:
        model.add_adapter(name)
    metrics: list[TrainingMetric] = []
    for index, name in enumerate(adapter_names):
        model.activate_adapter(name)
        model.freeze_base()
        offset = index + 1
        tokens = torch.tensor(
            [
                [offset, offset + 1, offset + 2, offset + 3, offset + 4],
                [offset + 4, offset + 3, offset + 2, offset + 1, offset],
            ]
        )

        metrics.append(
            _optimize(
                [model],
                partial(_decoder_fixture_loss, model, tokens),
                steps=steps,
                learning_rate=0.03,
            )
        )
    metric = TrainingMetric(
        initial_loss=sum(item.initial_loss for item in metrics) / len(metrics),
        final_loss=sum(item.final_loss for item in metrics) / len(metrics),
        learned_parameters=sum(
            tensor.numel()
            for name in adapter_names
            for tensor in model.adapter_state_dict(name).values()
        ),
        gradient_update_observed=all(item.gradient_update_observed for item in metrics),
        loss_decreased=all(item.loss_decreased for item in metrics),
        finite=all(item.finite for item in metrics),
    )
    return metric, {
        "adapter_state_dicts": {
            name: model.adapter_state_dict(name) for name in adapter_names
        },
        "adapter_names": adapter_names,
        "config": config.model_dump(mode="json"),
    }


def _decoder_fixture_loss(model: AdapterCausalDecoder, tokens: Tensor) -> Tensor:
    return causal_language_model_loss(model(tokens), tokens)


def _run_hermes(steps: int) -> tuple[TrainingMetric, dict[str, bool], dict[str, Any]]:
    model = HermesGroundingModule(vocab_size=32, hidden_size=16)
    query_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 3, 5, 7], [2, 4, 6, 8]])
    evidence_ids = torch.tensor(
        [
            [[1, 2, 9], [10, 11, 12], [3, 4, 13]],
            [[5, 6, 14], [15, 16, 17], [7, 8, 18]],
            [[1, 3, 19], [20, 21, 22], [5, 7, 23]],
            [[2, 4, 24], [25, 26, 27], [6, 8, 28]],
        ]
    )
    attribution_targets = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])

    def loss() -> Tensor:
        output = model(
            query_ids,
            evidence_ids,
            attribution_targets=attribution_targets,
            correctness_targets=torch.ones(4),
            memory_targets=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        )
        return cast(Tensor, output["loss"])

    metric = _optimize([model], loss, steps=steps, learning_rate=0.02)
    public = HermesEvidence.from_text(
        "mars-public", "Mars has two moons.", "urn:smoke:mars", acl_labels=("public",)
    )
    private = HermesEvidence.from_text(
        "mars-private",
        "The private launch code is 991.",
        "urn:smoke:private",
        acl_labels=("private",),
    )
    service = HermesGroundedService(minimum_support=0.5)
    answer = service.answer("Mars moons", [public], propose_memory=True)
    denied = service.answer("What is the private launch code?", [private])
    checks = {
        "citation_span_integrity": bool(
            answer.citations
            and answer.citations[0].quote
            == public.text[answer.citations[0].span_start : answer.citations[0].span_end]
        ),
        "memory_is_proposal_only": bool(
            answer.memory_write_proposals
            and answer.memory_write_proposals[0].requires_approval
        ),
        "unauthorized_evidence_abstains": denied.abstained,
    }
    return metric, checks, {"grounding_module": model.state_dict()}


def _run_atlas(steps: int) -> tuple[TrainingMetric, dict[str, bool], dict[str, Any]]:
    model = OlympusAtlasRetriever(vocab_size=40, hidden_size=16, temperature=0.2)
    query_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    passage_ids = torch.tensor(
        [
            [[1, 2, 3, 13], [14, 15, 16, 17], [18, 19, 20, 21]],
            [[4, 5, 6, 14], [15, 16, 17, 18], [19, 20, 21, 22]],
            [[7, 8, 9, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
            [[10, 11, 12, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ]
    )
    metric = _optimize(
        [model],
        lambda: model(query_ids, passage_ids, targets=torch.zeros(4, dtype=torch.long))[
            "loss"
        ],
        steps=steps,
        learning_rate=0.015,
    )
    service = OlympusAtlasService(dense_dimensions=16)
    version = service.build_index(
        [
            AtlasDocument(
                document_id="public",
                text="Saturn has rings made of ice and rock.",
                source_uri="urn:smoke:saturn",
                source_version="1",
                acquired_at="2026-09-06T00:00:00Z",
                license_id="test-only",
                acl_labels=("public",),
            ),
            AtlasDocument(
                document_id="private",
                text="Confidential launch phrase cobalt lantern.",
                source_uri="urn:smoke:private",
                source_version="1",
                acquired_at="2026-09-06T00:00:00Z",
                license_id="test-only",
                acl_labels=("private",),
            ),
        ]
    )
    result = service.retrieve(AuthorizedQuery("What are Saturn rings made of?"), top_k=1)
    private_query = service.retrieve(AuthorizedQuery("cobalt lantern"), top_k=1)
    checks = {
        "version_is_replayable": service.get_index(version) is not None,
        "provenance_is_complete": isinstance(result, EvidenceSet)
        and bool(result.hits[0].passage.passage_hash),
        "acl_filtered_before_results": not isinstance(private_query, EvidenceSet)
        or all(hit.passage.document_id != "private" for hit in private_query.hits),
    }
    return metric, checks, {"retriever": model.state_dict()}


def _run_prometheus(
    steps: int,
) -> tuple[TrainingMetric, dict[str, bool], dict[str, Any]]:
    proposer = TrainableBranchProposer(context_dim=4, branch_dim=3, hidden_dim=16)
    verifier = TrainableProcessVerifier(context_dim=4, transition_dim=2, hidden_dim=16)
    context = torch.tensor([0.2, -0.1, 0.7, 0.3])
    branch_features = torch.tensor([[0.0, 0.1, 0.2], [0.8, 0.7, 0.9], [-0.2, 0.3, 0.1]])
    transition_features = torch.tensor([[0.8, 0.2], [-0.5, 0.7], [0.4, 0.6]])

    def loss() -> Tensor:
        return prometheus_training_loss(
            proposer_logits=proposer(context, branch_features),
            target_branch=1,
            verifier_logits=verifier(context, transition_features),
            verifier_labels=torch.tensor([1.0, 0.0, 1.0]),
            branch_embeddings=branch_features,
        )

    metric = _optimize([proposer, verifier], loss, steps=steps, learning_rate=0.025)
    valid = _smoke_branch("valid", observed=4)
    invalid = _smoke_branch("invalid", observed=5)
    model_identity = ModelIdentity(
        model_id="prometheus-smoke",
        family="Prometheus",
        version="smoke",
        base_model_id="synthetic/none",
        base_revision="smoke",
        base_sha256="0" * 64,
    )
    receipt_evidence = [
        _smoke_transition_evidence(branch, model_identity)
        for branch in (valid, invalid)
    ]
    workspace = WorkspaceState(
        workspace_id="prometheus-smoke",
        objective="Select a checked fixture branch.",
        evidence=receipt_evidence,
        model_identity=model_identity,
    )
    selector = PrometheusSelector(
        minimum_score=0.6,
        trusted_receipt_sha256={item.content_sha256 for item in receipt_evidence},
    )
    selected = selector.select(
        [
            BranchAssessment(branch=invalid, proposer_score=0.99, verifier_scores=[0.99]),
            BranchAssessment(branch=valid, proposer_score=0.8, verifier_scores=[0.9]),
        ],
        workspace,
    )
    rejected = selector.select(
        [BranchAssessment(branch=invalid, proposer_score=0.99, verifier_scores=[0.99])],
        workspace,
    )
    checks = {
        "checked_branch_selected": selected.branch_id == "valid",
        "failed_check_abstains": rejected.decision == "abstained",
    }
    return metric, checks, {
        "branch_proposer": proposer.state_dict(),
        "process_verifier": verifier.state_dict(),
    }


def _smoke_branch(branch_id: str, *, observed: int) -> HypothesisBranch:
    return HypothesisBranch(
        branch_id=branch_id,
        claim=f"fixture claim {branch_id}",
        transitions=[
            CheckableTransition(
                step=0,
                operation="check two plus two",
                check="exact",
                expected=4,
                observed=observed,
                evidence_ids=[f"receipt-{branch_id}"],
            )
        ],
    )


def _smoke_transition_evidence(
    branch: HypothesisBranch, model_identity: ModelIdentity
) -> EvidenceItem:
    transition = branch.transitions[0]
    receipt = TransitionReceipt(
        workspace_id="prometheus-smoke",
        model_identity_sha256=canonical_sha256(model_identity),
        branch_id=branch.branch_id,
        claim_sha256=hashlib.sha256(branch.claim.encode()).hexdigest(),
        operation=transition.operation,
        check=transition.check,
        expected=transition.expected,
        observed=transition.observed,
        tolerance=transition.tolerance,
        issued_by="synthetic-smoke-verifier",
    )
    text = receipt.canonical_text()
    evidence_id = transition.evidence_ids[0]
    return EvidenceItem(
        evidence_id=evidence_id,
        source_uri=f"urn:smoke:{evidence_id}",
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        acquired_at=datetime(2026, 9, 6, tzinfo=UTC),
        license_id="test-only",
        acl_labels={"public"},
        text=text,
    )


class _RecordingExecutor:
    guarantees_idempotency = True

    def __init__(self) -> None:
        self.calls = 0

    def execute(
        self, tool_id: str, arguments: Mapping[str, Any], transaction_id: str
    ) -> Mapping[str, Any]:
        del tool_id, transaction_id
        self.calls += 1
        return {"echo": arguments["value"]}


def _run_perseus(steps: int) -> tuple[TrainingMetric, dict[str, bool], dict[str, Any]]:
    action = TrainableActionPolicy(state_dim=5, tool_count=3, argument_dim=2, hidden_dim=16)
    recovery = TrainableRecoveryPolicy(failure_dim=4, tool_count=3, hidden_dim=16)
    action_states = torch.tensor(
        [[1.0, 0.0, 0.2, 0.1, 0.3], [0.0, 1.0, 0.4, 0.2, 0.1], [0.2] * 5]
    )
    failure_states = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )

    def loss() -> Tensor:
        tool_logits, arguments = action(action_states)
        class_logits, next_tools = recovery(failure_states)
        return action_policy_loss(
            tool_logits,
            torch.tensor([0, 1, 2]),
            arguments,
            torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        ) + recovery_policy_loss(
            class_logits,
            torch.tensor(
                [RecoveryClass.RETRYABLE, RecoveryClass.COMPENSATABLE, RecoveryClass.FATAL]
            ),
            next_tools,
            torch.tensor([1, 2, 0]),
        )

    metric = _optimize([action, recovery], loss, steps=steps, learning_rate=0.02)
    definition = ToolDefinition(
        tool_id="smoke.echo",
        schema_version="1",
        arguments={"value": ArgumentRule(kind=ArgumentKind.STRING)},
        required_capabilities={"smoke.execute"},
        required_preconditions={"fixture-ready"},
        approval_required=True,
    )
    envelope = ActionEnvelope(
        tool_id="smoke.echo",
        schema_version="1",
        arguments={"value": "ok"},
        preconditions=["fixture-ready"],
        idempotency_key="smoke:perseus:1",
    )
    now = datetime(2026, 9, 6, tzinfo=UTC)
    approval = ActionApproval(
        approval_id=f"approval_{'c' * 16}",
        action_sha256=action_sha256(envelope),
        issued_by="smoke-reviewer",
        expires_at=datetime(2027, 9, 6, tzinfo=UTC),
    )
    manager = TransactionManager(
        CapabilityKernel(
            [definition], trusted_approvals=[approval], clock=lambda: now
        )
    )
    denied = False
    try:
        manager.prepare(envelope, ExecutionContext(satisfied_preconditions={"fixture-ready"}))
    except AuthorizationError:
        denied = True
    snapshot = manager.prepare(
        envelope,
        ExecutionContext(
            capabilities={"smoke.execute"},
            satisfied_preconditions={"fixture-ready"},
            approval_ids={approval.approval_id},
        ),
    )
    executor = _RecordingExecutor()
    receipt = manager.commit(snapshot.transaction_id, executor)
    replay = manager.commit(snapshot.transaction_id, executor)
    checks = {
        "unauthorized_action_rejected": denied,
        "authorized_transaction_committed": receipt.output == {"echo": "ok"},
        "idempotent_replay_executes_once": replay.replayed and executor.calls == 1,
    }
    return metric, checks, {
        "action_policy": action.state_dict(),
        "recovery_policy": recovery.state_dict(),
    }


def _run_kronos(steps: int) -> tuple[TrainingMetric, dict[str, bool], dict[str, Any]]:
    config = KronosConfig(
        feature_dim=4,
        hidden_dim=16,
        output_dim=1,
        horizons=2,
        plan_steps=2,
        action_count=3,
        attention_heads=4,
    )
    model = KronosTemporalPlanner(config)
    events = torch.randn(4, 6, 4)
    time_deltas = torch.tensor([[0.1, 0.5, 1.0, 0.2, 2.0, 0.3]]).repeat(4, 1)
    mask = torch.ones(4, 6, dtype=torch.bool)
    forecasts = torch.stack((events[:, -1, 0], events[:, -1, 1]), dim=1).unsqueeze(-1)
    action_targets = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]])
    metric = _optimize(
        [model],
        lambda: kronos_loss(
            model(events, time_deltas, mask), forecasts, action_targets
        ).total,
        steps=steps,
        learning_rate=0.01,
    )
    with TemporaryDirectory(prefix="olympus-kronos-smoke-") as temporary:
        manager = TemporalAdapterManager(Path(temporary))
        decision = manager.promote(
            manager.root / "missing.pt",
            AdaptationMetrics(
                candidate_sha256=hashlib.sha256(b"missing-candidate").hexdigest(),
                candidate_loss=1.0,
                baseline_loss=1.0,
                forgetting=0.0,
                evaluation_report_sha256=hashlib.sha256(b"untrusted-evaluation").hexdigest(),
                rollback_receipt_sha256=hashlib.sha256(b"untrusted-rollback").hexdigest(),
                deletion_replay_receipt_sha256=hashlib.sha256(
                    b"untrusted-deletion-replay"
                ).hexdigest(),
            ),
        )
    checks = {
        "forecast_shape_valid": model(events, time_deltas, mask).forecast_mean.shape
        == forecasts.shape,
        "promotion_gate_fails_closed": not decision.promoted and len(decision.reasons) >= 3,
    }
    return metric, checks, {"temporal_planner": model.state_dict(), "config": config.model_dump()}


def _run_aion(steps: int) -> tuple[TrainingMetric, dict[str, bool], dict[str, Any]]:
    config = AionRouterConfig(state_dim=8, hidden_dim=16)
    router = AionRouter(config)
    stages = [ResearchStage.QUESTION, ResearchStage.AUDIT]
    features = torch.tensor([[0.1] * 8, [0.2, 0.1, 0.3, 0.0, 0.4, 0.2, 0.1, 0.5]])
    mask = router.allowed_mask(stages)
    indexes = {stage: index for index, stage in enumerate(router.stages)}
    targets = torch.tensor(
        [indexes[ResearchStage.EVIDENCE_MAP], indexes[ResearchStage.STOP]]
    )
    metric = _optimize(
        [router],
        lambda: aion_router_loss(
            router(features, mask), targets, torch.tensor([0.4, 0.2])
        ).total,
        steps=steps,
        learning_rate=0.02,
    )
    protocol = FrozenProtocol(
        protocol_id="smoke-protocol",
        version=1,
        question="Does the deterministic controller preserve its protocol chain?",
        hypothesis_ids=["fixture-hypothesis"],
        evidence_ids=["fixture-evidence"],
        procedure=["map", "freeze", "execute", "audit"],
        falsification_criterion="Any invalid hash or unauthorized execution fails the smoke.",
        frozen_at=datetime(2026, 9, 6, tzinfo=UTC),
    )
    now = datetime(2026, 9, 6, tzinfo=UTC)
    approval = Approval(
        approval_id=f"approval_{'b' * 16}",
        authority=AuthorityKind.HUMAN,
        actor="smoke-reviewer",
        scope="execute",
        run_id=f"aion_{'a' * 16}",
        protocol_sha256=protocol.sha256,
        expected_head_sha256="0" * 64,
    )
    controller = AionController(clock=lambda: now)
    state = controller.create_run(f"aion_{'a' * 16}", protocol)
    for target in (
        ResearchStage.EVIDENCE_MAP,
        ResearchStage.HYPOTHESES,
        ResearchStage.PREREGISTERED_PROTOCOL,
    ):
        state = controller.transition(
            state, target, actor="smoke-controller", authority=AuthorityKind.SYSTEM
        )
    approval = approval.model_copy(
        update={"expected_head_sha256": state.receipts[-1].receipt_sha256}
    )
    controller.register_approval(approval)
    state = controller.transition(
        state,
        ResearchStage.AUTHORIZED_EXECUTION,
        actor="smoke-reviewer",
        authority=AuthorityKind.HUMAN,
        approval=approval,
        tool_calls=1,
    )
    illegal_rejected = False
    try:
        controller.transition(
            state,
            ResearchStage.PROMOTION_CANDIDATE,
            actor="model",
            authority=AuthorityKind.MODEL,
        )
    except ValueError:
        illegal_rejected = True
    checks = {
        "receipt_chain_valid": controller.verify_chain(state),
        "router_masks_illegal_targets": bool(
            torch.isneginf(router(features, mask).logits[0]).any()
        ),
        "illegal_promotion_transition_rejected": illegal_rejected,
    }
    return metric, checks, {"router": router.state_dict(), "config": config.model_dump()}


def _optimize(
    modules: list[nn.Module],
    loss_function: Callable[[], Tensor],
    *,
    steps: int,
    learning_rate: float,
) -> TrainingMetric:
    parameters = [parameter for module in modules for parameter in module.parameters()]
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    with torch.no_grad():
        initial = float(loss_function())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function()
        torch.autograd.backward(loss)
        optimizer.step()
    with torch.no_grad():
        final = float(loss_function())
    updated = any(
        not torch.equal(old, new.detach()) for old, new in zip(before, trainable, strict=True)
    )
    finite = bool(torch.isfinite(torch.tensor([initial, final])).all())
    return TrainingMetric(
        initial_loss=initial,
        final_loss=final,
        learned_parameters=sum(parameter.numel() for parameter in trainable),
        gradient_update_observed=updated,
        loss_decreased=final < initial,
        finite=finite,
    )


def _save_checkpoint(
    path: Path,
    *,
    family: str,
    implementation_classes: list[str],
    state_dicts: dict[str, Any],
    seed: int,
    steps: int,
    source_sha256: str,
) -> CheckpointArtifact:
    payload = {
        "format_version": 1,
        "family": family,
        "implementation_classes": implementation_classes,
        "status": _SMOKE_STATUS,
        "qualifying_checkpoint": False,
        "promoted": False,
        "fixture_scope": "deterministic_synthetic_contract_fixtures",
        "seed": seed,
        "optimization_steps": steps,
        "source_sha256": source_sha256,
        "state_dicts": state_dicts,
    }
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return CheckpointArtifact(
        relative_path=path.name,
        sha256=_file_digest(path),
        bytes=path.stat().st_size,
    )


def _write_json_atomic(path: Path, manifest: FamilySmokeManifest) -> None:
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


def _source_digest() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _MODEL_SOURCE_NAMES:
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
