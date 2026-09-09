from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

import pytest
import torch
from pydantic import ValidationError

from olympus.models.aion import (
    AionComparison,
    AionController,
    AionRouter,
    AionRouterConfig,
    AionRunState,
    Approval,
    AuditFinding,
    AuthorityKind,
    FrozenProtocol,
    ResearchStage,
    aion_router_loss,
    learned_router_adds_value,
)
from olympus.models.kronos import (
    AdaptationMetrics,
    EventKind,
    KronosConfig,
    KronosTemporalPlanner,
    TemporalAdapterManager,
    TemporalEvent,
    adaptation_evidence_binding_sha256,
    kronos_loss,
    load_staged_kronos,
    tensorize_event_stream,
)


class _AdaptationEvidenceValues(TypedDict):
    candidate_sha256: str
    candidate_loss: float
    baseline_loss: float
    forgetting: float


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _event(index: int, *, environment: str = "env-v1") -> TemporalEvent:
    return TemporalEvent(
        event_id=f"event-{index}",
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index),
        kind=EventKind.INTERVENTION if index == 1 else EventKind.OBSERVATION,
        features=[float(index), float(index % 2), 1.0, 0.25],
        environment_version=environment,
        source_sha256=_digest(f"event-{index}"),
        intervention="set-control" if index == 1 else None,
    )


def test_temporal_events_and_tensorization_fail_closed() -> None:
    stream = [_event(index) for index in range(4)]
    features, deltas, mask, metadata = tensorize_event_stream([stream], feature_dim=4)
    assert features.shape == (1, 4, 4)
    assert deltas.shape == mask.shape == (1, 4)
    assert mask.all()
    assert torch.all(deltas > 0)
    assert metadata.event_ids == [[f"event-{index}" for index in range(4)]]

    with pytest.raises(ValueError, match="chronological"):
        tensorize_event_stream([[stream[1], stream[0]]], feature_dim=4)
    with pytest.raises(ValueError, match="environment versions"):
        tensorize_event_stream([[stream[0], _event(1, environment="env-v2")]], feature_dim=4)
    with pytest.raises(ValueError, match="unique"):
        tensorize_event_stream([[stream[0]], [stream[0]]], feature_dim=4)
    with pytest.raises(ValidationError, match="must identify"):
        TemporalEvent(
            event_id="invalid",
            occurred_at=datetime.now(UTC),
            kind=EventKind.INTERVENTION,
            features=[1.0],
            environment_version="env",
            source_sha256=_digest("invalid"),
        )
    with pytest.raises(ValidationError, match="timezone-aware"):
        TemporalEvent(
            event_id="naive",
            occurred_at=datetime(2026, 1, 1),
            kind=EventKind.OBSERVATION,
            features=[1.0],
            environment_version="env",
            source_sha256=_digest("naive"),
        )
    with pytest.raises(ValidationError, match="finite"):
        TemporalEvent(
            event_id="nonfinite",
            occurred_at=datetime.now(UTC),
            kind=EventKind.OBSERVATION,
            features=[float("nan")],
            environment_version="env",
            source_sha256=_digest("nonfinite"),
        )


def test_kronos_forecasting_planning_and_gradient_training_are_real() -> None:
    torch.manual_seed(71)
    config = KronosConfig(
        feature_dim=4,
        hidden_dim=16,
        output_dim=1,
        horizons=2,
        plan_steps=3,
        action_count=4,
        attention_heads=4,
    )
    model = KronosTemporalPlanner(config)
    events = torch.randn(5, 6, 4)
    deltas = torch.full((5, 6), 0.5)
    mask = torch.ones(5, 6, dtype=torch.bool)
    forecast_targets = events[:, -1:, :1].repeat(1, 2, 1)
    action_targets = torch.zeros(5, 3, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    with torch.no_grad():
        initial = kronos_loss(
            model(events, deltas, mask), forecast_targets, action_targets
        ).total.item()
    for _ in range(24):
        optimizer.zero_grad()
        losses = kronos_loss(model(events, deltas, mask), forecast_targets, action_targets)
        torch.autograd.backward(losses.total)
        optimizer.step()
    with torch.no_grad():
        output = model(events, deltas, mask)
        final = kronos_loss(output, forecast_targets, action_targets).total.item()
    assert output.forecast_mean.shape == (5, 2, 1)
    assert output.action_logits.shape == (5, 3, 4)
    assert final < initial * 0.35

    with pytest.raises(ValueError, match="negative"):
        model(events, -deltas, mask)
    with pytest.raises(ValueError, match="unmasked event"):
        model(events, deltas, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="non-negative"):
        kronos_loss(
            model(events, deltas, mask),
            forecast_targets,
            action_targets,
            forecast_weight=-1.0,
        )
    with pytest.raises(ValueError, match="integer tensor"):
        kronos_loss(
            model(events, deltas, mask),
            forecast_targets,
            action_targets.float(),
        )


def test_kronos_adapter_promotion_requires_improvement_retention_and_rollback(
    tmp_path: Path,
) -> None:
    config = KronosConfig(feature_dim=3, hidden_dim=8, attention_heads=2)
    model = KronosTemporalPlanner(config)
    root = tmp_path / "adapters"
    candidate = TemporalAdapterManager(root).stage(
        model, {"base": "test-only", "status": "experimental"}
    )
    candidate_sha256 = _digest_file(candidate)
    accepted_values: _AdaptationEvidenceValues = {
        "candidate_sha256": candidate_sha256,
        "candidate_loss": 0.7,
        "baseline_loss": 0.9,
        "forgetting": 0.01,
    }
    evidence_hashes = {
        adaptation_evidence_binding_sha256(kind, **accepted_values)
        for kind in ("evaluation", "rollback", "deletion_replay")
    }
    manager = TemporalAdapterManager(
        root,
        maximum_forgetting=0.03,
        trusted_evidence_sha256=evidence_hashes,
    )

    rejected = manager.promote(
        candidate,
        AdaptationMetrics(
            candidate_sha256=candidate_sha256,
            candidate_loss=0.8,
            baseline_loss=0.9,
            forgetting=0.04,
            evaluation_report_sha256=_digest("unbound-evaluation"),
            rollback_receipt_sha256=_digest("untrusted-rollback"),
            deletion_replay_receipt_sha256=_digest("unbound-deletion"),
        ),
    )
    assert not rejected.promoted
    assert any("forgetting" in reason for reason in rejected.reasons)
    assert any("rollback" in reason for reason in rejected.reasons)

    accepted = manager.promote(
        candidate,
        AdaptationMetrics(
            candidate_sha256=candidate_sha256,
            candidate_loss=accepted_values["candidate_loss"],
            baseline_loss=accepted_values["baseline_loss"],
            forgetting=accepted_values["forgetting"],
            evaluation_report_sha256=adaptation_evidence_binding_sha256(
                "evaluation", **accepted_values
            ),
            rollback_receipt_sha256=adaptation_evidence_binding_sha256(
                "rollback", **accepted_values
            ),
            deletion_replay_receipt_sha256=adaptation_evidence_binding_sha256(
                "deletion_replay", **accepted_values
            ),
        ),
    )
    assert accepted.promoted
    assert accepted.checkpoint_sha256 == _digest_file(candidate)
    assert accepted.checkpoint_sha256 is not None
    loaded, metadata = load_staged_kronos(
        tmp_path / "adapters/promoted.pt",
        config,
        expected_sha256=accepted.checkpoint_sha256,
    )
    assert metadata["status"] == "experimental"
    assert set(loaded.state_dict()) == set(model.state_dict())

    with pytest.raises(ValueError, match="expected SHA-256"):
        load_staged_kronos(
            tmp_path / "adapters/promoted.pt",
            config,
            expected_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="finite JSON"):
        manager.stage(model, {"bad": float("nan")})

    replacement = KronosTemporalPlanner(config)
    swapped = manager.stage(replacement, {"base": "substituted"})
    swapped_decision = manager.promote(
        swapped,
        AdaptationMetrics(
            candidate_sha256=candidate_sha256,
            candidate_loss=0.7,
            baseline_loss=0.9,
            forgetting=0.01,
            evaluation_report_sha256=adaptation_evidence_binding_sha256(
                "evaluation", **accepted_values
            ),
            rollback_receipt_sha256=adaptation_evidence_binding_sha256(
                "rollback", **accepted_values
            ),
            deletion_replay_receipt_sha256=adaptation_evidence_binding_sha256(
                "deletion_replay", **accepted_values
            ),
        ),
    )
    assert not swapped_decision.promoted
    assert any("bound to the supplied metrics" in reason for reason in swapped_decision.reasons)


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _protocol() -> FrozenProtocol:
    return FrozenProtocol(
        protocol_id="protocol-v1",
        version=1,
        question="Does the bounded controller preserve protocol integrity?",
        hypothesis_ids=["hypothesis-1"],
        evidence_ids=["evidence-1"],
        procedure=["freeze", "execute", "audit"],
        falsification_criterion="Any unauthorized transition falsifies the control design.",
        frozen_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _approval(
    protocol: FrozenProtocol,
    scope: str,
    authority: AuthorityKind,
    *,
    run_id: str,
    expected_head_sha256: str,
    marker: str = "a",
) -> Approval:
    return Approval(
        approval_id=f"approval_{marker * 16}",
        authority=authority,
        actor="reviewer",
        scope=scope,
        run_id=run_id,
        protocol_sha256=protocol.sha256,
        expected_head_sha256=expected_head_sha256,
        expires_at=datetime(2027, 1, 1, tzinfo=UTC),
    )


def test_aion_controller_enforces_authority_audit_and_hash_chain() -> None:
    protocol = _protocol()
    run_id = f"aion_{'b' * 16}"
    fixed_now = datetime(2026, 2, 1, tzinfo=UTC)
    controller = AionController(
        clock=lambda: fixed_now,
    )
    state = controller.create_run(run_id, protocol)

    for target in (
        ResearchStage.EVIDENCE_MAP,
        ResearchStage.HYPOTHESES,
        ResearchStage.PREREGISTERED_PROTOCOL,
    ):
        state = controller.transition(
            state,
            target,
            actor="controller",
            authority=AuthorityKind.SYSTEM,
        )

    execution_approval = _approval(
        protocol,
        "execute",
        AuthorityKind.HUMAN,
        run_id=run_id,
        expected_head_sha256=state.receipts[-1].receipt_sha256,
    )
    registered_execution = execution_approval.model_copy(deep=True)
    controller.register_approval(registered_execution)
    registered_execution.scope = "mutated-after-registration"

    untrusted_execution = execution_approval.model_copy(
        update={"approval_id": f"approval_{'9' * 16}"}
    )
    with pytest.raises(PermissionError, match="trusted approval"):
        controller.preflight_approval(
            state,
            ResearchStage.AUTHORIZED_EXECUTION,
            approval=untrusted_execution,
        )
    controller.preflight_approval(
        state,
        ResearchStage.AUTHORIZED_EXECUTION,
        approval=execution_approval,
    )
    assert state.stage is ResearchStage.PREREGISTERED_PROTOCOL
    assert controller.verify_chain(state)

    with pytest.raises(PermissionError, match="approval"):
        controller.transition(
            state,
            ResearchStage.AUTHORIZED_EXECUTION,
            actor="model",
            authority=AuthorityKind.MODEL,
        )
    with pytest.raises(PermissionError, match="approval"):
        controller.transition(
            state,
            ResearchStage.AUTHORIZED_EXECUTION,
            actor="model",
            authority=AuthorityKind.MODEL,
            approval=_approval(
                protocol,
                "execute",
                AuthorityKind.MODEL,
                run_id=run_id,
                expected_head_sha256=state.receipts[-1].receipt_sha256,
                marker="c",
            ),
        )
    with pytest.raises(PermissionError, match="actor and authority"):
        controller.transition(
            state,
            ResearchStage.AUTHORIZED_EXECUTION,
            actor="different-operator",
            authority=AuthorityKind.HUMAN,
            approval=execution_approval,
        )
    with pytest.raises(ValueError, match="outside the frozen protocol"):
        controller.transition(
            state,
            ResearchStage.AUTHORIZED_EXECUTION,
            actor="reviewer",
            authority=AuthorityKind.HUMAN,
            approval=execution_approval,
            evidence_ids=["post-hoc-evidence"],
        )

    state = controller.transition(
        state,
        ResearchStage.AUTHORIZED_EXECUTION,
        actor="reviewer",
        authority=AuthorityKind.HUMAN,
        approval=execution_approval,
        tool_calls=2,
    )
    state = controller.transition(
        state,
        ResearchStage.AUDIT,
        actor="pantheon",
        authority=AuthorityKind.SYSTEM,
    )
    promotion_approval = _approval(
        protocol,
        "promote",
        AuthorityKind.HUMAN,
        run_id=run_id,
        expected_head_sha256=state.receipts[-1].receipt_sha256,
        marker="b",
    )
    passed_audit = AuditFinding(
        audit_id=f"audit_{'d' * 16}",
        run_id=run_id,
        protocol_sha256=protocol.sha256,
        audited_head_sha256=state.receipts[-1].receipt_sha256,
        evidence_artifact_sha256=[_digest("aion-passed-audit-report")],
        passed=True,
        claim_precision=1.0,
        evidence_coverage=1.0,
        protected_data_leakage=False,
        unauthorized_effects=False,
    )
    registered_promotion = promotion_approval.model_copy(deep=True)
    registered_audit = passed_audit.model_copy(deep=True)
    controller.register_approval(registered_promotion)
    controller.register_audit(registered_audit)
    registered_promotion.scope = "mutated-after-registration"
    registered_audit.passed = False
    failed_audit = AuditFinding(
        audit_id=f"audit_{'c' * 16}",
        run_id=run_id,
        protocol_sha256=protocol.sha256,
        audited_head_sha256=state.receipts[-1].receipt_sha256,
        evidence_artifact_sha256=[_digest("aion-failed-audit-report")],
        passed=False,
        claim_precision=0.5,
        evidence_coverage=1.0,
        protected_data_leakage=False,
        unauthorized_effects=False,
    )
    with pytest.raises(PermissionError, match="unsafe audit"):
        controller.transition(
            state,
            ResearchStage.PROMOTION_CANDIDATE,
            actor="reviewer",
            authority=AuthorityKind.HUMAN,
            approval=promotion_approval,
            audit=failed_audit,
        )

    state = controller.transition(
        state,
        ResearchStage.PROMOTION_CANDIDATE,
        actor="reviewer",
        authority=AuthorityKind.HUMAN,
        approval=promotion_approval,
        audit=passed_audit,
    )
    assert state.stage is ResearchStage.PROMOTION_CANDIDATE
    assert state.tool_calls_used == 2
    assert controller.verify_chain(state)

    tampered = state.model_copy(deep=True)
    tampered.receipts[-1].receipt_sha256 = "f" * 64
    assert not controller.verify_chain(tampered)


def test_aion_rejects_protocols_claiming_a_future_freeze_time() -> None:
    fixed_now = datetime(2026, 2, 1, tzinfo=UTC)
    future_protocol = _protocol().model_copy(
        update={"frozen_at": datetime(2026, 2, 2, tzinfo=UTC)}
    )

    with pytest.raises(ValueError, match="freeze time cannot be in the future"):
        AionController(clock=lambda: fixed_now).create_run(
            f"aion_{'9' * 16}", future_protocol
        )


def test_aion_rejects_unregistered_forged_and_inconsistent_state() -> None:
    protocol = _protocol()
    fixed_now = datetime(2026, 2, 1, tzinfo=UTC)
    controller = AionController(clock=lambda: fixed_now)
    with pytest.raises(ValidationError, match="receipt-free run"):
        AionRunState(
            run_id=f"aion_{'0' * 16}",
            protocol=protocol,
            stage=ResearchStage.AUDIT,
        )
    unregistered = controller.create_run(f"aion_{'e' * 16}", protocol).model_copy(
        update={"run_id": f"aion_{'f' * 16}"}
    )
    with pytest.raises(ValueError, match="chain or authority"):
        controller.transition(
            unregistered,
            ResearchStage.EVIDENCE_MAP,
            actor="forger",
            authority=AuthorityKind.MODEL,
        )

    state = controller.create_run(f"aion_{'1' * 16}", protocol)
    with pytest.raises(ValidationError, match="finite JSON"):
        controller.transition(
            state,
            ResearchStage.EVIDENCE_MAP,
            actor="controller",
            authority=AuthorityKind.SYSTEM,
            metadata={"when": datetime(2026, 2, 1, tzinfo=UTC)},
        )
    substituted_protocol = protocol.model_copy(update={"question": "A substituted protocol"})
    substituted = state.model_copy(update={"protocol": substituted_protocol})
    assert not controller.verify_chain(substituted)
    with pytest.raises(ValueError, match="chain or authority"):
        controller.transition(
            substituted,
            ResearchStage.EVIDENCE_MAP,
            actor="forger",
            authority=AuthorityKind.MODEL,
        )

    inconsistent = state.model_copy(update={"stage": ResearchStage.AUDIT})
    assert not controller.verify_chain(inconsistent)
    with pytest.raises(ValueError, match="chain or authority"):
        controller.transition(
            inconsistent,
            ResearchStage.PROMOTION_CANDIDATE,
            actor="forger",
            authority=AuthorityKind.MODEL,
        )


def test_aion_router_masks_illegal_transitions_and_trains() -> None:
    torch.manual_seed(73)
    router = AionRouter(AionRouterConfig(state_dim=8, hidden_dim=12))
    stages = [ResearchStage.QUESTION, ResearchStage.AUDIT]
    features = torch.randn(2, 8)
    mask = router.allowed_mask(stages)
    output = router(features, mask)
    stage_index = {stage: index for index, stage in enumerate(router.stages)}
    assert torch.isneginf(output.logits[0, stage_index[ResearchStage.AUDIT]])

    targets = torch.tensor(
        [stage_index[ResearchStage.EVIDENCE_MAP], stage_index[ResearchStage.STOP]]
    )
    optimizer = torch.optim.Adam(router.parameters(), lr=0.02)
    initial = aion_router_loss(output, targets, torch.tensor([0.4, 0.2])).total.item()
    for _ in range(20):
        optimizer.zero_grad()
        loss = aion_router_loss(
            router(features, mask), targets, torch.tensor([0.4, 0.2])
        ).total
        torch.autograd.backward(loss)
        optimizer.step()
    final = aion_router_loss(
        router(features, mask), targets, torch.tensor([0.4, 0.2])
    ).total.item()
    assert final < initial

    forbidden = targets.clone()
    forbidden[0] = stage_index[ResearchStage.AUDIT]
    with pytest.raises(ValueError, match="forbidden"):
        aion_router_loss(router(features, mask), forbidden, torch.tensor([0.4, 0.2]))


def test_aion_learned_router_falsification_rule() -> None:
    matched = AionComparison(
        learned_valid_experiment_rate=0.8,
        deterministic_valid_experiment_rate=0.8,
        learned_false_promotion_rate=0.01,
        deterministic_false_promotion_rate=0.01,
        learned_budget=10,
        deterministic_budget=10,
    )
    assert not learned_router_adds_value(matched)
    improved = matched.model_copy(update={"learned_valid_experiment_rate": 0.9})
    assert learned_router_adds_value(improved)
