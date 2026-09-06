import hashlib
from datetime import UTC, datetime

import pytest
import torch
from pydantic import ValidationError

from olympus.models.prometheus import (
    BranchAssessment,
    CheckableTransition,
    HypothesisBranch,
    PrometheusSelector,
    TrainableBranchProposer,
    TrainableProcessVerifier,
    TransitionReceipt,
    prometheus_training_loss,
    transition_from_receipt,
)
from olympus.models.substrate import (
    EvidenceItem,
    ModelIdentity,
    PermissionSet,
    WorkspaceState,
    canonical_sha256,
)


def _receipt_evidence(
    evidence_id: str,
    *,
    observed: int,
    operation: str = "evaluate a held-out equality",
) -> EvidenceItem:
    receipt = TransitionReceipt(
        workspace_id="prometheus-test",
        model_identity_sha256=canonical_sha256(_model_identity()),
        branch_id=evidence_id.removeprefix("receipt-"),
        claim_sha256=hashlib.sha256(
            f"claim {evidence_id.removeprefix('receipt-')}".encode()
        ).hexdigest(),
        operation=operation,
        check="exact",
        expected=4,
        observed=observed,
        issued_by="test-verifier",
    )
    text = receipt.canonical_text()
    return EvidenceItem(
        evidence_id=evidence_id,
        source_uri=f"urn:test:{evidence_id}",
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        acquired_at=datetime(2026, 9, 6, tzinfo=UTC),
        license_id="test-only",
        acl_labels={"public"},
        text=text,
    )


def _workspace(
    *evidence: EvidenceItem, granted_acl: set[str] | None = None
) -> WorkspaceState:
    return WorkspaceState(
        workspace_id="prometheus-test",
        objective="Select only a checked, evidence-backed branch.",
        evidence=list(evidence),
        permissions=PermissionSet(granted_acl_labels=granted_acl or set()),
        model_identity=_model_identity(),
    )


def _model_identity() -> ModelIdentity:
    return ModelIdentity(
        model_id="prometheus-test",
        family="Prometheus",
        version="test",
        base_model_id="local/test",
        base_revision="test",
        base_sha256="0" * 64,
    )


def _branch(branch_id: str, *, observed: int, evidence: bool = True) -> HypothesisBranch:
    return HypothesisBranch(
        branch_id=branch_id,
        claim=f"claim {branch_id}",
        transitions=[
            CheckableTransition(
                step=0,
                operation="evaluate a held-out equality",
                check="exact",
                expected=4,
                observed=observed,
                evidence_ids=[f"receipt-{branch_id}"] if evidence else [],
            )
        ],
    )


def test_prometheus_proposer_and_verifier_receive_gradient_updates() -> None:
    torch.manual_seed(7)
    proposer = TrainableBranchProposer(context_dim=4, branch_dim=3)
    verifier = TrainableProcessVerifier(context_dim=4, transition_dim=2)
    optimizer = torch.optim.SGD(
        list(proposer.parameters()) + list(verifier.parameters()), lr=0.2
    )
    context = torch.tensor([0.2, -0.1, 0.7, 0.3])
    branch_features = torch.tensor([[0.0, 0.1, 0.2], [0.8, 0.7, 0.9], [-0.2, 0.3, 0.1]])
    transition_features = torch.tensor([[0.8, 0.2], [-0.5, 0.7], [0.4, 0.6]])
    parameters = [*proposer.parameters(), *verifier.parameters()]
    before = [parameter.detach().clone() for parameter in parameters]

    optimizer.zero_grad()
    proposer_logits = proposer(context, branch_features)
    verifier_logits = verifier(context, transition_features)
    loss = prometheus_training_loss(
        proposer_logits=proposer_logits,
        target_branch=1,
        verifier_logits=verifier_logits,
        verifier_labels=torch.tensor([1.0, 0.0, 1.0]),
        branch_embeddings=branch_features,
    )
    torch.autograd.backward(loss)
    optimizer.step()

    after = [*proposer.parameters(), *verifier.parameters()]
    assert loss.item() > 0.0
    assert all(parameter.grad is not None for parameter in after)
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after, strict=True))


def test_proposer_enforces_branch_budget() -> None:
    proposer = TrainableBranchProposer(context_dim=2, branch_dim=2)
    candidates = [_branch("a", observed=4), _branch("b", observed=4), _branch("c", observed=4)]
    result = proposer.propose(
        torch.zeros(2),
        candidates,
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        branch_budget=2,
    )
    assert len(result.branches) == 2
    assert result.branch_budget == 2
    assert len({branch.branch_id for branch in result.branches}) == 2


def test_selector_rejects_failed_checks_and_can_abstain() -> None:
    valid = _branch("valid", observed=4)
    invalid = _branch("invalid", observed=5)
    valid_evidence = _receipt_evidence("receipt-valid", observed=4)
    invalid_evidence = _receipt_evidence("receipt-invalid", observed=5)
    selector = PrometheusSelector(
        minimum_score=0.7,
        trusted_receipt_sha256={
            valid_evidence.content_sha256,
            invalid_evidence.content_sha256,
        },
    )
    result = selector.select(
        [
            BranchAssessment(branch=invalid, proposer_score=0.99, verifier_scores=[0.99]),
            BranchAssessment(branch=valid, proposer_score=0.8, verifier_scores=[0.9]),
        ],
        _workspace(valid_evidence, invalid_evidence),
    )
    abstained = selector.select(
        [BranchAssessment(branch=invalid, proposer_score=0.99, verifier_scores=[0.99])],
        _workspace(invalid_evidence),
    )
    assert result.decision == "selected"
    assert result.branch_id == "valid"
    assert abstained.decision == "abstained"
    assert abstained.branch_id is None

    unresolved = selector.select(
        [BranchAssessment(branch=valid, proposer_score=0.99, verifier_scores=[0.99])],
        _workspace(),
    )
    assert unresolved.decision == "abstained"

    unrelated_evidence = _receipt_evidence(
        "receipt-valid", observed=4, operation="measure bird migration"
    )
    unrelated = PrometheusSelector(
        trusted_receipt_sha256={unrelated_evidence.content_sha256}
    ).select(
        [BranchAssessment(branch=valid, proposer_score=0.99, verifier_scores=[0.99])],
        _workspace(unrelated_evidence),
    )
    assert unrelated.decision == "abstained"


def test_transition_is_derived_from_canonical_receipt_evidence() -> None:
    evidence = _receipt_evidence("receipt-derived", observed=4)
    transition = transition_from_receipt(step=0, evidence=evidence)
    assert transition.operation == "evaluate a held-out equality"
    assert transition.observed == 4
    assert transition.evidence_ids == ["receipt-derived"]


def test_transition_contract_rejects_uncheckable_values_and_unordered_steps() -> None:
    with pytest.raises(ValidationError, match="numeric checks require numeric"):
        CheckableTransition(
            step=0,
            operation="bad numeric check",
            check="numeric",
            expected="four",
            observed=4,
        )
    with pytest.raises(ValidationError, match="contiguous"):
        HypothesisBranch(
            branch_id="bad",
            claim="unordered",
            transitions=[
                CheckableTransition(
                    step=1,
                    operation="out of order",
                    check="boolean",
                    expected=True,
                    observed=True,
                )
            ],
        )


def test_selector_revalidates_mutated_nested_assessments() -> None:
    branch = _branch("mutable", observed=4)
    assessment = BranchAssessment(
        branch=branch,
        proposer_score=0.9,
        verifier_scores=[0.9],
    )
    assessment.branch.transitions.clear()

    with pytest.raises(ValidationError, match="at least 1 item"):
        PrometheusSelector().select([assessment], _workspace())
