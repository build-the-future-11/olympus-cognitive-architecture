from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
import torch
from pydantic import ValidationError

from olympus.models.substrate import (
    ActionOutput,
    AdapterCausalDecoder,
    AuthorizedWorkspaceView,
    Claim,
    ClaimOutput,
    ClaimStatus,
    DecoderConfig,
    EvidenceItem,
    ModelIdentity,
    ModelLifecycle,
    PermissionSet,
    PlanStep,
    TextOutput,
    ToolInvocation,
    ToolSpec,
    WorkspaceEvent,
    WorkspaceState,
    authorized_workspace_view,
    canonical_json_bytes,
    canonical_sha256,
    causal_language_model_loss,
    parse_output_envelope,
    tool_invocation_sha256,
    validate_envelope_against_workspace,
    validate_workspace_snapshot,
)

_ZERO_HASH = "0" * 64


def _evidence(evidence_id: str = "evidence:1", *, acl: set[str] | None = None) -> EvidenceItem:
    text = "The measured value was 42."
    return EvidenceItem(
        evidence_id=evidence_id,
        source_uri="urn:test:measurement",
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        acquired_at=datetime(2026, 9, 6, tzinfo=UTC),
        license_id="test-only",
        acl_labels=acl or {"public"},
        span_start=0,
        span_end=len(text),
        text=text,
    )


def _workspace(
    *,
    acl: set[str] | None = None,
    approval_bindings: dict[str, str] | None = None,
) -> WorkspaceState:
    evidence = _evidence(acl=acl)
    return WorkspaceState(
        workspace_id="workspace:1",
        objective="Report the governed measurement.",
        evidence=[evidence],
        claims=[
            Claim(
                claim_id="claim:1",
                text="The value is 42.",
                status=ClaimStatus.SUPPORTED,
                support_evidence_ids=[evidence.evidence_id],
                confidence=0.9,
            )
        ],
        plan=[PlanStep(step_id="step:1", description="Read evidence", tool_id="tool:read")],
        tools=[
            ToolSpec(
                tool_id="tool:read",
                schema_version="1",
                argument_schema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                        "mode": {"type": "string", "enum": ["exact", "summary"]},
                    },
                    "required": ["limit", "mode"],
                    "additionalProperties": False,
                },
                required_capabilities={"read"},
                approval_required=True,
            )
        ],
        permissions=PermissionSet(
            granted_acl_labels={"private"},
            allowed_tool_ids={"tool:read"},
            capabilities={"read"},
            approval_bindings=approval_bindings or {},
        ),
        model_identity=ModelIdentity(
            model_id="model:test",
            family="Hermes",
            version="0.0-test",
            base_model_id="local/test-base",
            base_revision="frozen-test-revision",
            base_sha256=_ZERO_HASH,
            lifecycle=ModelLifecycle.IMPLEMENTED,
        ),
    )


def _bound_text(workspace: WorkspaceState) -> TextOutput:
    return TextOutput(
        workspace_sha256=workspace.sha256,
        model_identity_sha256=canonical_sha256(workspace.model_identity),
        text="The value is 42.",
        citation_evidence_ids=["evidence:1"],
        confidence=0.9,
    )


def test_workspace_enforces_hashes_references_and_plan_dag() -> None:
    workspace = _workspace()
    assert len(workspace.sha256) == 64

    with pytest.raises(ValidationError, match="content hash"):
        EvidenceItem(**{**_evidence().model_dump(), "content_sha256": "f" * 64})

    with pytest.raises(ValidationError, match="unknown evidence"):
        WorkspaceState(
            **{
                **workspace.model_dump(),
                "claims": [
                    Claim(
                        claim_id="claim:bad",
                        text="Unsupported",
                        status=ClaimStatus.SUPPORTED,
                        support_evidence_ids=["missing"],
                    )
                ],
            }
        )

    with pytest.raises(ValidationError, match="acyclic"):
        WorkspaceState(
            **{
                **workspace.model_dump(),
                "plan": [
                    PlanStep(step_id="a", description="a", depends_on=["b"]),
                    PlanStep(step_id="b", description="b", depends_on=["a"]),
                ],
            }
        )


def test_workspace_event_log_is_hash_linked_contiguous_and_monotonic() -> None:
    first = WorkspaceEvent.from_payload(
        event_id="event:1",
        sequence=1,
        kind="observation",
        actor="test",
        occurred_at=datetime(2026, 9, 6, tzinfo=UTC),
        payload={"value": 1},
        previous_event_sha256="0" * 64,
    )
    second = WorkspaceEvent.from_payload(
        event_id="event:2",
        sequence=2,
        kind="observation",
        actor="test",
        occurred_at=datetime(2026, 9, 6, 0, 1, tzinfo=UTC),
        payload={"value": 2},
        previous_event_sha256=first.event_sha256,
    )
    base = _workspace()
    workspace = WorkspaceState(
        **{**base.model_dump(), "events": [first, second], "event_sequence": 2}
    )
    assert workspace.events[-1].previous_event_sha256 == first.event_sha256

    with pytest.raises(ValidationError, match="payload hash"):
        WorkspaceEvent(**{**first.model_dump(), "payload": {"value": 99}})
    with pytest.raises(ValidationError, match="finite JSON"):
        WorkspaceEvent.from_payload(
            event_id="event:python-only",
            sequence=1,
            kind="observation",
            actor="test",
            occurred_at=datetime(2026, 9, 6, tzinfo=UTC),
            payload={"when": datetime(2026, 9, 6, tzinfo=UTC)},
            previous_event_sha256="0" * 64,
        )
    with pytest.raises(ValidationError, match="event_sequence"):
        WorkspaceState(**{**base.model_dump(), "events": [first]})

    workspace.events[0].payload["value"] = 99
    with pytest.raises(ValidationError, match="payload hash"):
        validate_workspace_snapshot(workspace)


def test_canonical_hash_sorts_sets_and_output_parser_discriminates() -> None:
    left = PermissionSet(granted_acl_labels={"z", "a"}, capabilities={"two", "one"})
    right = PermissionSet(granted_acl_labels={"a", "z"}, capabilities={"one", "two"})
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_sha256(left) == canonical_sha256(right)

    workspace = _workspace()
    parsed = parse_output_envelope(_bound_text(workspace).model_dump_json())
    assert isinstance(parsed, TextOutput)
    with pytest.raises(ValidationError):
        parse_output_envelope({"kind": "not-an-envelope"})
    with pytest.raises(TypeError, match="string keys"):
        canonical_json_bytes({1: "ambiguous"})


def test_envelope_boundary_revalidates_mutated_nested_claims() -> None:
    workspace = _workspace()
    output = ClaimOutput(
        workspace_sha256=workspace.sha256,
        model_identity_sha256=canonical_sha256(workspace.model_identity),
        claims=[workspace.claims[0].model_copy(deep=True)],
    )

    output.claims[0].support_evidence_ids.clear()

    with pytest.raises(ValidationError, match="supported claims require supporting evidence"):
        validate_envelope_against_workspace(output, workspace)


def test_authorized_workspace_view_removes_private_model_inputs() -> None:
    public_text = "Public evidence."
    private_text = "Secret launch code 991."
    workspace = WorkspaceState(
        workspace_id="workspace:projected",
        objective="Use only authorized evidence.",
        evidence=[
            EvidenceItem(
                evidence_id="evidence:public",
                source_uri="urn:test:public",
                content_sha256=hashlib.sha256(public_text.encode()).hexdigest(),
                acquired_at=datetime(2026, 9, 6, tzinfo=UTC),
                license_id="test-only",
                acl_labels={"public"},
                text=public_text,
            ),
            EvidenceItem(
                evidence_id="evidence:private",
                source_uri="urn:test:private",
                content_sha256=hashlib.sha256(private_text.encode()).hexdigest(),
                acquired_at=datetime(2026, 9, 6, tzinfo=UTC),
                license_id="test-only",
                acl_labels={"private"},
                text=private_text,
            ),
        ],
        model_identity=_workspace().model_identity,
    )
    view = authorized_workspace_view(workspace)

    assert isinstance(view, AuthorizedWorkspaceView)
    assert [item.evidence_id for item in view.evidence] == ["evidence:public"]
    serialized = view.model_dump_json()
    assert private_text not in serialized
    assert "evidence:private" not in serialized


def test_tool_schema_is_prevalidated_and_recursively_enforced() -> None:
    with pytest.raises(ValidationError, match="material-effect tools"):
        ToolSpec(
            tool_id="tool:unsafe-write",
            schema_version="1",
            argument_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            material_effect=True,
        )

    with pytest.raises(ValidationError, match="unsupported schema keywords"):
        ToolSpec(
            tool_id="tool:bad",
            schema_version="1",
            argument_schema={"type": "object", "$ref": "untrusted"},
        )
    with pytest.raises(ValidationError, match="cannot be empty"):
        ToolSpec(
            tool_id="tool:empty-nested-schema",
            schema_version="1",
            argument_schema={
                "type": "object",
                "properties": {"payload": {}},
            },
        )
    with pytest.raises(ValidationError, match="does not match integer"):
        ToolSpec(
            tool_id="tool:boolean-integer-enum",
            schema_version="1",
            argument_schema={
                "type": "object",
                "properties": {"count": {"type": "integer", "enum": [True]}},
            },
        )

    tool = ToolSpec(
        tool_id="tool:nested",
        schema_version="1",
        argument_schema={
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "items": {"type": "number"}}
                    },
                    "required": ["values"],
                    "additionalProperties": False,
                }
            },
            "required": ["payload"],
            "additionalProperties": False,
        },
    )
    base = _workspace()
    workspace = WorkspaceState(
        **{
            **base.model_dump(),
            "tools": [tool],
            "plan": [],
            "permissions": PermissionSet(allowed_tool_ids={"tool:nested"}),
        }
    )
    invocation = ToolInvocation(
        tool_id="tool:nested",
        schema_version="1",
        arguments={"payload": {"values": [1.0, "not-a-number"]}},
        idempotency_key="nested:0001",
    )
    action = ActionOutput(
        workspace_sha256=workspace.sha256,
        model_identity_sha256=canonical_sha256(workspace.model_identity),
        invocation=invocation,
    )
    with pytest.raises(ValueError, match="wrong type"):
        validate_envelope_against_workspace(action, workspace)


def test_envelope_validation_binds_identity_acl_tool_schema_and_approval() -> None:
    with pytest.raises(ValidationError, match="finite JSON"):
        ToolInvocation(
            tool_id="tool:read",
            schema_version="1",
            arguments={"when": datetime(2026, 9, 6, tzinfo=UTC)},
            idempotency_key="request:python-only",
        )

    invocation = ToolInvocation(
        tool_id="tool:read",
        schema_version="1",
        arguments={"limit": 3, "mode": "exact"},
        idempotency_key="request:0001",
        approval_id="approval:1",
    )
    workspace = _workspace(
        acl={"private"},
        approval_bindings={"approval:1": tool_invocation_sha256(invocation)},
    )
    validate_envelope_against_workspace(_bound_text(workspace), workspace)

    unauthorized = _workspace(acl={"secret"})
    with pytest.raises(PermissionError, match="unauthorized evidence"):
        validate_envelope_against_workspace(_bound_text(unauthorized), unauthorized)

    action = ActionOutput(
        workspace_sha256=workspace.sha256,
        model_identity_sha256=canonical_sha256(workspace.model_identity),
        invocation=invocation,
    )
    validate_envelope_against_workspace(action, workspace)

    without_approval = action.model_copy(
        update={"invocation": action.invocation.model_copy(update={"approval_id": None})}
    )
    with pytest.raises(PermissionError, match="granted approval"):
        validate_envelope_against_workspace(without_approval, workspace)

    invalid_arguments = action.model_copy(
        update={
            "invocation": action.invocation.model_copy(
                update={"arguments": {"limit": 99, "mode": "invalid"}}
            )
        }
    )
    with pytest.raises(ValueError, match="maximum|enum"):
        validate_envelope_against_workspace(invalid_arguments, workspace)


def test_adapter_decoder_has_real_gradient_path_and_frozen_base() -> None:
    torch.manual_seed(7)
    model = AdapterCausalDecoder(
        DecoderConfig(
            vocab_size=32,
            width=16,
            layers=1,
            heads=4,
            max_sequence_tokens=8,
            adapter_rank=4,
        )
    )
    model.add_adapter("hermes")
    model.add_adapter("prometheus")
    model.activate_adapter("hermes")
    model.freeze_base()

    assert model.trainable_parameter_count() > 0
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".adapters.hermes." not in name
    )
    initial = model.adapter_state_dict("hermes")
    inactive_initial = model.adapter_state_dict("prometheus")
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.05
    )
    losses: list[float] = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = causal_language_model_loss(model(tokens), tokens)
        torch.autograd.backward(loss)
        optimizer.step()
        losses.append(float(loss.detach()))

    trained = model.adapter_state_dict("hermes")
    assert all(torch.isfinite(torch.tensor(losses)))
    assert any(not torch.equal(initial[key], trained[key]) for key in initial)
    assert all(
        torch.equal(value, model.adapter_state_dict("prometheus")[key])
        for key, value in inactive_initial.items()
    )

    model.activate_adapter("prometheus")
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".adapters.hermes." in name
    )
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if ".adapters.prometheus." in name
    )

    clone = AdapterCausalDecoder(model.config)
    clone.add_adapter("hermes")
    clone.load_adapter_state_dict("hermes", trained)
    for key, value in trained.items():
        assert torch.equal(value, clone.adapter_state_dict("hermes")[key])


def test_decoder_rejects_invalid_shapes_and_empty_lm_targets() -> None:
    model = AdapterCausalDecoder(
        DecoderConfig(vocab_size=16, width=8, layers=1, heads=2, adapter_rank=2)
    )
    with pytest.raises(ValueError, match="at least one"):
        model(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="integer tensor"):
        model(torch.ones((1, 2), dtype=torch.float32))
    with pytest.raises(ValueError, match="at least two"):
        output = model(torch.tensor([[1]]))
        causal_language_model_loss(output, torch.tensor([[1]]))
    output = model(torch.tensor([[1, 2]]))
    with pytest.raises(ValueError, match="supervised token"):
        causal_language_model_loss(
            output,
            torch.tensor([[1, 2]]),
            attention_mask=torch.tensor([[True, False]]),
        )
