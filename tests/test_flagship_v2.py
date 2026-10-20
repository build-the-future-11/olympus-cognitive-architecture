from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from olympus.models.flagship import (
    AionFlagshipModel,
    AtlasFlagshipModel,
    HermesFlagshipModel,
    KronosFlagshipModel,
    OlympusFlagshipSystem,
    PerseusFlagshipModel,
    PrometheusFlagshipModel,
)
from olympus.models.olympus_system import OlympusSystemConfig, build_olympus_system
from olympus.models.parameter_growth import JEPAParameterGrowth, widen_feedforward
from olympus.models.reference_families import (
    AionReferenceConfig,
    AtlasReferenceConfig,
    HermesReferenceConfig,
    KronosReferenceConfig,
    PerseusReferenceConfig,
    PrometheusReferenceConfig,
    ReferenceTransformerConfig,
)
from olympus.models.spatiotemporal_lazy import (
    PlannedShard,
    ResidencyPlan,
    SpatiotemporalParameterStore,
    SpatiotemporalQuery,
    build_parameter_manifest,
    parameter_shard_record,
)


@pytest.fixture(autouse=True)
def deterministic_fixture() -> object:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(219)
        try:
            yield
        finally:
            torch.set_num_threads(threads)


def tiny_config() -> OlympusSystemConfig:
    backbone = ReferenceTransformerConfig(
        vocab_size=16,
        model_dim=8,
        layers=1,
        heads=2,
        feedforward_dim=16,
        max_sequence_length=8,
    )
    return OlympusSystemConfig(
        architecture_revision="flagship_v2",
        hermes=HermesReferenceConfig(backbone=backbone, tool_count=3),
        prometheus=PrometheusReferenceConfig(backbone=backbone, domains=2, adapter_dim=4),
        perseus=PerseusReferenceConfig(
            backbone=backbone, experts=3, active_experts=2, expert_dim=8
        ),
        atlas=AtlasReferenceConfig(backbone=backbone, patch_size=2, retrieval_dim=4),
        kronos=KronosReferenceConfig(
            state_dim=3,
            action_dim=2,
            hidden_dim=4,
            discrete_actions=3,
            planning_horizon=3,
        ),
        aion=AionReferenceConfig(state_dim=4, hidden_dim=4, research_stages=3, stop_stage=2),
        fusion_dim=8,
        decision_classes=3,
    )


def test_prometheus_causality_and_adapter_trust_radius() -> None:
    model = PrometheusFlagshipModel(tiny_config().prometheus).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    changed = torch.tensor([[1, 2, 9, 8]])
    first, second = model(tokens), model(changed)
    torch.testing.assert_close(
        first.token_logits[:, :2], second.token_logits[:, :2], rtol=0, atol=0
    )
    hidden = model.backbone(tokens)
    radius = (1 + first.adapter_disagreement).reciprocal()
    assert bool(((first.hidden_states - hidden).norm(dim=-1) <= radius + 1e-6).all())
    with pytest.raises(ValueError, match="domain_ids"):
        model(tokens, domain_ids=torch.tensor([0.5]))


def test_hermes_host_mask_and_monotone_abstention() -> None:
    model = HermesFlagshipModel(tiny_config().hermes)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    allowed = torch.tensor([[True, False, False], [False, False, False]])
    output = model(tokens, allowed_tools=allowed)
    assert torch.equal(output.tool_probabilities, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    assert output.tool_available.tolist() == [True, False]
    raw = model.abstention_head(output.hidden_states[:, -1]).squeeze(-1)
    assert bool((output.abstention_logit >= raw).all())
    with pytest.raises(ValueError, match="Boolean"):
        model(tokens, allowed_tools=allowed.float())
    (output.token_logits.square().mean() + output.tool_disagreement.sum()).backward()
    assert model.context_tool_head.weight.grad is not None


def test_perseus_never_executes_unavailable_experts_or_padding() -> None:
    model = PerseusFlagshipModel(tiny_config().perseus)
    counts = [0, 0, 0]
    handles = []
    for index, expert in enumerate(model.experts):

        def record(_module: nn.Module, args: tuple[torch.Tensor, ...], i: int = index) -> None:
            counts[i] += len(args[0])

        handles.append(expert.register_forward_pre_hook(record))
    tokens = torch.tensor([[1, 2, 0], [3, 0, 0]])
    active = tokens != 0
    output = model(tokens, active, resident_experts=torch.tensor([True, False, True]))
    assert counts == [3, 0, 3]
    assert torch.equal(output.routing_probabilities[..., 1], torch.zeros_like(tokens).float())
    assert bool((output.active_expert_indices[~active] == -1).all())
    assert torch.isfinite(output.load_balance_loss)
    for handle in handles:
        handle.remove()
    with pytest.raises(ValueError, match="top-k"):
        model(tokens, active, resident_experts=torch.tensor([True, False, False]))


def test_atlas_masked_patch_has_zero_influence_and_gradient() -> None:
    model = AtlasFlagshipModel(tiny_config().atlas).eval()
    tokens = torch.tensor([[1, 2]])
    image = torch.rand(1, 3, 4, 4, requires_grad=True)
    mask = torch.tensor([[True, True, True, False]])
    changed = image.detach().clone()
    changed[:, :, 2:, 2:] = 1 - changed[:, :, 2:, 2:]
    output = model(tokens, image, patch_mask=mask)
    other = model(tokens, changed, patch_mask=mask)
    torch.testing.assert_close(output.token_logits, other.token_logits, rtol=0, atol=0)
    assert bool((output.grounding_weights[..., -1] == 0).all())
    output.token_logits.square().mean().backward()
    assert image.grad is not None and bool((image.grad[:, :, 2:, 2:] == 0).all())
    with pytest.raises(ValueError, match="at least one"):
        model(tokens, image.detach(), patch_mask=torch.zeros_like(mask))
    with pytest.raises(ValueError, match="divisible"):
        model(tokens, torch.rand(1, 3, 5, 4))


def test_kronos_rolls_forward_states_and_accumulates_uncertainty() -> None:
    model = KronosFlagshipModel(tiny_config().kronos)
    # Constant residual proves the state advances at every planning step.
    nn.init.zeros_(model.next_state_head.weight)
    nn.init.constant_(model.next_state_head.bias, 0.25)
    states, actions = torch.rand(1, 3, 3), torch.rand(1, 3, 2)
    mask = torch.tensor([[True, False, False]])
    output = model(states, actions, mask)
    expected = states[:, :1] + torch.tensor([0.25, 0.5, 0.75]).view(1, 3, 1)
    torch.testing.assert_close(output.plan_states, expected)
    assert bool((output.cumulative_plan_uncertainty.diff(dim=1) >= 0).all())
    assert bool((output.next_state[:, 1:] == 0).all())


def test_aion_budget_intersects_authority_and_forces_authorized_stop() -> None:
    model = AionFlagshipModel(tiny_config().aion)
    state, risk = torch.rand(2, 4), torch.tensor([0.1, 0.1])
    allowed = torch.ones(2, 3, dtype=torch.bool)
    costs, budget = torch.tensor([[2.0, 3.0, 0.0], [2.0, 3.0, 0.0]]), torch.tensor([2.0, 0.0])
    output = model(state, allowed, risk, transition_costs=costs, remaining_budget=budget)
    assert output.transition_probabilities[0, 1] == 0
    assert output.selected_stage[1] == 2 and output.forced_stop[1]
    allowed[1, 2] = False
    with pytest.raises(ValueError, match="permit|STOP"):
        model(state, allowed, risk, transition_costs=costs, remaining_budget=budget)
    with pytest.raises(ValueError, match="together"):
        model(state, allowed, risk, transition_costs=costs)


def test_core_revision_roundtrip_and_monotone_risk() -> None:
    config = tiny_config()
    with pytest.raises(ValueError, match="architecture_revision"):
        OlympusFlagshipSystem(config.model_copy(update={"architecture_revision": "reference_v1"}))
    model = build_olympus_system(config)
    assert isinstance(model, OlympusFlagshipSystem)
    args = (
        torch.tensor([[1, 2], [3, 4]]),
        torch.rand(2, 3, 4, 4),
        torch.rand(2, 3, 3),
        torch.rand(2, 3, 2),
        torch.ones(2, 3, dtype=torch.bool),
    )
    low = model(*args, torch.tensor([0.0, 0.1]))
    high = model(*args, torch.tensor([0.9, 1.0]))
    assert bool((high.effective_risk >= low.effective_risk).all())
    assert not bool(high.decision_available.any())
    assert bool((high.decision_logits == 0).all())
    assert low.evidence_risk_terms.shape == (2, 3)
    clone = build_olympus_system(OlympusSystemConfig.model_validate_json(config.model_dump_json()))
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(
        clone(*args, torch.tensor([0.0, 0.1])).raw_decision_logits,
        low.raw_decision_logits,
        rtol=0,
        atol=0,
    )
    objective = (
        low.raw_decision_logits.square().mean()
        + low.effective_risk.mean()
        + low.kronos.plan_logits.square().mean()
        + low.hermes.token_logits.square().mean()
        + low.prometheus.evidence_logits.square().mean()
        + low.perseus.token_logits.square().mean()
        + low.atlas.token_logits.square().mean()
    )
    objective.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("width", [5, 8, 11])
@pytest.mark.parametrize("activation", [nn.GELU, nn.SiLU, nn.ReLU])
def test_growth_preserves_function_input_gradient_and_original(
    width: int,
    activation: type[nn.Module],
) -> None:
    block = nn.Sequential(nn.Linear(3, 4), activation(), nn.Linear(4, 2)).double().eval()
    original = copy.deepcopy(block.state_dict())
    before_rng = torch.get_rng_state().clone()
    grown, receipt = widen_feedforward(block, width, max_parameter_bytes=10000)
    assert torch.equal(before_rng, torch.get_rng_state())
    inputs = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    expected, observed = block(inputs), grown(inputs)
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)
    first = torch.autograd.grad(expected.sum(), inputs)[0]
    second = torch.autograd.grad(observed.sum(), inputs)[0]
    torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)
    assert receipt.expanded_parameters > receipt.original_parameters
    assert receipt.optimizer_rebuild_required and not receipt.trained_target_scale
    assert not grown.training
    for name, value in original.items():
        assert torch.equal(value, block.state_dict()[name])


def test_growth_refuses_over_budget_and_nonzero_dropout() -> None:
    block = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2))
    with pytest.raises(ValueError, match="budget"):
        widen_feedforward(block, 8, max_parameter_bytes=1)
    with pytest.raises(ValueError, match="zero-dropout"):
        widen_feedforward(
            nn.Sequential(block[0], block[1], nn.Dropout(0.1), block[2]),
            8,
            max_parameter_bytes=10000,
        )


def test_jepa_parameter_growth_identity_mask_isolation_and_ema() -> None:
    module = JEPAParameterGrowth(4, 8, momentum=0.5)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[False, True, False], [True, False, False]])
    assert torch.equal(module(hidden), hidden)
    objective = module.masked_objective(hidden, mask)
    objective.backward()  # type: ignore[no-untyped-call]
    assert hidden.grad is not None and bool((hidden.grad[mask] == 0).all())
    assert all(p.grad is None and not p.requires_grad for p in module.target_encoder.parameters())
    before = module.target_encoder.weight.detach().clone()
    with torch.no_grad():
        module.context_encoder.weight.add_(2)
    module.update_target()
    torch.testing.assert_close(module.target_encoder.weight, before + 1)
    with pytest.raises(ValueError, match="visible and target"):
        module.masked_objective(hidden, torch.ones_like(mask))


def _store(tmp_path: Path) -> tuple[SpatiotemporalParameterStore, ResidencyPlan]:
    path = tmp_path / "weights.pt"
    torch.save({"weight": torch.eye(2), "bias": torch.ones(2)}, path)
    record = parameter_shard_record("layer", path, root=tmp_path, always_load=True)
    manifest = tmp_path / "manifest.json"
    build_parameter_manifest([record], manifest)
    store = SpatiotemporalParameterStore(manifest, cache_budget_bytes=record.residency_bytes)
    plan = store.plan(
        SpatiotemporalQuery(position=(0, 0, 0), timestamp=0, resolution=1),
        budget_bytes=record.residency_bytes,
    )
    return store, plan


@pytest.mark.parametrize("case", ["total", "budget", "duplicate", "missing", "unknown"])
def test_ghost_rejects_forged_plan_before_disk_access(tmp_path: Path, case: str) -> None:
    store, plan = _store(tmp_path)
    if case == "total":
        plan.planned_bytes = 0
    elif case == "budget":
        plan.budget_bytes = 2 * plan.budget_bytes
    elif case == "duplicate":
        plan.shards.append(plan.shards[0])
    elif case == "missing":
        plan.shards.clear()
    else:
        plan.shards = [PlannedShard(shard_id="unknown", score=0, byte_count=1)]
    with pytest.raises(ValueError):
        store.materialize(plan)
    assert store.disk_loads == 0 and store.resident_bytes == 0


@pytest.mark.parametrize("fail", [False, True])
def test_ghost_lease_restores_meta_on_success_and_exception(tmp_path: Path, fail: bool) -> None:
    store, plan = _store(tmp_path)
    module = nn.Linear(2, 2, device="meta").eval()
    with torch.inference_mode():
        try:
            with store.inference_lease(module, plan):
                torch.testing.assert_close(module(torch.zeros(1, 2)), torch.ones(1, 2))
                assert store.resident_bytes <= plan.budget_bytes
                if fail:
                    raise RuntimeError("deliberate interrupted inference")
        except RuntimeError:
            assert fail
    assert all(value.is_meta for value in module.state_dict().values())
    assert store.resident_bytes == 0
    with pytest.raises(ValueError, match="gradients disabled"):
        with store.inference_lease(module, plan):
            pass


def test_ghost_parent_symlink_and_tampering_are_rejected(tmp_path: Path) -> None:
    store, plan = _store(tmp_path)
    directory = tmp_path / "actual"
    directory.mkdir()
    (tmp_path / "linked").symlink_to(directory, target_is_directory=True)
    (tmp_path / "weights.pt").rename(directory / "weights.pt")
    manifest = json.loads(store.manifest_path.read_text())
    manifest["shards"][0]["path"] = "linked/weights.pt"
    store.manifest_path.write_text(json.dumps(manifest))
    linked = SpatiotemporalParameterStore(store.manifest_path, cache_budget_bytes=plan.budget_bytes)
    with pytest.raises(ValueError, match="symlink"):
        linked.materialize(plan)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_ghost_query_rejects_nonfinite_coordinates(value: float) -> None:
    with pytest.raises(ValueError):
        SpatiotemporalQuery(position=(value, 0, 0), timestamp=0, resolution=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware is unavailable")
def test_flagship_cuda_forward_backward() -> None:
    model = OlympusFlagshipSystem(tiny_config()).cuda()
    output = model(
        torch.tensor([[1, 2]], device="cuda"),
        torch.rand(1, 3, 4, 4, device="cuda"),
        torch.rand(1, 3, 3, device="cuda"),
        torch.rand(1, 3, 2, device="cuda"),
        torch.ones(1, 3, dtype=torch.bool, device="cuda"),
        torch.zeros(1, device="cuda"),
    )
    output.raw_decision_logits.square().mean().backward()
    assert torch.isfinite(output.raw_decision_logits).all()
