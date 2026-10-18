from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from olympus.data.olympus_dataset import write_olympus_dataset_manifest
from olympus.evaluation.olympus_system import evaluate_olympus_system
from olympus.models.olympus_system import (
    FAMILY_VECTOR_NAMES,
    OlympusReferenceSystem,
    OlympusSystemConfig,
    olympus_system_loss,
)
from olympus.models.reference_families import (
    AionReferenceConfig,
    AtlasReferenceConfig,
    HermesReferenceConfig,
    KronosReferenceConfig,
    PerseusReferenceConfig,
    PrometheusReferenceConfig,
    ReferenceTransformerConfig,
)
from olympus.models.runtime import (
    OlympusRuntimeConfig,
    OlympusSystemRuntime,
    load_olympus_checkpoint,
    save_olympus_checkpoint,
    train_olympus_step,
)
from olympus.training.olympus_system import (
    OlympusSystemTrainingConfig,
    train_olympus_system,
)


def _config() -> OlympusSystemConfig:
    backbone = ReferenceTransformerConfig(
        vocab_size=32,
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=24,
        max_sequence_length=8,
    )
    return OlympusSystemConfig(
        hermes=HermesReferenceConfig(backbone=backbone, tool_count=3),
        prometheus=PrometheusReferenceConfig(
            backbone=backbone, domains=3, adapter_dim=8, evidence_classes=2
        ),
        perseus=PerseusReferenceConfig(
            backbone=backbone, experts=3, active_experts=2, expert_dim=24
        ),
        atlas=AtlasReferenceConfig(
            backbone=backbone, image_channels=3, patch_size=2, retrieval_dim=8
        ),
        kronos=KronosReferenceConfig(
            state_dim=5,
            action_dim=3,
            hidden_dim=8,
            discrete_actions=4,
            planning_horizon=3,
        ),
        aion=AionReferenceConfig(
            state_dim=6,
            hidden_dim=8,
            research_stages=4,
            stop_stage=3,
            kill_threshold=0.75,
        ),
        fusion_dim=12,
        decision_classes=4,
    )


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[True, True, True, False], [True, True, False, False]]),
        "image": torch.rand(2, 3, 8, 8),
        "states": torch.rand(2, 4, 5),
        "actions": torch.rand(2, 4, 3),
        "temporal_mask": torch.tensor([[True, True, True, True], [True, True, False, False]]),
        "allowed_transitions": torch.tensor([[True, True, False, True], [False, True, True, True]]),
        "risk_score": torch.tensor([0.2, 0.9]),
        "domain_ids": torch.tensor([0, 2]),
        "timestamp": torch.tensor([0.0, 1.0]),
    }


def _dataset_batch() -> dict[str, torch.Tensor]:
    inputs = _inputs()
    return {
        **inputs,
        "decision_targets": torch.tensor([2, 1]),
        "transition_targets": torch.tensor([1, 3]),
        "next_state_targets": torch.rand(2, 4, 5),
        "intrinsic_risk_targets": torch.tensor([0.1, 0.8]),
    }


def test_olympus_system_composes_all_families_and_preserves_stop_authority() -> None:
    torch.manual_seed(21)
    model = OlympusReferenceSystem(_config())
    inputs = _inputs()
    output = model(**inputs)
    assert FAMILY_VECTOR_NAMES == ("hermes", "prometheus", "perseus", "atlas", "kronos")
    assert output.family_gates.shape == (2, 5)
    assert torch.allclose(output.family_gates.sum(dim=-1), torch.ones(2))
    assert output.fused_state.shape == (2, 12)
    assert output.component_names == FAMILY_VECTOR_NAMES
    assert output.component_states.shape == (2, 5, 12)
    assert output.counterfactual_decision_logits.shape == (2, 5, 4)
    assert output.raw_decision_logits.shape == (2, 4)
    assert torch.equal(output.decision_available, torch.tensor([True, False]))
    assert torch.equal(output.decision_logits[1], torch.zeros(4))
    assert output.aion.selected_stage[1].item() == 3
    assert bool((output.uncertainty > 0.0).all())
    assert bool((output.effective_risk >= inputs["risk_score"]).all())
    bounded_disagreement = (0.0 <= output.family_disagreement) & (output.family_disagreement <= 1.0)
    assert bool(bounded_disagreement.all())

    loss = olympus_system_loss(
        output,
        decision_targets=torch.tensor([2, 1]),
        transition_targets=torch.tensor([1, 3]),
        next_state_targets=torch.rand(2, 4, 5),
        intrinsic_risk_targets=torch.tensor([0.1, 0.8]),
        temporal_mask=inputs["temporal_mask"],
    )
    assert torch.isfinite(loss.total)
    assert loss.gate_balance >= 0.0
    assert loss.counterfactual_consistency >= 0.0
    loss.total.backward()  # type: ignore[no-untyped-call]
    assert model.hermes.backbone.token_embedding.weight.grad is not None
    assert model.prometheus.backbone.token_embedding.weight.grad is not None
    assert model.perseus.backbone.token_embedding.weight.grad is not None
    assert model.atlas.patch_embedding.weight.grad is not None
    assert model.kronos.transition.weight_ih_l0.grad is not None
    assert next(model.aion.router.parameters()).grad is not None
    assert model.gate_router[1].weight.grad is not None
    final_risk_layer = model.intrinsic_risk_head[-1]
    assert isinstance(final_risk_layer, torch.nn.Linear)
    assert final_risk_layer.weight.grad is not None


def test_olympus_system_rejects_forbidden_targets_and_misaligned_batches() -> None:
    model = OlympusReferenceSystem(_config())
    inputs = _inputs()
    output = model(**inputs)
    with pytest.raises(ValueError, match="forbidden"):
        olympus_system_loss(
            output,
            decision_targets=torch.tensor([2, 1]),
            transition_targets=torch.tensor([2, 3]),
            next_state_targets=torch.rand(2, 4, 5),
        )

    inputs["image"] = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="same batch size"):
        model(**inputs)


def test_olympus_system_extensions_and_loss_validation() -> None:
    config = _config().model_copy(update={"extension_dims": {"sensor": 7}})
    model = OlympusReferenceSystem(config)
    inputs = _inputs()
    with pytest.raises(ValueError, match=r"missing=\['sensor'\]"):
        model(**inputs)

    output = model(**inputs, extension_features={"sensor": torch.rand(2, 7)})
    assert output.component_names == (*FAMILY_VECTOR_NAMES, "sensor")
    assert output.family_gates.shape == (2, 6)

    decision_targets = torch.tensor([2, 1])
    transition_targets = torch.tensor([1, 3])
    next_state_targets = torch.rand(2, 4, 5)
    with pytest.raises(ValueError, match="decision targets fall outside"):
        olympus_system_loss(
            output,
            decision_targets=torch.tensor([9, 1]),
            transition_targets=transition_targets,
            next_state_targets=next_state_targets,
        )
    with pytest.raises(ValueError, match="transition targets fall outside"):
        olympus_system_loss(
            output,
            decision_targets=decision_targets,
            transition_targets=torch.tensor([8, 3]),
            next_state_targets=next_state_targets,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        olympus_system_loss(
            output,
            decision_targets=decision_targets,
            transition_targets=transition_targets,
            next_state_targets=next_state_targets,
            counterfactual_weight=-1.0,
        )
    with pytest.raises(ValueError, match="intrinsic-risk targets"):
        olympus_system_loss(
            output,
            decision_targets=decision_targets,
            transition_targets=transition_targets,
            next_state_targets=next_state_targets,
            intrinsic_risk_targets=torch.tensor([0.1, 2.0]),
        )


def test_olympus_parameter_matched_mechanism_controls() -> None:
    inputs = _inputs()
    learned = OlympusReferenceSystem(_config())
    uniform = OlympusReferenceSystem(
        _config().model_copy(update={"fusion_strategy": "uniform", "disagreement_risk": False})
    )
    uniform.load_state_dict(learned.state_dict())

    learned_output = learned(**inputs)
    uniform_output = uniform(**inputs)
    expected = torch.full_like(uniform_output.family_gates, 1.0 / len(FAMILY_VECTOR_NAMES))
    assert torch.allclose(uniform_output.family_gates, expected)
    assert sum(parameter.numel() for parameter in learned.parameters()) == sum(
        parameter.numel() for parameter in uniform.parameters()
    )
    assert not torch.allclose(learned_output.family_gates, uniform_output.family_gates)

    loss = olympus_system_loss(
        uniform_output,
        decision_targets=torch.tensor([2, 1]),
        transition_targets=torch.tensor([1, 3]),
        next_state_targets=torch.rand(2, 4, 5),
    )
    loss.total.backward()  # type: ignore[no-untyped-call]
    router_weight = uniform.gate_router[-1]
    assert isinstance(router_weight, torch.nn.Linear)
    assert router_weight.weight.grad is not None
    assert torch.count_nonzero(router_weight.weight.grad) == 0


def test_olympus_runtime_training_and_digest_pinned_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(29)
    model = OlympusReferenceSystem(_config())
    runtime = OlympusSystemRuntime(
        model,
        OlympusRuntimeConfig(
            device="cpu",
            mixed_precision=False,
            resource_preflight=False,
        ),
    )
    inputs = _inputs()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    inference = runtime.infer(inputs)
    assert torch.are_deterministic_algorithms_enabled() is deterministic_before
    assert inference.device == "cpu"
    assert inference.elapsed_seconds >= 0.0
    assert inference.mixed_precision_effective is False

    checkpoint = tmp_path / "olympus.pt"
    record = save_olympus_checkpoint(model, checkpoint, metadata={"purpose": "test"})
    loaded, loaded_record = load_olympus_checkpoint(checkpoint, expected_sha256=record.sha256)
    assert loaded_record == record
    with torch.inference_mode():
        reloaded_output = loaded(**inputs)
    assert torch.allclose(
        inference.output.raw_decision_logits,
        reloaded_output.raw_decision_logits,
    )
    with pytest.raises(ValueError, match="digest"):
        load_olympus_checkpoint(checkpoint, expected_sha256="0" * 64)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_olympus_checkpoint(model, checkpoint)

    optimizer = torch.optim.AdamW(loaded.parameters(), lr=1e-3)
    result = train_olympus_step(
        loaded,
        optimizer,
        _inputs(),
        {
            "decision_targets": torch.tensor([2, 1]),
            "transition_targets": torch.tensor([1, 3]),
            "next_state_targets": torch.rand(2, 4, 5),
            "intrinsic_risk_targets": torch.tensor([0.1, 0.8]),
        },
    )
    assert torch.isfinite(result.loss.total)
    assert result.gradient_norm > 0.0


def test_olympus_composed_training_evaluation_and_immutable_report(tmp_path: Path) -> None:
    torch.manual_seed(31)
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    train_path = dataset_root / "train.pt"
    evaluation_path = dataset_root / "evaluation.pt"
    torch.save(_dataset_batch(), train_path)
    torch.save(_dataset_batch(), evaluation_path)
    manifest_path = dataset_root / "manifest.json"
    manifest = write_olympus_dataset_manifest(
        manifest_path,
        dataset_id="olympus-system-test",
        version="1.0.0",
        license_name="LicenseRef-Proprietary",
        train_batch=train_path,
        evaluation_batch=evaluation_path,
        train_source_ids=["train-0", "train-1"],
        evaluation_source_ids=["evaluation-0", "evaluation-1"],
    )
    config = OlympusSystemTrainingConfig(
        seed=31,
        steps=2,
        batch_size=2,
        learning_rate=1e-3,
        warmup_steps=1,
        device="cpu",
        resource_preflight=False,
        model=_config(),
    )
    output = tmp_path / "run"
    report = train_olympus_system(manifest_path, output, config=config)
    assert report.status == "COMPLETED_DEVELOPMENT_NOT_PROMOTED"
    assert report.dataset_manifest_sha256 == manifest.manifest_sha256
    assert report.completed_steps == 2
    assert report.promotion_authorized is False
    assert math.isfinite(report.training.first_loss)
    assert math.isfinite(report.training.final_loss)
    assert report.training.maximum_gradient_norm > 0.0
    assert report.final_evaluation.examples == 2
    assert report.final_evaluation.routing.mean_gates.keys() == set(FAMILY_VECTOR_NAMES)
    assert report.final_evaluation.risk.intrinsic_brier is not None
    assert len(report.final_evaluation.risk.risk_coverage) == 4
    assert Path(report.checkpoint_path).is_file()
    serialized = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert serialized["checkpoint_sha256"] == report.checkpoint_sha256
    assert serialized["training_config_sha256"] == report.training_config_sha256
    assert serialized["training_config"]["seed"] == 31

    reloaded, record = load_olympus_checkpoint(
        output / "checkpoint.pt", expected_sha256=report.checkpoint_sha256
    )
    evaluation = evaluate_olympus_system(reloaded, _dataset_batch(), [])
    assert record.metadata["dataset_manifest_sha256"] == manifest.manifest_sha256
    assert record.metadata["training_config_sha256"] == report.training_config_sha256
    assert evaluation.examples == 2
    assert math.isfinite(evaluation.total_loss)
    components = evaluation.loss_components
    reconstructed = (
        components["decision"]
        + config.loss.transition_weight * components["transition"]
        + config.loss.temporal_weight * components["temporal"]
        + config.loss.uncertainty_weight * components["uncertainty_calibration"]
        + config.loss.risk_weight * components["risk_calibration"]
        + config.loss.gate_balance_weight * components["gate_balance"]
        + config.loss.counterfactual_weight * components["counterfactual_consistency"]
    )
    assert reconstructed == pytest.approx(evaluation.total_loss, abs=1e-6)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        train_olympus_system(manifest_path, output, config=config)
