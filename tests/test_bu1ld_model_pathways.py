from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from olympus.models.encoder_quantization import (
    EncoderQuantizationConfig,
    QuantizedEncoder,
)
from olympus.models.reference_families import (
    AionReferenceConfig,
    AionReferenceModel,
    AtlasReferenceConfig,
    AtlasReferenceModel,
    HermesReferenceConfig,
    HermesReferenceModel,
    KronosReferenceConfig,
    KronosReferenceModel,
    PerseusReferenceConfig,
    PerseusReferenceModel,
    PrometheusReferenceConfig,
    PrometheusReferenceModel,
    ReferenceTransformerConfig,
    build_reference_model,
    default_reference_configs,
)
from olympus.models.spatiotemporal_lazy import (
    SpatialBounds,
    SpatiotemporalParameterStore,
    SpatiotemporalQuery,
    TemporalBounds,
    build_parameter_manifest,
    parameter_shard_record,
)
from olympus.models.specifications import BU1LD_MODEL_SPECS, model_spec
from olympus.training.encoder_quantization import (
    EncoderQuantizationTrainingConfig,
    train_encoder_quantizer,
    write_encoder_dataset_manifest,
)
from olympus.upscaler.jepa import (
    JEPAUpscaler,
    JEPAUpscalerConfig,
    jepa_upscaler_loss,
    upscale_self_ensemble,
    upscale_tiled,
)
from olympus.upscaler.training import (
    JEPAUpscalerTrainingConfig,
    load_jepa_upscaler_checkpoint,
    train_jepa_upscaler,
    write_super_resolution_manifest,
)


def test_bu1ld_target_specifications_are_not_capability_claims() -> None:
    assert [spec.family_id for spec in BU1LD_MODEL_SPECS] == [
        "hermes",
        "prometheus",
        "perseus",
        "atlas",
        "kronos",
        "aion",
    ]
    assert model_spec("hermes").target_parameters == 12_000_000_000
    assert model_spec("prometheus").target_parameters is None
    assert model_spec("prometheus").target_parameters_min == 12_000_000_000
    assert model_spec("prometheus").target_parameters_max == 20_000_000_000
    assert model_spec("perseus").target_parameters == 1_024_000_000_000
    assert model_spec("atlas").target_parameters is None
    assert model_spec("kronos").target_parameters == 8_000_000_000_000
    assert model_spec("aion").target_parameters is None
    assert all(not spec.capability_claimed for spec in BU1LD_MODEL_SPECS)
    assert [config.family_id for config in default_reference_configs()] == [
        spec.family_id for spec in BU1LD_MODEL_SPECS
    ]


def test_all_family_reference_architectures_are_executable_and_differentiable() -> None:
    torch.manual_seed(10)
    backbone = ReferenceTransformerConfig(
        vocab_size=32,
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=24,
        max_sequence_length=8,
    )
    tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    hermes = HermesReferenceModel(HermesReferenceConfig(backbone=backbone, tool_count=3))
    hermes_output = hermes(tokens, mask)
    assert hermes_output.token_logits.shape == (2, 4, 32)
    assert hermes_output.tool_logits.shape == (2, 3)
    assert hermes_output.abstention_logit.shape == (2,)

    prometheus = PrometheusReferenceModel(
        PrometheusReferenceConfig(backbone=backbone, domains=3, adapter_dim=8, evidence_classes=2)
    )
    prometheus_output = prometheus(tokens, mask, domain_ids=torch.tensor([0, 2]))
    assert prometheus_output.evidence_logits.shape == (2, 4, 2)
    assert torch.equal(
        prometheus_output.domain_weights,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )

    perseus = PerseusReferenceModel(
        PerseusReferenceConfig(backbone=backbone, experts=3, active_experts=2, expert_dim=24)
    )
    perseus_output = perseus(tokens, mask)
    assert perseus_output.active_expert_indices.shape == (2, 4, 2)
    assert torch.allclose(perseus_output.routing_probabilities.sum(dim=-1), torch.ones(2, 4))
    assert torch.isfinite(perseus_output.load_balance_loss)

    atlas = AtlasReferenceModel(
        AtlasReferenceConfig(backbone=backbone, image_channels=3, patch_size=2, retrieval_dim=6)
    )
    atlas_output = atlas(tokens, torch.rand(2, 3, 8, 8), mask, timestamp=torch.tensor([0.0, 1.0]))
    assert atlas_output.grounding_weights.shape == (2, 4, 16)
    assert atlas_output.retrieval_embedding.shape == (2, 6)
    assert torch.allclose(atlas_output.retrieval_embedding.norm(dim=-1), torch.ones(2), atol=1e-5)

    kronos = KronosReferenceModel(
        KronosReferenceConfig(
            state_dim=5,
            action_dim=3,
            hidden_dim=8,
            discrete_actions=4,
            planning_horizon=3,
        )
    )
    kronos_output = kronos(
        torch.rand(2, 4, 5),
        torch.rand(2, 4, 3),
        torch.tensor([[True, True, True, True], [True, True, False, False]]),
    )
    assert kronos_output.next_state.shape == (2, 4, 5)
    assert kronos_output.plan_logits.shape == (2, 3, 4)
    assert bool((kronos_output.uncertainty[0] > 0.0).all())
    assert bool((kronos_output.uncertainty[1, :2] > 0.0).all())
    assert torch.equal(kronos_output.uncertainty[1, 2:], torch.zeros(2, 5))

    aion = AionReferenceModel(
        AionReferenceConfig(
            state_dim=5,
            hidden_dim=8,
            research_stages=4,
            stop_stage=3,
            kill_threshold=0.75,
        )
    )
    allowed = torch.tensor([[True, True, False, True], [False, True, True, True]])
    aion_output = aion(
        torch.rand(2, 5),
        allowed,
        torch.tensor([0.2, 0.9]),
    )
    assert aion_output.transition_logits.shape == (2, 4)
    assert aion_output.selected_stage[1].item() == 3
    assert torch.equal(aion_output.forced_stop, torch.tensor([False, True]))
    assert bool(torch.isneginf(aion_output.transition_logits[0, 2]))

    total = (
        hermes_output.token_logits.mean()
        + prometheus_output.token_logits.mean()
        + perseus_output.token_logits.mean()
        + atlas_output.token_logits.mean()
        + kronos_output.next_state.mean()
        + aion_output.transition_probabilities[0].mean()
    )
    total.backward()
    assert hermes.backbone.token_embedding.weight.grad is not None
    assert next(prometheus.adapters[2].parameters()).grad is not None
    assert perseus.router.weight.grad is not None
    assert atlas.patch_embedding.weight.grad is not None
    assert kronos.transition.weight_ih_l0.grad is not None
    assert next(aion.router.parameters()).grad is not None
    assert all(build_reference_model(config) is not None for config in default_reference_configs())


def test_family_reference_contracts_reject_invalid_inputs() -> None:
    backbone = ReferenceTransformerConfig(
        vocab_size=16,
        model_dim=8,
        layers=1,
        heads=2,
        feedforward_dim=16,
        max_sequence_length=4,
    )
    hermes = HermesReferenceModel(HermesReferenceConfig(backbone=backbone))
    with pytest.raises(ValueError, match="integer dtype"):
        hermes(torch.ones(1, 2))
    with pytest.raises(ValueError, match="at least one active token"):
        hermes(torch.ones(1, 2, dtype=torch.long), torch.zeros(1, 2, dtype=torch.bool))

    prometheus = PrometheusReferenceModel(PrometheusReferenceConfig(backbone=backbone))
    with pytest.raises(ValueError, match="outside"):
        prometheus(torch.ones(1, 2, dtype=torch.long), domain_ids=torch.tensor([99]))

    atlas = AtlasReferenceModel(AtlasReferenceConfig(backbone=backbone, patch_size=4))
    with pytest.raises(ValueError, match="smaller"):
        atlas(torch.ones(1, 2, dtype=torch.long), torch.rand(1, 3, 2, 2))

    kronos = KronosReferenceModel(KronosReferenceConfig(state_dim=3, action_dim=2))
    with pytest.raises(ValueError, match="contiguous prefix"):
        kronos(
            torch.rand(1, 3, 3),
            torch.rand(1, 3, 2),
            torch.tensor([[True, False, True]]),
        )

    aion = AionReferenceModel(AionReferenceConfig(state_dim=3, research_stages=3, stop_stage=2))
    with pytest.raises(ValueError, match="requires STOP"):
        aion(
            torch.rand(1, 3),
            torch.tensor([[True, True, False]]),
            torch.tensor([0.9]),
        )


def test_encoder_quantization_is_masked_differentiable_and_residual() -> None:
    torch.manual_seed(11)
    config = EncoderQuantizationConfig(
        input_dim=6,
        hidden_dim=12,
        latent_dim=4,
        codebook_size=8,
        residual_codebooks=3,
    )
    encoder = QuantizedEncoder(config)
    inputs = torch.randn(2, 5, 6, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    output = encoder(inputs, mask=mask)
    assert output.quantized.shape == (2, 5, 4)
    assert output.indices.shape == (2, 5, 3)
    assert torch.equal(output.indices[~mask], torch.full((5, 3), -1))
    assert 0.0 < output.utilization.item() <= 1.0
    assert torch.isfinite(output.loss)
    (output.quantized.square().mean() + output.loss).backward()
    assert inputs.grad is not None
    assert encoder.quantizer.codebooks.grad is not None
    with pytest.raises(ValueError, match="mask shape"):
        encoder(inputs, mask=torch.ones((2, 4), dtype=torch.bool))


def test_hash_bound_encoder_quantizer_training_and_negative_result_retention(
    tmp_path: Path,
) -> None:
    torch.manual_seed(12)
    dataset = tmp_path / "encoder-dataset"
    dataset.mkdir()
    train_features = torch.rand(6, 3, 4)
    evaluation_features = torch.rand(2, 3, 4)
    torch.save(train_features, dataset / "train.pt")
    torch.save(evaluation_features, dataset / "evaluation.pt")
    manifest_path = dataset / "manifest.json"
    write_encoder_dataset_manifest(
        manifest_path,
        license_name="LicenseRef-Proprietary",
        train_source_ids=[f"train-{index}" for index in range(6)],
        evaluation_source_ids=[f"evaluation-{index}" for index in range(2)],
        train_features=dataset / "train.pt",
        evaluation_features=dataset / "evaluation.pt",
    )
    config = EncoderQuantizationTrainingConfig(
        seed=12,
        steps=2,
        batch_size=2,
        encoder=EncoderQuantizationConfig(
            input_dim=4,
            hidden_dim=8,
            latent_dim=3,
            codebook_size=8,
            residual_codebooks=2,
        ),
    )
    output = tmp_path / "encoder-run"
    report = train_encoder_quantizer(manifest_path, output, config=config)
    assert report.status == "COMPLETED_DEVELOPMENT_NOT_PROMOTED"
    assert report.completed_steps == 2
    assert report.promotion_authorized is False
    assert math.isfinite(report.final_training_loss)
    assert math.isfinite(report.evaluation.reconstruction_mse)
    assert 0.0 <= report.evaluation.codebook_utilization <= 1.0
    assert Path(report.checkpoint_path).is_file()
    assert (output / "report.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        train_encoder_quantizer(manifest_path, output, config=config)

    (dataset / "train.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest changed"):
        train_encoder_quantizer(manifest_path, tmp_path / "tampered-encoder-run", config=config)


def test_jepa_upscaler_combines_geometry_target_prediction_and_reconstruction() -> None:
    torch.manual_seed(13)
    config = JEPAUpscalerConfig(
        scale=2,
        input_channels=3,
        feature_channels=8,
        residual_blocks=1,
        target_momentum=0.9,
    )
    model = JEPAUpscaler(config)
    low_resolution = torch.rand(2, 3, 8, 10)
    target = torch.rand(2, 3, 16, 20)
    timestamp = torch.tensor([0.0, 2.0])
    output = model(low_resolution, target, timestamp=timestamp)
    baseline = torch.nn.functional.interpolate(
        low_resolution, scale_factor=2, mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    assert torch.allclose(output.upscaled, baseline)
    assert output.predicted_latent.shape == (2, 8, 8, 10)
    assert output.target_latent is not None
    assert output.residual.shape == target.shape
    assert output.geometry.shape == (2, 4, 8, 10)
    assert not bool(output.context_mask.any())
    assert not torch.equal(output.geometry[0, 3], output.geometry[1, 3])
    loss = jepa_upscaler_loss(output, target)
    assert math.isfinite(loss.total.item())
    assert math.isfinite(loss.frequency.item())
    assert math.isfinite(loss.structure.item())
    assert math.isfinite(loss.pyramid.item())
    assert math.isfinite(loss.spectral_coherence.item())
    loss.total.backward()  # type: ignore[no-untyped-call]
    assert model.context_encoder.layers[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(0.5)
    model.update_target_encoder(momentum=0.0)
    assert all(
        torch.equal(target_parameter, online_parameter)
        for target_parameter, online_parameter in zip(
            model.target_encoder.parameters(), model.context_encoder.parameters(), strict=True
        )
    )
    with pytest.raises(ValueError, match="target resolution"):
        model(low_resolution, torch.rand(2, 3, 15, 20))
    with pytest.raises(ValueError, match="finite values"):
        model(torch.full((1, 3, 4, 4), torch.nan))


def test_jepa_upscaler_tiled_and_self_ensemble_runtime_paths() -> None:
    torch.manual_seed(15)
    model = JEPAUpscaler(JEPAUpscalerConfig(scale=2, feature_channels=8, residual_blocks=1)).eval()
    low_resolution = torch.rand(1, 3, 9, 11)
    direct = model(low_resolution).upscaled
    untiled = upscale_tiled(model, low_resolution, tile_size=16, overlap=2)
    tiled = upscale_tiled(model, low_resolution, tile_size=6, overlap=2)
    ensemble = upscale_self_ensemble(model, low_resolution)
    assert torch.equal(untiled, direct)
    assert tiled.shape == direct.shape
    assert ensemble.shape == direct.shape
    assert bool(torch.isfinite(tiled).all())
    assert bool(torch.isfinite(ensemble).all())
    with pytest.raises(ValueError, match="overlap"):
        upscale_tiled(model, low_resolution, tile_size=4, overlap=4)


def test_jepa_upscaler_masks_context_and_targets_high_resolution_detail() -> None:
    torch.manual_seed(14)
    model = JEPAUpscaler(JEPAUpscalerConfig(scale=2, feature_channels=8, residual_blocks=1))
    low = torch.full((1, 3, 4, 4), 0.5)
    first = torch.zeros((1, 3, 8, 8))
    first[..., ::2, ::2] = 1.0
    first[..., 1::2, 1::2] = 1.0
    second = 1.0 - first
    assert torch.allclose(torch.nn.functional.interpolate(first, size=(4, 4), mode="area"), low)
    assert torch.allclose(torch.nn.functional.interpolate(second, size=(4, 4), mode="area"), low)
    mask = torch.tensor(
        [[[True, True, False, False], [True, True, False, False], [False] * 4, [False] * 4]]
    )
    first_output = model(low, first, context_mask=mask)
    second_output = model(low, second, context_mask=mask)
    assert first_output.target_latent is not None
    assert second_output.target_latent is not None
    assert not torch.allclose(first_output.target_latent, second_output.target_latent)
    assert torch.equal(first_output.context_mask, mask)
    assert torch.isfinite(jepa_upscaler_loss(first_output, first).total)
    with pytest.raises(ValueError, match="leave at least one visible"):
        model(low, first, context_mask=torch.ones(1, 4, 4, dtype=torch.bool))


def _write_shard(path: Path, name: str, value: float) -> None:
    torch.save({name: torch.full((2, 2), value)}, path)


def test_spatiotemporal_lazy_store_plans_verifies_and_caches(tmp_path: Path) -> None:
    base_path = tmp_path / "base.pt"
    near_path = tmp_path / "near.pt"
    far_path = tmp_path / "far.pt"
    _write_shard(base_path, "base.weight", 1.0)
    _write_shard(near_path, "near.weight", 2.0)
    _write_shard(far_path, "far.weight", 3.0)
    base = parameter_shard_record("base", base_path, root=tmp_path, priority=1.0, always_load=True)
    near = parameter_shard_record(
        "near",
        near_path,
        root=tmp_path,
        spatial=SpatialBounds(minimum=(-1.0, -1.0, -1.0), maximum=(1.0, 1.0, 1.0)),
        temporal=TemporalBounds(start=0.0, end=10.0),
        resolution_min=0.5,
        resolution_max=2.0,
        priority=10.0,
    )
    far = parameter_shard_record(
        "far",
        far_path,
        root=tmp_path,
        spatial=SpatialBounds(minimum=(99.0, -1.0, -1.0), maximum=(101.0, 1.0, 1.0)),
        temporal=TemporalBounds(start=90.0, end=110.0),
        resolution_min=0.5,
        resolution_max=2.0,
        priority=10.0,
    )
    manifest_path = tmp_path / "manifest.json"
    build_parameter_manifest([base, near, far], manifest_path)
    store = SpatiotemporalParameterStore(
        manifest_path,
        cache_budget_bytes=base.byte_count + near.byte_count + far.byte_count,
    )
    near_query = SpatiotemporalQuery(
        position=(0.0, 0.0, 0.0),
        timestamp=5.0,
        resolution=1.0,
        velocity=(1.0, 0.0, 0.0),
        lookahead_seconds=0.5,
    )
    near_plan = store.plan(near_query, budget_bytes=base.byte_count + near.byte_count)
    assert [item.shard_id for item in near_plan.shards] == ["base", "near"]
    tensors = store.materialize(near_plan)
    assert set(tensors) == {"base.weight", "near.weight"}
    assert store.disk_loads == 2
    assert store.materialize(near_plan)["near.weight"][0, 0].item() == 2.0
    assert store.disk_loads == 2

    far_query = SpatiotemporalQuery(position=(100.0, 0.0, 0.0), timestamp=100.0, resolution=1.0)
    far_plan = store.plan(far_query, budget_bytes=base.byte_count + far.byte_count)
    assert [item.shard_id for item in far_plan.shards] == ["base", "far"]
    far_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="byte count changed|digest changed"):
        store.materialize(far_plan)


def test_lazy_store_materializes_only_selected_meta_parameters(tmp_path: Path) -> None:
    shard_path = tmp_path / "layer.pt"
    torch.save(
        {
            "0.weight": torch.eye(2),
            "0.bias": torch.tensor([1.0, 2.0]),
        },
        shard_path,
    )
    shard = parameter_shard_record("active-layer", shard_path, root=tmp_path, always_load=True)
    manifest = tmp_path / "manifest.json"
    build_parameter_manifest([shard], manifest)
    store = SpatiotemporalParameterStore(manifest, cache_budget_bytes=shard.byte_count)
    plan = store.plan(
        SpatiotemporalQuery(position=(0.0, 0.0, 0.0), timestamp=0.0, resolution=1.0),
        budget_bytes=shard.byte_count,
    )
    module = torch.nn.Sequential(
        torch.nn.Linear(2, 2, device="meta"),
        torch.nn.Linear(2, 1, device="meta"),
    )
    receipt = store.materialize_module(module, plan)
    assert receipt.loaded_tensors == ["0.bias", "0.weight"]
    assert receipt.remaining_meta_tensors == ["1.bias", "1.weight"]
    assert torch.equal(module[0](torch.zeros(1, 2)), torch.tensor([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="leaves module tensors unloaded"):
        store.materialize_module(module, plan, require_complete=True)


def test_hash_bound_jepa_upscaler_training_and_negative_result_retention(
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    train_high = torch.rand(4, 3, 8, 8)
    evaluation_high = torch.rand(2, 3, 8, 8)
    train_low = torch.nn.functional.interpolate(train_high, size=(4, 4), mode="area")
    evaluation_low = torch.nn.functional.interpolate(evaluation_high, size=(4, 4), mode="area")
    paths = {
        "train-low": train_low,
        "train-high": train_high,
        "evaluation-low": evaluation_low,
        "evaluation-high": evaluation_high,
    }
    for name, tensor in paths.items():
        torch.save(tensor, dataset / f"{name}.pt")
    manifest_path = dataset / "manifest.json"
    write_super_resolution_manifest(
        manifest_path,
        license_name="LicenseRef-Proprietary",
        train_source_ids=[f"train-{index}" for index in range(4)],
        evaluation_source_ids=[f"evaluation-{index}" for index in range(2)],
        train_low_resolution=dataset / "train-low.pt",
        train_high_resolution=dataset / "train-high.pt",
        evaluation_low_resolution=dataset / "evaluation-low.pt",
        evaluation_high_resolution=dataset / "evaluation-high.pt",
    )
    output = tmp_path / "run"
    report = train_jepa_upscaler(
        manifest_path,
        output,
        config=JEPAUpscalerTrainingConfig(
            seed=17,
            steps=2,
            batch_size=2,
            model=JEPAUpscalerConfig(
                scale=2,
                input_channels=3,
                feature_channels=8,
                residual_blocks=1,
            ),
        ),
    )
    assert report.status == "COMPLETED_DEVELOPMENT_NOT_PROMOTED"
    assert report.completed_steps == 2
    assert report.promotion_authorized is False
    assert math.isfinite(report.final_training_loss)
    assert math.isfinite(report.evaluation.psnr_improvement)
    assert -1.0 <= report.evaluation.model_ssim <= 1.0
    assert math.isfinite(report.evaluation.model_frequency_error)
    assert set(report.final_loss_components) == {
        "reconstruction",
        "edge",
        "joint_embedding",
        "latent_geometry",
        "variance",
        "frequency",
        "structure",
        "pyramid",
        "spectral_coherence",
    }
    assert Path(report.checkpoint_path).is_file()
    assert (output / "report.json").is_file()
    loaded_model, loaded_config = load_jepa_upscaler_checkpoint(
        Path(report.checkpoint_path),
        expected_sha256=report.checkpoint_sha256,
        expected_dataset_manifest_sha256=report.dataset_manifest_sha256,
    )
    assert loaded_config == JEPAUpscalerTrainingConfig(
        seed=17,
        steps=2,
        batch_size=2,
        model=JEPAUpscalerConfig(
            scale=2,
            input_channels=3,
            feature_channels=8,
            residual_blocks=1,
        ),
    )
    assert loaded_model(evaluation_low).upscaled.shape == evaluation_high.shape
    with pytest.raises(ValueError, match="digest"):
        load_jepa_upscaler_checkpoint(Path(report.checkpoint_path), expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="dataset manifest"):
        load_jepa_upscaler_checkpoint(
            Path(report.checkpoint_path), expected_dataset_manifest_sha256="0" * 64
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        train_jepa_upscaler(
            manifest_path,
            output,
            config=JEPAUpscalerTrainingConfig(
                steps=1,
                batch_size=2,
                model=JEPAUpscalerConfig(feature_channels=8, residual_blocks=1),
            ),
        )

    (dataset / "train-low.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest changed"):
        train_jepa_upscaler(
            manifest_path,
            tmp_path / "tampered-run",
            config=JEPAUpscalerTrainingConfig(
                steps=1,
                batch_size=2,
                model=JEPAUpscalerConfig(feature_channels=8, residual_blocks=1),
            ),
        )
