from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from olympus.models.encoder_quantization import (
    EncoderQuantizationConfig,
    QuantizedEncoder,
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
from olympus.upscaler.jepa import JEPAUpscaler, JEPAUpscalerConfig, jepa_upscaler_loss
from olympus.upscaler.training import (
    JEPAUpscalerTrainingConfig,
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
    ]
    assert model_spec("hermes").target_parameters == 12_000_000_000
    assert model_spec("prometheus").target_parameters == 128_000_000_000
    assert model_spec("perseus").target_parameters == 1_024_000_000_000
    assert model_spec("atlas").target_parameters is None
    assert model_spec("kronos").target_parameters == 8_000_000_000_000
    assert all(not spec.capability_claimed for spec in BU1LD_MODEL_SPECS)


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
    assert output.geometry.shape == (2, 4, 8, 10)
    assert not torch.equal(output.geometry[0, 3], output.geometry[1, 3])
    loss = jepa_upscaler_loss(output, target)
    assert math.isfinite(loss.total.item())
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
    assert Path(report.checkpoint_path).is_file()
    assert (output / "report.json").is_file()
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
