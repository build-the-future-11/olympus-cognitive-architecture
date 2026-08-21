from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from olympus.foundry import resources
from olympus.foundry.data_pipeline import (
    InstructionExample,
    prepare_instruction_dataset,
    sha256_bytes,
    verify_dataset_manifest,
)
from olympus.foundry.eval_suite import (
    CategoryEvaluation,
    HeldOutEvaluation,
    WorkflowScore,
    evaluate_checkpoint,
)
from olympus.foundry.promotion import evaluate_promotion
from olympus.foundry.quantization import (
    QuantizationReport,
    load_quantized_model,
    quantize_checkpoint,
)
from olympus.foundry.resources import MemorySnapshot, ResourceGovernor
from olympus.foundry.sft import (
    ByteTokenizer,
    QLoRALinear,
    SFTConfig,
    TinyCausalLM,
    TinyModelConfig,
    _pack_nibbles,
    _unpack_nibbles,
    generate_text,
    load_trained_model,
    run_sft,
)


def _source() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets/hermes-smoke/source.jsonl"


def _small_config(*, mode: str = "full", epochs: int = 1) -> SFTConfig:
    return SFTConfig(
        mode=mode,  # type: ignore[arg-type]
        seed=31,
        epochs=epochs,
        batch_size=3,
        learning_rate=0.002,
        gradient_accumulation_steps=2,
        mixed_precision=True,
        model=TinyModelConfig(width=32, layers=1, heads=4, max_sequence_tokens=160),
    )


def test_dataset_preparation_is_content_addressed_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    manifest = prepare_instruction_dataset(_source(), tmp_path / "prepared")
    assert manifest.quality.accepted_records == 36
    assert manifest.quality.missing_category_split_pairs == []
    assert {split.records for split in manifest.splits} == {12}
    assert all(len(split.categories) == 12 for split in manifest.splits)
    assert verify_dataset_manifest(tmp_path / "prepared/manifest.json") == manifest
    second = prepare_instruction_dataset(_source(), tmp_path / "second-root")
    assert second.manifest_sha256 == manifest.manifest_sha256

    split = tmp_path / "prepared" / manifest.splits[0].path
    split.write_text(split.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="split hash mismatch"):
        verify_dataset_manifest(tmp_path / "prepared/manifest.json")


def test_dataset_rejects_duplicate_pii_license_leakage_and_missing_coverage(
    tmp_path: Path,
) -> None:
    lines = _source().read_text(encoding="utf-8").splitlines()
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text("\n".join(lines + [lines[0]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate record IDs"):
        prepare_instruction_dataset(duplicate, tmp_path / "duplicate-output")

    pii_payload = json.loads(lines[0])
    pii_payload["prompt"] = "Send this complete report to real.person@example.com immediately."
    pii = tmp_path / "pii.jsonl"
    pii.write_text(json.dumps(pii_payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="possible PII"):
        prepare_instruction_dataset(pii, tmp_path / "pii-output")

    license_payload = json.loads(lines[0])
    license_payload["license"] = "unknown"
    with pytest.raises(ValidationError, match="unsupported training license"):
        InstructionExample.model_validate(license_payload)

    incomplete = tmp_path / "incomplete.jsonl"
    incomplete.write_text(lines[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required category/split coverage"):
        prepare_instruction_dataset(incomplete, tmp_path / "incomplete-output")

    contaminated = [json.loads(line) for line in lines]
    contaminated[1]["prompt"] = contaminated[0]["prompt"] + " unique validation suffix"
    contaminated_path = tmp_path / "contaminated.jsonl"
    contaminated_path.write_text(
        "\n".join(json.dumps(item) for item in contaminated) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cross-split 8-token contamination"):
        prepare_instruction_dataset(contaminated_path, tmp_path / "contaminated-output")


def test_resource_governor_enforces_exclusivity_and_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe = MemorySnapshot(16_000, 8_000, 4_000, 1_000, 500)
    monkeypatch.setattr(resources, "memory_snapshot", lambda: safe)
    first = ResourceGovernor(tmp_path / "workload.lock", min_available_bytes=256 * 1024**2)
    first.min_available_bytes = 1_000
    with first:
        assert first.preflight == safe
        second = ResourceGovernor(tmp_path / "workload.lock", min_available_bytes=256 * 1024**2)
        second.min_available_bytes = 1_000
        with pytest.raises(RuntimeError, match="holds the lock"):
            second.acquire()

    low = MemorySnapshot(16_000, 500, 4_000, 1_000, 500)
    monkeypatch.setattr(resources, "memory_snapshot", lambda: low)
    governor = ResourceGovernor(tmp_path / "low.lock", min_available_bytes=256 * 1024**2)
    governor.min_available_bytes = 1_000
    with pytest.raises(RuntimeError, match="safety floor"):
        governor.acquire()

    swap = MemorySnapshot(16_000, 8_000, 4_000, 3_900, 500)
    monkeypatch.setattr(resources, "memory_snapshot", lambda: swap)
    governor = ResourceGovernor(
        tmp_path / "swap.lock", min_available_bytes=256 * 1024**2, max_swap_fraction=0.9
    )
    governor.min_available_bytes = 1_000
    with pytest.raises(RuntimeError, match="safety ceiling"):
        governor.acquire()
    with pytest.raises(ValueError, match="at least 256 MiB"):
        ResourceGovernor(tmp_path / "invalid.lock", min_available_bytes=1)


def test_tokenizer_model_and_nibble_quantization_are_real() -> None:
    text = "Olympus ✓"
    assert ByteTokenizer.decode(ByteTokenizer.encode(text, bos=True, eos=True)) == text
    values = torch.tensor([[1, 15, 8], [4, 9, 2]], dtype=torch.uint8)
    packed = _pack_nibbles(values)
    assert torch.equal(_unpack_nibbles(packed, values.numel()), values.flatten())
    with pytest.raises(ValidationError, match="divisible"):
        TinyModelConfig(width=30, heads=4)
    model = TinyCausalLM(TinyModelConfig(width=32, layers=1, heads=4, max_sequence_tokens=32))
    assert model(torch.ones((2, 12), dtype=torch.long)).shape == (2, 12, 260)
    with pytest.raises(ValueError, match="sequence limit"):
        model(torch.ones((1, 33), dtype=torch.long))
    qlora = QLoRALinear(torch.nn.Linear(32, 16), rank=2, alpha=4)
    output = qlora(torch.ones((2, 3, 32)))
    output.sum().backward()
    assert qlora.lora_a.grad is not None


@pytest.fixture(scope="module")
def trained_foundry(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("deep-foundry")
    prepare_instruction_dataset(_source(), root / "dataset")
    manifest = root / "dataset/manifest.json"
    full = run_sft(manifest, root / "training", config=_small_config())
    checkpoint = Path(full.checkpoint_path)
    resumed = run_sft(
        manifest,
        root / "training",
        config=_small_config(epochs=2),
        resume_checkpoint=checkpoint,
    )
    full_checkpoint = Path(resumed.checkpoint_path)
    lora = run_sft(
        manifest,
        root / "training",
        config=_small_config(mode="lora"),
        base_checkpoint=full_checkpoint,
    )
    qlora = run_sft(
        manifest,
        root / "training",
        config=_small_config(mode="qlora"),
        base_checkpoint=full_checkpoint,
    )
    return {
        "root": root,
        "manifest": manifest,
        "full": full_checkpoint,
        "lora": Path(lora.checkpoint_path),
        "qlora": Path(qlora.checkpoint_path),
        "adapter": Path(qlora.adapter_path or ""),
    }


def test_sft_full_lora_qlora_checkpoint_resume_and_generation(
    trained_foundry: dict[str, Path],
) -> None:
    assert trained_foundry["adapter"].is_file()
    full = load_trained_model(trained_foundry["full"])
    generated = generate_text(
        full, "Return a short answer.", device=torch.device("cpu"), max_new_tokens=4
    )
    assert isinstance(generated, str)
    assert load_trained_model(trained_foundry["lora"])
    assert load_trained_model(trained_foundry["qlora"])
    with pytest.raises(ValueError, match="require a base checkpoint"):
        run_sft(
            trained_foundry["manifest"],
            trained_foundry["root"] / "invalid-adapter",
            config=_small_config(mode="lora"),
        )
    with pytest.raises(ValueError, match="does not accept a base"):
        run_sft(
            trained_foundry["manifest"],
            trained_foundry["root"] / "invalid-full",
            config=_small_config(),
            base_checkpoint=trained_foundry["full"],
        )


def test_held_out_evaluation_quantization_and_negative_promotion(
    trained_foundry: dict[str, Path], tmp_path: Path
) -> None:
    evaluation_path = tmp_path / "evaluation.json"
    evaluation = evaluate_checkpoint(
        trained_foundry["full"],
        trained_foundry["manifest"],
        evaluation_path,
        max_generation_tokens=2,
    )
    assert len(evaluation.category_results) == 12
    assert evaluation.test_records == 12
    assert evaluation.checkpoint_sha256 == sha256_bytes(trained_foundry["full"].read_bytes())

    int8 = quantize_checkpoint(
        trained_foundry["full"], trained_foundry["manifest"], tmp_path / "int8", bits=8
    )
    int4 = quantize_checkpoint(
        trained_foundry["full"], trained_foundry["manifest"], tmp_path / "int4", bits=4
    )
    assert int8.quantized_bytes < int8.float_weights_bytes
    assert int4.quantized_bytes < int8.quantized_bytes
    assert load_quantized_model(Path(int4.artifact_path))
    with pytest.raises(ValueError, match="merged full checkpoint"):
        quantize_checkpoint(
            trained_foundry["qlora"],
            trained_foundry["manifest"],
            tmp_path / "invalid-quant",
            bits=4,
        )

    report = evaluate_promotion(
        requested_model_id="hermes-alpha",
        checkpoint_path=trained_foundry["full"],
        dataset_manifest_path=trained_foundry["manifest"],
        evaluation_path=evaluation_path,
        quantization_report_path=tmp_path / "int4/int4-report.json",
        model_card_path=Path(__file__),
        output_path=tmp_path / "promotion.json",
        approved_base_license="LicenseRef-Proprietary",
    )
    assert report.status == "NOT_PROMOTED"
    assert report.release_manifest_path is None
    assert "training_scale" in report.blockers


def test_promotion_gate_can_emit_a_hash_bound_release_manifest(
    trained_foundry: dict[str, Path], tmp_path: Path
) -> None:
    checkpoint = trained_foundry["full"]
    checkpoint_sha = sha256_bytes(checkpoint.read_bytes())
    manifest = verify_dataset_manifest(trained_foundry["manifest"])
    enlarged_splits = []
    for split in manifest.splits:
        split = split.model_copy(
            update={"path": str(trained_foundry["manifest"].parent / split.path)}
        )
        if split.name == "train":
            enlarged_splits.append(split.model_copy(update={"records": 10_000}))
        elif split.name == "test":
            enlarged_splits.append(
                split.model_copy(
                    update={"records": 1_200, "categories": {key: 100 for key in split.categories}}
                )
            )
        else:
            enlarged_splits.append(split)
    manifest = manifest.model_copy(update={"splits": enlarged_splits, "manifest_sha256": None})
    manifest.manifest_sha256 = sha256_bytes(manifest.canonical_bytes(include_hash=False))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest.canonical_bytes())

    category_results = [
        CategoryEvaluation(
            category=category,
            records=100,
            baseline_loss=2.0,
            candidate_loss=1.0,
            loss_change_fraction=-0.5,
            regressed=False,
        )
        for category in manifest.splits[-1].categories
    ]
    workflows = [
        WorkflowScore(
            name=name,
            categories=["planning"],
            examples=100,
            exact_match_rate=0.9,
            nonempty_rate=1.0,
        )
        for name in ("tool_workflow", "result_comparison", "multi_step_planning")
    ]
    evaluation = HeldOutEvaluation(
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_sha,
        dataset_manifest_sha256=manifest.manifest_sha256,
        baseline_identity="frozen",
        test_records=1_200,
        overall_baseline_loss=2.0,
        overall_candidate_loss=1.0,
        overall_baseline_perplexity=7.0,
        overall_candidate_perplexity=3.0,
        category_results=category_results,
        regression_count=0,
        exact_match_rate=0.9,
        format_compliance_rate=0.98,
        workflow_scores=workflows,
        median_generation_latency_ms=10,
        generated_bytes_per_second=100,
        peak_rss_bytes=1_000,
        passed_smoke_quality_gate=True,
        decision="fixture passed",
    )
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(evaluation.model_dump_json(), encoding="utf-8")
    quantization = QuantizationReport(
        format="fixture-int4",
        bits=4,
        source_checkpoint_sha256=checkpoint_sha,
        artifact_path="fixture",
        artifact_sha256="a" * 64,
        float_weights_bytes=100,
        quantized_bytes=40,
        size_reduction_fraction=0.6,
        source_loss=1.0,
        quantized_loss=1.01,
        loss_change_fraction=0.01,
        source_latency_ms=10,
        quantized_latency_ms=8,
        startup_latency_ms=5,
        max_context_tokens=1_024,
        context_limit_enforced=True,
        tool_exact_match_rate=0.8,
        peak_rss_bytes=1_000,
        passed_quality_gate=True,
    )
    quantization_path = tmp_path / "quantization.json"
    quantization_path.write_text(quantization.model_dump_json(), encoding="utf-8")
    card = tmp_path / "MODEL_CARD.md"
    card.write_text(
        f"# Test model\n\nCheckpoint: {checkpoint_sha}\n\nLicense: Apache-2.0\n\nLimitations\n",
        encoding="utf-8",
    )
    report = evaluate_promotion(
        requested_model_id="fixture-model",
        checkpoint_path=checkpoint,
        dataset_manifest_path=manifest_path,
        evaluation_path=evaluation_path,
        quantization_report_path=quantization_path,
        model_card_path=card,
        output_path=tmp_path / "promotion.json",
        approved_base_license="Apache-2.0",
        serving_verification={"passed": True, "checkpoint_sha256": checkpoint_sha},
    )
    assert report.status == "PROMOTED"
    assert Path(report.release_manifest_path or "").is_file()
