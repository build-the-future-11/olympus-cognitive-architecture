from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from subprocess import CalledProcessError

import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from olympus.foundry import promotion, resources
from olympus.foundry.attestation import (
    AttestationBundle,
    AttestationKind,
    AttestationStatement,
    AttestationTrustStore,
    PromotionEvidenceSubject,
    SignedAttestation,
    TrustedAttestationKey,
    sign_attestation,
)
from olympus.foundry.data_pipeline import (
    InstructionExample,
    prepare_instruction_dataset,
    sha256_bytes,
    verify_dataset_manifest,
)
from olympus.foundry.eval_suite import (
    WORKFLOW_CATEGORIES,
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
    _training_run_id,
    _unpack_nibbles,
    evaluate_loss,
    generate_text,
    load_trained_model,
    run_sft,
)


def _source() -> Path:
    return Path(__file__).resolve().parents[1] / "datasets/hermes-smoke/source.jsonl"


def test_validation_loss_weights_tokens_not_batches() -> None:
    torch.manual_seed(19)
    model = TinyCausalLM(TinyModelConfig(width=16, heads=2, layers=1, max_sequence_tokens=32))
    rows = [([1] + [5 + index] * (index + 1), [-100] + [5 + index] * (index + 1))
            for index in range(9)]
    expected = sum(
        evaluate_loss(model, [row], device=torch.device("cpu")) * (len(row[0]) - 1)
        for row in rows
    ) / sum(len(row[0]) - 1 for row in rows)
    assert evaluate_loss(model, rows, device=torch.device("cpu")) == pytest.approx(
        expected, rel=1e-6
    )
    assert evaluate_loss(model, list(reversed(rows)), device=torch.device("cpu")) == (
        pytest.approx(expected, rel=1e-6)
    )
    with pytest.raises(ValueError, match="no supervised tokens"):
        evaluate_loss(model, [([1, 2], [-100, -100])], device=torch.device("cpu"))


@pytest.mark.parametrize("weights", [
    {"saftey": 1.0}, {"safety": -1.0}, {"safety": 65.0}, {"safety": float("inf")},
])
def test_training_mixture_rejects_invalid_or_unbounded_weights(weights: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        SFTConfig(category_mix_weights=weights)


@pytest.mark.parametrize("vocabulary", [1, 259, 261, 2_000_000_000])
def test_model_vocabulary_matches_the_fixed_byte_tokenizer(vocabulary: int) -> None:
    with pytest.raises(ValidationError):
        TinyModelConfig(vocab_size=vocabulary)


def test_evaluation_and_quantization_ignore_training_sampling(
    trained_foundry: dict[str, Path], tmp_path: Path,
) -> None:
    original = evaluate_checkpoint(trained_foundry["full"], trained_foundry["manifest"],
                                   tmp_path / "original.json", max_generation_tokens=2)
    payload = torch.load(trained_foundry["full"], weights_only=True)
    payload["training_config"]["category_mix_weights"] = {"safety": 0, "reasoning": 8}
    mixed_checkpoint = tmp_path / "mixed.pt"
    torch.save(payload, mixed_checkpoint)
    mixed = evaluate_checkpoint(mixed_checkpoint, trained_foundry["manifest"],
                                tmp_path / "mixed.json", max_generation_tokens=2)
    assert mixed.category_results == original.category_results
    assert mixed.overall_candidate_loss == original.overall_candidate_loss
    assert mixed.test_records == 12
    quantized = quantize_checkpoint(mixed_checkpoint, trained_foundry["manifest"],
                                    tmp_path / "quantized", bits=4)
    assert quantized.source_loss == pytest.approx(original.overall_candidate_loss)
    assert quantized.dataset_manifest_sha256 == original.dataset_manifest_sha256
    payload["dataset_manifest_sha256"] = "f" * 64
    torch.save(payload, mixed_checkpoint)
    rejected_output = tmp_path / "wrong-dataset"
    with pytest.raises(ValueError, match="dataset hashes do not match"):
        quantize_checkpoint(mixed_checkpoint, trained_foundry["manifest"], rejected_output, bits=4)
    assert not rejected_output.exists()


def test_validation_population_is_independent_of_training_mixture(tmp_path: Path) -> None:
    prepare_instruction_dataset(_source(), tmp_path / "dataset")
    manifest = tmp_path / "dataset/manifest.json"
    original = run_sft(manifest, tmp_path / "original", config=_small_config())
    mixed = run_sft(manifest, tmp_path / "mixed", config=_small_config().model_copy(update={
        "category_mix_weights": {"safety": 0, "reasoning": 2},
    }))
    assert mixed.initial_validation_loss == original.initial_validation_loss


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


def _write_valid_attestations(
    root: Path,
    subject: PromotionEvidenceSubject,
) -> tuple[Path, Path]:
    evaluation_key = Ed25519PrivateKey.generate()
    release_key = Ed25519PrivateKey.generate()
    trust = AttestationTrustStore(
        keys=[
            TrustedAttestationKey(
                key_id="evaluation-key",
                issuer_id="evaluation-lab",
                runner_id="evaluation-runner",
                runner_source_sha256="f" * 64,
                public_key_base64=base64.b64encode(
                    evaluation_key.public_key().public_bytes(
                        serialization.Encoding.Raw,
                        serialization.PublicFormat.Raw,
                    )
                ).decode("ascii"),
                allowed_kinds={"evaluation", "quantization"},
            ),
            TrustedAttestationKey(
                key_id="release-key",
                issuer_id="release-lab",
                runner_id="release-runner",
                runner_source_sha256="f" * 64,
                public_key_base64=base64.b64encode(
                    release_key.public_key().public_bytes(
                        serialization.Encoding.Raw,
                        serialization.PublicFormat.Raw,
                    )
                ).decode("ascii"),
                allowed_kinds={"license_review", "serving"},
            ),
        ]
    )
    issued = datetime.now(UTC) - timedelta(minutes=1)
    expires = issued + timedelta(days=1)
    statements: tuple[tuple[AttestationKind, str, str, Ed25519PrivateKey], ...] = (
        ("evaluation", "evaluation-lab", "evaluation-key", evaluation_key),
        ("quantization", "evaluation-lab", "evaluation-key", evaluation_key),
        ("license_review", "release-lab", "release-key", release_key),
        ("serving", "release-lab", "release-key", release_key),
    )
    attestations: list[SignedAttestation] = []
    for kind, issuer, key_id, private_key in statements:
        evidence_sha = {
            "evaluation": subject.evaluation_sha256,
            "quantization": subject.quantization_report_sha256,
            "license_review": subject.model_card_sha256,
            "serving": subject.serving_verification_sha256,
        }[kind]
        statement = AttestationStatement(
            attestation_id=f"fixture-{kind}",
            kind=kind,
            issuer_id=issuer,
            runner_id=(
                "evaluation-runner"
                if kind in {"evaluation", "quantization"}
                else "release-runner"
            ),
            runner_source_sha256="f" * 64,
            issued_at=issued,
            expires_at=expires,
            outcome="approved" if kind == "license_review" else "passed",
            evidence_sha256=evidence_sha,
            subject=subject,
        )
        attestations.append(
            sign_attestation(statement, key_id=key_id, private_key=private_key)
        )
    trust_path = root / "attestation-trust.json"
    bundle_path = root / "attestation-bundle.json"
    trust_path.write_text(trust.model_dump_json(indent=2), encoding="utf-8")
    bundle_path.write_text(
        AttestationBundle(attestations=attestations).model_dump_json(indent=2),
        encoding="utf-8",
    )
    return bundle_path, trust_path


def test_training_output_identity_binds_full_configuration_and_base_checkpoint() -> None:
    dataset_sha = "d" * 64
    config = _small_config()
    run_id = _training_run_id(dataset_sha, config, None)
    assert run_id == _training_run_id(dataset_sha, config, None)
    assert run_id != _training_run_id(dataset_sha, config.model_copy(update={"epochs": 2}), None)
    assert run_id != _training_run_id(dataset_sha, config, "a" * 64)


def test_dataset_preparation_is_content_addressed_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    manifest = prepare_instruction_dataset(_source(), tmp_path / "prepared")
    manifest_path = tmp_path / "prepared/manifest.json"
    assert manifest.quality.accepted_records == 36
    assert manifest.quality.missing_category_split_pairs == []
    assert {split.records for split in manifest.splits} == {12}
    assert all(len(split.categories) == 12 for split in manifest.splits)
    assert verify_dataset_manifest(manifest_path) == manifest
    second = prepare_instruction_dataset(_source(), tmp_path / "second-root")
    assert second.manifest_sha256 == manifest.manifest_sha256

    forged_splits = [
        split.model_copy(update={"records": 10_000}) if split.name == "train" else split
        for split in manifest.splits
    ]
    forged = manifest.model_copy(update={"splits": forged_splits, "manifest_sha256": None})
    forged.manifest_sha256 = sha256_bytes(forged.canonical_bytes(include_hash=False))
    forged_path = tmp_path / "prepared/forged-manifest.json"
    forged_path.write_bytes(forged.canonical_bytes())
    with pytest.raises(ValueError, match="split record count mismatch"):
        verify_dataset_manifest(forged_path)

    missing_split = manifest.model_copy(
        update={
            "splits": [split for split in manifest.splits if split.name != "validation"],
            "manifest_sha256": None,
        }
    )
    missing_split.manifest_sha256 = sha256_bytes(missing_split.canonical_bytes(include_hash=False))
    missing_split_path = tmp_path / "prepared/missing-split-manifest.json"
    missing_split_path.write_bytes(missing_split.canonical_bytes())
    with pytest.raises(ValueError, match="exactly the train, validation, and test splits"):
        verify_dataset_manifest(missing_split_path)

    outside = tmp_path / "outside-train.jsonl"
    outside.write_bytes((tmp_path / "prepared" / manifest.splits[0].path).read_bytes())
    escaped_splits = [
        split.model_copy(update={"path": str(outside)})
        if split.name == manifest.splits[0].name
        else split
        for split in manifest.splits
    ]
    escaped = manifest.model_copy(
        update={"splits": escaped_splits, "manifest_sha256": None}
    )
    escaped.manifest_sha256 = sha256_bytes(escaped.canonical_bytes(include_hash=False))
    escaped_path = tmp_path / "prepared/escaped-manifest.json"
    escaped_path.write_bytes(escaped.canonical_bytes())
    with pytest.raises(ValueError, match="split path must be relative"):
        verify_dataset_manifest(escaped_path)

    split = tmp_path / "prepared" / manifest.splits[0].path
    split.write_text(split.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="split hash mismatch"):
        verify_dataset_manifest(manifest_path)


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

    def snapshot_failure() -> MemorySnapshot:
        raise RuntimeError("snapshot failed")

    lock_path = tmp_path / "snapshot-error.lock"
    monkeypatch.setattr(resources, "memory_snapshot", snapshot_failure)
    failing = ResourceGovernor(lock_path, min_available_bytes=256 * 1024**2)
    failing.min_available_bytes = 1_000
    with pytest.raises(RuntimeError, match="snapshot failed"):
        failing.acquire()

    monkeypatch.setattr(resources, "memory_snapshot", lambda: safe)
    retry = ResourceGovernor(lock_path, min_available_bytes=256 * 1024**2)
    retry.min_available_bytes = 1_000
    retry.acquire()
    retry.release()

    with pytest.raises(ValueError, match="at least 256 MiB"):
        ResourceGovernor(tmp_path / "invalid.lock", min_available_bytes=1)


def test_macos_memory_falls_back_when_sysctl_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vm_stat = """Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free: 100.
Pages inactive: 200.
Pages speculative: 10.
Pages purgeable: 5.
"""

    def run(command: list[str]) -> str:
        if command == ["vm_stat"]:
            return vm_stat
        raise CalledProcessError(1, command)

    monkeypatch.setattr(resources, "_run", run)
    monkeypatch.setattr(
        os,
        "sysconf",
        lambda name: {"SC_PHYS_PAGES": 1_000, "SC_PAGE_SIZE": 4_096}[name],
    )

    snapshot = resources._macos_memory()
    assert snapshot.total_bytes == 4_096_000
    assert snapshot.available_bytes == 315 * 4_096
    assert snapshot.swap_total_bytes == 0
    assert snapshot.swap_used_bytes == 0


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


def test_dropout_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    prepare_instruction_dataset(_source(), tmp_path / "dataset")
    manifest = tmp_path / "dataset/manifest.json"
    config = _small_config().model_copy(update={
        "model": _small_config().model.model_copy(update={"dropout": 0.2}),
    })
    first = run_sft(manifest, tmp_path / "split", config=config)
    target = config.model_copy(update={"epochs": 2})
    resumed = run_sft(manifest, tmp_path / "resumed", config=target,
                      resume_checkpoint=Path(first.checkpoint_path))
    uninterrupted = run_sft(manifest, tmp_path / "continuous", config=target)
    split_model = load_trained_model(Path(resumed.checkpoint_path))
    continuous_model = load_trained_model(Path(uninterrupted.checkpoint_path))
    for name, parameter in split_model.state_dict().items():
        torch.testing.assert_close(parameter, continuous_model.state_dict()[name], rtol=0, atol=0)


def test_gradient_accumulation_matches_full_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Observe gradients before Adam amplifies floating-point noise near zero.
    # Fixed weights isolate accumulation mathematics, including the partial group.
    gradients: list[list[torch.Tensor]] = []

    def capture_step(optimizer: torch.optim.AdamW, closure: object = None) -> None:
        gradients.append([
            parameter.grad.detach().clone()
            for group in optimizer.param_groups for parameter in group["params"]
            if parameter.grad is not None
        ])

    monkeypatch.setattr(torch.optim.AdamW, "step", capture_step)
    prepare_instruction_dataset(_source(), tmp_path / "dataset")
    manifest = tmp_path / "dataset/manifest.json"
    config = _small_config().model_copy(update={
        "pack_sequences": False, "batch_size": 1, "gradient_accumulation_steps": 5,
    })
    run_sft(manifest, tmp_path / "micro", config=config)
    micro_gradients = gradients.copy()
    gradients.clear()
    run_sft(manifest, tmp_path / "full", config=config.model_copy(update={
        "batch_size": 5, "gradient_accumulation_steps": 1,
    }))
    assert len(gradients) == len(micro_gradients) == 3
    for micro_group, full_group in zip(micro_gradients, gradients, strict=True):
        for micro_gradient, full_gradient in zip(micro_group, full_group, strict=True):
            torch.testing.assert_close(micro_gradient, full_gradient, rtol=1e-4, atol=1e-6)


def test_resume_rejects_caller_configuration_mismatch(tmp_path: Path) -> None:
    prepare_instruction_dataset(_source(), tmp_path / "dataset")
    manifest = tmp_path / "dataset/manifest.json"
    initial = run_sft(manifest, tmp_path / "training", config=_small_config())
    checkpoint = Path(initial.checkpoint_path)

    mismatched_model = _small_config(epochs=2).model_copy(
        update={"model": TinyModelConfig(width=48, layers=1, heads=4, max_sequence_tokens=160)}
    )
    with pytest.raises(ValueError, match="only the epoch target may change"):
        run_sft(
            manifest,
            tmp_path / "training",
            config=mismatched_model,
            resume_checkpoint=checkpoint,
        )

    mismatched_optimizer = _small_config(epochs=2).model_copy(update={"learning_rate": 0.004})
    with pytest.raises(ValueError, match="only the epoch target may change"):
        run_sft(
            manifest,
            tmp_path / "training",
            config=mismatched_optimizer,
            resume_checkpoint=checkpoint,
        )

    resumed = run_sft(
        manifest,
        tmp_path / "training",
        config=_small_config(epochs=2),
        resume_checkpoint=checkpoint,
    )
    payload = torch.load(resumed.checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["model_config"] == _small_config().model.model_dump(mode="json")
    assert payload["training_config"]["epochs"] == 2


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
    assert evaluation.passed_smoke_quality_gate is False
    for field, value in (
        ("schema_version", 1),
        ("regression_count", evaluation.regression_count + 1),
        ("category_results", []),
        ("workflow_scores", []),
        ("test_records", evaluation.test_records + 1),
        ("overall_baseline_perplexity", evaluation.overall_baseline_perplexity + 1),
        ("passed_smoke_quality_gate", True),
    ):
        payload = evaluation.model_dump()
        payload[field] = value
        with pytest.raises(ValidationError):
            HeldOutEvaluation.model_validate(payload)
    category_payload = evaluation.category_results[0].model_dump()
    category_payload["regressed"] = not category_payload["regressed"]
    with pytest.raises(ValidationError):
        CategoryEvaluation.model_validate(category_payload)

    int8 = quantize_checkpoint(
        trained_foundry["full"], trained_foundry["manifest"], tmp_path / "int8", bits=8
    )
    int4 = quantize_checkpoint(
        trained_foundry["full"], trained_foundry["manifest"], tmp_path / "int4", bits=4
    )
    assert int8.quantized_bytes < int8.float_weights_bytes
    assert int4.quantized_bytes < int8.quantized_bytes
    assert int4.passed_quality_gate is False
    for field, value in (
        ("schema_version", 1),
        ("passed_quality_gate", True),
        ("loss_change_fraction", int4.loss_change_fraction + 1),
        ("size_reduction_fraction", int4.size_reduction_fraction + 1),
    ):
        quantization_payload = int4.model_dump()
        quantization_payload[field] = value
        with pytest.raises(ValidationError):
            QuantizationReport.model_validate(quantization_payload)
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
    original = evaluation_path.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        evaluate_promotion(
            requested_model_id="hermes-alpha",
            checkpoint_path=trained_foundry["full"],
            dataset_manifest_path=trained_foundry["manifest"],
            evaluation_path=evaluation_path,
            quantization_report_path=tmp_path / "int4/int4-report.json",
            model_card_path=Path(__file__),
            output_path=evaluation_path,
            approved_base_license="LicenseRef-Proprietary",
        )
    assert evaluation_path.read_bytes() == original


def test_candidate_requires_valid_independent_attestation_to_promote(
    trained_foundry: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(promotion, "verify_dataset_manifest", lambda _: manifest)

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
            categories=list(categories),
            examples=100 * len(categories),
            exact_match_rate=0.9,
            nonempty_rate=1.0,
        )
        for name, categories in WORKFLOW_CATEGORIES.items()
    ]
    evaluation = HeldOutEvaluation(
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_sha,
        dataset_manifest_sha256=manifest.manifest_sha256,
        baseline_identity="frozen",
        test_records=1_200,
        overall_baseline_loss=2.0,
        overall_candidate_loss=1.0,
        overall_baseline_perplexity=math.exp(2),
        overall_candidate_perplexity=math.exp(1),
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
    quantized_artifact = tmp_path / "fixture-int4.bin"
    quantized_artifact.write_bytes(b"fixture quantized weights")
    quantization = QuantizationReport(
        dataset_manifest_sha256=manifest.manifest_sha256,
        format="fixture-int4",
        bits=4,
        source_checkpoint_sha256=checkpoint_sha,
        artifact_path=str(quantized_artifact),
        artifact_sha256=sha256_bytes(quantized_artifact.read_bytes()),
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
        (
            f"# Test model\n\nCheckpoint: {checkpoint_sha}\n\n"
            "## License\n\nApache-2.0\n\n"
            "## Intended use\n\nStructural promotion testing.\n\n"
            "## Limitations\n\nSynthetic fixture only.\n"
        ),
        encoding="utf-8",
    )
    serving_verification = {"passed": True, "checkpoint_sha256": checkpoint_sha}
    report = evaluate_promotion(
        requested_model_id="fixture-model",
        checkpoint_path=checkpoint,
        dataset_manifest_path=manifest_path,
        evaluation_path=evaluation_path,
        quantization_report_path=quantization_path,
        model_card_path=card,
        output_path=tmp_path / "promotion.json",
        approved_base_license="Apache-2.0",
        serving_verification=serving_verification,
    )
    assert report.status == "NOT_PROMOTED"
    assert report.release_manifest_path is None
    assert report.blockers == ["independent_evidence_authority"]
    assert not (tmp_path / "release-manifest.json").exists()

    serving_sha = hashlib.sha256(
        json.dumps(
            serving_verification,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    subject = PromotionEvidenceSubject(
        requested_model_id="fixture-model",
        checkpoint_sha256=checkpoint_sha,
        dataset_manifest_sha256=manifest.manifest_sha256,
        evaluation_sha256=sha256_bytes(evaluation_path.read_bytes()),
        quantization_report_sha256=sha256_bytes(quantization_path.read_bytes()),
        model_card_sha256=sha256_bytes(card.read_bytes()),
        serving_verification_sha256=serving_sha,
        approved_base_license="Apache-2.0",
    )
    bundle_path, trust_path = _write_valid_attestations(tmp_path, subject)
    monkeypatch.setenv("OLYMPUS_PROMOTION_TRUST_SHA256", sha256_bytes(trust_path.read_bytes()))
    promoted = evaluate_promotion(
        requested_model_id="fixture-model",
        checkpoint_path=checkpoint,
        dataset_manifest_path=manifest_path,
        evaluation_path=evaluation_path,
        quantization_report_path=quantization_path,
        model_card_path=card,
        output_path=tmp_path / "promotion-with-attestations.json",
        approved_base_license="Apache-2.0",
        serving_verification=serving_verification,
        attestation_bundle_path=bundle_path,
        attestation_trust_store_path=trust_path,
    )
    assert promoted.status == "PROMOTED"
    assert promoted.blockers == []
    monkeypatch.delenv("OLYMPUS_PROMOTION_TRUST_SHA256")
    unpinned = evaluate_promotion(
        requested_model_id="fixture-model",
        checkpoint_path=checkpoint,
        dataset_manifest_path=manifest_path,
        evaluation_path=evaluation_path,
        quantization_report_path=quantization_path,
        model_card_path=card,
        output_path=tmp_path / "unpinned.json",
        approved_base_license="Apache-2.0",
        serving_verification=serving_verification,
        attestation_bundle_path=bundle_path,
        attestation_trust_store_path=trust_path,
    )
    assert unpinned.blockers == ["independent_evidence_authority"]
    monkeypatch.setenv("OLYMPUS_PROMOTION_TRUST_SHA256", sha256_bytes(trust_path.read_bytes()))
    assert promoted.release_manifest_path is not None
    release_path = Path(promoted.release_manifest_path)
    release_manifest = json.loads(release_path.read_text(encoding="utf-8"))
    assert release_manifest["promotion_report_sha256"] == sha256_bytes(
        (tmp_path / "promotion-with-attestations.json").read_bytes()
    )

    quantized_artifact.write_bytes(b"tampered quantized weights")
    rejected_after_tamper = evaluate_promotion(
        requested_model_id="fixture-model",
        checkpoint_path=checkpoint,
        dataset_manifest_path=manifest_path,
        evaluation_path=evaluation_path,
        quantization_report_path=quantization_path,
        model_card_path=card,
        output_path=tmp_path / "promotion-with-attestations.json",
        approved_base_license="Apache-2.0",
        serving_verification=serving_verification,
        attestation_bundle_path=bundle_path,
        attestation_trust_store_path=trust_path,
    )
    assert rejected_after_tamper.blockers == ["quantized_artifact_identity"]
    assert rejected_after_tamper.release_manifest_path is None
    assert release_path.is_file()  # Historical evidence is never deleted by a later run.
    assert not (tmp_path / "release-manifest.json").exists()
