"""Artificial optimizer trajectories, continuation admission and attempt ownership."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from olympus.foundry import sft
from olympus.foundry.data_pipeline import REQUIRED_CATEGORIES, prepare_instruction_dataset


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    return cast(dict[str, Any], torch.load(path, map_location="cpu", weights_only=True))


def config(*, mode: str = "full", epochs: int = 1, dropout: float = 0.25) -> sft.SFTConfig:
    return sft.SFTConfig(
        mode=mode,  # type: ignore[arg-type]
        seed=113,
        epochs=epochs,
        batch_size=3,
        learning_rate=0.002,
        gradient_accumulation_steps=2,
        pack_sequences=False,
        mixed_precision=False,
        model=sft.TinyModelConfig(width=16, layers=1, heads=2,
                                  max_sequence_tokens=128, dropout=dropout),
    )


@pytest.fixture(scope="module")
def manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("continuation-artificial-data")
    rows = [{
        "id": f"{split}-{category}", "category": category,
        "prompt": f"Fixture {split}_{category} question",
        "response": f"Answer {split}_{category}", "license": "MIT",
        "source": "artificial://continuation-mechanism", "split": split,
    } for split in ("train", "validation", "test") for category in REQUIRED_CATEGORIES]
    source = root / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    prepare_instruction_dataset(source, root / "prepared")
    return root / "prepared/manifest.json"


@pytest.fixture(scope="module")
def full_first(manifest: Path, tmp_path_factory: pytest.TempPathFactory) -> sft.TrainingSummary:
    return sft.run_sft(manifest, tmp_path_factory.mktemp("continuation-first"), config=config())


def assert_same_tensors(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_tensors(left[key], right[key])
    elif isinstance(left, list | tuple):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_same_tensors(first, second)
    else:
        assert left == right


@pytest.mark.parametrize("mode", ["full", "lora", "qlora"])
def test_dropout_resume_matches_uninterrupted_model_optimizer_and_metrics(
    manifest: Path, full_first: sft.TrainingSummary, tmp_path: Path, mode: str,
) -> None:
    base = Path(full_first.checkpoint_path) if mode != "full" else None
    complete = sft.run_sft(manifest, tmp_path / "complete", config=config(mode=mode, epochs=3),
                           base_checkpoint=base)
    first = sft.run_sft(manifest, tmp_path / "split", config=config(mode=mode),
                        base_checkpoint=base)
    first_bytes = Path(first.checkpoint_path).read_bytes()
    first_log = Path(first.log_path).read_bytes()
    # Unrelated stochastic work must not change an exact continuation.
    torch.rand(79)
    second = sft.run_sft(manifest, tmp_path / "split", config=config(mode=mode, epochs=3),
                         resume_checkpoint=Path(first.checkpoint_path))
    expected = read_checkpoint(complete.checkpoint_path)
    actual = read_checkpoint(second.checkpoint_path)
    assert_same_tensors(actual["model_state"], expected["model_state"])
    assert_same_tensors(actual["optimizer_state"], expected["optimizer_state"])
    assert_same_tensors(actual["generator_state"], expected["generator_state"])
    assert actual["validation_loss"] == expected["validation_loss"]
    assert second.optimizer_steps == complete.optimizer_steps == 6
    assert Path(first.checkpoint_path).read_bytes() == first_bytes
    assert Path(first.log_path).read_bytes() == first_log
    assert second.checkpoint_path != first.checkpoint_path
    assert actual["resume_checkpoint_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    records = [json.loads(line) for line in Path(second.log_path).read_text().splitlines()]
    reference = [json.loads(line) for line in Path(complete.log_path).read_text().splitlines()]
    assert records == reference[1:]


@pytest.mark.parametrize("change", [
    {"seed": 117}, {"batch_size": 4}, {"learning_rate": 0.1},
    {"gradient_accumulation_steps": 1}, {"gradient_clip_norm": 2.0},
    {"weight_decay": 0.0}, {"pack_sequences": True}, {"mixed_precision": True},
    {"category_mix_weights": {"science": 2.0}}, {"lora_rank": 2},
    {"lora_alpha": 4.0}, {"model": {"width": 32}}, {"model": {"dropout": 0.0}},
])
def test_resume_rejects_non_epoch_configuration_changes_before_artifacts(
    manifest: Path, full_first: sft.TrainingSummary, tmp_path: Path, change: dict[str, Any],
) -> None:
    values = config(epochs=2).model_dump()
    if "model" in change:
        values["model"].update(change["model"])
    else:
        values.update(change)
    with pytest.raises(ValueError, match="resume checkpoint .*does not match"):
        sft.run_sft(manifest, tmp_path, config=sft.SFTConfig.model_validate(values),
                    resume_checkpoint=Path(full_first.checkpoint_path))
    assert not list(tmp_path.rglob("metrics.jsonl"))
    assert not list(tmp_path.rglob("checkpoint.pt"))


def test_completed_schedule_cannot_be_republished_as_new_training(
    manifest: Path, full_first: sft.TrainingSummary, tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="epochs must exceed"):
        sft.run_sft(manifest, tmp_path, config=config(),
                    resume_checkpoint=Path(full_first.checkpoint_path))
    assert not list(tmp_path.rglob("checkpoint.pt"))


def test_identical_fresh_runs_have_distinct_attempts_and_preserve_prior_bytes(
    manifest: Path, tmp_path: Path,
) -> None:
    first = sft.run_sft(manifest, tmp_path, config=config())
    retained = {p: p.read_bytes() for p in Path(first.checkpoint_path).parent.iterdir()}
    second = sft.run_sft(manifest, tmp_path, config=config())
    assert first.run_id != second.run_id
    assert first.checkpoint_path != second.checkpoint_path
    assert_same_tensors(read_checkpoint(first.checkpoint_path)["model_state"],
                        read_checkpoint(second.checkpoint_path)["model_state"])
    assert all(path.read_bytes() == content for path, content in retained.items())


def test_checkpoint_rejects_disagreement_between_model_and_training_config(
    full_first: sft.TrainingSummary, tmp_path: Path,
) -> None:
    checkpoint = read_checkpoint(full_first.checkpoint_path)
    checkpoint["training_config"]["model"]["dropout"] = 0.0
    path = tmp_path / "mislabeled.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="model configuration does not match"):
        sft.load_trained_model(path)


@pytest.mark.parametrize("field", ["rng_state", "torch_cpu", "python", "device_type"])
def test_missing_stochastic_state_is_rejected_for_new_checkpoints(
    manifest: Path, full_first: sft.TrainingSummary, tmp_path: Path, field: str,
) -> None:
    checkpoint = read_checkpoint(full_first.checkpoint_path)
    if field == "rng_state":
        checkpoint.pop(field, None)
    else:
        checkpoint.setdefault("rng_state", {}).pop(field, None)
    path = tmp_path / "missing-state.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="random state"):
        sft.run_sft(manifest, tmp_path / "rejected", config=config(epochs=2),
                    resume_checkpoint=path)
    assert not list((tmp_path / "rejected").rglob("checkpoint.pt"))


def test_legacy_dropout_checkpoint_remains_loadable_but_cannot_claim_exact_resume(
    manifest: Path, full_first: sft.TrainingSummary, tmp_path: Path,
) -> None:
    checkpoint = read_checkpoint(full_first.checkpoint_path)
    checkpoint["schema_version"] = 2
    checkpoint.pop("rng_state", None)
    path = tmp_path / "legacy-dropout.pt"
    torch.save(checkpoint, path)
    assert sft.load_trained_model(path).config.dropout == 0.25
    with pytest.raises(ValueError, match="random state"):
        sft.run_sft(manifest, tmp_path / "rejected", config=config(epochs=2),
                    resume_checkpoint=path)


def test_legacy_deterministic_checkpoint_can_continue_without_reinterpreting_objective(
    manifest: Path, tmp_path: Path,
) -> None:
    first = sft.run_sft(manifest, tmp_path / "first", config=config(dropout=0.0))
    payload = read_checkpoint(first.checkpoint_path)
    payload["schema_version"] = 2
    payload.pop("rng_state", None)
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    continuation = sft.run_sft(manifest, tmp_path / "continued",
                               config=config(epochs=2, dropout=0.0),
                               resume_checkpoint=legacy)
    reference = sft.run_sft(manifest, tmp_path / "reference", config=config(epochs=2, dropout=0.0))
    assert_same_tensors(read_checkpoint(continuation.checkpoint_path)["model_state"],
                        read_checkpoint(reference.checkpoint_path)["model_state"])


def test_base_and_resume_are_mutually_exclusive(
    manifest: Path, full_first: sft.TrainingSummary, tmp_path: Path,
) -> None:
    path = Path(full_first.checkpoint_path)
    with pytest.raises(ValueError, match="base and resume"):
        sft.run_sft(manifest, tmp_path, config=config(mode="lora", epochs=2),
                    base_checkpoint=path, resume_checkpoint=path)


def test_failed_attempt_is_retained_when_retry_succeeds(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = sft._loss

    def fail_training(model: sft.TinyCausalLM, *args: Any, **kwargs: Any) -> torch.Tensor:
        if model.training:
            raise RuntimeError("artificial training interruption")
        return original(model, *args, **kwargs)

    monkeypatch.setattr(sft, "_loss", fail_training)
    with pytest.raises(RuntimeError, match="artificial training interruption"):
        sft.run_sft(manifest, tmp_path, config=config())
    paths = list(tmp_path.rglob("attempt.json"))
    assert len(paths) == 1
    failed = paths[0].read_bytes()
    payload = json.loads(failed)
    assert payload["status"] == "FAILED"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "artificial training interruption"
    monkeypatch.setattr(sft, "_loss", original)
    retry = sft.run_sft(manifest, tmp_path, config=config())
    assert paths[0].read_bytes() == failed
    assert retry.attempt_path != str(paths[0])
    assert json.loads(Path(retry.attempt_path or "").read_text())["status"] == "COMPLETED"


def test_resume_identity_binds_the_bytes_deserialized_before_path_replacement(
    manifest: Path, full_first: sft.TrainingSummary, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_bytes = Path(full_first.checkpoint_path).read_bytes()
    parent = tmp_path / "parent.pt"
    parent.write_bytes(parent_bytes)
    original = Path.read_bytes
    reads: list[bytes] = []

    def replace_after_read(path: Path) -> bytes:
        payload = original(path)
        if path == parent:
            reads.append(payload)
            parent.write_bytes(b"artificial replacement after verified snapshot")
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    summary = sft.run_sft(manifest, tmp_path / "child", config=config(epochs=2),
                          resume_checkpoint=parent)
    expected_hash = hashlib.sha256(parent_bytes).hexdigest()
    assert reads == [parent_bytes]
    assert summary.resume_checkpoint_sha256 == expected_hash
    payload = read_checkpoint(summary.checkpoint_path)
    assert payload["resume_checkpoint_sha256"] == expected_hash
    assert json.loads(Path(summary.attempt_path or "").read_text())[
        "resume_checkpoint_sha256"
    ] == expected_hash


@pytest.mark.parametrize("field,value", [
    ("completed_epochs", 1.5), ("completed_epochs", True), ("completed_epochs", -1),
    ("completed_epochs", 2), ("optimizer_steps", 1.5), ("optimizer_steps", True),
    ("optimizer_steps", -1), ("optimizer_steps", 17),
])
def test_resume_requires_exact_consistent_counters(
    manifest: Path, full_first: sft.TrainingSummary, tmp_path: Path, field: str, value: Any,
) -> None:
    payload = read_checkpoint(full_first.checkpoint_path)
    payload[field] = value
    path = tmp_path / "invalid-counter.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checkpoint"):
        sft.run_sft(manifest, tmp_path / "rejected", config=config(epochs=3),
                    resume_checkpoint=path)
    assert not list((tmp_path / "rejected").rglob("checkpoint.pt"))


@pytest.mark.parametrize("field", ["model", "optimizer", "missing_optimizer"])
def test_resume_rejects_nonfinite_or_missing_optimization_state(
    manifest: Path, full_first: sft.TrainingSummary, tmp_path: Path, field: str,
) -> None:
    payload = read_checkpoint(full_first.checkpoint_path)
    if field == "model":
        next(iter(payload["model_state"].values())).flatten()[0] = float("nan")
    elif field == "optimizer":
        next(iter(payload["optimizer_state"]["state"].values()))["exp_avg"].flatten()[0] = (
            float("inf")
        )
    else:
        payload["optimizer_state"]["state"] = {}
    path = tmp_path / "invalid-state.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="checkpoint .*state"):
        sft.run_sft(manifest, tmp_path / "rejected", config=config(epochs=2),
                    resume_checkpoint=path)
    assert not list((tmp_path / "rejected").rglob("checkpoint.pt"))


@pytest.mark.parametrize("field,value", [
    ("lr", 0.0), ("weight_decay", 0.5), ("betas", (0.8, 0.99)), ("eps", 1e-4),
    ("amsgrad", True), ("maximize", True), ("foreach", True), ("capturable", True),
    ("differentiable", True), ("fused", True), ("decoupled_weight_decay", False),
    ("initial_lr", 0.25), ("missing_eps", None), ("reordered_params", None),
])
def test_resume_rejects_optimizer_policy_drift_before_load(
    manifest: Path, full_first: sft.TrainingSummary, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any,
) -> None:
    payload = read_checkpoint(full_first.checkpoint_path)
    group = payload["optimizer_state"]["param_groups"][0]
    if field == "missing_eps":
        del group["eps"]
    elif field == "reordered_params":
        group["params"] = list(reversed(group["params"]))
    else:
        group[field] = value
    path = tmp_path / "invalid-optimizer-policy.pt"
    torch.save(payload, path)
    source_bytes = path.read_bytes()

    def unexpected_load(*args: Any, **kwargs: Any) -> None:
        pytest.fail("mismatched optimizer policy reached load_state_dict")

    monkeypatch.setattr(torch.optim.AdamW, "load_state_dict", unexpected_load)
    with pytest.raises(ValueError, match="checkpoint optimizer policy"):
        sft.run_sft(manifest, tmp_path / "rejected", config=config(epochs=2),
                    resume_checkpoint=path)
    assert path.read_bytes() == source_bytes
    assert not list((tmp_path / "rejected").rglob("checkpoint.pt"))


@pytest.mark.parametrize("failure", ["loss", "gradient", "updated_weight"])
def test_nonfinite_training_is_failed_and_retained(
    manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    if failure == "loss":
        original_loss = sft._loss

        def bad_loss(model: sft.TinyCausalLM, *args: Any, **kwargs: Any) -> torch.Tensor:
            value = original_loss(model, *args, **kwargs)
            return value * float("nan") if model.training else value

        monkeypatch.setattr(sft, "_loss", bad_loss)
    elif failure == "gradient":
        original_init = sft.TinyCausalLM.__init__

        def bad_gradient(self: sft.TinyCausalLM, model_config: sft.TinyModelConfig) -> None:
            original_init(self, model_config)
            next(self.parameters()).register_hook(  # type: ignore[no-untyped-call]
                lambda gradient: gradient * float("nan")
            )

        monkeypatch.setattr(sft.TinyCausalLM, "__init__", bad_gradient)
    else:
        original_step = torch.optim.AdamW.step

        def bad_step(self: torch.optim.AdamW, *args: Any, **kwargs: Any) -> Any:
            result = original_step(self, *args, **kwargs)
            with torch.no_grad():
                self.param_groups[0]["params"][0].flatten()[0] = float("nan")
            return result

        monkeypatch.setattr(torch.optim.AdamW, "step", bad_step)
    with pytest.raises(ValueError, match="finite"):
        sft.run_sft(manifest, tmp_path, config=config())
    receipts = list(tmp_path.rglob("attempt.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "FAILED"
    assert not list(tmp_path.rglob("checkpoint.pt"))

