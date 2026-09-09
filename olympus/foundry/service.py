from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from olympus.foundry.bigram import CharacterBigramModel, uniform_baseline_metrics
from olympus.foundry.resources import ResourceGovernor
from olympus.foundry.schemas import (
    ArtifactStatus,
    CheckpointRecord,
    DatasetRecord,
    EvaluationRecord,
    ExperimentRecord,
    FoundryPipelineResult,
    GenerationResult,
    ModelRecord,
)
from olympus.foundry.store import FoundryStore

LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
        temporary_path = None
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class FoundryCancelled(RuntimeError):
    """Raised only at artifact-safe boundaries when a local run is cancelled."""


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise FoundryCancelled("foundry run cancelled at an artifact-safe boundary")


class FoundryService:
    """End-to-end Foundry operations backed by immutable artifacts and SQLite."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = FoundryStore(self.root / "registry.sqlite3")

    @staticmethod
    def _code_commit(repository: Path) -> str:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "unavailable"
        if commit.returncode != 0:
            return "uncommitted"
        commit_sha = commit.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40,64}", commit_sha) is None:
            return "unavailable"
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=repository,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return f"{commit_sha}+status-unavailable"
        suffix = "+dirty" if status.returncode != 0 or status.stdout.strip() else ""
        return f"{commit_sha}{suffix}"

    @staticmethod
    def _hardware() -> dict[str, Any]:
        return {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "processor": platform.processor() or "unknown",
        }

    def _evidence(
        self, entity_type: str, entity_id: str, action: str, payload: dict[str, Any]
    ) -> int:
        return self.store.append_evidence(
            timestamp=_utc_now(),
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            payload=payload,
        )

    def _read_checkpoint_payload(self, checkpoint: CheckpointRecord) -> bytes:
        expected = self.root / "checkpoints" / f"{checkpoint.checkpoint_id}.json"
        declared = Path(checkpoint.path)
        if declared != expected:
            raise RuntimeError("checkpoint path is not canonical")
        try:
            resolved = declared.resolve(strict=True)
        except OSError as error:
            raise RuntimeError("checkpoint artifact is missing") from error
        if not resolved.is_relative_to(self.root):
            raise RuntimeError("checkpoint path resolves outside the Foundry root")
        if not resolved.is_file():
            raise RuntimeError("checkpoint artifact is missing")
        if resolved.stat().st_size != checkpoint.byte_count:
            raise RuntimeError("checkpoint byte count mismatch")
        with resolved.open("rb") as handle:
            payload = handle.read(checkpoint.byte_count + 1)
        if len(payload) != checkpoint.byte_count:
            raise RuntimeError("checkpoint byte count mismatch")
        if _sha256(payload) != checkpoint.sha256:
            raise RuntimeError("checkpoint hash mismatch")
        return payload

    def register_text_dataset(
        self,
        source_path: Path,
        *,
        dataset_id: str,
        version: str,
        source: str,
        owner: str,
        license_name: str,
        provenance: str,
        privacy_classification: Literal["public", "internal", "private", "restricted"],
        synthetic: bool,
        generator: str | None = None,
    ) -> DatasetRecord:
        payload = source_path.read_bytes()
        if not payload:
            raise ValueError("dataset is empty")
        if b"\x00" in payload:
            raise ValueError("text dataset contains NUL bytes")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("text dataset must be valid UTF-8") from error
        if len(text) < 64:
            raise ValueError("text dataset must contain at least 64 characters")
        digest = _sha256(payload)
        materialized = self.root / "datasets" / dataset_id / f"{version}-{digest}.txt"
        record = DatasetRecord(
            dataset_id=dataset_id,
            version=version,
            sha256=digest,
            materialized_path=str(materialized),
            source=source,
            owner=owner,
            license=license_name,
            provenance=provenance,
            privacy_classification=privacy_classification,
            synthetic=synthetic,
            generator=generator,
            byte_count=len(payload),
            character_count=len(text),
            created_at=_utc_now(),
        )
        if materialized.exists() and _sha256(materialized.read_bytes()) != digest:
            raise ValueError(f"materialized dataset hash mismatch: {materialized}")
        if not materialized.exists():
            _atomic_write(materialized, payload)
        inserted = self.store.register_dataset(record)
        if inserted:
            self._evidence(
                "dataset",
                f"{dataset_id}@{version}",
                "registered",
                {"sha256": digest, "byte_count": len(payload), "source": source},
            )
            return record
        registered = self.store.dataset(dataset_id, version)
        if registered is None:  # pragma: no cover - protected by the completed registration
            raise RuntimeError("registered dataset disappeared")
        return registered

    def run_bigram_experiment(
        self,
        dataset: DatasetRecord,
        *,
        hypothesis: str,
        seed: int = 7,
        train_fraction: float = 0.8,
        smoothing: float = 0.25,
        repository: Path | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> FoundryPipelineResult:
        if not 0.5 <= train_fraction <= 0.95:
            raise ValueError("train_fraction must be between 0.5 and 0.95")
        if not math.isfinite(smoothing) or not 0 < smoothing <= 10:
            raise ValueError("smoothing must be finite and between 0 and 10")
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 2^32-1")
        registered_dataset = self.store.dataset(dataset.dataset_id, dataset.version)
        if registered_dataset is None:
            raise ValueError(f"dataset is not registered: {dataset.dataset_id}@{dataset.version}")
        if registered_dataset.sha256 != dataset.sha256:
            raise ValueError("dataset content identity does not match the registered dataset")
        expected_path = (
            self.root
            / "datasets"
            / registered_dataset.dataset_id
            / f"{registered_dataset.version}-{registered_dataset.sha256}.txt"
        )
        declared_path = Path(registered_dataset.materialized_path)
        if declared_path != expected_path:
            raise ValueError("registered dataset path is outside its canonical Foundry location")
        materialized = declared_path.resolve(strict=True)
        if not materialized.is_relative_to(self.root):
            raise ValueError("registered dataset path resolves outside the Foundry root")
        dataset = registered_dataset
        payload = materialized.read_bytes()
        if _sha256(payload) != dataset.sha256:
            raise ValueError("dataset content no longer matches the registry hash")
        text = payload.decode("utf-8")
        split_index = max(2, min(len(text) - 2, int(len(text) * train_fraction)))
        training_text = text[:split_index]
        evaluation_text = text[split_index - 1 :]
        timestamp = datetime.now(UTC)
        experiment_id = f"FND_{timestamp.strftime('%Y%m%dT%H%M%S%f')}"
        root_repository = repository or Path(__file__).resolve().parents[2]
        experiment = ExperimentRecord(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            model_family="FoundryVerificationBigram",
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.version,
            dataset_sha256=dataset.sha256,
            seed=seed,
            config={
                "model_type": "character-bigram",
                "train_fraction": train_fraction,
                "smoothing": smoothing,
                "training_characters": len(training_text),
                "evaluation_characters": len(evaluation_text),
            },
            code_commit=self._code_commit(root_repository),
            hardware=self._hardware(),
            status=ArtifactStatus.RUNNING_EXPERIMENT,
            created_at=timestamp.isoformat(),
            updated_at=timestamp.isoformat(),
        )
        self.store.register_experiment(experiment)
        self._evidence("experiment", experiment_id, "started", experiment.model_dump(mode="json"))
        try:
            _check_cancelled(cancelled)
            model = CharacterBigramModel.train(
                training_text,
                smoothing=smoothing,
                seed=seed,
            )
            model.checkpoint.training_provenance = {
                "experiment_id": experiment_id,
                "dataset_id": dataset.dataset_id,
                "dataset_version": dataset.version,
                "dataset_sha256": dataset.sha256,
                "code_commit": experiment.code_commit,
            }
            _check_cancelled(cancelled)
            checkpoint_payload = model.checkpoint.canonical_bytes()
            checkpoint_sha = _sha256(checkpoint_payload)
            checkpoint_id = f"ckpt_{checkpoint_sha[:16]}"
            checkpoint_path = self.root / "checkpoints" / f"{checkpoint_id}.json"
            _atomic_write(checkpoint_path, checkpoint_payload)
            training_metrics = model.evaluate(training_text)
            checkpoint = CheckpointRecord(
                checkpoint_id=checkpoint_id,
                experiment_id=experiment_id,
                sha256=checkpoint_sha,
                path=str(checkpoint_path),
                format="olympus-character-bigram-v1",
                byte_count=len(checkpoint_payload),
                metrics=training_metrics.as_dict(),
                created_at=_utc_now(),
            )
            self.store.register_checkpoint(checkpoint)
            experiment.status = ArtifactStatus.CHECKPOINTED
            experiment.updated_at = _utc_now()
            self.store.update_experiment(experiment)
            self._evidence(
                "checkpoint",
                checkpoint_id,
                "saved",
                {"sha256": checkpoint_sha, "path": str(checkpoint_path)},
            )

            _check_cancelled(cancelled)

            candidate = model.evaluate(evaluation_text)
            baseline = uniform_baseline_metrics(evaluation_text, len(model.checkpoint.alphabet))
            passed = (
                candidate.negative_log_likelihood < baseline.negative_log_likelihood
                and candidate.perplexity < baseline.perplexity
            )
            decision = (
                "Candidate beat the frozen uniform baseline on held-out negative "
                "log-likelihood and perplexity."
                if passed
                else "Candidate did not beat the frozen uniform baseline on held-out metrics."
            )
            evaluation_payload = json.dumps(
                {
                    "checkpoint_id": checkpoint_id,
                    "dataset_sha256": dataset.sha256,
                    "baseline": baseline.as_dict(),
                    "candidate": candidate.as_dict(),
                    "passed": passed,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            evaluation = EvaluationRecord(
                evaluation_id=f"eval_{_sha256(evaluation_payload)[:16]}",
                checkpoint_id=checkpoint_id,
                suite="held-out-character-model-v1",
                dataset_sha256=dataset.sha256,
                baseline_metrics=baseline.as_dict(),
                candidate_metrics=candidate.as_dict(),
                passed=passed,
                decision=decision,
                created_at=_utc_now(),
            )
            self.store.register_evaluation(evaluation)
            self._evidence(
                "evaluation",
                evaluation.evaluation_id,
                "completed",
                evaluation.model_dump(mode="json"),
            )
            _check_cancelled(cancelled)
            if not passed:
                experiment.status = ArtifactStatus.NEGATIVE_RESULT
                experiment.updated_at = _utc_now()
                self.store.update_experiment(experiment)
                raise RuntimeError(decision)

            export_directory = self.root / "exports" / checkpoint_id
            export_path = export_directory / "model.json"
            _atomic_write(export_path, checkpoint_payload)
            export_manifest = {
                "schema_version": 1,
                "checkpoint_id": checkpoint_id,
                "sha256": checkpoint_sha,
                "format": checkpoint.format,
                "source_experiment": experiment_id,
                "dataset_sha256": dataset.sha256,
            }
            _atomic_write(
                export_directory / "manifest.json",
                json.dumps(export_manifest, indent=2, sort_keys=True).encode("utf-8"),
            )
            model_record = ModelRecord(
                model_id=f"foundry-verification-bigram-{checkpoint_sha[:8]}",
                family="FoundryVerificationBigram",
                version=checkpoint_sha[:12],
                checkpoint_id=checkpoint_id,
                status=ArtifactStatus.VERIFIED,
                runtime="olympus-character-bigram-v1",
                capabilities=["text-generation", "foundry-lifecycle-verification"],
                intended_use=(
                    "Low-cost verification of the Olympus dataset, experiment, checkpoint, "
                    "evaluation, export, registry, and serving lifecycle."
                ),
                limitations=[
                    "Character bigram model; not an assistant or reasoning model.",
                    "Must not be presented as Hermes or as evidence of frontier capability.",
                ],
                created_at=_utc_now(),
            )
            self.store.register_model(model_record)
            experiment.status = ArtifactStatus.VERIFIED
            experiment.updated_at = _utc_now()
            self.store.update_experiment(experiment)
            self._evidence(
                "model",
                model_record.model_id,
                "promoted",
                {
                    "checkpoint_id": checkpoint_id,
                    "evaluation_id": evaluation.evaluation_id,
                    "export_path": str(export_path),
                },
            )
            return FoundryPipelineResult(
                dataset=dataset,
                experiment=experiment,
                checkpoint=checkpoint,
                evaluation=evaluation,
                model=model_record,
                export_path=str(export_path),
            )
        except BaseException as error:
            expected_failure = isinstance(error, FoundryCancelled) or (
                experiment.status is ArtifactStatus.NEGATIVE_RESULT
            )
            if isinstance(error, FoundryCancelled):
                persisted_reason = "Foundry experiment was cancelled."
            elif expected_failure:
                persisted_reason = str(error)
            else:
                LOGGER.error(
                    "Foundry experiment failed (%s)",
                    type(error).__name__,
                )
                persisted_reason = (
                    f"{type(error).__name__}: Foundry experiment failed; "
                    "inspect server logs for details."
                )
            if experiment.status not in {
                ArtifactStatus.NEGATIVE_RESULT,
                ArtifactStatus.VERIFIED,
            }:
                experiment.status = (
                    ArtifactStatus.CANCELLED
                    if isinstance(error, FoundryCancelled)
                    else ArtifactStatus.FAILED
                )
                experiment.failure_reason = persisted_reason
                experiment.updated_at = _utc_now()
                self.store.update_experiment(experiment)
            self._evidence(
                "experiment",
                experiment_id,
                "failed",
                {"status": experiment.status.value, "reason": persisted_reason},
            )
            raise

    def run_verification_pipeline(
        self,
        sample_path: Path,
        *,
        repository: Path | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> FoundryPipelineResult:
        # All entrypoints share the same process-independent lease and safety floor.
        with ResourceGovernor(self.root / "verification.lock"):
            _check_cancelled(cancelled)
            return self._run_verification_pipeline(
                sample_path, repository=repository, cancelled=cancelled
            )

    def _run_verification_pipeline(
        self,
        sample_path: Path,
        *,
        repository: Path | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> FoundryPipelineResult:
        dataset = self.register_text_dataset(
            sample_path,
            dataset_id="foundry-verification-corpus",
            version="1.0.0",
            source="repository://olympus/foundry/foundry_verification.txt",
            owner="BU1LD Olympus",
            license_name="LicenseRef-Proprietary",
            provenance=(
                "Deterministic repository-owned synthetic corpus written specifically to "
                "verify Foundry infrastructure; it is not capability-training data."
            ),
            privacy_classification="internal",
            synthetic=True,
            generator="Human-authored deterministic Foundry verification fixture v1",
        )
        _check_cancelled(cancelled)
        return self.run_bigram_experiment(
            dataset,
            hypothesis=(
                "A trained character-bigram model will outperform a frozen uniform baseline "
                "on held-out transitions from the structured verification corpus."
            ),
            repository=repository,
            cancelled=cancelled,
        )

    def list_models(self) -> list[ModelRecord]:
        return self.store.models()

    def model_health(self, model_id: str) -> dict[str, object]:
        model = self.store.model(model_id)
        if model is None:
            return {"status": "missing", "model": model_id}
        if model.status is not ArtifactStatus.VERIFIED:
            return {"status": "corrupt", "model": model_id, "reason": "model is not verified"}
        checkpoint = self.store.checkpoint(model.checkpoint_id)
        if checkpoint is None:
            return {"status": "corrupt", "model": model_id, "reason": "missing checkpoint"}
        try:
            payload = self._read_checkpoint_payload(checkpoint)
        except RuntimeError as error:
            return {"status": "corrupt", "model": model_id, "reason": str(error)}
        actual_sha = _sha256(payload)
        return {
            "status": "ok",
            "model": model_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "sha256": actual_sha,
        }

    def generate(
        self,
        model_id: str,
        prompt: str,
        *,
        max_characters: int = 160,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> GenerationResult:
        model_record = self.store.model(model_id)
        if model_record is None:
            raise KeyError(f"unknown model: {model_id}")
        if model_record.status is not ArtifactStatus.VERIFIED:
            raise RuntimeError(f"model is not verified: {model_id}")
        checkpoint_record = self.store.checkpoint(model_record.checkpoint_id)
        if checkpoint_record is None:
            raise RuntimeError(f"missing checkpoint: {model_record.checkpoint_id}")
        payload = self._read_checkpoint_payload(checkpoint_record)
        from olympus.foundry.bigram import BigramCheckpoint

        model = CharacterBigramModel(BigramCheckpoint.from_bytes(payload))
        content = model.generate(
            prompt,
            max_characters=max_characters,
            temperature=temperature,
            seed=seed,
        )
        self._evidence(
            "model",
            model_id,
            "generated",
            {
                "prompt_sha256": _sha256(prompt.encode("utf-8")),
                "prompt_characters": len(prompt),
                "generated_characters": len(content),
                "seed": seed,
                "temperature": temperature,
            },
        )
        return GenerationResult(
            model=model_id,
            content=content,
            finish_reason="length",
            prompt_characters=len(prompt),
            generated_characters=len(content),
            evidence={
                "checkpoint_id": checkpoint_record.checkpoint_id,
                "checkpoint_sha256": checkpoint_record.sha256,
                "runtime": model_record.runtime,
            },
        )

    def export_checkpoint(self, model_id: str, destination: Path) -> Path:
        model = self.store.model(model_id)
        if model is None:
            raise KeyError(f"unknown model: {model_id}")
        if model.status is not ArtifactStatus.VERIFIED:
            raise RuntimeError(f"model is not verified: {model_id}")
        checkpoint = self.store.checkpoint(model.checkpoint_id)
        if checkpoint is None:
            raise RuntimeError(f"missing checkpoint: {model.checkpoint_id}")
        payload = self._read_checkpoint_payload(checkpoint)
        _atomic_write(destination, payload)
        return destination

    def status(self) -> dict[str, object]:
        return {
            "integrity": self.store.integrity_check(),
            "datasets": len(self.store.datasets()),
            "experiments": len(self.store.experiments()),
            "checkpoints": len(self.store.checkpoints()),
            "evaluations": len(self.store.evaluations()),
            "models": len(self.store.models()),
            "evidence_events": len(self.store.evidence()),
        }

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> FoundryService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
