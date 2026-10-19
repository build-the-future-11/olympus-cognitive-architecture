from __future__ import annotations

import json
import math
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from olympus.core.schemas import StrictModel
from olympus.core.sqlite_journal import apply_durable_journal_mode
from olympus.foundry.admission import (
    evaluation_passes,
    load_registered_checkpoint,
    read_registered_dataset,
    validate_model_identity,
)
from olympus.foundry.bigram import CharacterBigramModel, uniform_baseline_metrics
from olympus.foundry.schemas import (
    ArtifactStatus,
    CheckpointRecord,
    DatasetRecord,
    EvaluationRecord,
    EvidenceRecord,
    ExperimentRecord,
    ModelRecord,
)


class FoundryStore:
    """SQLite registry with foreign keys and append-only evidence."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        self._lock = threading.RLock()
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        apply_durable_journal_mode(self._connection)
        self._connection.execute("PRAGMA synchronous = FULL")
        try:
            self._migrate()
        except BaseException:
            # Construction must not leak a live database handle when schema
            # creation fails (for example, on a read-only artifact root).
            self._connection.close()
            raise

    def _migrate(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT NOT NULL,
                version TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY (dataset_id, version),
                UNIQUE (sha256)
            );
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                status TEXT NOT NULL,
                record_json TEXT NOT NULL,
                FOREIGN KEY (dataset_id, dataset_version)
                    REFERENCES datasets(dataset_id, version)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                record_json TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            );
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                checkpoint_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id)
            );
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                checkpoint_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                record_json TEXT NOT NULL,
                FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id)
            );
            CREATE TABLE IF NOT EXISTS evidence (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                action TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (1, datetime.now(UTC).isoformat()),
        )
        self._migrate_verification_dataset_source()
        self._connection.commit()

    def _migrate_verification_dataset_source(self) -> None:
        version = self._connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = 2"
        ).fetchone()
        if version is not None:
            return
        legacy_source = "repository://datasets/samples/foundry_verification.txt"
        canonical_source = "repository://olympus/foundry/foundry_verification.txt"
        row = self._connection.execute(
            """SELECT record_json FROM datasets
            WHERE dataset_id = ? AND version = ? AND sha256 = ?""",
            (
                "foundry-verification-corpus",
                "1.0.0",
                "6f4823e0503426bda11aa4bffe8357963f3bc97a62ea40c2b69700af166a8eab",
            ),
        ).fetchone()
        if row is not None:
            record = DatasetRecord.model_validate_json(row["record_json"])
            if record.source == legacy_source:
                record.source = canonical_source
                self._connection.execute(
                    """UPDATE datasets SET record_json = ?
                    WHERE dataset_id = ? AND version = ?""",
                    (self._json(record), record.dataset_id, record.version),
                )
                self._connection.execute(
                    """INSERT INTO evidence(
                        timestamp, entity_type, entity_id, action, payload_json
                    ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        datetime.now(UTC).isoformat(),
                        "dataset",
                        "foundry-verification-corpus@1.0.0",
                        "metadata_migrated",
                        json.dumps(
                            {"field": "source", "from": legacy_source, "to": canonical_source},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    ),
                )
        self._connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (2, datetime.now(UTC).isoformat()),
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    @staticmethod
    def _json(
        record: (
            DatasetRecord | ExperimentRecord | CheckpointRecord | EvaluationRecord | ModelRecord
        ),
    ) -> str:
        return record.model_dump_json()

    def register_dataset(self, record: DatasetRecord) -> None:
        with self.transaction() as connection:
            existing = connection.execute(
                """SELECT sha256, record_json FROM datasets
                WHERE dataset_id = ? AND version = ?""",
                (record.dataset_id, record.version),
            ).fetchone()
            if existing is not None:
                if existing["sha256"] != record.sha256:
                    raise ValueError(
                        f"dataset identity already points to different content: "
                        f"{record.dataset_id}@{record.version}"
                    )
                registered = DatasetRecord.model_validate_json(existing["record_json"])
                registered_metadata = registered.model_dump(exclude={"created_at"})
                incoming_metadata = record.model_dump(exclude={"created_at"})
                if registered_metadata != incoming_metadata:
                    raise ValueError(
                        f"dataset identity already has different metadata: "
                        f"{record.dataset_id}@{record.version}"
                    )
                return
            connection.execute(
                """INSERT INTO datasets(dataset_id, version, sha256, record_json)
                VALUES (?, ?, ?, ?)""",
                (record.dataset_id, record.version, record.sha256, self._json(record)),
            )

    def register_experiment(self, record: ExperimentRecord) -> None:
        with self.transaction() as connection:
            dataset = self.dataset(record.dataset_id, record.dataset_version)
            if dataset is None or dataset.sha256 != record.dataset_sha256:
                raise ValueError("experiment does not reference the registered dataset hash")
            connection.execute(
                """INSERT INTO experiments(
                    experiment_id, dataset_id, dataset_version, status, record_json
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    record.experiment_id,
                    record.dataset_id,
                    record.dataset_version,
                    record.status.value,
                    self._json(record),
                ),
            )

    def update_experiment(self, record: ExperimentRecord) -> None:
        with self.transaction() as connection:
            existing = self.experiment(record.experiment_id)
            mutable = {"status", "failure_reason", "updated_at"}
            if existing is None:
                raise KeyError(f"unknown experiment: {record.experiment_id}")
            if existing.model_dump(exclude=mutable) != record.model_dump(exclude=mutable):
                raise ValueError("registered experiment identity and configuration are immutable")
            changed = connection.execute(
                "UPDATE experiments SET status = ?, record_json = ? WHERE experiment_id = ?",
                (record.status.value, self._json(record), record.experiment_id),
            ).rowcount
            if changed != 1:
                raise KeyError(f"unknown experiment: {record.experiment_id}")

    def register_checkpoint(self, record: CheckpointRecord) -> None:
        with self.transaction() as connection:
            experiment, dataset = self._checkpoint_context(record)
            load_registered_checkpoint(record, experiment, dataset)
            connection.execute(
                """INSERT INTO checkpoints(
                    checkpoint_id, experiment_id, sha256, record_json
                ) VALUES (?, ?, ?, ?)""",
                (record.checkpoint_id, record.experiment_id, record.sha256, self._json(record)),
            )

    def register_evaluation(self, record: EvaluationRecord) -> None:
        with self.transaction() as connection:
            checkpoint = self.checkpoint(record.checkpoint_id)
            if checkpoint is None:
                raise ValueError("evaluation references a missing checkpoint")
            experiment, dataset = self._checkpoint_context(checkpoint)
            _, model = load_registered_checkpoint(checkpoint, experiment, dataset)
            evaluation_passes(record, checkpoint, experiment, model)
            connection.execute(
                """INSERT INTO evaluations(evaluation_id, checkpoint_id, record_json)
                VALUES (?, ?, ?)""",
                (record.evaluation_id, record.checkpoint_id, self._json(record)),
            )

    def register_model(self, record: ModelRecord) -> None:
        if record.status is not ArtifactStatus.VERIFIED:
            raise ValueError("only verified checkpoints can be promoted to the model registry")
        with self.transaction() as connection:
            self._load_admitted_checkpoint(record, verify_metrics=True)
            connection.execute(
                """INSERT INTO models(model_id, checkpoint_id, status, record_json)
                VALUES (?, ?, ?, ?)""",
                (record.model_id, record.checkpoint_id, record.status.value, self._json(record)),
            )

    def _checkpoint_context(
        self, checkpoint: CheckpointRecord
    ) -> tuple[ExperimentRecord, DatasetRecord]:
        experiment = self.experiment(checkpoint.experiment_id)
        if experiment is None:
            raise ValueError("missing checkpoint experiment")
        dataset = self.dataset(experiment.dataset_id, experiment.dataset_version)
        if dataset is None:
            raise ValueError("missing registered dataset")
        return experiment, dataset

    def _load_admitted_checkpoint(
        self, record: ModelRecord, *, verify_metrics: bool = False
    ) -> tuple[CheckpointRecord, bytes, CharacterBigramModel]:
        checkpoint = self.checkpoint(record.checkpoint_id)
        if checkpoint is None:
            raise ValueError("missing checkpoint")
        validate_model_identity(record, checkpoint)
        experiment, dataset = self._checkpoint_context(checkpoint)
        if experiment.status not in {
            ArtifactStatus.CHECKPOINTED,
            ArtifactStatus.EVALUATED,
            ArtifactStatus.VERIFIED,
        }:
            raise ValueError("checkpoint experiment is not eligible for serving")
        raw, model = load_registered_checkpoint(checkpoint, experiment, dataset)
        rows = self._connection.execute(
            "SELECT record_json FROM evaluations WHERE checkpoint_id = ?",
            (record.checkpoint_id,),
        ).fetchall()
        passing: list[EvaluationRecord] = []
        for row in rows:
            evaluation = EvaluationRecord.model_validate_json(row["record_json"])
            if evaluation_passes(evaluation, checkpoint, experiment, model):
                passing.append(evaluation)
        if not passing:
            raise ValueError("checkpoint has no matching passing held-out evaluation")
        if verify_metrics:
            text = read_registered_dataset(dataset)
            fraction = experiment.config.get("train_fraction")
            if not isinstance(fraction, (int, float)) or not 0.5 <= fraction <= 0.95:
                raise ValueError("invalid experiment split fraction")
            split_index = max(2, min(len(text) - 2, int(len(text) * fraction)))
            held_out = text[split_index - 1 :]
            if model.checkpoint.training_characters != split_index or len(
                held_out
            ) != experiment.config.get("evaluation_characters"):
                raise ValueError("experiment split counts do not match the registered dataset")
            expected = CharacterBigramModel.train(
                text[:split_index], smoothing=model.checkpoint.smoothing, seed=experiment.seed
            ).checkpoint
            if (
                model.checkpoint.alphabet != expected.alphabet
                or model.checkpoint.counts != expected.counts
            ):
                raise ValueError("checkpoint was not fitted to the registered training split")
            observed = model.evaluate(held_out).as_dict()
            baseline = uniform_baseline_metrics(held_out, len(expected.alphabet)).as_dict()
            if not all(
                observed[key] < baseline[key] for key in ("negative_log_likelihood", "perplexity")
            ):
                raise ValueError("checkpoint does not beat the frozen held-out baseline")
            if not any(
                all(
                    math.isclose(observed[key], value, rel_tol=1e-12, abs_tol=1e-12)
                    for key, value in evaluation.candidate_metrics.items()
                )
                for evaluation in passing
            ):
                raise ValueError("held-out evaluation metrics do not match the checkpoint")
        return checkpoint, raw, model

    def load_admitted_checkpoint(
        self, record: ModelRecord
    ) -> tuple[CheckpointRecord, bytes, CharacterBigramModel]:
        with self._lock:
            return self._load_admitted_checkpoint(record)

    def append_evidence(
        self,
        *,
        timestamp: str,
        entity_type: str,
        entity_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> int:
        with self.transaction() as connection:
            cursor = connection.execute(
                """INSERT INTO evidence(
                    timestamp, entity_type, entity_id, action, payload_json
                ) VALUES (?, ?, ?, ?, ?)""",
                (
                    timestamp,
                    entity_type,
                    entity_id,
                    action,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return an evidence sequence")
            return cursor.lastrowid

    def _records(
        self,
        table: str,
        model: (
            type[DatasetRecord]
            | type[ExperimentRecord]
            | type[CheckpointRecord]
            | type[EvaluationRecord]
            | type[ModelRecord]
        ),
    ) -> list[Any]:
        allowed = {"datasets", "experiments", "checkpoints", "evaluations", "models"}
        if table not in allowed:
            raise ValueError(f"unsupported registry table: {table}")
        with self._lock:
            rows = self._connection.execute(f"SELECT * FROM {table}").fetchall()
        records = [model.model_validate_json(row["record_json"]) for row in rows]
        for row, record in zip(rows, records, strict=True):
            self._validate_registry_columns(row, record)
        return records

    @staticmethod
    def _validate_registry_columns(row: sqlite3.Row, record: StrictModel) -> None:
        values = record.model_dump(mode="json")
        if any(row[key] != values[key] for key in row.keys() if key in values):
            raise ValueError("registry columns contradict the stored record identity")

    def datasets(self) -> list[DatasetRecord]:
        return self._records("datasets", DatasetRecord)

    def dataset(self, dataset_id: str, version: str) -> DatasetRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM datasets WHERE dataset_id = ? AND version = ?",
                (dataset_id, version),
            ).fetchone()
        if row is None:
            return None
        record = DatasetRecord.model_validate_json(row["record_json"])
        self._validate_registry_columns(row, record)
        return record

    def experiment(self, experiment_id: str) -> ExperimentRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        if row is None:
            return None
        record = ExperimentRecord.model_validate_json(row["record_json"])
        self._validate_registry_columns(row, record)
        return record

    def experiments(self) -> list[ExperimentRecord]:
        return self._records("experiments", ExperimentRecord)

    def checkpoints(self) -> list[CheckpointRecord]:
        return self._records("checkpoints", CheckpointRecord)

    def evaluations(self) -> list[EvaluationRecord]:
        return self._records("evaluations", EvaluationRecord)

    def models(self) -> list[ModelRecord]:
        return self._records("models", ModelRecord)

    def model(self, model_id: str) -> ModelRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
        if row is None:
            return None
        record = ModelRecord.model_validate_json(row["record_json"])
        self._validate_registry_columns(row, record)
        return record

    def checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
            ).fetchone()
        if row is None:
            return None
        record = CheckpointRecord.model_validate_json(row["record_json"])
        self._validate_registry_columns(row, record)
        return record

    def evidence(self) -> list[EvidenceRecord]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT sequence, timestamp, entity_type, entity_id, action, payload_json
                FROM evidence ORDER BY sequence"""
            ).fetchall()
        return [
            EvidenceRecord(
                sequence=row["sequence"],
                timestamp=row["timestamp"],
                entity_type=row["entity_type"],
                entity_id=row["entity_id"],
                action=row["action"],
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        ]

    def integrity_check(self) -> str:
        with self._lock:
            result = self._connection.execute("PRAGMA integrity_check").fetchone()
        return str(result[0])

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> FoundryStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
