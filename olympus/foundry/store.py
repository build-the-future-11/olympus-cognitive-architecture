from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from olympus.core.sqlite_journal import apply_durable_journal_mode
from olympus.foundry.bigram import BigramCheckpoint, CharacterBigramModel
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
            changed = connection.execute(
                "UPDATE experiments SET status = ?, record_json = ? WHERE experiment_id = ?",
                (record.status.value, self._json(record), record.experiment_id),
            ).rowcount
            if changed != 1:
                raise KeyError(f"unknown experiment: {record.experiment_id}")

    def register_checkpoint(self, record: CheckpointRecord) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO checkpoints(
                    checkpoint_id, experiment_id, sha256, record_json
                ) VALUES (?, ?, ?, ?)""",
                (record.checkpoint_id, record.experiment_id, record.sha256, self._json(record)),
            )

    def register_evaluation(self, record: EvaluationRecord) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO evaluations(evaluation_id, checkpoint_id, record_json)
                VALUES (?, ?, ?)""",
                (record.evaluation_id, record.checkpoint_id, self._json(record)),
            )

    def register_model(self, record: ModelRecord) -> None:
        with self.transaction() as connection:
            self.load_admitted_checkpoint(record)
            connection.execute(
                """INSERT INTO models(model_id, checkpoint_id, status, record_json)
                VALUES (?, ?, ?, ?)""",
                (record.model_id, record.checkpoint_id, record.status.value, self._json(record)),
            )

    @staticmethod
    def _passing_bigram_evaluation(evaluation: EvaluationRecord) -> bool:
        if not evaluation.passed or evaluation.suite != "held-out-character-model-v1":
            return False
        baseline, candidate = evaluation.baseline_metrics, evaluation.candidate_metrics
        keys = ("negative_log_likelihood", "perplexity", "evaluated_transitions")
        if any(
            key not in metrics or not math.isfinite(metrics[key])
            for metrics in (baseline, candidate)
            for key in keys
        ):
            return False
        transitions = candidate["evaluated_transitions"]
        return (
            transitions > 0
            and transitions.is_integer()
            and transitions == baseline["evaluated_transitions"]
            and 0 <= candidate["negative_log_likelihood"] < baseline["negative_log_likelihood"]
            and 1 <= candidate["perplexity"] < baseline["perplexity"]
        )

    def _admission_records(self, record: ModelRecord) -> tuple[CheckpointRecord, ExperimentRecord]:
        if record.status is not ArtifactStatus.VERIFIED:
            raise ValueError("only verified checkpoints can be promoted to the model registry")
        if record.family != "FoundryVerificationBigram" or (
            record.runtime != "olympus-character-bigram-v1"
        ):
            raise ValueError("unsupported model family or runtime for the verification registry")
        # One query provides a consistent registry snapshot, including reads from
        # a separate service connection. Register_model also holds a write transaction.
        rows = self._connection.execute(
            """SELECT c.record_json AS checkpoint_json, e.record_json AS experiment_json,
                      d.record_json AS dataset_json, v.record_json AS evaluation_json
               FROM checkpoints c
               LEFT JOIN experiments e ON e.experiment_id = c.experiment_id
               LEFT JOIN datasets d ON d.dataset_id = e.dataset_id
                                   AND d.version = e.dataset_version
               LEFT JOIN evaluations v ON v.checkpoint_id = c.checkpoint_id
               WHERE c.checkpoint_id = ?""",
            (record.checkpoint_id,),
        ).fetchall()
        if not rows:
            raise ValueError("missing checkpoint")
        checkpoint = CheckpointRecord.model_validate_json(rows[0]["checkpoint_json"])
        if (
            checkpoint.checkpoint_id != record.checkpoint_id
            or checkpoint.checkpoint_id != f"ckpt_{checkpoint.sha256[:16]}"
            or checkpoint.format != record.runtime
            or record.model_id != f"foundry-verification-bigram-{checkpoint.sha256[:8]}"
            or record.version != checkpoint.sha256[:12]
        ):
            raise ValueError("model identity or checkpoint runtime does not match its artifact")
        if rows[0]["experiment_json"] is None:
            raise ValueError("missing checkpoint experiment")
        experiment = ExperimentRecord.model_validate_json(rows[0]["experiment_json"])
        if (
            experiment.experiment_id != checkpoint.experiment_id
            or experiment.model_family != record.family
            or experiment.status
            not in {ArtifactStatus.CHECKPOINTED, ArtifactStatus.EVALUATED, ArtifactStatus.VERIFIED}
        ):
            raise ValueError("checkpoint experiment is ineligible for model admission")
        if rows[0]["dataset_json"] is None:
            raise ValueError("missing experiment dataset")
        dataset = DatasetRecord.model_validate_json(rows[0]["dataset_json"])
        if (
            dataset.dataset_id != experiment.dataset_id
            or dataset.version != experiment.dataset_version
            or dataset.sha256 != experiment.dataset_sha256
        ):
            raise ValueError("experiment dataset identity does not match the registry")
        for row in rows:
            if row["evaluation_json"] is None:
                continue
            try:
                evaluation = EvaluationRecord.model_validate_json(row["evaluation_json"])
            except ValueError:
                continue
            if (
                evaluation.checkpoint_id == checkpoint.checkpoint_id
                and evaluation.dataset_sha256 == dataset.sha256
                and self._passing_bigram_evaluation(evaluation)
            ):
                return checkpoint, experiment
        raise ValueError("checkpoint has no passing, identity-bound held-out evaluation")

    def load_admitted_checkpoint(self, record: ModelRecord) -> tuple[CheckpointRecord, bytes]:
        """Validate current evidence and return the same artifact bytes that were checked.

        Historical rows are checked too; an old VERIFIED label is not proof of
        admission. This is a consistency boundary for the local registry, not
        authentication of evidence supplied by a hostile database writer.
        """
        with self._lock:
            checkpoint, experiment = self._admission_records(record)
        payload = Path(checkpoint.path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != checkpoint.sha256:
            raise RuntimeError("checkpoint hash mismatch")
        if len(payload) != checkpoint.byte_count:
            raise RuntimeError("checkpoint byte count mismatch")
        parsed = BigramCheckpoint.from_bytes(payload)
        CharacterBigramModel(parsed)
        if parsed.schema_version != 1 or parsed.model_type != "character-bigram":
            raise ValueError("unsupported checkpoint schema or model type")
        expected_provenance = {
            "experiment_id": experiment.experiment_id,
            "dataset_id": experiment.dataset_id,
            "dataset_version": experiment.dataset_version,
            "dataset_sha256": experiment.dataset_sha256,
            "code_commit": experiment.code_commit,
        }
        if parsed.training_provenance != expected_provenance:
            raise ValueError("checkpoint provenance does not match its admitted experiment")
        return checkpoint, payload

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
            rows = self._connection.execute(f"SELECT record_json FROM {table}").fetchall()
        return [model.model_validate_json(row["record_json"]) for row in rows]

    def datasets(self) -> list[DatasetRecord]:
        return self._records("datasets", DatasetRecord)

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
                "SELECT record_json FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
        return None if row is None else ModelRecord.model_validate_json(row["record_json"])

    def checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT record_json FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
            ).fetchone()
        return None if row is None else CheckpointRecord.model_validate_json(row["record_json"])

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
