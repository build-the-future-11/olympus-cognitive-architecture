from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from olympus.foundry.schemas import (
    ArtifactStatus,
    CheckpointRecord,
    DatasetRecord,
    EvaluationRecord,
    EvidenceRecord,
    ExperimentRecord,
    ModelRecord,
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_EXPERIMENT_TRANSITIONS: dict[ArtifactStatus, frozenset[ArtifactStatus]] = {
    ArtifactStatus.RUNNING_EXPERIMENT: frozenset(
        {
            ArtifactStatus.CHECKPOINTED,
            ArtifactStatus.CANCELLED,
            ArtifactStatus.FAILED,
        }
    ),
    ArtifactStatus.CHECKPOINTED: frozenset(
        {
            ArtifactStatus.EVALUATED,
            ArtifactStatus.VERIFIED,
            ArtifactStatus.NEGATIVE_RESULT,
            ArtifactStatus.INCONCLUSIVE,
            ArtifactStatus.CANCELLED,
            ArtifactStatus.FAILED,
        }
    ),
    ArtifactStatus.EVALUATED: frozenset(
        {
            ArtifactStatus.VERIFIED,
            ArtifactStatus.NEGATIVE_RESULT,
            ArtifactStatus.INCONCLUSIVE,
            ArtifactStatus.CANCELLED,
            ArtifactStatus.FAILED,
        }
    ),
    ArtifactStatus.DESIGNED: frozenset(),
    ArtifactStatus.VERIFIED: frozenset(),
    ArtifactStatus.NEGATIVE_RESULT: frozenset(),
    ArtifactStatus.INCONCLUSIVE: frozenset(),
    ArtifactStatus.CANCELLED: frozenset(),
    ArtifactStatus.FAILED: frozenset(),
}


class FoundryStore:
    """SQLite registry with foreign keys and append-only evidence."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._connection = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        self._lock = threading.RLock()
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._migrate()

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
        self._migrate_append_only_evidence()
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

    def _migrate_append_only_evidence(self) -> None:
        version = self._connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = 3"
        ).fetchone()
        if version is not None:
            return
        self._connection.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS evidence_reject_update
            BEFORE UPDATE ON evidence
            BEGIN
                SELECT RAISE(ABORT, 'foundry evidence is append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS evidence_reject_delete
            BEFORE DELETE ON evidence
            BEGIN
                SELECT RAISE(ABORT, 'foundry evidence is append-only');
            END;
            """
        )
        self._connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (3, datetime.now(UTC).isoformat()),
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
        try:
            return json.dumps(
                record.model_dump(mode="python"),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("registry record must contain finite JSON data") from error

    def register_dataset(self, record: DatasetRecord) -> bool:
        """Register immutable dataset metadata and report whether it was inserted."""

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
                return False
            connection.execute(
                """INSERT INTO datasets(dataset_id, version, sha256, record_json)
                VALUES (?, ?, ?, ?)""",
                (record.dataset_id, record.version, record.sha256, self._json(record)),
            )
            return True

    def register_experiment(self, record: ExperimentRecord) -> None:
        if record.status is not ArtifactStatus.RUNNING_EXPERIMENT:
            raise ValueError("new experiments must start in RUNNING_EXPERIMENT")
        with self.transaction() as connection:
            dataset = connection.execute(
                """SELECT sha256 FROM datasets
                WHERE dataset_id = ? AND version = ?""",
                (record.dataset_id, record.dataset_version),
            ).fetchone()
            if dataset is None or dataset["sha256"] != record.dataset_sha256:
                raise ValueError("experiment dataset identity does not match the registry")
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
            row = connection.execute(
                """SELECT dataset_id, dataset_version, status, record_json FROM experiments
                WHERE experiment_id = ?""",
                (record.experiment_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown experiment: {record.experiment_id}")
            registered = ExperimentRecord.model_validate_json(row["record_json"])
            mutable_fields = {"status", "failure_reason", "updated_at"}
            if registered.model_dump(exclude=mutable_fields) != record.model_dump(
                exclude=mutable_fields
            ):
                raise ValueError("experiment immutable metadata cannot be changed")
            if (
                registered.dataset_id != row["dataset_id"]
                or registered.dataset_version != row["dataset_version"]
            ):
                raise ValueError("stored experiment linkage is inconsistent")
            if registered.status.value != row["status"]:
                raise ValueError("stored experiment status is inconsistent")
            if (
                record.status is not registered.status
                and record.status not in _EXPERIMENT_TRANSITIONS[registered.status]
            ):
                raise ValueError(
                    f"invalid experiment status transition: "
                    f"{registered.status.value} -> {record.status.value}"
                )
            artifact_count = connection.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE experiment_id = ?",
                (record.experiment_id,),
            ).fetchone()[0]
            if (
                record.status
                in {
                    ArtifactStatus.CHECKPOINTED,
                    ArtifactStatus.EVALUATED,
                    ArtifactStatus.VERIFIED,
                    ArtifactStatus.NEGATIVE_RESULT,
                    ArtifactStatus.INCONCLUSIVE,
                }
                and artifact_count < 1
            ):
                raise ValueError("experiment status requires a registered checkpoint")
            evaluation_count = connection.execute(
                """SELECT COUNT(*) FROM evaluations
                JOIN checkpoints USING(checkpoint_id)
                WHERE checkpoints.experiment_id = ?""",
                (record.experiment_id,),
            ).fetchone()[0]
            if (
                record.status
                in {
                    ArtifactStatus.EVALUATED,
                    ArtifactStatus.VERIFIED,
                    ArtifactStatus.NEGATIVE_RESULT,
                    ArtifactStatus.INCONCLUSIVE,
                }
                and evaluation_count < 1
            ):
                raise ValueError("experiment status requires a registered evaluation")
            model_count = connection.execute(
                """SELECT COUNT(*) FROM models
                JOIN checkpoints USING(checkpoint_id)
                WHERE checkpoints.experiment_id = ?""",
                (record.experiment_id,),
            ).fetchone()[0]
            if record.status is ArtifactStatus.VERIFIED and model_count < 1:
                raise ValueError("verified experiment requires a registered model")
            changed = connection.execute(
                "UPDATE experiments SET status = ?, record_json = ? WHERE experiment_id = ?",
                (record.status.value, self._json(record), record.experiment_id),
            ).rowcount
            if changed != 1:  # pragma: no cover - guarded by the lookup in this transaction
                raise RuntimeError(f"experiment update failed: {record.experiment_id}")

    def register_checkpoint(self, record: CheckpointRecord) -> None:
        path = Path(record.path)
        if not path.is_file():
            raise ValueError("checkpoint artifact is missing")
        if path.stat().st_size != record.byte_count or _file_sha256(path) != record.sha256:
            raise ValueError("checkpoint artifact does not match its registered identity")
        with self.transaction() as connection:
            experiment = connection.execute(
                "SELECT status FROM experiments WHERE experiment_id = ?",
                (record.experiment_id,),
            ).fetchone()
            if (
                experiment is None
                or experiment["status"] != ArtifactStatus.RUNNING_EXPERIMENT.value
            ):
                raise ValueError("checkpoint requires a running experiment")
            connection.execute(
                """INSERT INTO checkpoints(
                    checkpoint_id, experiment_id, sha256, record_json
                ) VALUES (?, ?, ?, ?)""",
                (record.checkpoint_id, record.experiment_id, record.sha256, self._json(record)),
            )

    def register_evaluation(self, record: EvaluationRecord) -> None:
        with self.transaction() as connection:
            lineage = connection.execute(
                """SELECT experiments.status, experiments.record_json
                FROM checkpoints
                JOIN experiments USING(experiment_id)
                WHERE checkpoints.checkpoint_id = ?""",
                (record.checkpoint_id,),
            ).fetchone()
            if lineage is None or lineage["status"] != ArtifactStatus.CHECKPOINTED.value:
                raise ValueError("evaluation requires a checkpointed experiment")
            experiment = ExperimentRecord.model_validate_json(lineage["record_json"])
            if record.dataset_sha256 != experiment.dataset_sha256:
                raise ValueError("evaluation dataset does not match its experiment")
            connection.execute(
                """INSERT INTO evaluations(evaluation_id, checkpoint_id, record_json)
                VALUES (?, ?, ?)""",
                (record.evaluation_id, record.checkpoint_id, self._json(record)),
            )

    def register_model(self, record: ModelRecord) -> None:
        if record.status is not ArtifactStatus.VERIFIED:
            raise ValueError("only verified checkpoints can be promoted to the model registry")
        with self.transaction() as connection:
            lineage = connection.execute(
                """SELECT checkpoints.record_json AS checkpoint_json,
                          experiments.status AS experiment_status
                FROM checkpoints
                JOIN experiments USING(experiment_id)
                WHERE checkpoints.checkpoint_id = ?""",
                (record.checkpoint_id,),
            ).fetchone()
            if lineage is None or lineage["experiment_status"] not in {
                ArtifactStatus.CHECKPOINTED.value,
                ArtifactStatus.EVALUATED.value,
            }:
                raise ValueError("model registration requires an evaluated checkpoint lineage")
            checkpoint = CheckpointRecord.model_validate_json(lineage["checkpoint_json"])
            path = Path(checkpoint.path)
            if (
                not path.is_file()
                or path.stat().st_size != checkpoint.byte_count
                or _file_sha256(path) != checkpoint.sha256
            ):
                raise ValueError("model checkpoint artifact does not match the registry")
            evaluations = connection.execute(
                "SELECT record_json FROM evaluations WHERE checkpoint_id = ?",
                (record.checkpoint_id,),
            ).fetchall()
            if not any(
                EvaluationRecord.model_validate_json(row["record_json"]).passed
                for row in evaluations
            ):
                raise ValueError("model registration requires a passing evaluation")
            connection.execute(
                """INSERT INTO models(model_id, checkpoint_id, status, record_json)
                VALUES (?, ?, ?, ?)""",
                (record.model_id, record.checkpoint_id, record.status.value, self._json(record)),
            )

    def append_evidence(
        self,
        *,
        timestamp: str,
        entity_type: str,
        entity_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> int:
        try:
            payload_json = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("evidence payload must be finite JSON data") from error
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
                    payload_json,
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
            rows = self._connection.execute(
                f"SELECT record_json FROM {table} ORDER BY rowid"
            ).fetchall()
        return [model.model_validate_json(row["record_json"]) for row in rows]

    def datasets(self) -> list[DatasetRecord]:
        return self._records("datasets", DatasetRecord)

    def dataset(self, dataset_id: str, version: str) -> DatasetRecord | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT record_json FROM datasets
                WHERE dataset_id = ? AND version = ?""",
                (dataset_id, version),
            ).fetchone()
        return None if row is None else DatasetRecord.model_validate_json(row["record_json"])

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
                payload=json.loads(
                    row["payload_json"],
                    parse_constant=_reject_json_constant,
                ),
            )
            for row in rows
        ]

    def integrity_check(self) -> str:
        with self._lock:
            result = self._connection.execute("PRAGMA integrity_check").fetchone()
            physical_result = str(result[0])
            if physical_result != "ok":
                return physical_result
            if self._connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                return "foreign key violation"
            checks: tuple[tuple[str, tuple[str, ...], type[Any]], ...] = (
                ("datasets", ("dataset_id", "version", "sha256"), DatasetRecord),
                (
                    "experiments",
                    ("experiment_id", "dataset_id", "dataset_version", "status"),
                    ExperimentRecord,
                ),
                (
                    "checkpoints",
                    ("checkpoint_id", "experiment_id", "sha256"),
                    CheckpointRecord,
                ),
                ("evaluations", ("evaluation_id", "checkpoint_id"), EvaluationRecord),
                ("models", ("model_id", "checkpoint_id", "status"), ModelRecord),
            )
            for table, columns, model in checks:
                selected = ", ".join((*columns, "record_json"))
                rows = self._connection.execute(f"SELECT {selected} FROM {table}").fetchall()
                for row in rows:
                    try:
                        record = model.model_validate_json(row["record_json"])
                    except TypeError, ValueError:
                        return f"invalid {table} record"
                    for column in columns:
                        value = getattr(record, column)
                        stored = row[column]
                        if isinstance(value, ArtifactStatus):
                            value = value.value
                        if value != stored:
                            return f"{table} column/record mismatch: {column}"

            experiment_rows = self._connection.execute(
                """SELECT experiments.record_json AS experiment_json,
                          datasets.record_json AS dataset_json
                FROM experiments
                JOIN datasets ON datasets.dataset_id = experiments.dataset_id
                    AND datasets.version = experiments.dataset_version"""
            ).fetchall()
            for row in experiment_rows:
                experiment = ExperimentRecord.model_validate_json(row["experiment_json"])
                dataset = DatasetRecord.model_validate_json(row["dataset_json"])
                if experiment.dataset_sha256 != dataset.sha256:
                    return "experiment dataset content identity mismatch"
                checkpoint_count = self._connection.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE experiment_id = ?",
                    (experiment.experiment_id,),
                ).fetchone()[0]
                evaluation_rows = self._connection.execute(
                    """SELECT evaluations.record_json
                    FROM evaluations
                    JOIN checkpoints USING(checkpoint_id)
                    WHERE checkpoints.experiment_id = ?""",
                    (experiment.experiment_id,),
                ).fetchall()
                evaluations = [
                    EvaluationRecord.model_validate_json(row["record_json"])
                    for row in evaluation_rows
                ]
                model_count = self._connection.execute(
                    """SELECT COUNT(*) FROM models
                    JOIN checkpoints USING(checkpoint_id)
                    WHERE checkpoints.experiment_id = ?""",
                    (experiment.experiment_id,),
                ).fetchone()[0]
                artifact_statuses = {
                    ArtifactStatus.CHECKPOINTED,
                    ArtifactStatus.EVALUATED,
                    ArtifactStatus.VERIFIED,
                    ArtifactStatus.NEGATIVE_RESULT,
                    ArtifactStatus.INCONCLUSIVE,
                }
                evaluation_statuses = {
                    ArtifactStatus.EVALUATED,
                    ArtifactStatus.VERIFIED,
                    ArtifactStatus.NEGATIVE_RESULT,
                    ArtifactStatus.INCONCLUSIVE,
                }
                if experiment.status in artifact_statuses and checkpoint_count < 1:
                    return "experiment status is missing its checkpoint"
                if experiment.status in evaluation_statuses and not evaluation_rows:
                    return "experiment status is missing its evaluation"
                if experiment.status is ArtifactStatus.VERIFIED and model_count < 1:
                    return "verified experiment is missing its model"
                if experiment.status is ArtifactStatus.VERIFIED and not any(
                    evaluation.passed for evaluation in evaluations
                ):
                    return "verified experiment has no passing evaluation"
                if experiment.status is ArtifactStatus.NEGATIVE_RESULT and any(
                    evaluation.passed for evaluation in evaluations
                ):
                    return "negative experiment has a passing evaluation"

            evaluation_rows = self._connection.execute(
                """SELECT evaluations.record_json AS evaluation_json,
                          experiments.record_json AS experiment_json
                FROM evaluations
                JOIN checkpoints USING(checkpoint_id)
                JOIN experiments USING(experiment_id)"""
            ).fetchall()
            for row in evaluation_rows:
                evaluation = EvaluationRecord.model_validate_json(row["evaluation_json"])
                experiment = ExperimentRecord.model_validate_json(row["experiment_json"])
                if evaluation.dataset_sha256 != experiment.dataset_sha256:
                    return "evaluation dataset content identity mismatch"

            model_rows = self._connection.execute(
                """SELECT models.record_json AS model_json,
                          evaluations.record_json AS evaluation_json
                FROM models
                LEFT JOIN evaluations USING(checkpoint_id)
                ORDER BY models.model_id"""
            ).fetchall()
            model_evaluations: dict[str, list[EvaluationRecord]] = {}
            for row in model_rows:
                model_record = ModelRecord.model_validate_json(row["model_json"])
                if model_record.status is not ArtifactStatus.VERIFIED:
                    return "model registry contains a non-verified model"
                evaluations = model_evaluations.setdefault(model_record.model_id, [])
                if row["evaluation_json"] is not None:
                    evaluations.append(
                        EvaluationRecord.model_validate_json(row["evaluation_json"])
                    )
            if any(
                not any(evaluation.passed for evaluation in evaluations)
                for evaluations in model_evaluations.values()
            ):
                return "model registry lineage has no passing evaluation"

            evidence_rows = self._connection.execute(
                """SELECT sequence, timestamp, entity_type, entity_id, action, payload_json
                FROM evidence ORDER BY sequence"""
            ).fetchall()
            for row in evidence_rows:
                try:
                    payload = json.loads(
                        row["payload_json"],
                        parse_constant=_reject_json_constant,
                    )
                    EvidenceRecord(
                        sequence=row["sequence"],
                        timestamp=row["timestamp"],
                        entity_type=row["entity_type"],
                        entity_id=row["entity_id"],
                        action=row["action"],
                        payload=payload,
                    )
                except TypeError, ValueError:
                    return "invalid evidence record"
        return "ok"

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> FoundryStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
