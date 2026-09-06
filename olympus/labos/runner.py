from __future__ import annotations

import json
import os
import shlex
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from olympus.labos.manifest import ProjectManifest, ProjectStatus, ResourceProfile

SENSITIVE_ENV_FRAGMENTS = (
    "API_KEY",
    "AUTH",
    "COOKIE",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)


@dataclass(slots=True)
class RunResult:
    project_id: str
    command: str
    return_code: int
    stdout: str
    stderr: str
    status: ProjectStatus
    started_at: str
    finished_at: str
    profile: str
    cwd: str
    classification: str


class RunDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                command TEXT NOT NULL,
                profile TEXT NOT NULL,
                cwd TEXT NOT NULL,
                return_code INTEGER NOT NULL,
                status TEXT NOT NULL,
                classification TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                stdout TEXT NOT NULL,
                stderr TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> RunDatabase:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def insert(self, result: RunResult) -> None:
        self.connection.execute(
            """
            INSERT INTO runs (
                project_id, command, profile, cwd, return_code, status,
                classification, started_at, finished_at, stdout, stderr
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.project_id,
                result.command,
                result.profile,
                result.cwd,
                result.return_code,
                result.status.value,
                result.classification,
                result.started_at,
                result.finished_at,
                result.stdout,
                result.stderr,
            ),
        )
        self.connection.commit()

    def failed_runs(self) -> list[tuple[str, str]]:
        rows = self.connection.execute(
            "SELECT project_id, command FROM runs WHERE return_code != 0 ORDER BY id DESC"
        ).fetchall()
        return [(str(row[0]), str(row[1])) for row in rows]

    def rows(self) -> list[dict[str, object]]:
        cursor = self.connection.execute(
            """
            SELECT project_id, command, profile, cwd, return_code, status,
                   classification, started_at, finished_at
            FROM runs ORDER BY id
            """
        )
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def detailed_rows(self) -> list[dict[str, object]]:
        cursor = self.connection.execute(
            """
            SELECT project_id, command, profile, cwd, return_code, status,
                   classification, started_at, finished_at, stdout, stderr
            FROM runs ORDER BY id
            """
        )
        columns = [item[0] for item in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


class PortfolioRunner:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root
        self.run_db = RunDatabase(artifact_root / "labos_runs.sqlite3")
        self.event_logger = EventLogger(artifact_root / "labos_events.jsonl")

    def close(self) -> None:
        self.run_db.close()

    def run_manifest(
        self,
        manifest: ProjectManifest,
        profile: ResourceProfile = ResourceProfile.SMOKE,
        timeout_seconds: int = 120,
    ) -> list[RunResult]:
        commands = [entry for entry in manifest.entry_points if entry.profile == profile]
        if not commands and profile == ResourceProfile.SMOKE:
            return []
        results: list[RunResult] = []
        for entry in commands:
            result = self.run_command(
                project_id=manifest.project_id,
                command=entry.command,
                cwd=manifest.path,
                profile=profile,
                timeout_seconds=entry.timeout_seconds or timeout_seconds,
                extra_env=entry.env,
            )
            results.append(result)
        return results

    def run_command(
        self,
        project_id: str,
        command: str,
        cwd: Path,
        profile: ResourceProfile,
        timeout_seconds: float = 120,
        extra_env: dict[str, str] | None = None,
    ) -> RunResult:
        started_at = datetime.now(UTC).isoformat()
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTEST_ADDOPTS"] = "-p no:cacheprovider"
        if extra_env:
            env.update(extra_env)
        try:
            completed = subprocess.run(
                shlex.split(command),
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            return_code = 124
            stdout = _decode_timeout_stream(exc.stdout)
            stderr = _decode_timeout_stream(exc.stderr)
            stderr = (stderr + "\n" if stderr else "") + (
                f"Command timed out after {timeout_seconds} seconds."
            )
        finished_at = datetime.now(UTC).isoformat()
        classification = classify_run(return_code, stderr)
        safe_stdout = _redact_sensitive_values(stdout, extra_env)
        safe_stderr = _redact_sensitive_values(stderr, extra_env)
        status = status_from_run(return_code, classification, profile)
        result = RunResult(
            project_id=project_id,
            command=command,
            return_code=return_code,
            stdout=safe_stdout[-20_000:],
            stderr=safe_stderr[-20_000:],
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            profile=profile.value,
            cwd=str(cwd),
            classification=classification,
        )
        self.run_db.insert(result)
        self.event_logger.log(
            {
                "project_id": project_id,
                "command": command,
                "profile": profile.value,
                "cwd": str(cwd),
                "return_code": return_code,
                "classification": classification,
                "status": status.value,
                "started_at": started_at,
                "finished_at": finished_at,
                "env_overrides": _redact_env_overrides(extra_env),
            }
        )
        return result


def classify_run(return_code: int, stderr: str) -> str:
    lowered = stderr.lower()
    if return_code == 0:
        return "ok"
    if return_code == 124 or "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "permission denied" in lowered or "operation not permitted" in lowered:
        return "permission"
    if "no module named" in lowered or "command not found" in lowered:
        return "dependency"
    if "not found" in lowered and "dataset" in lowered:
        return "external_dataset"
    if "credential" in lowered or "api key" in lowered:
        return "credential"
    return "failed_command"


def status_from_run(
    return_code: int,
    classification: str,
    profile: ResourceProfile,
) -> ProjectStatus:
    if return_code == 0 and profile == ResourceProfile.SMOKE:
        return ProjectStatus.SMOKE_TESTED
    if return_code == 0:
        return ProjectStatus.SCIENTIFICALLY_RUNNABLE
    if classification == "external_dataset":
        return ProjectStatus.BLOCKED_BY_EXTERNAL_DATASET
    if classification == "credential":
        return ProjectStatus.BLOCKED_BY_CREDENTIAL
    if classification == "permission":
        return ProjectStatus.BLOCKED_BY_PERMISSION
    if classification == "timeout":
        return ProjectStatus.FULL_BENCHMARK_PENDING_COMPUTE
    return ProjectStatus.DISCOVERED_ONLY


def _decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _is_sensitive_env_key(key: str) -> bool:
    normalized = key.upper()
    return any(fragment in normalized for fragment in SENSITIVE_ENV_FRAGMENTS)


def _redact_env_overrides(values: dict[str, str] | None) -> dict[str, str]:
    if not values:
        return {}
    return {
        key: "<redacted>" if _is_sensitive_env_key(key) else value
        for key, value in values.items()
    }


def _redact_sensitive_values(text: str, values: dict[str, str] | None) -> str:
    if not values:
        return text
    redacted = text
    for key, value in values.items():
        if _is_sensitive_env_key(key) and value:
            redacted = redacted.replace(value, "<redacted>")
    return redacted
