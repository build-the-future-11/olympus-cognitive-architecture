"""Stdlib-only, fail-closed accounting for one explicitly shared Kaggle account.

The file lock coordinates processes using this exact ledger file, not other
Kaggle sessions or private copies. Callers must inspect the platform themselves.
No function allocates or stops a platform session. A terminated child process
does NOT stop GPU billing: sessions accrue until record_platform_stop receives
an explicit stopped observation and a final displayed balance. Do not delete or
replace the ledger when changing projects or restarting a notebook.
"""

from __future__ import annotations

import codecs
import contextlib
import fcntl
import json
import math
import os
import signal
import subprocess
import tempfile
import time
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_OBSERVATION_AGE_SECONDS = 300
SHUTDOWN_MARGIN_SECONDS = 120


class BudgetError(RuntimeError):
    """An observation or spending request failed the account guard."""


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise BudgetError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _date(value: str | datetime) -> datetime:
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.tzinfo is None:
        raise BudgetError("timestamps require an explicit timezone")
    return result.astimezone(UTC)


def _now(now: str | datetime | None) -> datetime:
    return datetime.now(UTC) if now is None else _date(now)


def _observation(quota: float, observed_at: str | datetime, now: datetime) -> tuple[float, str]:
    value = _number(quota, "displayed quota")
    observed = _date(observed_at)
    age = (now - observed).total_seconds()
    if not 0 <= age <= MAX_OBSERVATION_AGE_SECONDS:
        raise BudgetError("quota observation is future-dated or older than five minutes")
    return value, observed.isoformat()


def _other_sessions(count: int | None) -> None:
    if type(count) is not int or count != 0:
        raise BudgetError("other active account sessions must be observed and equal zero")


@contextlib.contextmanager
def _locked(path: str | Path) -> Iterator[Path]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(destination.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield destination


def _read(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise BudgetError(f"nonfinite ledger value: {value}")

    ledger = json.loads(path.read_text(), parse_constant=reject_constant)
    if ledger.get("schema_version") != 1:
        raise BudgetError("unsupported account ledger schema")
    return dict(ledger)


def _write(path: Path, ledger: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(ledger, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _snapshot(ledger: dict[str, Any], now: datetime) -> dict[str, Any]:
    wall = physical = estimated = 0.0
    active = None
    for session in ledger["sessions"]:
        start = _date(session["allocation_started_at"])
        end = _date(session["observed_stopped_at"]) if session["observed_stopped_at"] else now
        if end < start:
            raise BudgetError("clock precedes an allocation start; cannot account safely")
        hours = (end - start).total_seconds() / 3600
        wall += hours
        physical += hours * session["device_count"]
        estimated += hours * session["conservative_charge_rate"]
        if session["observed_stopped_at"] is None:
            if active is not None:
                raise BudgetError("ledger contains multiple active allocations")
            active = session
    observed = 0.0
    debit = estimated
    for observation in ledger["quota_observations"]:
        observed_debit = max(0.0, ledger["initial_quota_hours"] - observation["quota_hours"])
        observed = max(observed, observed_debit)
        after_observation = 0.0
        for session in ledger["sessions"]:
            start = max(_date(session["allocation_started_at"]), _date(observation["observed_at"]))
            end = _date(session["observed_stopped_at"]) if session["observed_stopped_at"] else now
            after_observation += (
                max(0.0, (end - start).total_seconds()) / 3600 * session["conservative_charge_rate"]
            )
        debit = max(debit, observed_debit + after_observation)
    remaining = min(
        ledger["target_budget_hours"] - debit, ledger["initial_quota_hours"] - 24 - debit
    )
    rate = active["conservative_charge_rate"] if active else 1.0
    reserve = ledger["target_budget_hours"] * 0.1 + SHUTDOWN_MARGIN_SECONDS / 3600 * rate
    return {
        "at": now.isoformat(),
        "deadline": ledger["deadline"],
        "initial_quota_hours": ledger["initial_quota_hours"],
        "preferred_budget_hours": ledger["preferred_budget_hours"],
        "possible_expanded_budget_hours": ledger["possible_expanded_budget_hours"],
        "target_budget_hours": ledger["target_budget_hours"],
        "session_wall_hours": wall,
        "physical_device_hours": physical,
        "conservative_estimated_debit_hours": estimated,
        "observed_quota_debit_hours": observed,
        "accounted_debit_hours": debit,
        "latest_observed_quota_hours": ledger["latest_quota_hours"],
        "latest_quota_observed_at": ledger["latest_quota_observed_at"],
        "remaining_budget_hours": max(0.0, remaining),
        "reserved_hours": reserve,
        "seconds_to_deadline": (_date(ledger["deadline"]) - now).total_seconds(),
        "active_session_id": active["id"] if active else None,
        "platform_stop_verified": active is None and bool(ledger["sessions"]),
        "jobs": ledger["jobs"],
    }


def initialize(
    path: str | Path,
    *,
    quota_hours: float,
    observed_at: str | datetime,
    deadline: str | datetime,
    other_active_sessions: int | None,
    now: str | datetime | None = None,
) -> dict[str, Any]:
    """Create once. Re-entry cannot replace the original budget or deadline."""
    current = _now(now)
    quota, observation = _observation(quota_hours, observed_at, current)
    _other_sessions(other_active_sessions)
    end = _date(deadline)
    if end <= current:
        raise BudgetError("explicit campaign deadline has passed")
    with _locked(path) as destination:
        if destination.exists():
            ledger = _read(destination)
            if ledger["initial_quota_hours"] != quota or ledger["deadline"] != end.isoformat():
                raise BudgetError("existing account budget/deadline cannot be reset")
        else:
            preferred = max(0.0, min(6.0, quota - 24.0))
            ledger = {
                "schema_version": 1,
                "created_at": current.isoformat(),
                "initial_quota_hours": quota,
                "initial_quota_observed_at": observation,
                "latest_quota_hours": quota,
                "latest_quota_observed_at": observation,
                "quota_observations": [{"quota_hours": quota, "observed_at": observation}],
                "preferred_budget_hours": preferred,
                "possible_expanded_budget_hours": max(0.0, min(10.0, quota - 20.0)),
                "target_budget_hours": min(4.0, preferred),
                "deadline": end.isoformat(),
                "sessions": [],
                "jobs": [],
            }
            _write(destination, ledger)
        return _snapshot(ledger, current)


def snapshot(path: str | Path, *, now: str | datetime | None = None) -> dict[str, Any]:
    with _locked(path) as destination:
        return _snapshot(_read(destination), _now(now))


def _add_observation(ledger: dict[str, Any], quota: float, observation: str) -> None:
    if _date(observation) < _date(ledger["latest_quota_observed_at"]):
        raise BudgetError("quota observations must be chronological")
    ledger["latest_quota_hours"], ledger["latest_quota_observed_at"] = quota, observation
    ledger["quota_observations"].append({"quota_hours": quota, "observed_at": observation})


def observe_quota(
    path: str | Path,
    *,
    quota_hours: float,
    observed_at: str | datetime,
    other_active_sessions: int | None,
    now: str | datetime | None = None,
) -> dict[str, Any]:
    """Append an actual displayed balance without resetting past debit estimates."""
    current = _now(now)
    quota, observation = _observation(quota_hours, observed_at, current)
    _other_sessions(other_active_sessions)
    with _locked(path) as destination:
        ledger = _read(destination)
        _add_observation(ledger, quota, observation)
        _write(destination, ledger)
        return _snapshot(ledger, current)


def start_session(
    path: str | Path,
    *,
    project: str,
    quota_hours: float,
    observed_at: str | datetime,
    other_active_sessions: int | None,
    allocation_started_at: str | datetime,
    device_count: int,
    charge_rate: float | None = None,
    charge_evidence: str | None = None,
    now: str | datetime | None = None,
) -> dict[str, Any]:
    """Record actual allocation start, using a fresh pre-allocation observation.

    Unknown charging uses device_count quota hours per wall hour. A verified
    different rate requires explicit evidence. This function does not allocate.
    An already-started session is recorded even if budget/deadline are exhausted;
    admit() then blocks work. Check initialize()/snapshot() BEFORE allocating.
    """
    current = _now(now)
    quota, observation = _observation(quota_hours, observed_at, current)
    _other_sessions(other_active_sessions)
    started = _date(allocation_started_at)
    if not _date(observation) <= started <= current:
        raise BudgetError("allocation start must follow quota observation and not be future-dated")
    if type(device_count) is not int or device_count < 1 or not project.strip():
        raise BudgetError("positive device count and project name required")
    if charge_rate is not None and not (charge_evidence and charge_evidence.strip()):
        raise BudgetError("a platform charging rate requires observation evidence")
    rate = (
        float(device_count)
        if charge_rate is None
        else _number(charge_rate, "charge rate", positive=True)
    )
    with _locked(path) as destination:
        ledger = _read(destination)
        before = _snapshot(ledger, current)
        if before["active_session_id"]:
            raise BudgetError("an allocation remains active; observe platform stop before another")
        _add_observation(ledger, quota, observation)
        ledger["sessions"].append(
            {
                "id": uuid.uuid4().hex,
                "project": project,
                "allocation_started_at": started.isoformat(),
                "device_count": device_count,
                "conservative_charge_rate": rate,
                "charge_evidence": charge_evidence,
                "preallocation_quota_hours": quota,
                "preallocation_observed_at": observation,
                "observed_stopped_at": None,
            }
        )
        _write(destination, ledger)
        return _snapshot(ledger, current)


def _admit(ledger: dict[str, Any], estimated_seconds: float, now: datetime) -> dict[str, Any]:
    duration = _number(estimated_seconds, "estimated seconds", positive=True)
    state = _snapshot(ledger, now)
    if not state["active_session_id"]:
        raise BudgetError("no recorded active allocation")
    if any(job["status"] == "RUNNING" for job in ledger["jobs"]):
        raise BudgetError("a job is already running or requires interruption reconciliation")
    session = next(item for item in ledger["sessions"] if item["id"] == state["active_session_id"])
    if duration + SHUTDOWN_MARGIN_SECONDS >= state["seconds_to_deadline"]:
        raise BudgetError("job estimate does not leave deadline shutdown margin")
    cost = duration / 3600 * session["conservative_charge_rate"]
    if cost + state["reserved_hours"] > state["remaining_budget_hours"]:
        raise BudgetError("job estimate does not fit remaining budget and reserve")
    return state


def admit(
    path: str | Path, *, estimated_seconds: float, now: str | datetime | None = None
) -> dict[str, Any]:
    with _locked(path) as destination:
        return _admit(_read(destination), estimated_seconds, _now(now))


def record_platform_stop(
    path: str | Path,
    *,
    observed_stopped_at: str | datetime,
    final_quota_hours: float,
    final_quota_observed_at: str | datetime,
    evidence: str,
    now: str | datetime | None = None,
) -> dict[str, Any]:
    """Record externally observed stop; evidence must identify the UI/API receipt.

    Passing values is an operator attestation, not independent platform checking.
    Never invoke merely because a subprocess or notebook cell finished.
    """
    current = _now(now)
    quota, observation = _observation(final_quota_hours, final_quota_observed_at, current)
    stopped = _date(observed_stopped_at)
    if not evidence.strip() or not stopped <= _date(observation) <= current:
        raise BudgetError("stopped observation and subsequent quota evidence required")
    with _locked(path) as destination:
        ledger = _read(destination)
        active = [item for item in ledger["sessions"] if item["observed_stopped_at"] is None]
        if len(active) != 1 or stopped < _date(active[0]["allocation_started_at"]):
            raise BudgetError("no matching active allocation for stop observation")
        active[0].update(observed_stopped_at=stopped.isoformat(), stop_evidence=evidence)
        _add_observation(ledger, quota, observation)
        for job in ledger["jobs"]:
            if job["status"] == "RUNNING":
                job.update(
                    status="INTERRUPTED",
                    finished_at=stopped.isoformat(),
                    reason="platform stop observed before successful job receipt",
                )
        _write(destination, ledger)
        return _snapshot(ledger, current)


def run_command(
    path: str | Path,
    command: Sequence[str],
    *,
    estimated_seconds: float,
    timeout_seconds: float,
    heartbeat_seconds: float = 15,
    terminate_grace_seconds: float = 10,
    cwd: str | Path | None = None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run once, without a shell; timeout kills its process group, NOT Kaggle.

    Logs combine stdout/stderr and are tailed to the console in bounded chunks.
    Heartbeats report real PID/elapsed time to stdout.
    A failed job remains in the denominator; this helper never retries.
    """
    if (
        isinstance(command, (str, bytes))
        or not command
        or any(not isinstance(x, str) for x in command)
    ):
        raise BudgetError("command must be a nonempty list of argument strings")
    timeout = _number(timeout_seconds, "timeout", positive=True)
    heartbeat = _number(heartbeat_seconds, "heartbeat", positive=True)
    grace = _number(terminate_grace_seconds, "termination grace")
    estimate = max(_number(estimated_seconds, "estimate", positive=True), timeout) + grace
    job_id = uuid.uuid4().hex
    output = Path(log_path) if log_path else Path(path).parent / "job-logs" / f"{job_id}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path) as destination:
        ledger = _read(destination)
        current = datetime.now(UTC)
        try:
            state = _admit(ledger, estimate, current)
        except BudgetError as error:
            ledger["jobs"].append(
                {
                    "id": job_id,
                    "command": list(command),
                    "status": "SKIPPED_BUDGET",
                    "reason": str(error),
                    "at": current.isoformat(),
                    "exit_code": None,
                }
            )
            _write(destination, ledger)
            raise
        job = {
            "id": job_id,
            "session_id": state["active_session_id"],
            "command": list(command),
            "status": "RUNNING",
            "started_at": current.isoformat(),
            "log_path": str(output.resolve()),
            "timeout_seconds": timeout,
        }
        ledger["jobs"].append(job)
        _write(destination, ledger)
    started = time.monotonic()
    process = None
    reason = None
    code = None
    interrupted = False
    tail = None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    next_heartbeat = 0.0

    def forward_log() -> int:
        if tail is None:
            return 0
        chunk = tail.read(16 * 1024)
        if chunk:
            print(decoder.decode(chunk), end="", flush=True)
        return len(chunk)

    try:
        with output.open("x") as log:
            tail = output.open("rb")
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
            while process.poll() is None:
                elapsed = time.monotonic() - started
                forward_log()
                current_state = snapshot(path)
                if (
                    elapsed >= timeout
                    or current_state["seconds_to_deadline"] <= SHUTDOWN_MARGIN_SECONDS + grace
                    or current_state["remaining_budget_hours"] <= current_state["reserved_hours"]
                ):
                    reason = "timeout/deadline/budget guard"
                    interrupted = True
                    break
                if elapsed >= next_heartbeat:
                    print(
                        json.dumps(
                            {
                                "job": job_id,
                                "status": "RUNNING",
                                "pid": process.pid,
                                "elapsed_seconds": round(elapsed, 3),
                                "log_path": str(output),
                            }
                        ),
                        flush=True,
                    )
                    next_heartbeat = elapsed + heartbeat
                time.sleep(min(heartbeat, 1.0, max(0.01, timeout - elapsed)))
    except BaseException as error:
        reason = f"{type(error).__name__}: {error}"
        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
        raise
    finally:
        if process is not None:
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    # The leader can exit on SIGTERM while descendants ignore it.
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            else:
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    reason = "command left live descendants; process group terminated"
                    interrupted = True
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
            code = process.returncode
        with _locked(path) as destination:
            ledger = _read(destination)
            job = next(item for item in ledger["jobs"] if item["id"] == job_id)
            job.update(
                status="SUCCEEDED"
                if code == 0 and reason is None
                else ("INTERRUPTED" if interrupted else "FAILED"),
                exit_code=code,
                reason=reason,
                finished_at=datetime.now(UTC).isoformat(),
                elapsed_seconds=time.monotonic() - started,
            )
            _write(destination, ledger)
        if tail is not None:
            try:
                # Bound console work even if a faulty child produced enormous logs.
                # The complete byte-for-byte log remains the authoritative artifact.
                for _ in range(64):
                    if not forward_log():
                        break
                if tail.tell() < os.fstat(tail.fileno()).st_size:
                    print(
                        f"[Console backlog truncated; complete child output: {output}]", flush=True
                    )
                print(decoder.decode(b"", final=True), end="", flush=True)
            finally:
                tail.close()
        print(json.dumps(job), flush=True)
    return job
