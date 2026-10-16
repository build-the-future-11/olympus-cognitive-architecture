"""Account safety checks use synthetic observations, never allocate a GPU."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from notebooks.kaggle_support import account_guard as guard

NOW = datetime(2026, 10, 16, 8, tzinfo=UTC)
DEADLINE = "2026-10-16T22:00:00+05:30"


def create(path: Path, *, quota: float = 30, now: datetime = NOW) -> dict:
    return guard.initialize(
        path,
        quota_hours=quota,
        observed_at=now,
        deadline=DEADLINE,
        other_active_sessions=0,
        now=now,
    )


def start(path: Path, *, now: datetime = NOW, devices: int = 1, **kwargs: object) -> dict:
    arguments = dict(
        project="candidate",
        quota_hours=30,
        observed_at=now,
        other_active_sessions=0,
        allocation_started_at=now,
        device_count=devices,
        now=now,
    )
    arguments.update(kwargs)
    return guard.start_session(path, **arguments)


@pytest.mark.parametrize(
    "quota,preferred,expanded,target",
    [
        (30, 6, 10, 4),
        (27, 3, 7, 3),
        (24, 0, 4, 0),
        (20, 0, 0, 0),
        (35, 6, 10, 4),
    ],
)
def test_actual_quota_formula(
    tmp_path: Path, quota: float, preferred: float, expanded: float, target: float
) -> None:
    state = create(tmp_path / "account.json", quota=quota)
    assert state["preferred_budget_hours"] == preferred
    assert state["possible_expanded_budget_hours"] == expanded
    assert state["target_budget_hours"] == target
    assert state["platform_stop_verified"] is False


@pytest.mark.parametrize("quota", [float("inf"), float("nan"), -1, True, "30"])
def test_invalid_quota_rejected(tmp_path: Path, quota: float) -> None:
    with pytest.raises(guard.BudgetError):
        create(tmp_path / "account.json", quota=quota)
    assert not (tmp_path / "account.json").exists()


def test_stale_future_naive_quota_observations_rejected(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    create(path)
    for observation in [
        NOW - timedelta(seconds=301),
        NOW + timedelta(seconds=1),
        NOW.replace(tzinfo=None),
    ]:
        with pytest.raises(guard.BudgetError):
            start(path, observed_at=observation)
    assert guard.snapshot(path, now=NOW)["active_session_id"] is None


@pytest.mark.parametrize("other", [None, 1, -1, True])
def test_unknown_or_uncoordinated_other_sessions_block(tmp_path: Path, other: int | None) -> None:
    path = tmp_path / "account.json"
    create(path)
    with pytest.raises(guard.BudgetError):
        start(path, other_active_sessions=other)


def test_unknown_dual_gpu_charge_counts_idle_setup_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    create(path)
    first = start(path, devices=2)
    state = guard.snapshot(path, now=NOW + timedelta(hours=1))
    assert state["session_wall_hours"] == 1
    assert state["physical_device_hours"] == 2
    assert state["conservative_estimated_debit_hours"] == 2
    assert state["remaining_budget_hours"] == 2
    assert state["active_session_id"] == first["active_session_id"]
    assert state["reserved_hours"] == pytest.approx(0.4 + 240 / 3600)
    assert not state["platform_stop_verified"]
    with pytest.raises(guard.BudgetError, match="active"):
        start(path, now=NOW + timedelta(hours=1))


def test_verified_charge_rate_needs_evidence(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    create(path)
    with pytest.raises(guard.BudgetError, match="evidence"):
        start(path, devices=2, charge_rate=1)
    start(path, devices=2, charge_rate=1, charge_evidence="Platform billing policy receipt")
    state = guard.snapshot(path, now=NOW + timedelta(hours=1))
    assert state["physical_device_hours"] == 2
    assert state["conservative_estimated_debit_hours"] == 1


def test_reinitialization_never_resets_budget_or_deadline(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    create(path)
    start(path)
    before = path.read_bytes()
    create(path)
    assert path.read_bytes() == before
    with pytest.raises(guard.BudgetError, match="cannot be reset"):
        create(path, quota=35)
    with pytest.raises(guard.BudgetError, match="cannot be reset"):
        guard.initialize(
            path,
            quota_hours=30,
            observed_at=NOW,
            deadline="2026-10-17T22:00:00+05:30",
            other_active_sessions=0,
            now=NOW,
        )
    assert path.read_bytes() == before


def test_reserve_and_deadline_gate_admission(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    create(path)
    start(path)
    guard.admit(path, estimated_seconds=3 * 3600, now=NOW)
    with pytest.raises(guard.BudgetError, match="reserve"):
        guard.admit(path, estimated_seconds=3.6 * 3600, now=NOW)
    with pytest.raises(guard.BudgetError, match="deadline"):
        guard.admit(path, estimated_seconds=30, now=datetime(2026, 10, 16, 16, 28, tzinfo=UTC))


def test_observed_debit_dominates_optimistic_time_estimate(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    create(path)
    start(path, quota_hours=26.5)
    state = guard.snapshot(path, now=NOW)
    assert state["observed_quota_debit_hours"] == 3.5
    with pytest.raises(guard.BudgetError, match="reserve"):
        guard.admit(path, estimated_seconds=360, now=NOW)


def test_expired_or_missing_deadline_is_never_defaulted(tmp_path: Path) -> None:
    with pytest.raises(guard.BudgetError, match="deadline"):
        create(tmp_path / "account.json", now=datetime(2026, 10, 17, tzinfo=UTC))
    with pytest.raises(TypeError):
        guard.initialize(
            tmp_path / "account.json",
            quota_hours=30,
            observed_at=NOW,
            other_active_sessions=0,
            now=NOW,
        )


def test_stop_requires_explicit_evidence_and_final_quota(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    create(path)
    start(path)
    end = NOW + timedelta(hours=1)
    with pytest.raises(guard.BudgetError, match="evidence"):
        guard.record_platform_stop(
            path,
            observed_stopped_at=end,
            final_quota_hours=29,
            final_quota_observed_at=end,
            evidence="",
            now=end,
        )
    state = guard.record_platform_stop(
        path,
        observed_stopped_at=end,
        final_quota_hours=28.8,
        final_quota_observed_at=end,
        evidence="UI shows stopped",
        now=end,
    )
    assert state["platform_stop_verified"]
    later = guard.snapshot(path, now=end + timedelta(hours=2))
    assert later["session_wall_hours"] == 1
    assert later["observed_quota_debit_hours"] == pytest.approx(1.2)
    start(path, now=end, quota_hours=28.8, project="another explicitly selected project")
    assert guard.snapshot(path, now=end)["remaining_budget_hours"] == pytest.approx(2.8)


def test_parallel_starts_cannot_both_allocate_ledger_slot(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    create(path)

    def attempt() -> bool:
        try:
            start(path)
            return True
        except guard.BudgetError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: attempt(), range(2))) == [False, True]
    assert len(json.loads(path.read_text())["sessions"]) == 1


def runtime_ledger(path: Path) -> None:
    current = datetime.now(UTC)
    guard.initialize(
        path,
        quota_hours=30,
        observed_at=current,
        deadline=current + timedelta(hours=4),
        other_active_sessions=0,
        now=current,
    )
    start(path, now=current)


def test_command_success_failure_and_logs_do_not_claim_platform_stop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "account.json"
    runtime_ledger(path)
    success = guard.run_command(
        path,
        [sys.executable, "-u", "-c", "print('measured output')"],
        estimated_seconds=2,
        timeout_seconds=2,
        heartbeat_seconds=0.01,
    )
    assert success["status"] == "SUCCEEDED"
    assert success["exit_code"] == 0
    assert "measured output" in Path(success["log_path"]).read_text()
    assert "measured output\n" in capsys.readouterr().out
    failure = guard.run_command(
        path,
        [sys.executable, "-u", "-c", "raise SystemExit(7)"],
        estimated_seconds=2,
        timeout_seconds=2,
        heartbeat_seconds=0.01,
    )
    assert failure["status"] == "FAILED"
    assert failure["exit_code"] == 7
    state = guard.snapshot(path)
    assert [job["status"] for job in state["jobs"]] == ["SUCCEEDED", "FAILED"]
    assert state["active_session_id"] and not state["platform_stop_verified"]


def test_live_child_stdout_and_stderr_are_forwarded_and_preserved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "account.json"
    runtime_ledger(path)
    script = (
        "import sys,time;print('actual step=1 loss=0.125');"
        "time.sleep(0.04);print('actual validation complete',file=sys.stderr)"
    )
    result = guard.run_command(
        path,
        [sys.executable, "-u", "-c", script],
        estimated_seconds=1,
        timeout_seconds=1,
        heartbeat_seconds=0.01,
    )
    output = capsys.readouterr().out
    log = Path(result["log_path"]).read_text()
    assert "actual step=1 loss=0.125\n" in output and "actual validation complete\n" in output
    assert log == "actual step=1 loss=0.125\nactual validation complete\n"
    assert result["status"] == "SUCCEEDED"


def test_timeout_terminates_command_and_records_interruption(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    runtime_ledger(path)
    result = guard.run_command(
        path,
        [sys.executable, "-u", "-c", "import time; time.sleep(20)"],
        estimated_seconds=1,
        timeout_seconds=0.05,
        heartbeat_seconds=0.01,
        terminate_grace_seconds=0.05,
    )
    assert result["status"] == "INTERRUPTED"
    assert result["exit_code"] < 0
    assert result["elapsed_seconds"] < 5
    assert guard.snapshot(path)["platform_stop_verified"] is False


def test_timeout_and_grace_both_reserved_before_child_starts(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    runtime_ledger(path)
    with pytest.raises(guard.BudgetError, match="reserve"):
        guard.run_command(
            path,
            [sys.executable, "-c", "raise RuntimeError"],
            estimated_seconds=1,
            timeout_seconds=3.5 * 3600,
            terminate_grace_seconds=400,
        )
    assert guard.snapshot(path)["jobs"][0]["status"] == "SKIPPED_BUDGET"


def test_no_shell_or_nan_command_controls(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    runtime_ledger(path)
    with pytest.raises(guard.BudgetError, match="argument strings"):
        guard.run_command(path, "python -c pass", estimated_seconds=1, timeout_seconds=1)
    with pytest.raises(guard.BudgetError, match="finite"):
        guard.run_command(path, ["true"], estimated_seconds=1, timeout_seconds=float("nan"))


def test_observed_debit_plus_subsequent_estimate_never_underaccounts(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    create(path)
    start(path)
    first_hour = NOW + timedelta(hours=1)
    guard.observe_quota(
        path, quota_hours=28, observed_at=first_hour, other_active_sessions=0, now=first_hour
    )
    state = guard.snapshot(path, now=NOW + timedelta(hours=2))
    assert state["conservative_estimated_debit_hours"] == 2
    assert state["observed_quota_debit_hours"] == 2
    assert state["accounted_debit_hours"] == 3
    # A displayed balance increase/reset cannot replenish the campaign budget.
    second_hour = NOW + timedelta(hours=2)
    guard.observe_quota(
        path, quota_hours=30, observed_at=second_hour, other_active_sessions=0, now=second_hour
    )
    assert guard.snapshot(path, now=second_hour)["accounted_debit_hours"] == 3
    assert guard.snapshot(path, now=second_hour)["remaining_budget_hours"] == 1


def test_actual_allocation_is_recorded_even_when_no_work_can_be_admitted(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    create(path, quota=24)
    state = start(path, quota_hours=24)
    assert state["active_session_id"]
    assert state["remaining_budget_hours"] == 0
    with pytest.raises(guard.BudgetError, match="reserve"):
        guard.admit(path, estimated_seconds=1, now=NOW)
    assert guard.snapshot(path, now=NOW + timedelta(seconds=60))["session_wall_hours"] > 0


def test_failed_launch_is_preserved_and_does_not_stop_platform(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    runtime_ledger(path)
    with pytest.raises(FileNotFoundError):
        guard.run_command(path, ["/nonexistent/no-command"], estimated_seconds=1, timeout_seconds=1)
    state = guard.snapshot(path)
    assert state["jobs"][0]["status"] == "FAILED"
    assert state["active_session_id"]


def test_descendants_cannot_survive_timeout_when_leader_exits(tmp_path: Path) -> None:
    path = tmp_path / "account.json"
    runtime_ledger(path)
    marker = tmp_path / "survived.txt"
    child = (
        "import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"time.sleep(0.6);pathlib.Path({str(marker)!r}).write_text('still running')"
    )
    parent = (
        f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-u','-c',{child!r}]);"
        "time.sleep(20)"
    )
    result = guard.run_command(
        path,
        [sys.executable, "-u", "-c", parent],
        estimated_seconds=1,
        timeout_seconds=0.2,
        heartbeat_seconds=0.01,
        terminate_grace_seconds=0.05,
    )
    assert result["status"] == "INTERRUPTED"
    time.sleep(0.65)
    assert not marker.exists()
