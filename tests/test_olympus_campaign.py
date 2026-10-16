from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import pytest

from olympus.training import campaign
from olympus.training.provenance import digest_json, file_sha256


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "campaign_id": "engineering-smoke",
        "purpose": "smoke",
        "max_wall_seconds": 600,
        "stop_reserve_seconds": 30,
        "cells": [
            {
                "id": "pilot",
                "dataset": "data/manifest_development.json",
                "training_config": {
                    "seed": 123,
                    "steps": 8,
                    "batch_size": 4,
                    "learning_rate": 0.0003,
                    "warmup_steps": 1,
                    "device": "cpu",
                    "precision": "fp32",
                    "cpu_threads": 1,
                    "evaluation_interval": 4,
                    "checkpoint_interval_seconds": 30,
                    "model": {"fusion_dim": 16},
                },
            }
        ],
    }


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _dataset(root: Path) -> Path:
    root.mkdir(parents=True)
    train = root / "train.pt"
    development = root / "development.pt"
    train.write_bytes(b"fixture train bytes, never deserialized")
    development.write_bytes(b"fixture development bytes, never deserialized")
    value = {
        "schema_version": 1,
        "dataset_id": "engineering-fixture",
        "version": "1.0.0",
        "license": "CC-BY-4.0",
        "extensions": [],
        "manifest_sha256": None,
        "train": {
            "path": "train.pt",
            "sha256": file_sha256(train),
            "source_ids": ["train-1"],
        },
        "evaluation": {
            "path": "development.pt",
            "sha256": file_sha256(development),
            "source_ids": ["dev-1"],
        },
    }
    value["manifest_sha256"] = hashlib.sha256(
        (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
    ).hexdigest()
    path = root / "manifest_development.json"
    _write(path, value)
    return path


def test_campaign_import_and_help_do_not_import_torch() -> None:
    code = "import sys; import olympus.training.campaign; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
    result = subprocess.run(
        [sys.executable, "-m", "olympus.training.campaign", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prepare-smoke" in result.stdout


def test_dry_run_needs_no_data_and_imports_no_torch(tmp_path: Path) -> None:
    manifest_path = tmp_path / "campaign.json"
    _write(manifest_path, _manifest())
    run_root = tmp_path / "not-created"
    code = (
        "import sys; from pathlib import Path; from olympus.training.campaign import dry_run; "
        "result=dry_run(Path(sys.argv[1]),Path(sys.argv[2])); "
        "assert result['status']=='DRY_RUN_NO_EXECUTION'; "
        "assert result['cells'][0]['data_status']=='NOT_PREPARED'; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(manifest_path), str(run_root)],
        check=True,
        capture_output=True,
    )
    assert not run_root.exists()


def test_preflight_memory_failure_returns_blocked_without_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"torch": "2.4.0", "numpy": "2.0.0", "pydantic": "2.10.0", "Pillow": "12.0.0"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)

    def blocked() -> None:
        raise PermissionError("resource snapshot denied")

    monkeypatch.setattr(campaign, "memory_snapshot", blocked)
    monkeypatch.setitem(sys.modules, "olympus.models.runtime", None)
    result = campaign.preflight(tmp_path, "cuda")
    assert result["status"] == "BLOCKED_RESOURCE_OR_DEVICE"
    assert result["package_versions"] == versions
    assert "resource snapshot denied" in result["reason"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(purpose="confirmatory"),
        lambda value: value.update(max_wall_seconds=6 * 3600 + 1),
        lambda value: value["cells"][0].update(dataset="protected_test.json"),
        lambda value: value["cells"][0]["training_config"].update(device="auto"),
        lambda value: value["cells"][0]["training_config"].update(precision="fp16"),
        lambda value: value["cells"][0]["training_config"].update(resource_preflight=False),
        lambda value: value["cells"][0]["training_config"].update(
            min_available_bytes=512 * 1024**2
        ),
        lambda value: value["cells"][0]["training_config"].pop("seed"),
        lambda value: value["cells"].append(copy.deepcopy(value["cells"][0])),
    ],
)
def test_campaign_rejects_ambiguous_or_unbounded_controls(
    mutation: Callable[[dict], object],
) -> None:
    value = _manifest()
    mutation(value)
    with pytest.raises(ValueError):
        campaign.validate_manifest(value)


def test_durable_budget_counts_active_crash_time_and_rejects_rollback() -> None:
    state = {"elapsed_seconds": 30, "active_started_at": 100}
    assert campaign.remaining_seconds(state, 600, now=180) == 490
    assert campaign.remaining_seconds(state, 600, now=900) == 0
    with pytest.raises(ValueError, match="backwards"):
        campaign.remaining_seconds(state, 600, now=99)


def test_binding_detects_changed_data_and_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "campaign.json"
    _write(path, _manifest())
    _dataset(tmp_path / "data")
    monkeypatch.setattr(campaign, "source_identity", lambda: {"sha256": "a" * 64})
    first = campaign.bind_contract(path, tmp_path / "runs")
    assert campaign.bind_contract(path, tmp_path / "relocated-runs") == first
    monkeypatch.setattr(campaign, "source_identity", lambda: {"sha256": "b" * 64})
    second = campaign.bind_contract(path, tmp_path / "runs")
    assert first != second
    (tmp_path / "data" / "train.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="dataset digest mismatch"):
        campaign.bind_contract(path, tmp_path / "runs")


def test_dataset_and_export_paths_reject_traversal_and_symlinks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contained"):
        campaign._contained(tmp_path, "../secret")
    (tmp_path / "target").write_bytes(b"data")
    (tmp_path / "alias").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="symlink"):
        campaign._contained(tmp_path, "alias")


def _completed_fixture(root: Path) -> tuple[dict, Path]:
    cell = _manifest()["cells"][0]
    cell["dataset_identity"] = {"manifest_sha256": "d" * 64}
    contract = {"manifest": {**_manifest(), "cells": [cell]}, "source_sha256": "s" * 64}
    _write(root / "campaign_contract.json", contract)
    run = root / "cells" / "pilot"
    run.mkdir(parents=True)
    (run / "checkpoint.pt").write_bytes(b"checkpoint fixture")
    report = {
        "status": "COMPLETED_DEVELOPMENT_NOT_PROMOTED",
        "promotion_authorized": False,
        "dataset_manifest_sha256": "d" * 64,
        "checkpoint_sha256": file_sha256(run / "checkpoint.pt"),
        "completed_steps": 8,
        "final_evaluation": {"decision_accuracy": 0.25},
        "training_config": cell["training_config"],
        "training_config_sha256": digest_json(cell["training_config"]),
        "source_identity": {"sha256": "s" * 64},
    }
    _write(run / "report.json", report)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "step-8-fixture.pt").write_bytes(b"optimizer checkpoint fixture")
    pointer = {
        "path": "step-8-fixture.pt",
        "sha256": file_sha256(checkpoints / "step-8-fixture.pt"),
        "completed_steps": 8,
        "binding": {
            "source_sha256": "s" * 64,
            "training_config_sha256": report["training_config_sha256"],
            "dataset_manifest_sha256": "d" * 64,
        },
    }
    _write(checkpoints / "latest.json", pointer)
    _write(checkpoints / "best.json", pointer)
    campaign._collect_completion(run, cell, "s" * 64, 1.0)
    return cell, run


def test_completion_rejects_tampered_report_and_checkpoint(tmp_path: Path) -> None:
    cell, run = _completed_fixture(tmp_path)
    assert campaign._completion(run, cell, "s" * 64)["completed_steps"] == 8
    (run / "checkpoint.pt").write_bytes(b"altered")
    with pytest.raises(ValueError, match="completed artifact changed"):
        campaign._completion(run, cell, "s" * 64)


def test_export_uses_completion_allowlist_and_never_archives_secrets(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _completed_fixture(root)
    (root / ".env").write_text("FAKE_SECRET=do-not-export", encoding="utf-8")
    (root / "unrelated.log").write_text("unrelated", encoding="utf-8")
    destination = tmp_path / "completed.tar.gz"
    result = campaign.export_completed(root, destination)
    with tarfile.open(destination) as archive:
        names = archive.getnames()
    assert names == result["files"]
    assert not any(".env" in name or "unrelated" in name for name in names)
    with pytest.raises(FileExistsError):
        campaign.export_completed(root, destination)


def test_blocked_admission_does_not_load_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Blocked:
        def __enter__(self) -> None:
            raise RuntimeError("available memory below floor")

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(campaign, "_admission", lambda _root: Blocked())
    with pytest.raises(RuntimeError, match="available memory"):
        campaign.execute_campaign(tmp_path / "missing.json", tmp_path / "runs")
    assert not (tmp_path / "runs" / "campaign_contract.json").exists()


def test_insufficient_cell_budget_stops_before_training_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    manifest["cells"][0]["estimated_seconds"] = 700
    contract = {"manifest": manifest, "source_sha256": "s" * 64}
    monkeypatch.setattr(campaign, "_admission", lambda _root: nullcontext())
    monkeypatch.setattr(campaign, "bind_contract", lambda _path, _root: contract)
    monkeypatch.setitem(sys.modules, "olympus.training.olympus_system", None)
    state = campaign.execute_campaign(tmp_path / "unused.json", tmp_path / "runs")
    assert state["status"] == "BUDGET_STOPPED"
    assert state["cells"] == {}


def test_recover_completed_report_after_crash_without_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runs"
    _cell, run = _completed_fixture(root)
    (run / "completion.json").unlink()
    contract = json.loads((root / "campaign_contract.json").read_text())
    _write(
        root / "campaign_state.json",
        {
            "schema_version": 1,
            "elapsed_seconds": 10.0,
            "contract_sha256": digest_json(contract),
            "active_started_at": time.time() - 4,
            "active_cell_id": "pilot",
            "cells": {"pilot": {"status": "RUNNING", "steps": 8, "elapsed_seconds": 2.0}},
        },
    )
    monkeypatch.setattr(campaign, "_admission", lambda _root: nullcontext())
    monkeypatch.setattr(campaign, "bind_contract", lambda _path, _root: contract)
    monkeypatch.setitem(sys.modules, "olympus.training.olympus_system", None)
    state = campaign.execute_campaign(tmp_path / "unused.json", root)
    assert state["status"] == "COMPLETED_DEVELOPMENT_NOT_PROMOTED"
    assert state["elapsed_seconds"] >= 14
    assert state["cells"]["pilot"]["elapsed_seconds"] >= 6
    assert (run / "completion.json").is_file()


@pytest.mark.parametrize(
    "status,code",
    [
        ("BLOCKED_ENVIRONMENT", 2),
        ("BLOCKED_RESOURCE_OR_DEVICE", 2),
        ("INTERRUPTED", 3),
        ("BUDGET_STOPPED", 3),
    ],
)
def test_cli_stopped_statuses_are_nonzero(
    status: str,
    code: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "preflight", lambda _root, _device: {"status": status})
    monkeypatch.setattr(sys, "argv", ["campaign", "preflight", "--run-root", str(tmp_path)])
    with pytest.raises(SystemExit) as raised:
        campaign.main()
    assert raised.value.code == code
