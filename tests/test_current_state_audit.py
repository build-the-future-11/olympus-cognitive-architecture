from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_current_state.py"
SPEC = importlib.util.spec_from_file_location("olympus_current_state_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def make_repo(path: Path) -> None:
    subprocess.run(
        ["git", "init", str(path)], check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://user:SECRET@github.com/owner/repo.git?token=SECRET#SECRET",
         "https://github.com/owner/repo.git"),
        ("git@github.com:owner/repo.git", "github.com:owner/repo.git"),
        ("ssh://user:SECRET@github.com/owner/repo", "ssh://github.com/owner/repo"),
        ("/private/path/SECRET", "[local or unrecognized remote omitted]"),
        ("file:///private/SECRET", "[local or unrecognized remote omitted]"),
    ],
)
def test_remote_redaction(raw: str, expected: str) -> None:
    assert audit.safe_remote(raw) == expected
    assert "SECRET" not in audit.safe_remote(raw)


def test_weights_are_present_not_claimed_runnable(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    (directory / "candidate.pt").write_bytes(b"untrusted-pickle-like-bytes")
    (directory / "empty.safetensors").touch()
    (directory / ".env").write_text("SECRET=do-not-read")
    (directory / "training.jsonl").write_text("private training examples")
    report = audit.scan_artifacts(tmp_path)
    states = {item["path"]: item["state"] for item in report["artifacts"]}
    assert states == {
        "checkpoints/candidate.pt": "PRESENT_NOT_LOADED",
        "checkpoints/empty.safetensors": "EMPTY_FILE",
    }
    assert "SECRET" not in json.dumps(report)
    assert "private training" not in json.dumps(report)


def test_lfs_pointer_is_not_a_checkpoint(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "model.gguf").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 100\n"
    )
    assert audit.scan_artifacts(tmp_path)["artifacts"][0]["state"] == "LFS_POINTER_ONLY"


@pytest.mark.parametrize("payload", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', '[]'])
def test_invalid_or_nonobject_json_is_not_evidence(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(payload)
    assert audit.inspect_json(path)["json_status"] in {"INVALID_JSON", "NOT_AN_OBJECT"}


def test_json_only_exports_allowlisted_evidence(tmp_path: Path) -> None:
    path = tmp_path / "release-manifest.json"
    path.write_text(json.dumps({
        "checkpoint_sha256": "b" * 64, "status": "PROMOTED",
        "prompt": "SECRET", "api_key": "SECRET", "model_id": "SECRET",
    }))
    report = audit.inspect_json(path)
    assert report["status_claim_not_verified"] == "PROMOTED"
    assert report["hash_claims_not_verified"] == {"checkpoint_sha256": "b" * 64}
    assert "SECRET" not in json.dumps(report)
    assert report["json_status"] == "PARSED_NOT_VERIFIED"


def test_metadata_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(b" " * (audit.MAX_JSON_BYTES + 1))
    assert audit.inspect_json(path)["json_status"] == "SKIPPED_SIZE_LIMIT"


def test_missing_shard_and_unsafe_paths(tmp_path: Path) -> None:
    (tmp_path / "one.safetensors").write_bytes(b"not-loaded")
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps({"weight_map": {
        "a": "one.safetensors", "b": "missing.safetensors", "c": "../outside.safetensors",
    }}))
    report = audit.inspect_json(path)
    assert report["shard_index_status"] == "INCOMPLETE_OR_UNSAFE"
    assert {item["state"] for item in report["shards"]} == {
        "PRESENT_NOT_LOADED", "MISSING", "UNSAFE_PATH",
    }


def test_complete_shards_still_are_not_load_verified(tmp_path: Path) -> None:
    (tmp_path / "one.safetensors").write_bytes(b"not-loaded")
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps({"weight_map": {"a": "one.safetensors"}}))
    assert audit.inspect_json(path)["shard_index_status"] == "FILES_PRESENT_NOT_LOAD_VERIFIED"


def test_symlinks_are_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "candidate.pt").write_text("SECRET")
    (tmp_path / "models").symlink_to(outside, target_is_directory=True)
    report = audit.scan_artifacts(tmp_path)
    assert report["artifacts"] == []
    assert report["scope_complete"] is False
    assert audit.shard_state(tmp_path, "models/candidate.pt") == "SYMLINK_NOT_FOLLOWED"
    with pytest.raises(OSError):
        audit.read_prefix(tmp_path / "models", 100)


def test_entry_limit_is_explicit(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    for number in range(5):
        (tmp_path / "models" / f"{number}.pt").touch()
    report = audit.scan_artifacts(tmp_path, max_entries=2)
    assert len(report["artifacts"]) <= 2
    assert report["scope_complete"] is False
    assert report["limitations"]


def test_shard_count_limit(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps({"weight_map": {
        str(number): f"{number}.safetensors" for number in range(audit.MAX_SHARDS + 1)
    }}))
    assert audit.inspect_json(path)["shard_index_status"] == "SKIPPED_SHARD_LIMIT"


def test_bounded_command_rejects_excessive_output() -> None:
    result = audit.bounded_command(
        [sys.executable, "-c", "print('a' * 200000)"], dict(os.environ),
    )
    assert result == {"status": "NOT_CHECKED", "reason": "OUTPUT_LIMIT"}


def test_bounded_command_does_not_publish_stderr() -> None:
    result = audit.bounded_command(
        [sys.executable, "-c", "import sys; sys.stderr.write('SECRET'); sys.exit(1)"],
        dict(os.environ),
    )
    assert result == {"status": "NOT_CHECKED", "exit_code": 1}


def test_end_to_end_new_report_does_not_claim_readiness(tmp_path: Path) -> None:
    make_repo(tmp_path)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "remote.origin.url",
         "https://user:SECRET@github.com/owner/repo.git?token=SECRET"], check=True,
    )
    (tmp_path / "artifacts").mkdir()
    checkpoint = tmp_path / "artifacts" / "model.pt"
    checkpoint.write_bytes(b"unchanged-and-never-loaded")
    report = audit.collect(tmp_path)
    destination = audit.write_report(tmp_path, report)
    saved = json.loads((destination / "EVIDENCE.json").read_text())
    assert saved["verification"]["model_release_readiness"] == "NOT_ESTABLISHED"
    assert saved["verification"]["runtime_generation"] == "NOT_RUN"
    assert saved["audit_time_ist"].endswith("+05:30")
    assert "SECRET" not in json.dumps(saved)
    assert checkpoint.read_bytes() == b"unchanged-and-never-loaded"
    assert stat.S_IMODE((destination / "EVIDENCE.json").stat().st_mode) == 0o600
    assert (destination / "CURRENT_STATUS.md").is_file()
    assert audit.write_report(tmp_path, report) != destination


def test_non_git_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Git working-tree root"):
        audit.collect(tmp_path)


def test_symlinked_report_parent_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "reports").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        audit.write_report(tmp_path, {})
    assert list(outside.iterdir()) == []


def test_total_metadata_budget_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "config.json").write_text('{"model_type":"tiny"}')
    monkeypatch.setattr(audit, "MAX_METADATA_BYTES", 1)
    report = audit.scan_artifacts(tmp_path)
    assert report["artifacts"][0]["json_status"] == "SKIPPED_TOTAL_METADATA_LIMIT"
    assert report["scope_complete"] is False


def test_lfs_shard_is_not_treated_as_complete(tmp_path: Path) -> None:
    (tmp_path / "one.safetensors").write_text("version https://git-lfs.github.com/spec/v1\n")
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps({"weight_map": {"a": "one.safetensors"}}))
    report = audit.inspect_json(path)
    assert report["shard_index_status"] == "INCOMPLETE_OR_UNSAFE"
    assert report["shards"][0]["state"] == "LFS_POINTER_ONLY"


@pytest.mark.skipif(os.name != "posix", reason="collector supports macOS/Linux")
def test_named_pipes_are_not_read(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    fifo = tmp_path / "models" / "candidate.pt"
    os.mkfifo(fifo)
    assert audit.scan_artifacts(tmp_path)["artifacts"] == []
    with pytest.raises(ValueError, match="not a regular file"):
        audit.read_prefix(fifo, 128)
