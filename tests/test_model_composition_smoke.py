from __future__ import annotations

import json
from importlib.resources import files as package_files
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from typer.testing import CliRunner

from olympus.cli import app
from olympus.models import composition_smoke as composition_smoke_module
from olympus.models.composition_smoke import (
    reference_replay_source_sha256,
    run_reference_replay_smoke,
)
from olympus.models.hermes import HermesWorkspaceRuntime
from olympus.models.kronos import KronosTemporalPlanner
from olympus.models.substrate import TextOutput, canonical_sha256


def test_composition_smoke_persists_sanitized_nonqualifying_replay(
    tmp_path: Path,
) -> None:
    manifest = run_reference_replay_smoke(tmp_path, seed=20_260_906)
    persisted = tmp_path / "reference_replay_manifest.json"

    assert manifest.passed
    assert manifest.scientific_claim == "contract_composition_only"
    assert manifest.promotion_authorized is False
    assert manifest.qualifying_result is False
    assert all(manifest.deterministic_checks.values())
    assert set(manifest.source_paths) >= {
        "olympus/models/composition.py",
        "olympus/models/substrate.py",
    }
    assert all(path.startswith("olympus/") for path in manifest.source_paths)
    assert "pyproject.toml" not in manifest.source_paths
    assert "uv.lock" not in manifest.source_paths
    assert persisted.is_file()
    payload = json.loads(persisted.read_text(encoding="utf-8"))
    assert payload == manifest.model_dump(mode="json")
    assert manifest.deterministic_checks[
        "approval_binds_action_policy_and_executor_profile"
    ]
    assert manifest.deterministic_checks["budget_accounting_is_exact"]
    assert manifest.deterministic_checks["transaction_snapshot_is_terminal"]
    assert manifest.deterministic_checks[
        "kronos_is_host_supplied_and_non_authoritative"
    ]
    assert manifest.result.budget_usage.model_calls_used == 4
    assert manifest.result.budget_usage.tool_calls_used == 1
    assert manifest.result.transaction.state.value == "committed"
    assert (
        manifest.result.kronos_observation.source_kind
        == "host_supplied_pre_execution_features"
    )
    assert manifest.result.kronos_observation.status == "completed"
    assert manifest.result.kronos_observation.source_authenticated is False
    persisted_text = persisted.read_text(encoding="utf-8")
    assert "PRIVATE-ZEPHYR" not in persisted_text
    assert "RAW-CALCULATION-RESULT-MUST-NOT-PERSIST" not in persisted_text
    assert '"private_result"' not in persisted_text
    assert not {"workspace", "aion_state", "claim_output"} & set(payload["result"])


def test_composition_smoke_replay_is_stable_except_execution_identity(tmp_path: Path) -> None:
    first = run_reference_replay_smoke(tmp_path / "first", seed=99)
    second = run_reference_replay_smoke(tmp_path / "second", seed=99)

    assert first.source_sha256 == second.source_sha256
    assert first.protocol_id == second.protocol_id
    assert first.result == second.result
    assert first.deterministic_checks == second.deterministic_checks
    assert first.execution_id != second.execution_id


def test_composition_source_digest_uses_only_installed_package_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = reference_replay_source_sha256()
    installed_package = tmp_path / "site-packages" / "olympus"
    source_package = package_files("olympus")
    for relative in composition_smoke_module._PACKAGE_SOURCE_PATHS:
        parts = PurePosixPath(relative).parts
        destination = installed_package.joinpath(*parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source_package.joinpath(*parts).read_bytes())

    def isolated_package_files(package: str) -> Path:
        assert package == "olympus"
        return installed_package

    monkeypatch.setattr(composition_smoke_module, "package_files", isolated_package_files)
    monkeypatch.setattr(
        composition_smoke_module,
        "__file__",
        str(installed_package / "models" / "composition_smoke.py"),
    )

    assert not (tmp_path / "pyproject.toml").exists()
    assert not (tmp_path / "uv.lock").exists()
    assert reference_replay_source_sha256() == expected


def test_composition_smoke_cli_is_explicitly_nonqualifying(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "models",
            "composition-smoke",
            "--output-dir",
            str(tmp_path),
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["passed"] is True
    assert payload["scientific_claim"] == "contract_composition_only"
    assert payload["promotion_authorized"] is False
    assert payload["qualifying_result"] is False
    assert payload["deterministic_checks"]["budget_accounting_is_exact"] is True
    assert payload["deterministic_checks"][
        "approval_binds_action_policy_and_executor_profile"
    ] is True
    assert payload["result"]["transaction"]["state"] == "committed"
    assert payload["result"]["kronos_observation"]["source_authenticated"] is False


def test_composition_smoke_fails_closed_on_component_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_component(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("injected component failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            KronosTemporalPlanner,
            "forward",
            fail_component,
        )
        kronos_failure = run_reference_replay_smoke(tmp_path / "kronos", seed=8)
    assert kronos_failure.passed is False
    assert kronos_failure.deterministic_checks[
        "kronos_is_host_supplied_and_non_authoritative"
    ] is False
    assert kronos_failure.result.kronos_observation.status == "failed"

    with monkeypatch.context() as scoped:
        scoped.setattr(
            HermesWorkspaceRuntime,
            "respond_after_execution",
            fail_component,
        )
        hermes_failure = run_reference_replay_smoke(tmp_path / "hermes", seed=9)
    assert hermes_failure.passed is False
    assert hermes_failure.deterministic_checks["hermes_returned_grounded_text"] is False
    assert hermes_failure.result.answer.producer == "runtime_fail_closed"


def test_composition_smoke_rejects_fabricated_text_with_a_real_citation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fabricate(
        runtime: Any,
        workspace: Any,
        request: str,
        **kwargs: Any,
    ) -> TextOutput:
        del runtime, request
        allowed = set(kwargs["allowed_evidence_ids"])
        evidence_id = next(
            item.evidence_id
            for item in workspace.evidence
            if item.evidence_id in allowed and item.text is not None
        )
        return TextOutput(
            workspace_sha256=workspace.sha256,
            model_identity_sha256=canonical_sha256(workspace.model_identity),
            text="FABRICATED CLAIM NOT PRESENT IN ANY EVIDENCE",
            citation_evidence_ids=[evidence_id],
            confidence=1.0,
        )

    monkeypatch.setattr(
        HermesWorkspaceRuntime,
        "respond_after_execution",
        fabricate,
    )
    manifest = run_reference_replay_smoke(tmp_path, seed=10)

    assert manifest.passed is False
    assert manifest.deterministic_checks["hermes_returned_grounded_text"] is False
    assert manifest.result.answer.producer == "runtime_fail_closed"
    assert "FABRICATED" not in manifest.model_dump_json()
