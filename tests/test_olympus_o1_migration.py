from __future__ import annotations

import hashlib
import json
from pathlib import Path


def test_o1_migration_is_byte_bound_and_not_authorized() -> None:
    root = Path(__file__).resolve().parents[1] / "research" / "olympus_o1"
    provenance = json.loads((root / "MIGRATION_PROVENANCE.json").read_text(encoding="utf-8"))

    assert provenance["status"] == "MIGRATED_PREREGISTRATION_NOT_RUN"
    assert provenance["execution_gate"] == {
        "status": "UNFROZEN_DO_NOT_RUN",
        "task_manifest_present": False,
        "provider_runtime_identity_frozen": False,
        "prompt_hashes_frozen": False,
        "execution_authorized": False,
    }
    for filename, expected_sha256 in provenance["files"].items():
        payload = (root / "upstream" / filename).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha256


def test_o1_templates_retain_null_freeze_and_four_arms() -> None:
    root = Path(__file__).resolve().parents[1] / "research" / "olympus_o1" / "upstream"
    protocol = json.loads((root / "olympus_o1_protocol_v1.json").read_text(encoding="utf-8"))
    freeze = json.loads(
        (root / "olympus_o1_execution_freeze_template_v1.json").read_text(encoding="utf-8")
    )

    assert protocol["status"] == "PREREGISTERED_NOT_RUN"
    assert protocol["task_count"] == 100
    assert len(protocol["arms"]) == 4
    assert freeze["status"] == "UNFROZEN_DO_NOT_RUN"
    assert freeze["task_manifest_sha256"] is None
    assert freeze["provider_model_exact_id"] is None
    assert set(freeze["system_prompt_hashes"]) == set(protocol["arms"])
