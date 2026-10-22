"""Apply one hash-pinned repair on one review branch; never run research."""
from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

BASE = "a50cad27c66a788c83fe93db63b522559c9e15dd"
BRANCH = "work/olympus-verified-repairs-20261022"
REPOSITORY = "build-the-future-11/olympus-cognitive-architecture"
PATCH_SHA = "833b8d60315270da179062c81164f0dd388b72c537dd8ded59cbb5a40fceb5a8"
EXPECTED = {
    ".github/workflows/ci.yml": "8a5fc211cbe2a0effc797c87dd01817d0004146b",
    ".github/workflows/release.yml": "7972c3e378237e662039955e47a5dc3cb4704139",
    "apps/forge-web/package-lock.json": "00372df3b978f17c9507a0b8ad2056de653d4fdc",
    "apps/forge-web/package.json": "32c911bc3f1690b363b1c736b243f68b592ae411",
    "docs/faithful-dynamics.md": "2689ba195f704c256d3c372f4e364ddb5a03bf37",
    "olympus/cli.py": "c1f899c18b3c2db7931a726b2318bf1fb09c404a",
    "olympus/models/faithful_dynamics.py": "0a440c3a45b2904f21a1160ac8a5a2001df1138a",
    "pyproject.toml": "bc6fd56594b0922370dc8b9771eb0284647e2fa5",
    "research/verification/ci_20261022/README.md": "ed14173e3270dce87a1e5339313c4e2ba01ef24f",
    "research/verification/ci_20261022/frozen-artifacts.sha256": "6f98622a43d16b121b2d37d605d3324846818344",
    "tests/test_faithful_dynamics.py": "a24750276bdacc4053e15c43e317df0befd97409",
    "tests/test_multi_horizon_dynamics_v1.py": "160c43a3caec996f73fe3fb9da3674a2ddcc7534",
    "tests/test_research_state.py": "d55457b566dcb3119ad06c5fd5c906d60155035b",
}
CHUNKS = [f".integration/olympus-20261022.{i:02d}.b64" for i in range(3)]
HELPERS = [*CHUNKS, ".integration/integrate_olympus.py", ".github/workflows/integrate-olympus-20261022.yml"]
RECORD = Path("research/verification/integration_20261022")
ENGINEERING = Path("research/faithful_dynamics_engineering_v1")


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object, *, new: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if new else "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=False)
        handle.write("\n")


def main() -> None:
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise RuntimeError("Repository is not the explicitly authorized target")
    if os.environ.get("GITHUB_REF") != "refs/heads/" + BRANCH:
        raise RuntimeError("Ref is not the isolated review branch")
    bootstrap = run("git", "rev-parse", "HEAD")
    if bootstrap != os.environ.get("GITHUB_SHA"):
        raise RuntimeError("Checkout does not match the triggering revision")
    if run("git", "status", "--porcelain"):
        raise RuntimeError("Checkout is not clean")
    run("git", "merge-base", "--is-ancestor", BASE, "HEAD")
    if set(run("git", "diff", "--name-only", BASE, "HEAD").splitlines()) != set(HELPERS):
        raise RuntimeError("Branch contains changes other than the five reviewed helpers")
    parts = [Path(path).read_text().strip() for path in CHUNKS]
    # Correct one observed transport transcription only; the exact final patch
    # SHA-256 and every applied Git blob below remain mandatory.
    if run("git", "hash-object", "--no-filters", CHUNKS[2]) == "7f8424772698b45ef05658d078598854ff1b5412":
        if parts[2].count("YLCS6+6xoru") != 1:
            raise RuntimeError("Unexpected transport transcription")
        parts[2] = parts[2].replace("YLCS6+6xoru", "YLCS6xoru", 1)
        print("Corrected recorded base64 transport transcription; verifying original patch digest")
    encoded = "".join(parts)
    if len(encoded) != 22196:
        raise RuntimeError("Truncated or unexpected integration payload")
    patch = gzip.decompress(base64.b64decode(encoded, validate=True))
    if len(patch) != 55670 or hashlib.sha256(patch).hexdigest() != PATCH_SHA:
        raise RuntimeError("Reviewed repair checksum mismatch")
    with tempfile.TemporaryDirectory() as temporary:
        patch_path = Path(temporary) / "reviewed.patch"
        patch_path.write_bytes(patch)
        changed = {line.split("\t", 2)[2] for line in run("git", "apply", "--numstat", str(patch_path)).splitlines()}
        if changed != set(EXPECTED):
            raise RuntimeError("Repair modifies unexpected paths")
        run("git", "apply", "--check", "--index", str(patch_path))
        run("git", "apply", "--index", str(patch_path))
    for relative, expected in EXPECTED.items():
        if run("git", "hash-object", "--no-filters", relative) != expected:
            raise RuntimeError(f"Applied bytes differ from reviewed candidate: {relative}")
    run("sha256sum", "--check", "research/verification/ci_20261022/frozen-artifacts.sha256")

    state_path = Path("RESEARCH_STATE.json")
    state = json.loads(state_path.read_text())
    old_state = copy.deepcopy(state)
    v3_path = Path("research/multi_horizon_dynamics_v3_heavy/RESEARCH_STATE.json")
    v3 = json.loads(v3_path.read_text())
    write_json(RECORD / "PREVIOUS_ROOT_STATE.json", old_state)
    write_json(RECORD / "PREVIOUS_V3_STATE.json", v3)
    v3["phase"] = "OUTCOME_RECOVERY_BLOCKED_AUTHENTICATION"
    v3["current_execution_status"] = "UNKNOWN"
    v3["scientific_outcomes_created"] = None
    v3["outcome_archive_retrieved"] = False
    v3["access_observation_20261022"] = {
        "browser": "Originating connected-browser context unavailable",
        "api_credentials": "No Kaggle API token or configuration present in this session",
        "last_retained_observation": "RUNNING on 2026-10-21; not a current status",
        "duplicate_submission": False,
    }
    write_json(v3_path, v3, new=False)
    state["updated_utc"] = "2026-10-22"
    state["software"]["working_tree"] = "REVIEW_BRANCH_REPAIR_PENDING_EXACT_HOSTED_VALIDATION"
    state["current_phase"] = "V3_OUTCOME_RECOVERY_BLOCKED_ENGINEERING_REPAIR_VALIDATION"
    state["active_scientific_disposition"] = "V3_OUTCOME_UNKNOWN_PENDING_AUTHENTICATED_RETRIEVAL"
    for study in state["studies"]:
        if study["id"] == "multi-horizon-dynamics-v3-heavy":
            study["phase"] = v3["phase"]
    successor = state["active_scientific_successor"]
    successor.update(status=v3["phase"], current_execution_status="UNKNOWN", scientific_outcomes_created=None, outcome_archive_retrieved=False)

    # Keep the historical Aion receipt intact. Bind the exact CLI type correction
    # without copying old whole-suite pass claims into a fresh validation record.
    aion = next(s for s in state["engineering_studies"] if s["id"] == "aion-executor-v1")
    old_receipt_path = Path(aion["receipt"])
    old_receipt = json.loads(old_receipt_path.read_text())
    bound = {}
    for relative, expected in old_receipt["source_sha256"].items():
        actual = sha(Path(relative))
        if relative != "olympus/cli.py" and actual != expected:
            raise RuntimeError(f"Unexpected drift in Aion evidence: {relative}")
        bound[relative] = actual
    aion_receipt = RECORD / "AION_SOURCE_BINDING.json"
    write_json(aion_receipt, {
        "schema_version": 1, "study_id": aion["id"], "phase": aion["phase"],
        "kind": "source_binding_only_pending_hosted_tests",
        "source_sha256": bound, "prior_receipt": str(old_receipt_path),
        "prior_receipt_sha256": sha(old_receipt_path),
        "change": "List literals in the deterministic CLI fixture become equivalent tuple literals; executor logic is unchanged.",
        "new_test_pass_claim": False, "scientific_outcomes_created": False,
    })
    aion["receipt"] = str(aion_receipt)

    engineering_phase = "ARTIFICIAL_COMPONENT_CHECKS_PASSED_HOSTED_VALIDATION_PENDING"
    engineering_sources = ["olympus/models/faithful_dynamics.py", "tests/test_faithful_dynamics.py", "docs/faithful-dynamics.md"]
    write_json(ENGINEERING / "RESEARCH_STATE.json", {
        "schema_version": 1, "id": "faithful-dynamics-engineering-v1", "phase": engineering_phase,
        "question": "Does detached variance training with disjoint optimizers and clipping preserve the mean-only optimizer trajectory?",
        "falsifier": "Any mean gradient or mean AdamW state differs under identical initialization, batches and execution conditions.",
        "scientific_training_epochs": 0, "protected_evaluations": 0,
        "external_downloads": 0, "paid_jobs": 0, "promotion_authorized": False,
        "claim_boundary": "Artificial engineering invariance only; no new scientific efficacy, calibration, novelty or target-scale claim.",
    })
    write_json(ENGINEERING / "RECEIPT.json", {
        "schema_version": 1, "study_id": "faithful-dynamics-engineering-v1", "phase": engineering_phase,
        "source_sha256": {path: sha(Path(path)) for path in engineering_sources},
        "local_component_verification": {
            "passed": 37, "failed": 0, "deselected": 1, "seconds": 10.58,
            "runtime": "Python 3.13.5; Torch 2.10.0+cpu; not the project Python 3.14 locked environment",
            "command": "PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_faithful_dynamics.py -k 'not reference_arithmetic'",
            "deselection_reason": "The complete frozen v2 runner was unavailable in the isolated local snapshot; the unchanged test is retained for hosted execution.",
            "whole_application_installation": False,
        },
        "full_hosted_validation": "NOT_YET_OBSERVED", "scientific_outcomes_created": False,
    })
    state["engineering_studies"].append({
        "id": "faithful-dynamics-engineering-v1", "phase": engineering_phase,
        "path": str(ENGINEERING / "RESEARCH_STATE.json"),
        "receipt": str(ENGINEERING / "RECEIPT.json"), "source_verification": "current_files",
    })
    receipt_path = RECORD / "INTEGRATION_RECEIPT.json"
    new_artifacts = [str(RECORD / "PREVIOUS_ROOT_STATE.json"), str(RECORD / "PREVIOUS_V3_STATE.json"), str(aion_receipt), str(receipt_path), str(ENGINEERING / "RESEARCH_STATE.json"), str(ENGINEERING / "RECEIPT.json"), *engineering_sources, "research/verification/ci_20261022/README.md", "research/verification/ci_20261022/frozen-artifacts.sha256"]
    # The validator expects regular files. Expand any old directory inventory
    # rather than deleting its evidence or weakening that invariant.
    expanded = []
    directory_expansions = {}
    for relative in state["artifacts"]:
        path = Path(relative)
        if path.is_symlink():
            raise RuntimeError(f"Symlink in canonical artifact inventory: {relative}")
        if path.is_dir():
            leaves = sorted(str(item) for item in path.rglob("*") if item.is_file())
            if not leaves or any(Path(leaf).is_symlink() for leaf in leaves):
                raise RuntimeError(f"Empty or symlink-bearing evidence directory: {relative}")
            directory_expansions[relative] = leaves
            expanded.extend(leaves)
        else:
            if not path.is_file():
                raise RuntimeError(f"Existing artifact is missing: {relative}")
            expanded.append(relative)
    state["artifacts"] = list(dict.fromkeys([*expanded, *new_artifacts]))
    state["history"].append({
        "date": "2026-10-22", "event": "Reviewed software repair integrated on isolated branch; exact-head CI pending",
        "record": str(receipt_path), "source_base_commit": BASE,
        "scientific_outcomes_created": False,
        "limits": "No production deployment, fresh scientific training, protected evaluation, favorable rerun, paper submission or model promotion. Existing v3 run outcome is unknown.",
    })
    state["verification"]["integration_20261022"] = {
        "receipt": str(receipt_path), "scope": "Software and metadata repair; hosted acceptance pending",
        "fresh_local_component_passes": 37,
    }
    write_json(state_path, state, new=False)
    write_json(receipt_path, {
        "schema_version": 1, "date": "2026-10-22", "repository": REPOSITORY, "branch": BRANCH,
        "base_commit": BASE, "bootstrap_commit": bootstrap, "patch_sha256": PATCH_SHA,
        "exact_reviewed_blob_identities": EXPECTED,
        "prior_complete_repair_patch_sha256": "25093614df1ceaeb954db4a21ad658883287934a57461b69f4d7d90b8eb307db",
        "scope": "13 software/CI/source-test files from the recovered repair, with additive canonical-state reconciliation",
        "frozen_files_verified": 9, "directory_artifact_expansions": directory_expansions,
        "local_component_receipt": str(ENGINEERING / "RECEIPT.json"),
        "hosted_ci": "NOT_YET_OBSERVED", "complete_research_release_integrated": False,
        "recovered_paper_binaries_published": False, "new_kaggle_launches": 0,
        "new_scientific_results": 0, "production_deployments": 0, "worldcastr_touched": False,
    })
    manifest = Path("MANIFEST.in")
    additions = [f"include {path}" for path in state["artifacts"] if path not in old_state["artifacts"]]
    # All paths above are local repository paths, never shell arguments.
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write("\n# Additive evidence required by the reviewed software repair.\n")
        handle.write("\n".join(additions) + "\n")
    audit = run(sys.executable, "-c", "import importlib.util,sys; from pathlib import Path; s=importlib.util.spec_from_file_location('state_audit', 'olympus/research_state.py'); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); print(m.audit_research_state(Path('.')).as_dict())")
    print("Canonical state audit:", audit)
    extra = ["RESEARCH_STATE.json", str(v3_path), "MANIFEST.in", *new_artifacts]
    run("git", "add", "--", *sorted(set([*EXPECTED, *extra])))
    run("git", "rm", "--", *HELPERS)
    actual = set(run("git", "diff", "--cached", "--name-only").splitlines())
    expected_changes = set(EXPECTED) | set(extra) | set(HELPERS)
    if not actual <= expected_changes or not set(EXPECTED) <= actual:
        raise RuntimeError("Unexpected staged changes; refusing commit")
    run("git", "diff", "--cached", "--check")
    run("git", "-c", "user.name=github-actions[bot]", "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com", "commit", "-m", "Integrate reviewed Olympus software repair; preserve frozen studies")
    remote = run("git", "ls-remote", "origin", "refs/heads/" + BRANCH).split()[0]
    if remote != bootstrap:
        raise RuntimeError("Review branch advanced concurrently; refusing push")
    run("git", "push", "origin", "HEAD:refs/heads/" + BRANCH)
    commit = run("git", "rev-parse", "HEAD")
    print(json.dumps({"integrated_commit": commit, "branch": BRANCH, "hosted_ci": "PENDING", "scientific_results": 0}))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"Reviewed repair committed to `{BRANCH}` at `{commit}`. Frozen artifacts preserved. Full CI remains pending. No scientific run or production release.\n")


if __name__ == "__main__":
    main()
