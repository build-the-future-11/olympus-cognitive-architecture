import importlib.util
from pathlib import Path

import pandas as pd


def _module():
    path = Path(__file__).parent / "scientific_readiness_gate.py"
    spec = importlib.util.spec_from_file_location("scientific_readiness_gate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_complete_artifact_can_fail_capability_floor():
    module = _module()
    questions = pd.DataFrame({"agent_a_answered": [0] * 20, "agent_b_answered": [0] * 20})
    tasks = pd.DataFrame({"agent_a_submitted_report": [False] * 17, "agent_b_submitted_report": [False] * 17})
    result = module.assess(questions, tasks, {"verdict": "PUBLICATION-READY"})
    assert result["artifact_complete"] is True
    assert result["capability_floor_passed"] is False
    assert result["verdict"] == "NEGATIVE_RESULT_ARTIFACT_COMPLETE_CAPABILITY_FLOOR_FAILED"


def test_capability_floor_requires_joint_answers_and_authored_reports():
    module = _module()
    questions = pd.DataFrame({"agent_a_answered": [1] * 5, "agent_b_answered": [1] * 5})
    tasks = pd.DataFrame({"agent_a_submitted_report": [True] * 5, "agent_b_submitted_report": [True] * 5})
    result = module.assess(questions, tasks, {"verdict": "PUBLICATION-READY"})
    assert result["capability_floor_passed"] is True
    assert result["verdict"] == "ARTIFACT_COMPLETE_CAPABILITY_FLOOR_PASSED"
