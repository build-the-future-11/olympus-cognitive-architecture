import json
from pathlib import Path

import pytest
from pantheon.external import agent_loop
from pantheon.external.providers import Turn

class FakeProvider:
    def __init__(self): self.i=0
    def turn(self,input_items,tools,instructions):
        self.i+=1
        if self.i==1:
            cmd="python3 -c 'import json; json.dump({\"q\": 7}, open(\"report.json\",\"w\"))'"
            return Turn('',[{'id':'c1','name':'shell','arguments':{'command':cmd,'timeout_sec':10}}],{})
        return Turn('done',[],{})

def test_agent_loop_creates_report(monkeypatch,tmp_path):
    monkeypatch.setattr(agent_loop,'make_provider',lambda name,model: FakeProvider())
    def fake_shell(workspace, command, timeout):
        (workspace / 'report.json').write_text('{"q": 7}')
        return 'exit_code=0'
    monkeypatch.setattr(agent_loop,'_run_shell',fake_shell)
    r=agent_loop.run_agent('fake','fake-model',tmp_path,'compute',['q'],max_turns=3)
    assert r['report_exists']
    assert r['answers']=={'q':7}
    assert r['turns']==2
    assert r['agent_report_exists']
    assert r['report_origin']=='agent'


class NonCompliantProvider:
    def __init__(self): self.inputs=[]
    def turn(self,input_items,tools,instructions):
        self.inputs.append(input_items)
        return Turn('I could not complete it.',[],{})


def test_agent_loop_normalizes_missing_report_to_explicit_nulls(monkeypatch,tmp_path):
    provider=NonCompliantProvider()
    monkeypatch.setattr(agent_loop,'make_provider',lambda name,model: provider)
    r=agent_loop.run_agent('fake','fake-model',tmp_path,'compute',['exact q'],max_turns=3)
    assert r['answers']=={'exact q':None}
    assert r['report_exists']
    assert not r['agent_report_exists']
    assert r['report_origin']=='harness_failure_normalization'
    assert r['turns']==2
    assert 'No report.json exists yet' in provider.inputs[1]
    assert json.loads((tmp_path/'report.json').read_text())=={'exact q':None}


@pytest.mark.skipif(not agent_loop.sandbox_available(), reason="macOS sandbox unavailable")
def test_shell_cannot_read_parent_repository(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repository_readme = Path(__file__).resolve().parents[1] / "README.md"
    result = agent_loop._run_shell(workspace, f"cat {repository_readme}", 10)
    assert "Pantheon" not in result
    assert "Operation not permitted" in result or "deny" in result.lower()


@pytest.mark.skipif(not agent_loop.sandbox_available(), reason="macOS sandbox unavailable")
def test_shell_can_execute_and_write_inside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = agent_loop._run_shell(workspace, "printf safe > result.txt && cat result.txt", 10)
    assert "exit_code=0" in result
    assert (workspace / "result.txt").read_text() == "safe"
