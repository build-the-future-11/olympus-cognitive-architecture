from __future__ import annotations
import json, os, platform, subprocess, time
from functools import lru_cache
from pathlib import Path
from typing import Any
from .providers import AnthropicProvider, ProviderError, make_provider

TOOLS=[{
  "type":"function","name":"shell",
  "description":"Run a shell command inside the isolated research workspace. Use it to inspect files, install dependencies, execute experiments, and extract results.",
  "parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout_sec":{"type":"integer","minimum":1,"maximum":900}},"required":["command"],"additionalProperties":False}
}]

SYSTEM="""You are an independent computational-reproducibility agent. Work only inside the provided workspace. The shell starts in the workspace root; use relative paths such as `.` and do not assume `/workspace` exists. Reproduce the requested result from the available code/data. Do not guess. Begin with the shell command `pwd; find . -maxdepth 2 -type f | head -100`. Use python3, not python. You may inspect files and execute included code, but network access is unavailable. Do not seek or infer hidden canonical answers. When finished, use the shell tool to write report.json in the workspace as a JSON object mapping each exact question string to your answer. If a result cannot be obtained, use null for that question and explain the failure in agent_notes.md. Do not merely print the report in chat. Never fabricate numerical results."""


@lru_cache(maxsize=1)
def sandbox_available() -> bool:
    executable = Path("/usr/bin/sandbox-exec")
    if platform.system() != "Darwin" or not executable.exists():
        return False
    try:
        probe = subprocess.run(
            [str(executable), "-p", "(version 1) (allow default)", "/usr/bin/true"],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def _run_shell(workspace: Path, command: str, timeout: int) -> str:
    start=time.time()
    workspace = workspace.resolve()
    home = workspace / ".pantheon_home"
    temp = workspace / ".pantheon_tmp"
    home.mkdir(exist_ok=True)
    temp.mkdir(exist_ok=True)
    env = {
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C.UTF-8",
        "PANTHEON_WORKSPACE": str(workspace),
        "HOME": str(home),
        "TMPDIR": str(temp),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if not sandbox_available():
        return "SANDBOX_UNAVAILABLE: external agent command refused"
    # The actual workspace lives under /private/tmp. Canonical answers and the
    # parent repository are on /Volumes/PRO-BLADE and are denied. Network is
    # denied so an OOD answer cannot be retrieved from the public benchmark.
    profile = '''(version 1)
(allow default)
(deny network*)
(deny file-read* (subpath "/Users/ryan") (subpath "/Volumes/PRO-BLADE"))
(deny file-write* (subpath "/Users/ryan") (subpath "/Volumes/PRO-BLADE"))
'''
    try:
        p=subprocess.run(
            ["/usr/bin/sandbox-exec", "-p", profile, "/bin/sh", "-lc", command],
            cwd=workspace, text=True, capture_output=True, timeout=timeout, env=env
        )
        out=f"exit_code={p.returncode}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
    except subprocess.TimeoutExpired as e:
        out=f"TIMEOUT after {timeout}s\nstdout:\n{e.stdout or ''}\nstderr:\n{e.stderr or ''}"
    return f"runtime_sec={time.time()-start:.3f}\n" + out[-20000:]


def run_agent(provider_name: str, model: str, workspace: Path, task_prompt: str, questions: list[str], *, max_turns: int=40, tool_timeout: int=600) -> dict[str,Any]:
    provider=make_provider(provider_name,model)
    user_prompt=("First inspect the current directory with the shell tool; it is the workspace root.\n\n"+task_prompt+"\n\nQuestions to answer exactly as JSON keys:\n"+"\n".join(f"- {q}" for q in questions))
    next_input=user_prompt; transcript=[]; provider_error=None; no_tool_reminders=0
    for turn_idx in range(max_turns):
        try:
            turn=provider.turn(next_input,TOOLS,SYSTEM)
        except ProviderError as exc:
            provider_error=f"{type(exc).__name__}: {exc}"
            transcript.append({"turn":turn_idx,"provider_error":provider_error})
            break
        transcript.append({"turn":turn_idx,"text":turn.text,"tool_calls":turn.tool_calls})
        if not turn.tool_calls:
            if not (workspace / "report.json").exists() and no_tool_reminders < 1 and turn_idx + 1 < max_turns:
                no_tool_reminders += 1
                next_input = "No report.json exists yet. Do not answer in chat. Use the shell tool now: inspect the workspace, reproduce the result, and write report.json with every exact question key. Use null plus agent_notes.md for any unresolved result."
                continue
            break
        results=[]
        for call in turn.tool_calls:
            if call.get("name")!="shell":
                output="Unsupported tool"
            else:
                args=call.get("arguments") or {}; output=_run_shell(workspace,str(args.get("command","")),min(int(args.get("timeout_sec",tool_timeout)),900))
            results.append({"call_id":call["id"],"output":output})
        if isinstance(provider,AnthropicProvider):
            provider.append_tool_results(results); next_input="Continue the reproduction."
        else:
            next_input=[{"type":"function_call_output","call_id":r["call_id"],"output":r["output"]} for r in results]
    report=workspace/'report.json'; agent_report_exists=report.exists(); report_origin="agent"
    answers={}
    if agent_report_exists:
        try:
            obj=json.loads(report.read_text()); answers=obj if isinstance(obj,dict) else {}
        except json.JSONDecodeError:
            pass
    normalized = {question: answers.get(question) for question in questions}
    if not agent_report_exists or answers != normalized:
        if agent_report_exists:
            (workspace / "report.agent.json").write_text(report.read_text())
            report_origin = "agent_plus_harness_key_normalization"
        else:
            report_origin = "harness_failure_normalization"
        report.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n")
        notes = workspace / "agent_notes.md"
        with notes.open("a", encoding="utf-8") as handle:
            handle.write("\nHarness note: the agent did not produce a valid complete exact-key report; unresolved keys were normalized to null.\n")
    return {"answers":normalized,"transcript":transcript,"report_exists":report.exists(),"agent_report_exists":agent_report_exists,"report_origin":report_origin,"turns":len(transcript),"provider_error":provider_error}
