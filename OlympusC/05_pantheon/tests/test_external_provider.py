import json

from pantheon.external import providers


def test_ollama_chat_provider_tool_roundtrip(monkeypatch):
    responses = [
        {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "shell", "arguments": json.dumps({"command": "pwd"})}}]}}]},
        {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
    ]
    seen = []

    def fake_post(url, headers, payload, timeout=180):
        seen.append((url, headers, json.loads(json.dumps(payload)), timeout))
        return responses.pop(0)

    monkeypatch.setattr(providers, "_post_json", fake_post)
    provider = providers.OpenAIChatProvider("qwen3:0.6b", "http://127.0.0.1:11434/v1")
    tools = [{"type": "function", "name": "shell", "description": "run", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}]
    first = provider.turn("solve", tools, "system")
    assert first.tool_calls == [{"id": "call-1", "name": "shell", "arguments": {"command": "pwd"}}]
    second = provider.turn([{"type": "function_call_output", "call_id": "call-1", "output": "/tmp"}], tools, "system")
    assert second.text == "done"
    assert seen[1][2]["messages"][-1] == {"role": "tool", "tool_call_id": "call-1", "content": "/tmp"}
    assert seen[0][2]["max_tokens"] == 768


def test_chat_provider_completion_cap_is_configurable(monkeypatch):
    seen = {}

    def fake_post(url, headers, payload, timeout=180):
        seen.update(payload)
        return {"choices": [{"message": {"role": "assistant", "content": "done"}}]}

    monkeypatch.setenv("PANTHEON_MAX_COMPLETION_TOKENS", "384")
    monkeypatch.setattr(providers, "_post_json", fake_post)
    provider = providers.OpenAIChatProvider("qwen3:0.6b", "http://127.0.0.1:11434/v1")
    provider.turn("solve", [], "system")
    assert seen["max_tokens"] == 384


def test_make_ollama_provider_needs_no_secret():
    provider = providers.make_provider("ollama", "gemma3:1b")
    assert isinstance(provider, providers.OpenAIChatProvider)


def test_ollama_nested_tool_arguments_are_normalized(monkeypatch):
    nested = {"function": "shell", "parameters": {"command": "pwd", "timeout_sec": "30"}}

    def fake_post(url, headers, payload, timeout=180):
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"id": "nested-1", "type": "function", "function": {"name": "shell", "arguments": json.dumps(nested)}}]}}]}

    monkeypatch.setattr(providers, "_post_json", fake_post)
    provider = providers.OpenAIChatProvider("llama3.2:1b", "http://127.0.0.1:11434/v1")
    turn = provider.turn("solve", [], "system")
    assert turn.tool_calls[0]["arguments"] == {"command": "pwd", "timeout_sec": "30"}
