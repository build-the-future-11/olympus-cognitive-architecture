from __future__ import annotations
import json, os, urllib.request, urllib.error
from dataclasses import dataclass
from typing import Any

class ProviderError(RuntimeError): pass


def _post_json(url: str, headers: dict[str,str], payload: dict[str,Any], timeout: int=180) -> dict[str,Any]:
    data=json.dumps(payload).encode()
    req=urllib.request.Request(url, data=data, headers={"Content-Type":"application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body=e.read().decode(errors='replace')
        raise ProviderError(f"HTTP {e.code}: {body[:4000]}") from e

@dataclass
class Turn:
    text: str
    tool_calls: list[dict[str,Any]]
    raw: dict[str,Any]

class OpenAIResponsesProvider:
    def __init__(self, model: str, api_key: str|None=None):
        self.model=model; self.api_key=api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key: raise ProviderError("OPENAI_API_KEY is not set")
        self.url="https://api.openai.com/v1/responses"
        self.previous_response_id: str|None=None

    def turn(self, input_items: Any, tools: list[dict[str,Any]], instructions: str) -> Turn:
        payload={"model":self.model,"instructions":instructions,"input":input_items,"tools":tools,"reasoning":{"effort":"high"}}
        if self.previous_response_id: payload["previous_response_id"]=self.previous_response_id
        raw=_post_json(self.url,{"Authorization":f"Bearer {self.api_key}"},payload)
        self.previous_response_id=raw.get("id")
        text=[]; calls=[]
        for item in raw.get("output",[]):
            if item.get("type")=="message":
                for c in item.get("content",[]):
                    if c.get("type")=="output_text": text.append(c.get("text",""))
            elif item.get("type")=="function_call":
                args=item.get("arguments",{})
                if isinstance(args,str):
                    try: args=json.loads(args)
                    except json.JSONDecodeError: args={"raw":args}
                calls.append({"id":item.get("call_id") or item.get("id"),"name":item.get("name"),"arguments":args})
        return Turn("\n".join(text),calls,raw)

class AnthropicProvider:
    def __init__(self, model: str, api_key: str|None=None):
        self.model=model; self.api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key: raise ProviderError("ANTHROPIC_API_KEY is not set")
        self.url="https://api.anthropic.com/v1/messages"; self.messages=[]

    def turn(self, input_items: Any, tools: list[dict[str,Any]], instructions: str) -> Turn:
        if isinstance(input_items,str): self.messages.append({"role":"user","content":input_items})
        elif isinstance(input_items,list): self.messages.append({"role":"user","content":input_items})
        atools=[{"name":t["name"],"description":t.get("description",""),"input_schema":t["parameters"]} for t in tools if t.get("type")=="function"]
        raw=_post_json(self.url,{"x-api-key":self.api_key,"anthropic-version":"2023-06-01"},{"model":self.model,"max_tokens":8192,"system":instructions,"messages":self.messages,"tools":atools})
        self.messages.append({"role":"assistant","content":raw.get("content",[])})
        text=[]; calls=[]
        for item in raw.get("content",[]):
            if item.get("type")=="text": text.append(item.get("text",""))
            elif item.get("type")=="tool_use": calls.append({"id":item["id"],"name":item["name"],"arguments":item.get("input",{})})
        return Turn("\n".join(text),calls,raw)

    def append_tool_results(self, results: list[dict[str,Any]]) -> None:
        content=[{"type":"tool_result","tool_use_id":r["call_id"],"content":r["output"]} for r in results]
        self.messages.append({"role":"user","content":content})


class OpenAIChatProvider:
    """Stateful OpenAI-compatible chat provider, including local Ollama."""

    def __init__(self, model: str, base_url: str, api_key: str | None = None):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.messages: list[dict[str, Any]] = []

    def turn(self, input_items: Any, tools: list[dict[str,Any]], instructions: str) -> Turn:
        if not self.messages:
            self.messages.append({"role": "system", "content": instructions})
        if isinstance(input_items, str):
            self.messages.append({"role": "user", "content": input_items})
        elif isinstance(input_items, list):
            for item in input_items:
                if item.get("type") != "function_call_output":
                    raise ProviderError(f"unsupported chat input item: {item.get('type')}")
                self.messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        chat_tools = [
            {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t["parameters"]}}
            for t in tools if t.get("type") == "function"
        ]
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        raw = _post_json(self.url, headers, {
            "model": self.model,
            "messages": self.messages,
            "tools": chat_tools,
            "temperature": 0,
            "max_tokens": int(os.getenv("PANTHEON_MAX_COMPLETION_TOKENS", "768")),
            "stream": False,
        }, timeout=int(os.getenv("PANTHEON_PROVIDER_TIMEOUT", "900")))
        choices = raw.get("choices") or []
        if not choices:
            raise ProviderError(f"chat provider returned no choices: {str(raw)[:1000]}")
        message = choices[0].get("message") or {}
        self.messages.append(message)
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            # Some Ollama tool templates wrap the declared arguments in a
            # second function/parameters envelope. Normalize that provider
            # representation before handing it to the agent loop.
            if isinstance(arguments, dict) and isinstance(arguments.get("parameters"), dict):
                arguments = arguments["parameters"]
            calls.append({"id": call.get("id"), "name": function.get("name"), "arguments": arguments})
        content = message.get("content") or ""
        return Turn(content if isinstance(content, str) else json.dumps(content), calls, raw)


def make_provider(name: str, model: str):
    n=name.lower()
    if n=="openai": return OpenAIResponsesProvider(model)
    if n=="anthropic": return AnthropicProvider(model)
    if n=="ollama":
        return OpenAIChatProvider(model, os.getenv("PANTHEON_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"))
    if n=="openai_chat":
        base=os.getenv("PANTHEON_OPENAI_CHAT_BASE_URL")
        if not base: raise ProviderError("PANTHEON_OPENAI_CHAT_BASE_URL is not set")
        key=os.getenv("PANTHEON_OPENAI_CHAT_API_KEY")
        if not key and not base.startswith(("http://127.0.0.1", "http://localhost")):
            raise ProviderError("PANTHEON_OPENAI_CHAT_API_KEY is not set for remote endpoint")
        return OpenAIChatProvider(model, base, key)
    raise ProviderError(f"unsupported provider: {name}")
