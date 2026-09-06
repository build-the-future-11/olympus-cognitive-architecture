#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def provider_issue(provider: str, model: str) -> str | None:
    provider = provider.lower()
    if provider == "openai":
        return None if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY is not set"
    if provider == "anthropic":
        return None if os.getenv("ANTHROPIC_API_KEY") else "ANTHROPIC_API_KEY is not set"
    if provider == "openai_chat":
        base = os.getenv("PANTHEON_OPENAI_CHAT_BASE_URL")
        if not base:
            return "PANTHEON_OPENAI_CHAT_BASE_URL is not set"
        if not base.startswith(("http://127.0.0.1", "http://localhost")) and not os.getenv("PANTHEON_OPENAI_CHAT_API_KEY"):
            return "PANTHEON_OPENAI_CHAT_API_KEY is not set for remote endpoint"
        return None
    if provider == "ollama":
        base = os.getenv("PANTHEON_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        tags_url = base.removesuffix("/v1") + "/api/tags"
        try:
            with urllib.request.urlopen(tags_url, timeout=3) as response:
                names = {item["name"] for item in json.load(response).get("models", [])}
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            return f"Ollama is unavailable at {tags_url}: {exc}"
        return None if model in names else f"Ollama model is not installed: {model}"
    return f"unsupported provider: {provider}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="external/corebench/manifest.jsonl")
    parser.add_argument("--agent-a-provider", default=os.getenv("PANTHEON_AGENT_A_PROVIDER", "openai"))
    parser.add_argument("--agent-a-model", default=os.getenv("PANTHEON_AGENT_A_MODEL", "gpt-5.6-terra"))
    parser.add_argument("--agent-b-provider", default=os.getenv("PANTHEON_AGENT_B_PROVIDER", "anthropic"))
    parser.add_argument("--agent-b-model", default=os.getenv("PANTHEON_AGENT_B_MODEL", "claude-sonnet-4-5"))
    args = parser.parse_args()
    issues: list[str] = []
    manifest = Path(args.manifest)
    if not manifest.exists():
        issues.append(f"missing manifest: {manifest}")
    else:
        tasks = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if not tasks:
            issues.append("manifest has zero tasks")
        for task in tasks:
            source = Path(task["source"])
            canonical = Path(task["canonical_answers"])
            if not source.exists():
                issues.append(f"missing source {source}")
            if not canonical.exists():
                issues.append(f"missing canonical {canonical}")
            if source.exists() and task.get("source_sha256"):
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if digest != task["source_sha256"]:
                    issues.append(f"source hash mismatch {source}")
    identity_a = (args.agent_a_provider.lower(), args.agent_a_model)
    identity_b = (args.agent_b_provider.lower(), args.agent_b_model)
    if identity_a == identity_b:
        issues.append("agent identities must differ")
    for provider, model in (identity_a, identity_b):
        issue = provider_issue(provider, model)
        if issue and issue not in issues:
            issues.append(issue)
    if issues:
        print("EXTERNAL READY CHECK: BLOCKED")
        for issue in issues:
            print("-", issue)
        return 2
    print("EXTERNAL READY CHECK: PASS")
    print(f"- agent A: {identity_a[0]}/{identity_a[1]}")
    print(f"- agent B: {identity_b[0]}/{identity_b[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
