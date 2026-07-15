from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PermissionPolicy:
    allow_network: bool = False
    allow_shell: bool = False
    allow_filesystem_write: bool = True
    allow_http_domains: tuple[str, ...] = ()


class PromptInjectionGuard:
    suspicious_tokens = ("ignore previous", "reveal system prompt", "exfiltrate", "<script")

    def inspect(self, content: str) -> bool:
        lowered = content.lower()
        return not any(token in lowered for token in self.suspicious_tokens)
