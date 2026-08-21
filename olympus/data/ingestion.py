from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field

from olympus.core.schemas import StrictModel
from olympus.core.security import PermissionPolicy

_SAFE_CONTENT_TYPES = frozenset(
    {"application/json", "application/xml", "text/csv", "text/plain", "text/xml"}
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise ValueError("HTTP redirects are disabled for ingestion")


class IngestedDocument(StrictModel):
    identifier: str
    text: str
    language: str = "unknown"
    chunks: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestionPipeline:
    def __init__(
        self,
        policy: PermissionPolicy | None = None,
        *,
        max_response_bytes: int = 1_000_000,
    ) -> None:
        if max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        self.policy = policy or PermissionPolicy()
        self.max_response_bytes = max_response_bytes

    def ingest_file(self, path: Path, chunk_size: int = 120) -> IngestedDocument:
        text = path.read_text(encoding="utf-8")
        return self._normalize(path.name, text, chunk_size)

    def ingest_http(self, url: str, chunk_size: int = 120) -> IngestedDocument:
        if not self.policy.allow_network:
            raise PermissionError("network ingestion is disabled")
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
            raise ValueError("ingestion URLs must be credential-free HTTPS URLs")
        allowed = {domain.lower().rstrip(".") for domain in self.policy.allow_http_domains}
        if hostname not in allowed:
            raise PermissionError(f"host is not allowlisted: {hostname}")
        self._reject_non_public_addresses(hostname, parsed.port or 443)

        request = urllib.request.Request(
            url,
            headers={"Accept": ", ".join(sorted(_SAFE_CONTENT_TYPES)), "User-Agent": "Olympus/0.1"},
        )
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(request, timeout=5) as response:
            content_type = response.headers.get_content_type().lower()
            if content_type not in _SAFE_CONTENT_TYPES:
                raise ValueError(f"unsupported HTTP content type: {content_type}")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > self.max_response_bytes:
                raise ValueError("HTTP response exceeds the configured byte limit")
            payload = response.read(self.max_response_bytes + 1)
            if len(payload) > self.max_response_bytes:
                raise ValueError("HTTP response exceeds the configured byte limit")
            charset = response.headers.get_content_charset("utf-8")
            text = payload.decode(charset)
        return self._normalize(url, text, chunk_size)

    @staticmethod
    def _reject_non_public_addresses(hostname: str, port: int) -> None:
        try:
            addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"unable to resolve ingestion host: {hostname}") from exc
        if not addresses:
            raise ValueError(f"unable to resolve ingestion host: {hostname}")
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise PermissionError(f"ingestion host resolves to a non-public address: {ip}")

    def _normalize(self, identifier: str, text: str, chunk_size: int) -> IngestedDocument:
        normalized = " ".join(text.split())
        chunks = [
            normalized[index : index + chunk_size]
            for index in range(0, len(normalized), chunk_size)
        ]
        return IngestedDocument(
            identifier=identifier,
            text=normalized,
            language="en" if normalized.isascii() else "mixed",
            chunks=chunks,
            metadata={"sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest()},
        )

    def shard(self, documents: list[IngestedDocument], output_path: Path) -> None:
        output_path.write_text(
            json.dumps([document.model_dump(mode="json") for document in documents], indent=2),
            encoding="utf-8",
        )
