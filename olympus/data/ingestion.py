from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import socket
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit

from pydantic import Field

from olympus.core.schemas import StrictModel
from olympus.core.security import PermissionPolicy

_SAFE_CONTENT_TYPES = frozenset(
    {"application/json", "application/xml", "text/csv", "text/plain", "text/xml"}
)


@dataclass(frozen=True)
class _ResolvedEndpoint:
    family: int
    socktype: int
    proto: int
    sockaddr: Any
    ip: str


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection whose TCP peer is fixed before TLS starts."""

    def __init__(
        self,
        hostname: str,
        port: int,
        endpoint: _ResolvedEndpoint,
        *,
        timeout: float,
    ) -> None:
        context = ssl.create_default_context()
        super().__init__(hostname, port=port, timeout=timeout, context=context)
        self._endpoint = endpoint
        self._connect_timeout = timeout
        self._tls_context = context

    def connect(self) -> None:
        sock = socket.socket(
            self._endpoint.family,
            self._endpoint.socktype,
            self._endpoint.proto,
        )
        sock.settimeout(self._connect_timeout)
        try:
            # Connect directly to the already-validated sockaddr. This deliberately avoids
            # socket.create_connection(), which would perform a second hostname resolution.
            sock.connect(self._endpoint.sockaddr)
            self.sock = self._tls_context.wrap_socket(sock, server_hostname=self.host)
        except BaseException:
            sock.close()
            raise


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
        port = parsed.port or 443
        endpoints = self._resolve_public_endpoints(hostname, port)
        payload, charset = self._fetch_https(parsed, hostname, port, endpoints)
        text = payload.decode(charset)
        return self._normalize(url, text, chunk_size)

    def _fetch_https(
        self,
        parsed: SplitResult,
        hostname: str,
        port: int,
        endpoints: tuple[_ResolvedEndpoint, ...],
    ) -> tuple[bytes, str]:
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        headers = {
            "Accept": ", ".join(sorted(_SAFE_CONTENT_TYPES)),
            "User-Agent": "Olympus/0.1",
        }
        last_error: Exception | None = None
        for endpoint in endpoints:
            connection = _PinnedHTTPSConnection(hostname, port, endpoint, timeout=5.0)
            try:
                connection.request("GET", target, headers=headers)
                response = connection.getresponse()
                if 300 <= response.status < 400:
                    raise ValueError("HTTP redirects are disabled for ingestion")
                if response.status < 200 or response.status >= 300:
                    raise ValueError(f"HTTP ingestion request failed with status {response.status}")
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
                return payload, charset
            except (OSError, http.client.HTTPException) as exc:
                last_error = exc
            finally:
                connection.close()
        if last_error is not None:
            raise ValueError(f"unable to connect to ingestion host: {hostname}") from last_error
        raise ValueError(f"unable to connect to ingestion host: {hostname}")

    @staticmethod
    def _resolve_public_endpoints(hostname: str, port: int) -> tuple[_ResolvedEndpoint, ...]:
        try:
            addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"unable to resolve ingestion host: {hostname}") from exc
        if not addresses:
            raise ValueError(f"unable to resolve ingestion host: {hostname}")

        endpoints: list[_ResolvedEndpoint] = []
        for family, socktype, proto, _canonical_name, sockaddr in addresses:
            ip = ipaddress.ip_address(sockaddr[0])
            if not ip.is_global:
                raise PermissionError(f"ingestion host resolves to a non-public address: {ip}")
            endpoints.append(
                _ResolvedEndpoint(
                    family=family,
                    socktype=socktype,
                    proto=proto,
                    sockaddr=sockaddr,
                    ip=str(ip),
                )
            )
        return tuple(endpoints)

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
