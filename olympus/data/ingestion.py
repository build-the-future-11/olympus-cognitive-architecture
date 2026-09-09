from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import socket
import ssl
import stat
import urllib.request
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import Field

from olympus.core.schemas import StrictModel
from olympus.core.security import PermissionPolicy

_SAFE_CONTENT_TYPES = frozenset(
    {"application/json", "application/xml", "text/csv", "text/plain", "text/xml"}
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise ValueError("HTTP redirects are disabled for ingestion")


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self, host: str, *, address: str, context: ssl.SSLContext | None = None, **kwargs: Any
    ) -> None:
        self.tls_context = context or ssl.create_default_context()
        super().__init__(host, context=self.tls_context, **kwargs)
        self.address = address

    def connect(self) -> None:
        if getattr(self, "_tunnel_host", None):
            raise PermissionError("ingestion proxy tunnels are disabled")
        # Connect to the checked numeric address without a second DNS lookup.
        family = socket.AF_INET6 if ":" in self.address else socket.AF_INET
        transport = socket.socket(family, socket.SOCK_STREAM)
        try:
            transport.settimeout(self.timeout)
            transport.connect((self.address, self.port))
            self.sock = self.tls_context.wrap_socket(transport, server_hostname=self.host)
        except BaseException:
            transport.close()
            raise


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, address: str) -> None:
        self.tls_context = ssl.create_default_context()
        super().__init__(context=self.tls_context)
        self.address = address

    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        return self.do_open(
            partial(_PinnedHTTPSConnection, address=self.address), req, context=self.tls_context
        )


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
        max_file_bytes: int = 1_000_000,
    ) -> None:
        if max_response_bytes < 1 or max_file_bytes < 1:
            raise ValueError("ingestion byte limits must be positive")
        self.policy = policy or PermissionPolicy()
        self.max_response_bytes = max_response_bytes
        self.max_file_bytes = max_file_bytes

    def ingest_file(self, path: Path, chunk_size: int = 120) -> IngestedDocument:
        if not self.policy.allow_filesystem_read:
            raise PermissionError("filesystem ingestion is disabled")
        resolved = path.resolve(strict=True)
        allowed_root = self._enforce_roots(
            resolved,
            self.policy.allowed_read_roots,
            "read",
        )
        payload = self._read_regular_file(resolved, allowed_root)
        text = payload.decode("utf-8")
        return self._normalize(resolved.name, text, chunk_size)

    def _read_regular_file(self, path: Path, allowed_root: Path | None) -> bytes:
        """Open a root-relative path without following attacker-swapped symlinks."""

        no_follow = getattr(os, "O_NOFOLLOW", 0)
        close_on_exec = getattr(os, "O_CLOEXEC", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        open_flags = os.O_RDONLY | close_on_exec | no_follow
        directory_flags = open_flags | directory

        if allowed_root is None:
            file_descriptor = os.open(path, open_flags)
        else:
            anchor = allowed_root if path != allowed_root else allowed_root.parent
            relative = path.relative_to(anchor)
            current_descriptor = os.open(anchor, directory_flags)
            try:
                for component in relative.parts[:-1]:
                    next_descriptor = os.open(
                        component,
                        directory_flags,
                        dir_fd=current_descriptor,
                    )
                    os.close(current_descriptor)
                    current_descriptor = next_descriptor
                file_descriptor = os.open(
                    relative.parts[-1],
                    open_flags,
                    dir_fd=current_descriptor,
                )
            finally:
                os.close(current_descriptor)

        try:
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("ingestion source must be a regular file")
            if metadata.st_size > self.max_file_bytes:
                raise ValueError("file exceeds the configured byte limit")
            with os.fdopen(file_descriptor, "rb", closefd=False) as handle:
                payload = handle.read(self.max_file_bytes + 1)
            if len(payload) > self.max_file_bytes:
                raise ValueError("file exceeds the configured byte limit")
            return payload
        finally:
            os.close(file_descriptor)

    def ingest_http(self, url: str, chunk_size: int = 120) -> IngestedDocument:
        if not self.policy.allow_network:
            raise PermissionError("network ingestion is disabled")
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower().rstrip(".")
        if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
            raise ValueError("ingestion URLs must be credential-free HTTPS URLs")
        if parsed.port not in {None, 443}:
            raise ValueError("ingestion HTTPS URLs must use port 443")
        allowed = {domain.lower().rstrip(".") for domain in self.policy.allow_http_domains}
        if hostname not in allowed:
            raise PermissionError(f"host is not allowlisted: {hostname}")
        address = self._reject_non_public_addresses(hostname, parsed.port or 443)

        request = urllib.request.Request(
            url,
            headers={"Accept": ", ".join(sorted(_SAFE_CONTENT_TYPES)), "User-Agent": "Olympus/0.1"},
        )
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect(), _PinnedHTTPSHandler(address)
        )
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
    def _reject_non_public_addresses(hostname: str, port: int) -> str:
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
        return str(addresses[0][4][0])

    def _normalize(self, identifier: str, text: str, chunk_size: int) -> IngestedDocument:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
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
        if not self.policy.allow_filesystem_write:
            raise PermissionError("filesystem writes are disabled")
        resolved = output_path.resolve()
        root = self._enforce_roots(resolved, self.policy.allowed_write_roots, "write")
        anchor = root if root is not None else Path(resolved.anchor)
        relative = resolved.relative_to(anchor)
        if not relative.parts:
            raise ValueError("shard output must be a file below the allowed root")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(anchor, flags)
        temporary = f".shard-{uuid4().hex}.tmp"
        created = False
        try:
            for component in relative.parts[:-1]:
                try:
                    os.mkdir(component, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            file_descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=descriptor,
            )
            created = True
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump([document.model_dump(mode="json") for document in documents],
                          handle, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, relative.parts[-1],
                       src_dir_fd=descriptor, dst_dir_fd=descriptor)
            created = False
            os.fsync(descriptor)
        finally:
            if created:
                os.unlink(temporary, dir_fd=descriptor)
            os.close(descriptor)

    @staticmethod
    def _enforce_roots(
        path: Path,
        roots: tuple[str, ...],
        operation: str,
    ) -> Path | None:
        if not roots:
            return None
        allowed = [Path(root).resolve() for root in roots]
        matching = [root for root in allowed if path == root or path.is_relative_to(root)]
        if not matching:
            raise PermissionError(f"filesystem {operation} is outside the allowed roots")
        return max(matching, key=lambda root: len(root.parts))
