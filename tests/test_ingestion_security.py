from __future__ import annotations

import socket
from email.message import Message
from unittest.mock import MagicMock, patch

import pytest

from olympus.core.security import PermissionPolicy
from olympus.data.ingestion import IngestionPipeline, _PinnedHTTPSConnection, _ResolvedEndpoint


class _Response:
    def __init__(
        self,
        payload: bytes,
        content_type: str = "text/plain",
        *,
        status: int = 200,
    ) -> None:
        self.payload = payload
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Content-Length"] = str(len(payload))

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


def _policy() -> PermissionPolicy:
    return PermissionPolicy(allow_network=True, allow_http_domains=("data.example.org",))


def _public_address(ip: str = "93.184.216.34") -> list[tuple[object, ...]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]


def _connection(response: _Response) -> MagicMock:
    connection = MagicMock()
    connection.getresponse.return_value = response
    return connection


def test_network_is_denied_by_default() -> None:
    with pytest.raises(PermissionError, match="disabled"):
        IngestionPipeline().ingest_http("https://data.example.org/file.txt")


@pytest.mark.parametrize(
    "url, error",
    [
        ("http://data.example.org/file.txt", ValueError),
        ("https://other.example.org/file.txt", PermissionError),
        ("https://user:secret@data.example.org/file.txt", ValueError),
    ],
)
def test_url_policy_rejects_unsafe_targets(url: str, error: type[Exception]) -> None:
    with pytest.raises(error):
        IngestionPipeline(_policy()).ingest_http(url)


def test_private_resolution_is_rejected() -> None:
    address = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
    with patch("socket.getaddrinfo", return_value=address):
        with pytest.raises(PermissionError, match="non-public"):
            IngestionPipeline(_policy()).ingest_http("https://data.example.org/file.txt")


def test_response_is_content_typed_and_bounded() -> None:
    address = _public_address()
    pipeline = IngestionPipeline(_policy(), max_response_bytes=12)
    with (
        patch("socket.getaddrinfo", return_value=address),
        patch(
            "olympus.data.ingestion._PinnedHTTPSConnection",
            return_value=_connection(_Response(b"hello world")),
        ),
    ):
        document = pipeline.ingest_http("https://data.example.org/file.txt")
    assert document.text == "hello world"

    with (
        patch("socket.getaddrinfo", return_value=address),
        patch(
            "olympus.data.ingestion._PinnedHTTPSConnection",
            return_value=_connection(_Response(b"not plain", "application/octet-stream")),
        ),
    ):
        with pytest.raises(ValueError, match="content type"):
            pipeline.ingest_http("https://data.example.org/file.bin")

    with (
        patch("socket.getaddrinfo", return_value=address),
        patch(
            "olympus.data.ingestion._PinnedHTTPSConnection",
            return_value=_connection(_Response(b"this payload is much too large")),
        ),
    ):
        with pytest.raises(ValueError, match="byte limit"):
            pipeline.ingest_http("https://data.example.org/file.txt")


def test_redirects_are_rejected_without_following_location() -> None:
    with (
        patch("socket.getaddrinfo", return_value=_public_address()),
        patch(
            "olympus.data.ingestion._PinnedHTTPSConnection",
            return_value=_connection(_Response(b"", status=302)),
        ),
    ):
        with pytest.raises(ValueError, match="redirects are disabled"):
            IngestionPipeline(_policy()).ingest_http("https://data.example.org/file.txt")


def test_connection_retries_only_prevalidated_endpoints() -> None:
    addresses = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.35", 443)),
    ]
    first = MagicMock()
    first.request.side_effect = OSError("first endpoint unavailable")
    second = _connection(_Response(b"fallback works"))
    with (
        patch("socket.getaddrinfo", return_value=addresses) as resolver,
        patch(
            "olympus.data.ingestion._PinnedHTTPSConnection",
            side_effect=[first, second],
        ) as connection_type,
    ):
        document = IngestionPipeline(_policy()).ingest_http(
            "https://data.example.org/path?q=1"
        )

    assert document.text == "fallback works"
    resolver.assert_called_once_with("data.example.org", 443, type=socket.SOCK_STREAM)
    assert connection_type.call_count == 2
    first_endpoint = connection_type.call_args_list[0].args[2]
    second_endpoint = connection_type.call_args_list[1].args[2]
    assert first_endpoint.ip == "93.184.216.34"
    assert second_endpoint.ip == "93.184.216.35"
    second.request.assert_called_once()
    assert second.request.call_args.args[:2] == ("GET", "/path?q=1")


def test_pinned_connection_does_not_reresolve_hostname() -> None:
    endpoint = _ResolvedEndpoint(
        family=socket.AF_INET,
        socktype=socket.SOCK_STREAM,
        proto=6,
        sockaddr=("93.184.216.34", 443),
        ip="93.184.216.34",
    )
    connection = _PinnedHTTPSConnection(
        "data.example.org",
        443,
        endpoint,
        timeout=5.0,
    )
    raw_socket = MagicMock()
    tls_socket = MagicMock()
    tls_context = MagicMock()
    tls_context.wrap_socket.return_value = tls_socket
    connection._tls_context = tls_context

    with (
        patch("olympus.data.ingestion.socket.socket", return_value=raw_socket) as socket_type,
        patch(
            "olympus.data.ingestion.socket.getaddrinfo",
            side_effect=AssertionError("hostname must not be resolved during connect"),
        ),
    ):
        connection.connect()

    socket_type.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM, 6)
    raw_socket.settimeout.assert_called_once_with(5.0)
    raw_socket.connect.assert_called_once_with(("93.184.216.34", 443))
    tls_context.wrap_socket.assert_called_once_with(
        raw_socket,
        server_hostname="data.example.org",
    )
    assert connection.sock is tls_socket
