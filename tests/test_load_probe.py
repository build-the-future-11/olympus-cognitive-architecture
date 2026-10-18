from __future__ import annotations

import asyncio

import pytest

from olympus.api import app
from olympus.foundry.load_probe import probe_asgi_app


def test_load_probe_records_percentiles_and_throughput() -> None:
    report = asyncio.run(probe_asgi_app(app, requests=20, concurrency=4))
    assert report.successful_requests == 20
    assert report.failed_requests == 0
    assert report.status_counts == {"200": 20}
    assert 0 <= report.p50_latency_ms <= report.p95_latency_ms <= report.p99_latency_ms
    assert report.throughput_requests_per_second > 0
    assert report.qualification == "BOUNDED_ENGINEERING_SMOKE_NOT_PRODUCTION_EVIDENCE"


@pytest.mark.parametrize(("requests", "concurrency"), [(0, 1), (2, 0), (2, 3)])
def test_load_probe_rejects_invalid_workload(requests: int, concurrency: int) -> None:
    with pytest.raises(ValueError):
        asyncio.run(probe_asgi_app(app, requests=requests, concurrency=concurrency))
