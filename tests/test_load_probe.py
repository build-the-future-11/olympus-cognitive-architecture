from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI

from olympus.api import app
from olympus.foundry.load_probe import _percentile, probe_asgi_app


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


def test_load_probe_counts_application_exceptions_as_failures() -> None:
    failing = FastAPI()

    @failing.get("/fail")
    def fail() -> None:
        raise RuntimeError("injected failure")

    report = asyncio.run(probe_asgi_app(failing, endpoint="/fail", requests=5, concurrency=2))
    assert report.successful_requests == 0
    assert report.failed_requests == 5
    assert report.error_rate == 1
    assert report.status_counts == {"error:RuntimeError": 5}


def test_percentile_uses_nearest_rank_without_underestimating_tail() -> None:
    assert _percentile([1, 2, 3, 4, 100], 0.95) == 100
