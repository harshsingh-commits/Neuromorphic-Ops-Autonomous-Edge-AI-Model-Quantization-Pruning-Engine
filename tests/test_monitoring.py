from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.monitoring import RuntimeMonitor


# ============================================================
# REQUEST MONITORING
# ============================================================

def test_record_successful_request() -> None:
    monitor = RuntimeMonitor()

    monitor.record_request(
        method="GET",
        path="/health",
        status_code=200,
        latency_ms=10.0,
    )

    snapshot = monitor.snapshot()

    assert snapshot["requests"]["total"] == 1
    assert snapshot["requests"]["successful"] == 1
    assert snapshot["requests"]["failed"] == 0
    assert snapshot["requests"]["success_rate_percent"] == 100.0
    assert snapshot["requests"]["average_latency_ms"] == 10.0
    assert snapshot["requests"]["max_latency_ms"] == 10.0


def test_record_failed_request() -> None:
    monitor = RuntimeMonitor()

    monitor.record_request(
        method="POST",
        path="/optimize",
        status_code=500,
        latency_ms=25.0,
    )

    snapshot = monitor.snapshot()

    assert snapshot["requests"]["total"] == 1
    assert snapshot["requests"]["successful"] == 0
    assert snapshot["requests"]["failed"] == 1
    assert snapshot["requests"]["success_rate_percent"] == 0.0
    assert snapshot["requests"]["average_latency_ms"] == 25.0
    assert snapshot["requests"]["max_latency_ms"] == 25.0


def test_request_latency_average_and_max() -> None:
    monitor = RuntimeMonitor()

    monitor.record_request(
        method="GET",
        path="/health",
        status_code=200,
        latency_ms=10.0,
    )

    monitor.record_request(
        method="GET",
        path="/health/live",
        status_code=200,
        latency_ms=30.0,
    )

    snapshot = monitor.snapshot()

    assert snapshot["requests"]["total"] == 2
    assert snapshot["requests"]["successful"] == 2
    assert snapshot["requests"]["failed"] == 0
    assert snapshot["requests"]["average_latency_ms"] == 20.0
    assert snapshot["requests"]["max_latency_ms"] == 30.0


# ============================================================
# ERROR MONITORING
# ============================================================

def test_record_error() -> None:
    monitor = RuntimeMonitor()

    monitor.record_error(
        "Test monitoring error"
    )

    snapshot = monitor.snapshot()

    assert snapshot["last_error"] == (
        "Test monitoring error"
    )

    assert snapshot["recent_events"]

    last_event = snapshot["recent_events"][-1]

    assert last_event["type"] == "error"
    assert last_event["error"] == (
        "Test monitoring error"
    )


# ============================================================
# OPTIMIZATION MONITORING
# ============================================================

def test_optimization_counters() -> None:
    monitor = RuntimeMonitor()

    monitor.record_optimization_started()
    monitor.record_optimization_completed()

    snapshot = monitor.snapshot()

    assert snapshot["optimization"]["started"] == 1
    assert snapshot["optimization"]["completed"] == 1
    assert snapshot["optimization"]["failed"] == 0


def test_optimization_failure_counter() -> None:
    monitor = RuntimeMonitor()

    monitor.record_optimization_started()

    monitor.record_optimization_failed(
        "Test optimization failure"
    )

    snapshot = monitor.snapshot()

    assert snapshot["optimization"]["started"] == 1
    assert snapshot["optimization"]["completed"] == 0
    assert snapshot["optimization"]["failed"] == 1
    assert snapshot["last_error"] == (
        "Test optimization failure"
    )


# ============================================================
# RESOURCE MONITORING
# ============================================================

def test_resource_snapshot_available() -> None:
    monitor = RuntimeMonitor()

    resources = monitor.resource_snapshot()

    assert resources["status"] == "available"

    assert isinstance(
        resources["process_id"],
        int,
    )

    assert isinstance(
        resources["cpu_percent"],
        float,
    )

    assert isinstance(
        resources["memory_rss_mb"],
        float,
    )

    assert isinstance(
        resources["memory_vms_mb"],
        float,
    )

    assert isinstance(
        resources["thread_count"],
        int,
    )

    assert isinstance(
        resources["process_uptime_seconds"],
        float,
    )

    assert resources["process_id"] > 0
    assert resources["memory_rss_mb"] >= 0.0
    assert resources["memory_vms_mb"] >= 0.0
    assert resources["thread_count"] >= 0
    assert resources["process_uptime_seconds"] >= 0.0


def test_snapshot_contains_resources() -> None:
    monitor = RuntimeMonitor()

    snapshot = monitor.snapshot()

    assert "resources" in snapshot
    assert snapshot["resources"]["status"] == (
        "available"
    )

    expected_fields = {
        "process_id",
        "cpu_percent",
        "memory_rss_mb",
        "memory_vms_mb",
        "thread_count",
        "process_uptime_seconds",
    }

    assert expected_fields.issubset(
        snapshot["resources"].keys()
    )


# ============================================================
# RECENT EVENTS
# ============================================================

def test_recent_request_event_is_recorded() -> None:
    monitor = RuntimeMonitor()

    monitor.record_request(
        method="GET",
        path="/monitoring",
        status_code=200,
        latency_ms=5.5,
    )

    snapshot = monitor.snapshot()

    assert len(
        snapshot["recent_events"]
    ) == 1

    event = snapshot["recent_events"][0]

    assert event["method"] == "GET"
    assert event["path"] == "/monitoring"
    assert event["status_code"] == 200
    assert event["latency_ms"] == 5.5
    assert "timestamp" in event


# ============================================================
# FASTAPI INTEGRATION
# ============================================================

def test_monitoring_endpoint() -> None:
    client = TestClient(app)

    response = client.get(
        "/monitoring"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"

    assert "requests" in data
    assert "optimization" in data
    assert "resources" in data
    assert "recent_events" in data

    assert data["resources"]["status"] == (
        "available"
    )


def test_health_endpoint_is_monitored() -> None:
    client = TestClient(app)

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"