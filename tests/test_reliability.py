from __future__ import annotations

from types import SimpleNamespace

from backend.graph.workflow import workflow
from backend.services.orchestrator import Orchestrator


def test_unexpected_workflow_exception_becomes_persisted_failure(
    monkeypatch,
):
    thread_id = "test-reliability-thread"

    persisted_state = {
        "workflow_status": "running",
        "deployment_status": "running",
        "approval_status": "pending",
        "errors": [],
        "run_id": "RUN-TEST-001",
    }

    def fake_invoke(*args, **kwargs):
        raise RuntimeError(
            "Simulated unexpected agent failure"
        )

    def fake_get_state(config):
        return SimpleNamespace(
            values=dict(persisted_state)
        )

    def fake_update_state(
        config,
        patch,
    ):
        persisted_state.update(
            patch
        )

    monkeypatch.setattr(
        workflow,
        "invoke",
        fake_invoke,
    )

    monkeypatch.setattr(
        workflow,
        "get_state",
        fake_get_state,
    )

    monkeypatch.setattr(
        workflow,
        "update_state",
        fake_update_state,
    )

    orchestrator = Orchestrator()

    # The real start() generates the thread ID internally,
    # so only the returned state is validated here.
    returned_thread_id, state = (
        orchestrator.start(
            model_path="/app/outputs/demo_model.pt",
            strategy="hybrid",
            pruning_ratio=0.2,
            accuracy_threshold=1.5,
            quantization_type="dynamic_int8",
        )
    )

    assert returned_thread_id
    assert returned_thread_id != thread_id

    assert state["workflow_status"] == "failed"
    assert state["deployment_status"] == "blocked"
    assert state["approval_status"] == "blocked"

    assert (
        "Workflow execution failed: "
        "Simulated unexpected agent failure"
        in state["errors"]
    )

    assert (
        state["error"]
        == (
            "Workflow execution failed: "
            "Simulated unexpected agent failure"
        )
    )

    assert (
        persisted_state["workflow_status"]
        == "failed"
    )

    assert (
        persisted_state["deployment_status"]
        == "blocked"
    )

    assert (
        persisted_state["approval_status"]
        == "blocked"
    )


def test_status_reads_persistent_state_without_memory_cache(
    monkeypatch,
):
    thread_id = (
        "7d13768a-7c3d-4491-9c9e-eb3d6c8f46c7"
    )

    persisted_state = {
        "workflow_status": "failed",
        "deployment_status": "blocked",
        "approval_status": "blocked",
        "run_id": "RUN-TEST-002",
        "errors": [
            "Simulated persisted failure"
        ],
    }

    def fake_get_state(config):
        return SimpleNamespace(
            values=dict(persisted_state)
        )

    monkeypatch.setattr(
        workflow,
        "get_state",
        fake_get_state,
    )

    orchestrator = Orchestrator()

    # Simulate a fresh backend process:
    # the in-memory cache is empty.
    assert thread_id not in orchestrator.states

    result = orchestrator.status(
        thread_id
    )

    assert (
        result["thread_id"]
        == thread_id
    )

    assert (
        result["status"]
        == "failed"
    )

    assert (
        result["state"]["run_id"]
        == "RUN-TEST-002"
    )

    assert (
        result["state"]["deployment_status"]
        == "blocked"
    )

    assert (
        result["state"]["approval_status"]
        == "blocked"
    )

    assert (
        result["state"]["errors"]
        == [
            "Simulated persisted failure"
        ]
    )


def test_health_behavior_is_not_affected_by_workflow_failure():
    """
    This is intentionally a lightweight regression test.

    The workflow failure mechanism must not modify FastAPI's
    health endpoint behavior.
    """

    from backend.main import app
    from fastapi.testclient import TestClient

    client = TestClient(app)

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"