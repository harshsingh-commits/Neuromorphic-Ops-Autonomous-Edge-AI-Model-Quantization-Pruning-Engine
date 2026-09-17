from __future__ import annotations

from backend.models.schemas import WorkflowState
from backend.services.metrics import measure_latency
from backend.services.model_io import load_model


def latency_node(state: WorkflowState) -> dict:
    model = load_model(state["model_path"])

    latency = measure_latency(model)

    baseline = state.get("baseline_latency_ms")

    improvement = None

    if baseline and baseline > 0:
        improvement = ((baseline - latency) / baseline) * 100.0

    return {
        "latency_ms": latency,
        "latency_improvement_percent": improvement,
        "latency_status": "success",
    }