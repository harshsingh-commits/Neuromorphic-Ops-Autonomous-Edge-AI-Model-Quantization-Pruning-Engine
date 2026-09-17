from __future__ import annotations

from backend.models.schemas import WorkflowState
from backend.services.metrics import measure_model_memory
from backend.services.model_io import load_model


def memory_node(state: WorkflowState) -> dict:
    model = load_model(state["model_path"])

    memory = measure_model_memory(model)

    baseline = state.get("baseline_memory_usage_mb")

    reduction = None

    if baseline and baseline > 0:
        reduction = ((baseline - memory) / baseline) * 100.0

    return {
        "memory_usage_mb": memory,
        "memory_reduction_percent": reduction,
        "memory_status": "success",
    }