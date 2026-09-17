from __future__ import annotations

from backend.models.schemas import WorkflowState


def accuracy_node(state: WorkflowState) -> dict:
    baseline = state.get("original_accuracy")

    if baseline is None:
        return {
            "optimized_accuracy": None,
            "accuracy_drop": None,
            "accuracy_status": "unavailable",
        }

    return {
        "optimized_accuracy": None,
        "accuracy_drop": None,
        "accuracy_status": "unavailable",
    }