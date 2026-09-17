from __future__ import annotations

from backend.models.schemas import WorkflowState


DEFAULT_REDUCTION = 0.05
MIN_PRUNING_RATIO = 0.05


def refine_model(state: WorkflowState) -> dict:
    """
    Make the optimization more conservative when accuracy
    degradation exceeds the configured threshold.
    """

    iteration = int(state.get("iteration", 0)) + 1

    current_pruning_ratio = float(
        state.get("pruning_ratio", 0.30)
    )

    new_pruning_ratio = max(
        MIN_PRUNING_RATIO,
        current_pruning_ratio - DEFAULT_REDUCTION,
    )

    accuracy_drop = state.get("accuracy_drop")
    threshold = float(
        state.get("accuracy_threshold", 1.5)
    )

    history = list(
        state.get("refinement_history", [])
    )

    history.append(
        {
            "iteration": iteration,
            "previous_pruning_ratio": current_pruning_ratio,
            "new_pruning_ratio": new_pruning_ratio,
            "accuracy_drop": accuracy_drop,
            "accuracy_threshold": threshold,
            "reason": (
                "Accuracy degradation exceeded threshold; "
                "reducing pruning aggressiveness."
            ),
        }
    )

    return {
        "iteration": iteration,
        "pruning_ratio": new_pruning_ratio,
        "refinement_history": history,
        "workflow_status": "refining",
    }