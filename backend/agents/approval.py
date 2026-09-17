from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from backend.models.schemas import WorkflowState


def request_approval(
    state: WorkflowState,
) -> dict[str, Any]:
    """
    Request human approval only when the optimized
    artifacts are valid.
    """

    export_status = state.get(
        "export_status"
    )

    onnx_validation_status = state.get(
        "onnx_validation_status",
        False,
    )

    onnx_runtime_status = state.get(
        "onnx_runtime_status",
        False,
    )

    if (
        export_status != "success"
        or not onnx_validation_status
        or not onnx_runtime_status
    ):
        return {
            "approval_status": "blocked",
            "deployment_status": "blocked",
            "workflow_status": "failed",
            "errors": [
                *state.get("errors", []),
                (
                    "Approval blocked because the ONNX "
                    "artifact was not successfully exported "
                    "and validated."
                ),
            ],
        }

    decision = interrupt(
        {
            "message": (
                "Approve optimized model for deployment?"
            ),
            "report_path": state.get(
                "report_path"
            ),
            "onnx_path": state.get(
                "onnx_path"
            ),
            "accuracy_drop": state.get(
                "accuracy_drop"
            ),
            "latency_improvement_percent": state.get(
                "latency_improvement_percent"
            ),
            "memory_reduction_percent": state.get(
                "memory_reduction_percent"
            ),
        }
    )

    normalized = (
        str(decision)
        .strip()
        .lower()
    )

    if normalized in {
        "approve",
        "approved",
        "yes",
        "true",
    }:
        return {
            "approval_status": "approved",
            "deployment_status": "pending",
        }

    return {
        "approval_status": "rejected",
        "deployment_status": "rejected",
    }


def mark_deployment(
    state: WorkflowState,
) -> dict[str, Any]:
    """
    Final deployment gate.
    """

    if state.get(
        "approval_status"
    ) != "approved":
        return {
            "deployment_status": "rejected",
            "workflow_status": "completed",
        }

    return {
        "deployment_status": "deployment_ready",
        "workflow_status": "completed",
    }