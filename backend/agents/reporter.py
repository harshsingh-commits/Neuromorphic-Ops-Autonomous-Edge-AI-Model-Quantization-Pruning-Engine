from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.models.schemas import WorkflowState
from backend.utils.config import REPORT_DIR


def _safe_size_mb(path: str | Path | None) -> float | None:
    if not path:
        return None

    file_path = Path(path)

    if not file_path.exists():
        return None

    return file_path.stat().st_size / (1024 * 1024)


def generate_report(state: WorkflowState) -> dict[str, Any]:
    """
    Generate JSON and Markdown optimization reports from
    the current workflow state.

    Only values already present in the workflow state are reported.
    Missing metrics are represented as None rather than fake zeroes.
    """

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    optimized_path = state.get(
        "optimized_model_path",
        state.get("model_path"),
    )

    optimized_size_mb = state.get("optimized_size_mb")

    if optimized_size_mb is None:
        optimized_size_mb = _safe_size_mb(optimized_path)

    original_size_mb = state.get("original_size_mb")

    compression_ratio = state.get("compression_ratio")

    if (
        compression_ratio is None
        and original_size_mb is not None
        and optimized_size_mb
        and optimized_size_mb > 0
    ):
        compression_ratio = (
            original_size_mb / optimized_size_mb
        )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    payload = {
        "project": {
            "name": "Neuromorphic-Ops",
            "timestamp": timestamp,
        },

        "model": {
            "name": state.get("model_name"),
            "type": state.get("model_type"),
        },

        "original_model": {
            "size_mb": original_size_mb,
            "parameters": state.get("original_parameters"),
            "accuracy": state.get("original_accuracy"),
            "latency_ms": state.get(
                "baseline_latency_ms"
            ),
            "memory_mb": state.get(
                "baseline_memory_usage_mb"
            ),
        },

        "optimized_model": {
            "path": str(optimized_path)
            if optimized_path
            else None,
            "size_mb": optimized_size_mb,
            "parameters": state.get(
                "optimized_parameters"
            ),
            "accuracy": state.get(
                "optimized_accuracy"
            ),
            "latency_ms": state.get(
                "latency_ms"
            ),
            "memory_mb": state.get(
                "memory_usage_mb"
            ),
        },

        "optimization": {
            "strategy": state.get(
                "optimization_strategy"
            ),
            "pruning_ratio": state.get(
                "pruning_ratio"
            ),
            "quantization_type": state.get(
                "quantization_type"
            ),
            "sparsity": state.get("sparsity"),
            "compression_ratio": compression_ratio,
        },

        "evaluation": {
            "accuracy_drop": state.get(
                "accuracy_drop"
            ),
            "latency_improvement_percent": state.get(
                "latency_improvement_percent"
            ),
            "memory_reduction_percent": state.get(
                "memory_reduction_percent"
            ),
        },

        "refinement": {
            "iteration": state.get(
                "iteration",
                0,
            ),
            "history": state.get(
                "refinement_history",
                [],
            ),
        },

        "onnx": {
            "path": state.get("onnx_path"),
            "validation_status": state.get(
                "onnx_validation_status"
            ),
            "runtime_status": state.get(
                "onnx_runtime_status"
            ),
        },

        "deployment": {
            "approval_status": state.get(
                "approval_status"
            ),
            "deployment_status": state.get(
                "deployment_status"
            ),
        },

        "workflow": {
            "status": state.get(
                "workflow_status"
            ),
            "errors": state.get(
                "errors",
                [],
            ),
        },
    }

    json_path = REPORT_DIR / "optimization_report.json"
    md_path = REPORT_DIR / "optimization_report.md"

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    def value(value: Any) -> str:
        return "Not available" if value is None else str(value)

    markdown = f"""# Neuromorphic-Ops Optimization Report

## Project

- Name: {payload["project"]["name"]}
- Timestamp: {payload["project"]["timestamp"]}

## Model

- Name: {value(payload["model"]["name"])}
- Type: {value(payload["model"]["type"])}

## Original Model

| Metric | Value |
|---|---|
| Size | {value(payload["original_model"]["size_mb"])} MB |
| Parameters | {value(payload["original_model"]["parameters"])} |
| Accuracy | {value(payload["original_model"]["accuracy"])} |
| Latency | {value(payload["original_model"]["latency_ms"])} ms |
| Memory | {value(payload["original_model"]["memory_mb"])} MB |

## Optimized Model

| Metric | Value |
|---|---|
| Path | {value(payload["optimized_model"]["path"])} |
| Size | {value(payload["optimized_model"]["size_mb"])} MB |
| Parameters | {value(payload["optimized_model"]["parameters"])} |
| Accuracy | {value(payload["optimized_model"]["accuracy"])} |
| Latency | {value(payload["optimized_model"]["latency_ms"])} ms |
| Memory | {value(payload["optimized_model"]["memory_mb"])} MB |

## Optimization

| Metric | Value |
|---|---|
| Strategy | {value(payload["optimization"]["strategy"])} |
| Pruning Ratio | {value(payload["optimization"]["pruning_ratio"])} |
| Quantization | {value(payload["optimization"]["quantization_type"])} |
| Sparsity | {value(payload["optimization"]["sparsity"])} |
| Compression Ratio | {value(payload["optimization"]["compression_ratio"])} |

## Evaluation

| Metric | Value |
|---|---|
| Accuracy Drop | {value(payload["evaluation"]["accuracy_drop"])} |
| Latency Improvement | {value(payload["evaluation"]["latency_improvement_percent"])}% |
| Memory Reduction | {value(payload["evaluation"]["memory_reduction_percent"])}% |

## Refinement

- Iteration: {value(payload["refinement"]["iteration"])}

## ONNX

- Path: {value(payload["onnx"]["path"])}
- Validation: {value(payload["onnx"]["validation_status"])}
- Runtime: {value(payload["onnx"]["runtime_status"])}

## Deployment

- Approval: {value(payload["deployment"]["approval_status"])}
- Deployment Status: {value(payload["deployment"]["deployment_status"])}

## Workflow

- Status: {value(payload["workflow"]["status"])}

### Errors

"""

    errors = payload["workflow"]["errors"]

    if errors:
        for error in errors:
            markdown += f"- {error}\n"
    else:
        markdown += "- None\n"

    md_path.write_text(
        markdown,
        encoding="utf-8",
    )

    return {
        "report_path": str(json_path),
        "markdown_report_path": str(md_path),
        "optimized_size_mb": optimized_size_mb,
        "compression_ratio": compression_ratio,
        "workflow_status": "report_generated",
    }