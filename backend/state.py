from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional

class WorkflowState(TypedDict, total=False):
    # Model Metadata
    model_path: str
    model_name: str
    model_type: str  # e.g., "resnet18", "cnn"

    # Size Metrics
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    sparsity: float  # ratio of zero weights to total weights

    # Parameter Metrics
    original_parameters: int
    optimized_parameters: int

    # Accuracy Metrics
    original_accuracy: float  # percentage, e.g., 98.5
    optimized_accuracy: float
    accuracy_drop: float  # baseline - optimized

    # Latency Metrics (ms)
    baseline_latency_ms: float
    latency_ms: float
    latency_improvement_percent: float

    # Memory Metrics (MB)
    baseline_memory_usage_mb: float
    memory_usage_mb: float
    memory_reduction_percent: float

    # Optimization Configuration & Strategy
    optimization_strategy: str  # "pruning", "quantization", "hybrid"
    optimization_reason: str
    pruning_ratio: float
    quantization_type: str  # e.g., "dynamic_int8", "static_int8"

    # Refinement Iteration Controls
    refinement_iteration: int
    max_refinement_iterations: int
    accuracy_threshold: float  # e.g., 1.5%

    # Workflow Execution Status
    workflow_status: str  # "analyzing", "selecting_strategy", "pruning", "quantizing", "evaluating", "refining", "exporting", "waiting_for_approval", "completed", "failed"
    approval_status: str  # "pending", "approved", "rejected"
    deployment_status: str  # "pending", "approved", "rejected", "failed"

    # Export Status
    onnx_path: str
    onnx_validation_status: str  # "pending", "passed", "failed"
    report_path: str

    # Execution History
    errors: List[str]
    logs: List[str]

def create_default_state(model_path: str, accuracy_threshold: float = 1.5, max_iterations: int = 5) -> WorkflowState:
    """Helper to initialize a workflow state with sensible defaults."""
    import os
    return {
        "model_path": model_path,
        "model_name": os.path.basename(model_path),
        "model_type": "unknown",
        "original_size_mb": 0.0,
        "optimized_size_mb": 0.0,
        "compression_ratio": 1.0,
        "sparsity": 0.0,
        "original_parameters": 0,
        "optimized_parameters": 0,
        "original_accuracy": 0.0,
        "optimized_accuracy": 0.0,
        "accuracy_drop": 0.0,
        "baseline_latency_ms": 0.0,
        "latency_ms": 0.0,
        "latency_improvement_percent": 0.0,
        "baseline_memory_usage_mb": 0.0,
        "memory_usage_mb": 0.0,
        "memory_reduction_percent": 0.0,
        "optimization_strategy": "unknown",
        "optimization_reason": "",
        "pruning_ratio": 0.0,
        "quantization_type": "none",
        "refinement_iteration": 0,
        "max_refinement_iterations": max_iterations,
        "accuracy_threshold": accuracy_threshold,
        "workflow_status": "started",
        "approval_status": "pending",
        "deployment_status": "pending",
        "onnx_path": "",
        "onnx_validation_status": "pending",
        "report_path": "",
        "errors": [],
        "logs": []
    }
