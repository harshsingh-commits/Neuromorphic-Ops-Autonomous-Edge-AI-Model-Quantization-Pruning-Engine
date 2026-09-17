from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class WorkflowState(TypedDict, total=False):
    # =========================================================
    # Model paths
    # =========================================================

    model_path: str
    original_model_path: str
    pruned_model_path: str
    pre_quantized_model_path: str
    optimized_model_path: str
    onnx_path: str

    # =========================================================
    # Model metrics
    # =========================================================

    original_size_mb: float
    optimized_size_mb: float
    original_parameters: int
    optimized_parameters: int
    compression_ratio: float | None

    # =========================================================
    # Accuracy
    # =========================================================

    original_accuracy: float | None
    optimized_accuracy: float | None
    accuracy_drop: float
    accuracy_threshold: float
    accuracy_status: str
    accuracy_reason: str

    # =========================================================
    # Evaluation configuration
    # =========================================================

    evaluation_dataset: str
    evaluation_data_root: str
    evaluation_batch_size: int
    evaluation_max_samples: int
    evaluation_samples: int

    # =========================================================
    # Baseline performance
    # =========================================================

    baseline_latency_ms: float | None
    baseline_memory_usage_mb: float | None

    # =========================================================
    # Optimized performance
    # =========================================================

    latency_ms: float | None
    latency_improvement_percent: float | None

    memory_usage_mb: float | None
    memory_reduction_percent: float | None

    # =========================================================
    # Optimization
    # =========================================================

    optimization_strategy: str
    pruning_method: str
    pruning_ratio: float
    sparsity: float

    # =========================================================
    # Quantization
    # =========================================================

    quantization_type: str
    quantization_status: str

    # =========================================================
    # ONNX
    # =========================================================

    onnx_validation_status: bool
    onnx_runtime_status: bool
    export_status: str
    export_source: str

    # =========================================================
    # Analysis and evaluation
    # =========================================================

    analysis: dict[str, Any]
    analysis_summary: dict[str, Any]
    metrics: dict[str, Any]
    evaluation: dict[str, Any]

    # =========================================================
    # Refinement
    # =========================================================

    max_refinement_iterations: int

    # =========================================================
    # PHASE 19C - EXPERIMENT TRACKING
    # =========================================================

    run_id: str
    run_timestamp: str
    random_seed: int
    model_hash: str

    optimization_config: dict[str, Any]
    warnings: list[str]

    # =========================================================
    # Workflow
    # =========================================================

    iteration: int
    workflow_status: str
    deployment_status: str
    approval_status: str

    # =========================================================
    # Reports
    # =========================================================

    report_path: str
    report_markdown_path: str

    # =========================================================
    # Errors
    # =========================================================

    errors: list[str]
    error: str | None

    # =========================================================
    # Thread
    # =========================================================

    thread_id: str


class AnalysisSummary(BaseModel):
    parameter_count: int
    trainable_parameters: int
    module_count: int
    layer_count: int
    file_size_mb: float
    architecture_name: str
    supported_pruning_layers: int
    supported_quantization_layers: int


class OptimizationRequest(BaseModel):
    model_path: str

    strategy: str | None = Field(
        default=None,
        pattern="^(pruning|quantization|hybrid)$",
    )

    accuracy_threshold: float = Field(
        default=1.5,
        ge=0.1,
        le=10.0,
    )

    pruning_ratio: float | None = Field(
        default=0.30,
        ge=0.05,
        le=0.90,
    )

    quantization_type: str | None = Field(
        default="dynamic_int8",
        pattern="^(dynamic_int8|static_int8)$",
    )


class ApprovalRequest(BaseModel):
    approved: bool


class WorkflowResponse(BaseModel):
    thread_id: str
    status: str
    state: dict[str, Any]


class JobStatusResponse(BaseModel):
    thread_id: str
    status: str
    deployment_status: str | None = None
    approval_status: str | None = None
    report_path: str | None = None
    onnx_path: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str