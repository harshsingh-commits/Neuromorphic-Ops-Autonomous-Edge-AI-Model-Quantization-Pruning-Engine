from __future__ import annotations

import random
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from backend.agents.analyzer import analyze_model
from backend.agents.approval import (
    mark_deployment,
    request_approval,
)
from backend.agents.evaluator import (
    evaluate_accuracy,
    evaluate_latency,
    evaluate_memory,
    merge_evaluation,
)
from backend.agents.exporter import export_onnx
from backend.agents.pruner import apply_pruning
from backend.agents.quantizer import apply_quantization
from backend.agents.refiner import refine_model
from backend.agents.reporter import generate_report
from backend.agents.strategy import select_strategy
from backend.models.schemas import WorkflowState
from backend.services.metrics import (
    benchmark_latency,
    measure_latency,
    measure_model_memory,
)
from backend.services.model_io import load_model
from backend.services.tracking import OptimizationTracker


DEFAULT_BENCHMARK_BATCH_SIZES = (1, 8, 16)
DEFAULT_RANDOM_SEED = 42


# ============================================================
# PHASE 23.2 - PERSISTENT LANGGRAPH CHECKPOINT STORAGE
# ============================================================

CHECKPOINT_DIRECTORY = Path(
    "runs"
)

CHECKPOINT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DB_PATH = (
    CHECKPOINT_DIRECTORY
    / "checkpoints.sqlite"
)

checkpoint_connection = sqlite3.connect(
    str(
        CHECKPOINT_DB_PATH
    ),
    check_same_thread=False,
    timeout=30,
)

checkpointer = SqliteSaver(
    checkpoint_connection
)

checkpointer.setup()


# ============================================================
# PHASE 19C - EXPERIMENT TRACKING INITIALIZATION
# ============================================================

def _generate_run_id() -> str:
    """
    Generate a unique experiment run ID.
    """

    return (
        f"RUN-"
        f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-"
        f"{uuid.uuid4().hex[:8]}"
    )


def initialize_tracking(
    state: WorkflowState,
) -> dict:
    """
    Initialize reproducibility metadata for the optimization run.

    Generates:
    - Run ID
    - UTC timestamp
    - Random seed
    - Original model SHA-256 hash
    """

    run_id = state.get(
        "run_id"
    )

    if not run_id:
        run_id = _generate_run_id()

    run_timestamp = state.get(
        "run_timestamp"
    )

    if not run_timestamp:
        run_timestamp = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    random_seed = int(
        state.get(
            "random_seed",
            DEFAULT_RANDOM_SEED,
        )
        or DEFAULT_RANDOM_SEED
    )

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    random.seed(
        random_seed
    )

    torch.manual_seed(
        random_seed
    )

    # --------------------------------------------------------
    # Model hash
    # --------------------------------------------------------

    model_path = (
        state.get(
            "original_model_path"
        )
        or state.get(
            "model_path"
        )
    )

    model_hash = (
        state.get(
            "model_hash"
        )
    )

    if not model_hash and model_path:
        model_hash = (
            OptimizationTracker.calculate_model_hash(
                model_path
            )
        )

    return {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "random_seed": random_seed,
        "model_hash": model_hash,
        "workflow_status": "tracking_initialized",
    }


# ============================================================
# BASELINE MEASUREMENT
# ============================================================

def measure_baseline_metrics(
    state: WorkflowState,
) -> dict:
    """
    Measure the original model's standard latency and memory
    before pruning or quantization.
    """

    original_model_path = state.get(
        "original_model_path"
    )

    if not original_model_path:
        return {
            "baseline_latency_ms": None,
            "baseline_memory_usage_mb": None,
            "workflow_status": "baseline_failed",
            "errors": [
                *state.get(
                    "errors",
                    [],
                ),
                "Baseline measurement failed: original model path is missing.",
            ],
        }

    try:

        model = load_model(
            original_model_path
        )

        model = (
            model
            .cpu()
            .eval()
        )

        baseline_latency = measure_latency(
            model
        )

        baseline_memory = measure_model_memory(
            model
        )

        return {
            "baseline_latency_ms": float(
                baseline_latency
            ),
            "baseline_memory_usage_mb": float(
                baseline_memory
            ),
            "workflow_status": "baseline_completed",
        }

    except Exception as exc:

        return {
            "baseline_latency_ms": None,
            "baseline_memory_usage_mb": None,
            "workflow_status": "baseline_failed",
            "errors": [
                *state.get(
                    "errors",
                    [],
                ),
                f"Baseline measurement failed: {exc}",
            ],
            "error": (
                f"Baseline measurement failed: {exc}"
            ),
        }


# ============================================================
# ROUTING
# ============================================================

def route_after_strategy(
    state: WorkflowState,
) -> str:
    """
    Safely route after strategy selection.

    Strategy-selection failures must never continue into
    pruning or quantization.
    """

    workflow_status = state.get(
        "workflow_status"
    )

    if workflow_status in {
        "failed",
        "strategy_failed",
        "baseline_failed",
    }:
        return "strategy_failure"

    strategy = state.get(
        "optimization_strategy"
    )

    if strategy == "quantization":
        return "quantize"

    if strategy in {
        "pruning",
        "hybrid",
    }:
        return "prune"

    return "strategy_failure"


def route_after_pruning(
    state: WorkflowState,
) -> str:
    strategy = state.get(
        "optimization_strategy",
        "pruning",
    )

    if strategy == "hybrid":
        return "quantize"

    if strategy == "pruning":
        return "evaluate"

    return "strategy_failure"


def route_after_evaluation(
    state: WorkflowState,
) -> str:
    """
    Decide whether to refine, export, or safely terminate.

    Safety rules:
    - Accuracy evaluation failure => block.
    - Benchmark failure => block.
    - Tracking failure => block.
    - Accuracy drop above threshold + iterations available => refine.
    - Accuracy drop above threshold at max iterations => block.
    - Only acceptable accuracy continues to export.
    """

    accuracy_status = state.get(
        "accuracy_status"
    )

    if accuracy_status in {
        "failed",
        "unavailable",
        "unsupported",
    }:
        return "accuracy_failure"

    workflow_status = state.get(
        "workflow_status"
    )

    if workflow_status == "benchmark_failed":
        return "optimization_failure"

    if workflow_status == "tracking_failed":
        return "tracking_failure"

    accuracy_drop = float(
        state.get(
            "accuracy_drop",
            0.0,
        )
        or 0.0
    )

    iteration = int(
        state.get(
            "iteration",
            0,
        )
        or 0
    )

    threshold = float(
        state.get(
            "accuracy_threshold",
            1.5,
        )
        or 1.5
    )

    max_iterations = int(
        state.get(
            "max_refinement_iterations",
            5,
        )
        or 5
    )

    if accuracy_drop > threshold:

        if iteration < max_iterations:
            return "refine"

        return "refinement_limit_failure"

    return "export"


# ============================================================
# PHASE 19A - BENCHMARK
# ============================================================

def run_benchmark_node(
    state: WorkflowState,
) -> dict:
    """
    Run CPU latency benchmark for original and optimized models.

    Batch sizes:
    - 1
    - 8
    - 16

    Methodology:
    - 10 warmup runs
    - 50 timed runs
    - median latency
    """

    batch_sizes = [
        int(batch_size)
        for batch_size
        in DEFAULT_BENCHMARK_BATCH_SIZES
    ]

    original_model_path = state.get(
        "original_model_path"
    )

    optimized_model_path = state.get(
        "model_path"
    )

    if not original_model_path:

        return {
            "benchmark_batch_sizes": batch_sizes,
            "benchmark_latency_ms": {},
            "benchmark_baseline_latency_ms": {},
            "benchmark_latency_improvement_percent": {},
            "workflow_status": "benchmark_failed",
            "errors": [
                *state.get(
                    "errors",
                    [],
                ),
                "Benchmark failed: original model path is missing.",
            ],
            "error": (
                "Benchmark failed: original model path is missing."
            ),
        }

    if not optimized_model_path:

        return {
            "benchmark_batch_sizes": batch_sizes,
            "benchmark_latency_ms": {},
            "benchmark_baseline_latency_ms": {},
            "benchmark_latency_improvement_percent": {},
            "workflow_status": "benchmark_failed",
            "errors": [
                *state.get(
                    "errors",
                    [],
                ),
                "Benchmark failed: optimized model path is missing.",
            ],
            "error": (
                "Benchmark failed: optimized model path is missing."
            ),
        }

    try:

        # ----------------------------------------------------
        # Original model
        # ----------------------------------------------------

        original_model = load_model(
            original_model_path
        )

        original_model = (
            original_model
            .cpu()
            .eval()
        )

        # ----------------------------------------------------
        # Optimized model
        # ----------------------------------------------------

        optimized_model = load_model(
            optimized_model_path
        )

        optimized_model = (
            optimized_model
            .cpu()
            .eval()
        )

        # ----------------------------------------------------
        # Baseline benchmark
        # ----------------------------------------------------

        baseline_benchmark = benchmark_latency(
            original_model,
            batch_sizes=tuple(
                batch_sizes
            ),
            warmup_runs=10,
            runs=50,
        )

        # ----------------------------------------------------
        # Optimized benchmark
        # ----------------------------------------------------

        optimized_benchmark = benchmark_latency(
            optimized_model,
            batch_sizes=tuple(
                batch_sizes
            ),
            warmup_runs=10,
            runs=50,
        )

        # ----------------------------------------------------
        # Improvement
        # ----------------------------------------------------

        improvement: dict[
            str,
            float | None,
        ] = {}

        for batch_size in batch_sizes:

            key = str(
                batch_size
            )

            baseline_value = (
                baseline_benchmark.get(
                    key
                )
            )

            optimized_value = (
                optimized_benchmark.get(
                    key
                )
            )

            if (
                isinstance(
                    baseline_value,
                    (int, float),
                )
                and isinstance(
                    optimized_value,
                    (int, float),
                )
                and baseline_value > 0
            ):

                improvement[key] = (
                    (
                        baseline_value
                        - optimized_value
                    )
                    / baseline_value
                ) * 100.0

            else:

                improvement[key] = None

        # ----------------------------------------------------
        # Persist benchmark inside metrics
        # ----------------------------------------------------

        metrics = dict(
            state.get(
                "metrics",
                {},
            )
        )

        metrics.update(
            {
                "benchmark_batch_sizes": batch_sizes,
                "benchmark_baseline_latency_ms": (
                    baseline_benchmark
                ),
                "benchmark_latency_ms": (
                    optimized_benchmark
                ),
                "benchmark_latency_improvement_percent": (
                    improvement
                ),
            }
        )

        return {
            "benchmark_batch_sizes": batch_sizes,
            "benchmark_baseline_latency_ms": (
                baseline_benchmark
            ),
            "benchmark_latency_ms": (
                optimized_benchmark
            ),
            "benchmark_latency_improvement_percent": (
                improvement
            ),
            "metrics": metrics,
            "workflow_status": "benchmark_completed",
        }

    except Exception as exc:

        return {
            "benchmark_batch_sizes": batch_sizes,
            "benchmark_latency_ms": {},
            "benchmark_baseline_latency_ms": {},
            "benchmark_latency_improvement_percent": {},
            "workflow_status": "benchmark_failed",
            "errors": [
                *state.get(
                    "errors",
                    [],
                ),
                f"Benchmark failed: {exc}",
            ],
            "error": (
                f"Benchmark failed: {exc}"
            ),
        }


# ============================================================
# PHASE 19C - EXPERIMENT TRACKING
# ============================================================

def track_experiment_node(
    state: WorkflowState,
) -> dict:
    """
    Persist experiment metadata and metrics for the current run.

    A run_id is normally created by initialize_tracking().
    A safe fallback is generated here so tracking does not fail
    if the state schema/process does not preserve run_id.
    """

    run_id = state.get(
        "run_id"
    )

    fallback_generated = False

    if not run_id:

        run_id = _generate_run_id()
        fallback_generated = True

    tracker = None

    try:

        random_seed = int(
            state.get(
                "random_seed",
                DEFAULT_RANDOM_SEED,
            )
            or DEFAULT_RANDOM_SEED
        )

        tracker = OptimizationTracker(
            run_id=run_id,
            random_seed=random_seed,
        )

        tracker.record(
            state,
            step=int(
                state.get(
                    "iteration",
                    0,
                )
                or 0
            ),
        )

        metadata = tracker.build_metadata(
            state
        )

        warnings = list(
            state.get(
                "warnings",
                [],
            )
        )

        if fallback_generated:

            warnings.append(
                "Tracking run_id was missing from workflow state; "
                "a fallback run_id was generated at tracking time."
            )

        return {
            "run_id": run_id,
            "run_timestamp": (
                state.get(
                    "run_timestamp"
                )
                or datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "random_seed": random_seed,
            "model_hash": state.get(
                "model_hash"
            ),
            "optimization_config": metadata.get(
                "optimization_config",
                {},
            ),
            "warnings": warnings,
            "workflow_status": "tracking_completed",
        }

    except Exception as exc:

        errors = list(
            state.get(
                "errors",
                [],
            )
        )

        error_message = (
            f"Experiment tracking failed: {exc}"
        )

        errors.append(
            error_message
        )

        return {
            "workflow_status": "tracking_failed",
            "errors": errors,
            "error": error_message,
        }

    finally:

        if tracker is not None:

            try:
                tracker.close()

            except Exception:
                pass


# ============================================================
# METRIC ROUTE
# ============================================================

def log_metrics_node(
    state: WorkflowState,
) -> dict:
    """
    Compatibility node for the existing workflow.

    ExperimentTracker already records TensorBoard metrics.
    """

    return {}


# ============================================================
# FAILURE HANDLERS
# ============================================================

def handle_strategy_failure(
    state: WorkflowState,
) -> dict:

    reason = state.get(
        "error",
        "Strategy selection failed.",
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    if reason not in errors:
        errors.append(
            reason
        )

    logs = list(
        state.get(
            "logs",
            [],
        )
    )

    logs.append(
        "Workflow blocked: strategy selection failure."
    )

    return {
        "workflow_status": "failed",
        "deployment_status": "blocked",
        "approval_status": "blocked",
        "errors": errors,
        "error": reason,
        "logs": logs,
    }


def handle_accuracy_failure(
    state: WorkflowState,
) -> dict:

    reason = state.get(
        "accuracy_reason",
        "Accuracy evaluation could not be completed.",
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    message = (
        "Accuracy evaluation blocked workflow: "
        f"{reason}"
    )

    if message not in errors:
        errors.append(
            message
        )

    return {
        "workflow_status": "failed",
        "deployment_status": "blocked",
        "approval_status": "blocked",
        "errors": errors,
        "error": reason,
    }


def handle_refinement_limit(
    state: WorkflowState,
) -> dict:
    """
    Block deployment after maximum refinement attempts
    when accuracy remains outside the threshold.
    """

    accuracy_drop = float(
        state.get(
            "accuracy_drop",
            0.0,
        )
        or 0.0
    )

    threshold = float(
        state.get(
            "accuracy_threshold",
            1.5,
        )
        or 1.5
    )

    iteration = int(
        state.get(
            "iteration",
            0,
        )
        or 0
    )

    max_iterations = int(
        state.get(
            "max_refinement_iterations",
            5,
        )
        or 5
    )

    reason = (
        "Accuracy threshold was not satisfied after the "
        f"maximum refinement iterations. "
        f"accuracy_drop={accuracy_drop:.4f} pp, "
        f"threshold={threshold:.4f} pp, "
        f"iterations={iteration}/{max_iterations}."
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    if reason not in errors:
        errors.append(
            reason
        )

    logs = list(
        state.get(
            "logs",
            [],
        )
    )

    logs.append(
        "Workflow blocked: maximum refinement iterations reached."
    )

    return {
        "workflow_status": "failed",
        "deployment_status": "blocked",
        "approval_status": "blocked",
        "errors": errors,
        "error": reason,
        "logs": logs,
    }


def handle_optimization_failure(
    state: WorkflowState,
) -> dict:
    """
    Safely terminate the workflow when benchmark or another
    optimization-stage failure prevents reliable deployment.
    """

    reason = state.get(
        "error",
        "Optimization workflow failed.",
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    benchmark_error = (
        "Optimization benchmark failed; "
        "deployment cannot continue safely."
    )

    if benchmark_error not in errors:
        errors.append(
            benchmark_error
        )

    return {
        "workflow_status": "failed",
        "deployment_status": "blocked",
        "approval_status": "blocked",
        "errors": errors,
        "error": reason,
    }


def handle_tracking_failure(
    state: WorkflowState,
) -> dict:
    """
    Block deployment when experiment tracking fails.
    """

    reason = state.get(
        "error",
        "Experiment tracking failed.",
    )

    errors = list(
        state.get(
            "errors",
            [],
        )
    )

    if reason not in errors:
        errors.append(
            reason
        )

    return {
        "workflow_status": "failed",
        "deployment_status": "blocked",
        "approval_status": "blocked",
        "errors": errors,
        "error": reason,
    }


# ============================================================
# BUILD WORKFLOW
# ============================================================

def build_workflow():

    graph = StateGraph(
        WorkflowState
    )

    # --------------------------------------------------------
    # Tracking initialization
    # --------------------------------------------------------

    graph.add_node(
        "initialize_tracking",
        initialize_tracking,
    )

    # --------------------------------------------------------
    # Core
    # --------------------------------------------------------

    graph.add_node(
        "analyze",
        analyze_model,
    )

    graph.add_node(
        "baseline",
        measure_baseline_metrics,
    )

    graph.add_node(
        "strategy",
        select_strategy,
    )

    graph.add_node(
        "prune",
        apply_pruning,
    )

    graph.add_node(
        "quantize",
        apply_quantization,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    graph.add_node(
        "evaluate_start",
        lambda state: {},
    )

    graph.add_node(
        "accuracy",
        evaluate_accuracy,
    )

    graph.add_node(
        "latency",
        evaluate_latency,
    )

    graph.add_node(
        "memory",
        evaluate_memory,
    )

    graph.add_node(
        "merge_evaluation",
        merge_evaluation,
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    graph.add_node(
        "benchmark",
        run_benchmark_node,
    )

    # --------------------------------------------------------
    # Tracking
    # --------------------------------------------------------

    graph.add_node(
        "track_experiment",
        track_experiment_node,
    )

    graph.add_node(
        "log_metrics",
        log_metrics_node,
    )

    # --------------------------------------------------------
    # Failure handlers
    # --------------------------------------------------------

    graph.add_node(
        "strategy_failure",
        handle_strategy_failure,
    )

    graph.add_node(
        "accuracy_failure",
        handle_accuracy_failure,
    )

    graph.add_node(
        "refinement_limit_failure",
        handle_refinement_limit,
    )

    graph.add_node(
        "optimization_failure",
        handle_optimization_failure,
    )

    graph.add_node(
        "tracking_failure",
        handle_tracking_failure,
    )

    # --------------------------------------------------------
    # Optimization / export
    # --------------------------------------------------------

    graph.add_node(
        "refine",
        refine_model,
    )

    graph.add_node(
        "export",
        export_onnx,
    )

    graph.add_node(
        "report",
        generate_report,
    )

    graph.add_node(
        "approval",
        request_approval,
    )

    graph.add_node(
        "deploy",
        mark_deployment,
    )

    # ========================================================
    # WORKFLOW EDGES
    # ========================================================

    graph.add_edge(
        START,
        "initialize_tracking",
    )

    graph.add_edge(
        "initialize_tracking",
        "analyze",
    )

    graph.add_edge(
        "analyze",
        "baseline",
    )

    graph.add_edge(
        "baseline",
        "strategy",
    )

    # --------------------------------------------------------
    # Strategy routing
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "strategy",
        route_after_strategy,
        {
            "prune": "prune",
            "quantize": "quantize",
            "strategy_failure": "strategy_failure",
        },
    )

    # --------------------------------------------------------
    # Pruning routing
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "prune",
        route_after_pruning,
        {
            "quantize": "quantize",
            "evaluate": "evaluate_start",
            "strategy_failure": "strategy_failure",
        },
    )

    # --------------------------------------------------------
    # Quantization
    # --------------------------------------------------------

    graph.add_edge(
        "quantize",
        "evaluate_start",
    )

    # --------------------------------------------------------
    # Evaluation fan-out
    # --------------------------------------------------------

    graph.add_edge(
        "evaluate_start",
        "accuracy",
    )

    graph.add_edge(
        "evaluate_start",
        "latency",
    )

    graph.add_edge(
        "evaluate_start",
        "memory",
    )

    # --------------------------------------------------------
    # Evaluation merge
    # --------------------------------------------------------

    graph.add_edge(
        "accuracy",
        "merge_evaluation",
    )

    graph.add_edge(
        "latency",
        "merge_evaluation",
    )

    graph.add_edge(
        "memory",
        "merge_evaluation",
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    graph.add_edge(
        "merge_evaluation",
        "benchmark",
    )

    # --------------------------------------------------------
    # Experiment tracking
    # --------------------------------------------------------

    def route_after_tracking(
        state: WorkflowState,
    ) -> str:

        if state.get(
            "workflow_status"
        ) == "tracking_failed":

            return "tracking_failure"

        return "continue"

    graph.add_conditional_edges(
        "benchmark",
        lambda state: "tracking",
        {
            "tracking": "track_experiment",
        },
    )

    graph.add_conditional_edges(
        "track_experiment",
        route_after_tracking,
        {
            "continue": "log_metrics",
            "tracking_failure": "tracking_failure",
        },
    )

    # --------------------------------------------------------
    # Evaluation decision
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "log_metrics",
        route_after_evaluation,
        {
            "refine": "refine",
            "export": "export",
            "accuracy_failure": "accuracy_failure",
            "refinement_limit_failure":
                "refinement_limit_failure",
            "optimization_failure":
                "optimization_failure",
            "tracking_failure":
                "tracking_failure",
        },
    )

    # --------------------------------------------------------
    # Failure termination
    # --------------------------------------------------------

    graph.add_edge(
        "strategy_failure",
        END,
    )

    graph.add_edge(
        "accuracy_failure",
        END,
    )

    graph.add_edge(
        "refinement_limit_failure",
        END,
    )

    graph.add_edge(
        "optimization_failure",
        END,
    )

    graph.add_edge(
        "tracking_failure",
        END,
    )

    # --------------------------------------------------------
    # Refinement
    # --------------------------------------------------------

    graph.add_edge(
        "refine",
        "prune",
    )

    # --------------------------------------------------------
    # Export → Report → Approval → Deployment
    # --------------------------------------------------------

    graph.add_edge(
        "export",
        "report",
    )

    graph.add_edge(
        "report",
        "approval",
    )

    graph.add_edge(
        "approval",
        "deploy",
    )

    graph.add_edge(
        "deploy",
        END,
    )

    # --------------------------------------------------------
    # PHASE 23.2
    # Persistent SQLite checkpointer
    # --------------------------------------------------------

    return graph.compile(
        checkpointer=checkpointer
    )


workflow = build_workflow()