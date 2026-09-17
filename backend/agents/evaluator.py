
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from backend.models.schemas import WorkflowState
from backend.services.metrics import (
    benchmark_latency,
    measure_latency,
    measure_model_memory,
)
from backend.services.model_io import load_model


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_EVAL_DATASET = "cifar10"
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_EVAL_MAX_SAMPLES = 1000

DEFAULT_BENCHMARK_BATCH_SIZES = (1, 8, 16)


# ============================================================
# DATASET
# ============================================================

def _build_cifar10_loader(
    data_root: str | Path = "data",
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    max_samples: int = DEFAULT_EVAL_MAX_SAMPLES,
) -> DataLoader:
    """
    Build the CIFAR-10 test dataloader.
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform,
    )

    if (
        max_samples > 0
        and max_samples < len(dataset)
    ):
        dataset = torch.utils.data.Subset(
            dataset,
            list(range(max_samples)),
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


# ============================================================
# ACCURACY
# ============================================================

def _evaluate_model_accuracy(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str = "cpu",
) -> float:
    """
    Calculate classification accuracy as a percentage.
    """

    model = model.to(device).eval()

    correct = 0
    total = 0

    with torch.inference_mode():

        for inputs, targets in data_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            if not isinstance(
                outputs,
                torch.Tensor,
            ):
                raise RuntimeError(
                    "Model output must be a Tensor "
                    "for classification evaluation."
                )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += targets.numel()

    if total == 0:
        raise RuntimeError(
            "Evaluation dataset contains no samples."
        )

    return (
        correct / total
    ) * 100.0


def evaluate_accuracy(
    state: WorkflowState,
) -> dict[str, Any]:
    """
    Evaluate original and optimized model accuracy.

    CIFAR-10 is used when configured.
    """

    dataset_name = str(
        state.get(
            "evaluation_dataset",
            "",
        )
    ).strip().lower()

    # --------------------------------------------------------
    # Compatibility/reference mode
    # --------------------------------------------------------

    if not dataset_name:

        baseline = state.get(
            "original_accuracy"
        )

        if baseline is None:
            return {
                "optimized_accuracy": None,
                "original_accuracy": None,
                "accuracy_drop": 0.0,
                "accuracy_status": "unavailable",
                "accuracy_reason": (
                    "No evaluation dataset or baseline "
                    "accuracy was supplied."
                ),
            }

        baseline = float(
            baseline
        )

        return {
            "optimized_accuracy": baseline,
            "original_accuracy": baseline,
            "accuracy_drop": 0.0,
            "accuracy_status": "reference",
            "accuracy_reason": (
                "No evaluation dataset was configured; "
                "the supplied baseline accuracy is used "
                "as the reference value."
            ),
        }

    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    if dataset_name != DEFAULT_EVAL_DATASET:

        return {
            "optimized_accuracy": None,
            "original_accuracy": None,
            "accuracy_drop": 0.0,
            "accuracy_status": "unsupported",
            "accuracy_reason": (
                f"Unsupported evaluation dataset: "
                f"{dataset_name}"
            ),
        }

    # --------------------------------------------------------
    # Model paths
    # --------------------------------------------------------

    original_model_path = state.get(
        "original_model_path"
    )

    optimized_model_path = state.get(
        "model_path"
    )

    if not original_model_path:

        return {
            "optimized_accuracy": None,
            "original_accuracy": None,
            "accuracy_drop": 0.0,
            "accuracy_status": "unavailable",
            "accuracy_reason": (
                "Original model path is missing."
            ),
        }

    if not optimized_model_path:

        return {
            "optimized_accuracy": None,
            "original_accuracy": None,
            "accuracy_drop": 0.0,
            "accuracy_status": "unavailable",
            "accuracy_reason": (
                "Optimized model path is missing."
            ),
        }

    # --------------------------------------------------------
    # Real evaluation
    # --------------------------------------------------------

    try:

        data_root = state.get(
            "evaluation_data_root",
            "data",
        )

        batch_size = int(
            state.get(
                "evaluation_batch_size",
                DEFAULT_EVAL_BATCH_SIZE,
            )
        )

        max_samples = int(
            state.get(
                "evaluation_max_samples",
                DEFAULT_EVAL_MAX_SAMPLES,
            )
        )

        loader = _build_cifar10_loader(
            data_root=data_root,
            batch_size=batch_size,
            max_samples=max_samples,
        )

        original_model = load_model(
            original_model_path
        )

        optimized_model = load_model(
            optimized_model_path
        )

        original_accuracy = (
            _evaluate_model_accuracy(
                original_model,
                loader,
            )
        )

        optimized_accuracy = (
            _evaluate_model_accuracy(
                optimized_model,
                loader,
            )
        )

        accuracy_drop = (
            original_accuracy
            - optimized_accuracy
        )

        if max_samples > 0:

            evaluation_samples = min(
                max_samples,
                len(loader.dataset),
            )

        else:

            evaluation_samples = len(
                loader.dataset
            )

        return {
            "optimized_accuracy": optimized_accuracy,
            "original_accuracy": original_accuracy,
            "accuracy_drop": accuracy_drop,
            "accuracy_status": "success",
            "accuracy_reason": (
                "Accuracy measured on the CIFAR-10 "
                "test dataset."
            ),
            "evaluation_samples": evaluation_samples,
        }

    except Exception as exc:

        return {
            "optimized_accuracy": None,
            "original_accuracy": None,
            "accuracy_drop": 0.0,
            "accuracy_status": "failed",
            "accuracy_reason": str(exc),
        }


# ============================================================
# STANDARD LATENCY
# ============================================================

def evaluate_latency(
    state: WorkflowState,
) -> dict[str, Any]:
    """
    Measure standard optimized-model CPU inference latency.

    The Phase 19A multi-batch benchmark is handled separately
    by evaluate_benchmark().
    """

    model_path = state.get(
        "model_path"
    )

    if not model_path:

        return {
            "latency_ms": None,
            "latency_improvement_percent": None,
            "latency_status": "unavailable",
        }

    try:

        model = load_model(
            model_path
        )

        latency = measure_latency(
            model
        )

        baseline = state.get(
            "baseline_latency_ms"
        )

        improvement = None

        if (
            isinstance(
                baseline,
                (int, float),
            )
            and baseline > 0
        ):

            improvement = (
                (baseline - latency)
                / baseline
            ) * 100.0

        return {
            "latency_ms": float(
                latency
            ),
            "latency_improvement_percent": (
                improvement
            ),
            "latency_status": "success",
        }

    except Exception as exc:

        return {
            "latency_ms": None,
            "latency_improvement_percent": None,
            "latency_status": "failed",
            "latency_error": str(exc),
        }


# ============================================================
# PHASE 19A BENCHMARK
# ============================================================

def evaluate_benchmark(
    state: WorkflowState,
) -> dict[str, Any]:
    """
    Run the Phase 19A multi-batch CPU latency benchmark.

    Both original and optimized models are tested using:
    - Batch size 1
    - Batch size 8
    - Batch size 16
    - warm-up runs
    - repeated timed runs
    - median latency
    """

    original_model_path = state.get(
        "original_model_path"
    )

    optimized_model_path = state.get(
        "model_path"
    )

    batch_sizes = (
        DEFAULT_BENCHMARK_BATCH_SIZES
    )

    empty_result = {
        "benchmark_batch_sizes": [
            int(batch_size)
            for batch_size
            in batch_sizes
        ],
        "benchmark_latency_ms": {},
        "benchmark_baseline_latency_ms": {},
        "benchmark_latency_improvement_percent": {},
    }

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not original_model_path:
        return empty_result

    if not optimized_model_path:
        return empty_result

    try:

        # ----------------------------------------------------
        # Original model
        # ----------------------------------------------------

        original_model = load_model(
            original_model_path
        )

        # ----------------------------------------------------
        # Optimized model
        # ----------------------------------------------------

        optimized_model = load_model(
            optimized_model_path
        )

        # ----------------------------------------------------
        # Baseline benchmark
        # ----------------------------------------------------

        baseline_results = benchmark_latency(
            original_model,
            batch_sizes=batch_sizes,
        )

        # ----------------------------------------------------
        # Optimized benchmark
        # ----------------------------------------------------

        optimized_results = benchmark_latency(
            optimized_model,
            batch_sizes=batch_sizes,
        )

        # ----------------------------------------------------
        # Improvement calculation
        # ----------------------------------------------------

        improvement_results: dict[
            str,
            float | None,
        ] = {}

        for batch_size in batch_sizes:

            key = str(
                batch_size
            )

            baseline = (
                baseline_results.get(
                    key
                )
            )

            optimized = (
                optimized_results.get(
                    key
                )
            )

            if (
                isinstance(
                    baseline,
                    (int, float),
                )
                and isinstance(
                    optimized,
                    (int, float),
                )
                and baseline > 0
            ):

                improvement_results[key] = (
                    (
                        baseline
                        - optimized
                    )
                    / baseline
                ) * 100.0

            else:

                improvement_results[key] = None

        return {
            "benchmark_batch_sizes": [
                int(batch_size)
                for batch_size
                in batch_sizes
            ],

            "benchmark_latency_ms": (
                optimized_results
            ),

            "benchmark_baseline_latency_ms": (
                baseline_results
            ),

            "benchmark_latency_improvement_percent": (
                improvement_results
            ),
        }

    except Exception as exc:

        errors = list(
            state.get(
                "errors",
                [],
            )
        )

        errors.append(
            f"Phase 19A benchmark failed: {exc}"
        )

        return {
            **empty_result,
            "errors": errors,
        }


# ============================================================
# MEMORY
# ============================================================

def evaluate_memory(
    state: WorkflowState,
) -> dict[str, Any]:
    """
    Measure optimized model memory footprint and compare
    it with the original-model baseline.
    """

    model_path = state.get(
        "model_path"
    )

    if not model_path:

        return {
            "memory_usage_mb": None,
            "memory_reduction_percent": None,
            "memory_status": "unavailable",
        }

    try:

        model = load_model(
            model_path
        )

        memory = measure_model_memory(
            model
        )

        baseline = state.get(
            "baseline_memory_usage_mb"
        )

        reduction = None

        if (
            isinstance(
                baseline,
                (int, float),
            )
            and baseline > 0
        ):

            reduction = (
                (baseline - memory)
                / baseline
            ) * 100.0

        return {
            "memory_usage_mb": float(
                memory
            ),
            "memory_reduction_percent": (
                reduction
            ),
            "memory_status": "success",
        }

    except Exception as exc:

        return {
            "memory_usage_mb": None,
            "memory_reduction_percent": None,
            "memory_status": "failed",
            "memory_error": str(exc),
        }


# ============================================================
# MERGE EVALUATION
# ============================================================

def merge_evaluation(
    state: WorkflowState,
) -> dict[str, Any]:
    """
    Merge standard accuracy, latency, and memory metrics.

    Phase 19A benchmark data is merged separately by the
    merge_benchmark node in workflow.py.
    """

    accuracy_drop = state.get(
        "accuracy_drop"
    )

    if accuracy_drop is None:
        accuracy_drop = 0.0

    return {
        "accuracy_drop": float(
            accuracy_drop
        ),

        "metrics": {
            # ------------------------------------------------
            # Accuracy
            # ------------------------------------------------

            "accuracy": state.get(
                "optimized_accuracy"
            ),

            "original_accuracy": state.get(
                "original_accuracy"
            ),

            "accuracy_drop": float(
                accuracy_drop
            ),

            # ------------------------------------------------
            # Standard latency
            # ------------------------------------------------

            "baseline_latency_ms": state.get(
                "baseline_latency_ms"
            ),

            "latency_ms": state.get(
                "latency_ms"
            ),

            "latency_improvement_percent": state.get(
                "latency_improvement_percent"
            ),

            # ------------------------------------------------
            # Memory
            # ------------------------------------------------

            "baseline_memory_mb": state.get(
                "baseline_memory_usage_mb"
            ),

            "memory_mb": state.get(
                "memory_usage_mb"
            ),

            "memory_reduction_percent": state.get(
                "memory_reduction_percent"
            ),
        },

        "evaluation": {
            "dataset": state.get(
                "evaluation_dataset"
            ),

            "samples": state.get(
                "evaluation_samples"
            ),

            "accuracy_status": state.get(
                "accuracy_status"
            ),

            "accuracy_reason": state.get(
                "accuracy_reason"
            ),
        },

        "workflow_status": (
            "evaluation_completed"
        ),
    }

