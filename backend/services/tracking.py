from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from backend.utils.config import OUTPUT_DIR


class OptimizationTracker:
    """
    Lightweight experiment tracker for optimization runs.

    Tracks:
    - Run ID
    - Timestamp
    - Random seed
    - Model SHA-256 hash
    - Optimization configuration
    - Core metrics
    - Benchmark metrics
    - Iteration metadata
    """

    def __init__(
        self,
        run_id: str,
        random_seed: int = 42,
    ) -> None:
        self.run_id = run_id
        self.random_seed = int(random_seed)
        self.timestamp = datetime.now(
            timezone.utc
        ).isoformat()

        self.run_dir = (
            OUTPUT_DIR
            / "tracking"
            / run_id
        )

        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.writer = SummaryWriter(
            log_dir=OUTPUT_DIR
            / "tensorboard"
            / run_id
        )

        self.metadata_path = (
            self.run_dir
            / "metadata.json"
        )

    # ========================================================
    # MODEL HASH
    # ========================================================

    @staticmethod
    def calculate_model_hash(
        model_path: str | Path | None,
    ) -> str | None:
        """
        Calculate SHA-256 hash for the model artifact.
        """

        if not model_path:
            return None

        path = Path(model_path)

        if not path.exists() or not path.is_file():
            return None

        sha256 = hashlib.sha256()

        with path.open(
            "rb"
        ) as model_file:
            for chunk in iter(
                lambda: model_file.read(
                    1024 * 1024
                ),
                b"",
            ):
                sha256.update(chunk)

        return sha256.hexdigest()

    # ========================================================
    # OPTIMIZATION CONFIG
    # ========================================================

    @staticmethod
    def extract_config(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract optimization configuration from workflow state.
        """

        return {
            "strategy": state.get(
                "optimization_strategy"
            ),
            "pruning_ratio": state.get(
                "pruning_ratio"
            ),
            "quantization_type": state.get(
                "quantization_type"
            ),
            "accuracy_threshold": state.get(
                "accuracy_threshold"
            ),
            "evaluation_dataset": state.get(
                "evaluation_dataset"
            ),
            "evaluation_batch_size": state.get(
                "evaluation_batch_size"
            ),
            "evaluation_max_samples": state.get(
                "evaluation_max_samples"
            ),
        }

    # ========================================================
    # METADATA
    # ========================================================

    def build_metadata(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build complete experiment metadata.
        """

        model_path = (
            state.get("original_model_path")
            or state.get("model_path")
        )

        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "random_seed": self.random_seed,
            "model_hash": self.calculate_model_hash(
                model_path
            ),
            "model_path": model_path,
            "optimization_config": self.extract_config(
                state
            ),
            "iteration": state.get(
                "iteration",
                0,
            ),
        }

    # ========================================================
    # RECORD METADATA
    # ========================================================

    def record_metadata(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Save run metadata to metadata.json.
        """

        metadata = self.build_metadata(
            state
        )

        with self.metadata_path.open(
            "w",
            encoding="utf-8",
        ) as metadata_file:
            json.dump(
                metadata,
                metadata_file,
                indent=2,
                default=str,
            )

        return metadata

    # ========================================================
    # RECORD METRICS
    # ========================================================

    def record(
        self,
        state: dict[str, Any],
        step: int,
    ) -> None:
        """
        Record standard and benchmark metrics
        to TensorBoard.
        """

        scalar_metrics = {
            "performance/optimized_accuracy":
                state.get(
                    "optimized_accuracy"
                ),
            "performance/original_accuracy":
                state.get(
                    "original_accuracy"
                ),
            "performance/accuracy_drop":
                state.get(
                    "accuracy_drop"
                ),
            "performance/latency_ms":
                state.get(
                    "latency_ms"
                ),
            "performance/baseline_latency_ms":
                state.get(
                    "baseline_latency_ms"
                ),
            "performance/latency_improvement_percent":
                state.get(
                    "latency_improvement_percent"
                ),
            "performance/memory_usage_mb":
                state.get(
                    "memory_usage_mb"
                ),
            "performance/baseline_memory_mb":
                state.get(
                    "baseline_memory_usage_mb"
                ),
            "performance/memory_reduction_percent":
                state.get(
                    "memory_reduction_percent"
                ),
            "performance/compression_ratio":
                state.get(
                    "compression_ratio"
                ),
            "optimization/pruning_ratio":
                state.get(
                    "pruning_ratio"
                ),
            "optimization/sparsity":
                state.get(
                    "sparsity"
                ),
        }

        # ----------------------------------------------------
        # Standard metrics
        # ----------------------------------------------------

        for key, value in scalar_metrics.items():
            if isinstance(
                value,
                (int, float),
            ):
                self.writer.add_scalar(
                    key,
                    float(value),
                    step,
                )

        # ----------------------------------------------------
        # Benchmark metrics
        # ----------------------------------------------------

        batch_sizes = state.get(
            "benchmark_batch_sizes",
            [],
        )

        optimized_benchmark = state.get(
            "benchmark_latency_ms",
            {},
        )

        baseline_benchmark = state.get(
            "benchmark_baseline_latency_ms",
            {},
        )

        improvement_benchmark = state.get(
            "benchmark_latency_improvement_percent",
            {},
        )

        for batch_size in batch_sizes:
            key = str(batch_size)

            optimized_value = (
                optimized_benchmark.get(key)
            )

            baseline_value = (
                baseline_benchmark.get(key)
            )

            improvement_value = (
                improvement_benchmark.get(key)
            )

            if isinstance(
                optimized_value,
                (int, float),
            ):
                self.writer.add_scalar(
                    f"benchmark/optimized_latency_batch_{key}",
                    float(optimized_value),
                    step,
                )

            if isinstance(
                baseline_value,
                (int, float),
            ):
                self.writer.add_scalar(
                    f"benchmark/baseline_latency_batch_{key}",
                    float(baseline_value),
                    step,
                )

            if isinstance(
                improvement_value,
                (int, float),
            ):
                self.writer.add_scalar(
                    f"benchmark/improvement_batch_{key}",
                    float(improvement_value),
                    step,
                )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        self.record_metadata(
            state
        )

        self.writer.flush()

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:
        """
        Close the TensorBoard writer safely.
        """

        self.writer.flush()
        self.writer.close()