
from __future__ import annotations

import statistics
import time
from pathlib import Path

import psutil
import torch
from torch import nn


# ============================================================
# MODEL SIZE
# ============================================================

def model_size_mb(path: str | Path) -> float:
    """
    Return model file size in megabytes.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {file_path}"
        )

    return file_path.stat().st_size / (
        1024 * 1024
    )


# ============================================================
# PARAMETER COUNT
# ============================================================

def parameter_count(model: nn.Module) -> int:
    """
    Return total number of model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


# ============================================================
# LAYER COUNT
# ============================================================

def layer_count(model: nn.Module) -> int:
    """
    Count leaf modules that contain no child modules.
    """

    return sum(
        1
        for module in model.modules()
        if len(list(module.children())) == 0
    )


# ============================================================
# LATENCY
# ============================================================

def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    warmup_runs: int = 10,
    runs: int = 50,
) -> float:
    """
    Measure CPU inference latency.

    Benchmark methodology:
    - CPU inference
    - warm-up runs
    - repeated timed runs
    - median used as final latency
    """

    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs must be >= 0"
        )

    if runs <= 0:
        raise ValueError(
            "runs must be > 0"
        )

    model = model.cpu().eval()

    sample = torch.randn(
        input_shape,
        device="cpu",
    )

    with torch.inference_mode():

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        for _ in range(warmup_runs):
            model(sample)

        # ----------------------------------------------------
        # Timed runs
        # ----------------------------------------------------

        measurements: list[float] = []

        for _ in range(runs):

            start = time.perf_counter()

            model(sample)

            end = time.perf_counter()

            measurements.append(
                (end - start) * 1000.0
            )

    return float(
        statistics.median(measurements)
    )


# ============================================================
# MULTI-BATCH LATENCY BENCHMARK
# ============================================================

def benchmark_latency(
    model: nn.Module,
    batch_sizes: tuple[int, ...] = (1, 8, 16),
    input_channels: int = 3,
    input_height: int = 32,
    input_width: int = 32,
    warmup_runs: int = 10,
    runs: int = 50,
) -> dict[str, float]:
    """
    Benchmark CPU inference latency across multiple batch sizes.

    Default batch sizes:
        1, 8, 16

    For each batch size:
        1. Create representative input tensor.
        2. Run warm-up inference.
        3. Run repeated timed inference.
        4. Calculate median latency.

    Returns:
        {
            "1": latency_ms,
            "8": latency_ms,
            "16": latency_ms,
        }
    """

    if not batch_sizes:
        raise ValueError(
            "batch_sizes must not be empty."
        )

    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs must be >= 0"
        )

    if runs <= 0:
        raise ValueError(
            "runs must be > 0"
        )

    model = model.cpu().eval()

    results: dict[str, float] = {}

    with torch.inference_mode():

        for batch_size in batch_sizes:

            if batch_size <= 0:
                raise ValueError(
                    "batch_size must be > 0"
                )

            sample = torch.randn(
                (
                    batch_size,
                    input_channels,
                    input_height,
                    input_width,
                ),
                device="cpu",
            )

            # ------------------------------------------------
            # Warm-up
            # ------------------------------------------------

            for _ in range(warmup_runs):
                model(sample)

            # ------------------------------------------------
            # Benchmark
            # ------------------------------------------------

            measurements: list[float] = []

            for _ in range(runs):

                start = time.perf_counter()

                model(sample)

                end = time.perf_counter()

                measurements.append(
                    (end - start) * 1000.0
                )

            results[str(batch_size)] = float(
                statistics.median(
                    measurements
                )
            )

    return results


# ============================================================
# PROCESS MEMORY
# ============================================================

def memory_usage_mb() -> float:
    """
    Return current Python process RSS memory usage.

    This helper is kept for process-level monitoring.
    """

    process = psutil.Process()

    memory_bytes = (
        process.memory_info().rss
    )

    return memory_bytes / (
        1024 * 1024
    )


# ============================================================
# MODEL MEMORY FOOTPRINT
# ============================================================

def measure_model_memory(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
) -> float:
    """
    Estimate model inference memory footprint.

    Includes:
    - parameter memory
    - buffer memory
    - input tensor memory
    - output tensor memory
    """

    model = model.cpu().eval()

    # --------------------------------------------------------
    # Parameter memory
    # --------------------------------------------------------

    parameter_memory = sum(
        parameter.numel()
        * parameter.element_size()
        for parameter in model.parameters()
    )

    # --------------------------------------------------------
    # Buffer memory
    # --------------------------------------------------------

    buffer_memory = sum(
        buffer.numel()
        * buffer.element_size()
        for buffer in model.buffers()
    )

    # --------------------------------------------------------
    # Input memory
    # --------------------------------------------------------

    input_tensor = torch.randn(
        input_shape,
        device="cpu",
    )

    input_memory = (
        input_tensor.numel()
        * input_tensor.element_size()
    )

    # --------------------------------------------------------
    # Output memory
    # --------------------------------------------------------

    output_memory = 0

    with torch.inference_mode():

        output = model(
            input_tensor
        )

        if isinstance(
            output,
            torch.Tensor,
        ):
            output_memory = (
                output.numel()
                * output.element_size()
            )

    # --------------------------------------------------------
    # Total memory footprint
    # --------------------------------------------------------

    total_bytes = (
        parameter_memory
        + buffer_memory
        + input_memory
        + output_memory
    )

    return total_bytes / (
        1024 * 1024
    )

