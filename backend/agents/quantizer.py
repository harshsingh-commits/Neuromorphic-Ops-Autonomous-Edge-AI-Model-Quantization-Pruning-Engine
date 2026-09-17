from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _get_model_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0

    return path.stat().st_size / (1024 * 1024)


def _count_supported_modules(
    model: nn.Module,
) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(module, nn.Linear)
    )


def quantize_model(
    model: nn.Module,
    output_path: str | Path,
) -> dict[str, Any]:

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_model = (
        copy.deepcopy(model)
        .cpu()
        .eval()
    )

    supported_count = (
        _count_supported_modules(
            original_model
        )
    )

    if supported_count == 0:
        return {
            "quantization_type": "dynamic_int8",
            "quantization_status": "unsupported",
            "supported_module_count": 0,
            "optimized_model_path": None,
            "optimized_size_mb": 0.0,
            "error": (
                "No supported Linear modules found."
            ),
        }

    try:
        quantized_model = (
            torch.ao.quantization
            .quantize_dynamic(
                original_model,
                {nn.Linear},
                dtype=torch.qint8,
            )
        )

        torch.save(
            quantized_model,
            output_path,
        )

        size_mb = _get_model_size_mb(
            output_path
        )

        return {
            "quantization_type": "dynamic_int8",
            "quantization_status": "success",
            "supported_module_count": supported_count,
            "optimized_model_path": str(
                output_path
            ),
            "optimized_size_mb": size_mb,
            "error": None,
        }

    except Exception as exc:
        logger.exception(
            "Quantization failed."
        )

        return {
            "quantization_type": "dynamic_int8",
            "quantization_status": "failed",
            "supported_module_count": supported_count,
            "optimized_model_path": None,
            "optimized_size_mb": 0.0,
            "error": str(exc),
        }


def quantization_node(
    state: dict[str, Any],
) -> dict[str, Any]:

    model = state.get("model")

    # If model is not directly in state, load it.
    if model is None and state.get("model_path"):
        from backend.services.model_io import load_model

        model = load_model(
            state["model_path"]
        )

    if model is None:
        return {
            "workflow_status": "failed",
            "errors": [
                *state.get("errors", []),
                "Quantization failed: model is missing.",
            ],
        }

    # IMPORTANT:
    # Preserve the current float/pruned model path
    # before replacing model_path with the quantized artifact.
    pre_quantized_model_path = state.get(
        "model_path"
    )

    output_path = (
        Path(
            state.get(
                "output_directory",
                "outputs",
            )
        )
        / "quantized_model.pt"
    )

    result = quantize_model(
        model,
        output_path,
    )

    errors = list(
        state.get("errors", [])
    )

    if result["error"]:
        errors.append(
            result["error"]
        )

    return {
        "model": (
            model
            if result["quantization_status"]
            != "success"
            else None
        ),

        # Active optimized artifact
        "model_path": (
            result["optimized_model_path"]
            or state.get("model_path")
        ),

        # Quantized artifact
        "optimized_model_path": (
            result["optimized_model_path"]
        ),

        # IMPORTANT:
        # Path of the float/pruned model before quantization
        "pre_quantized_model_path": (
            pre_quantized_model_path
        ),

        "optimized_size_mb": (
            result["optimized_size_mb"]
        ),

        "quantization_type": (
            result["quantization_type"]
        ),

        "quantization_status": (
            result["quantization_status"]
        ),

        "errors": errors,

        "workflow_status": (
            "quantized"
            if result["quantization_status"]
            == "success"
            else "failed"
        ),
    }


def apply_quantization(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Compatibility wrapper used by the workflow.
    """
    return quantization_node(state)