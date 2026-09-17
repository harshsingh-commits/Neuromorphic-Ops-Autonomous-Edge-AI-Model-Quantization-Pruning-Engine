from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from backend.models.schemas import WorkflowState
from backend.services.model_io import load_model
from backend.services.metrics import model_size_mb, parameter_count

logger = logging.getLogger(__name__)


def _prunable_modules(model: nn.Module) -> list[nn.Module]:
    supported = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
    )

    return [
        module
        for module in model.modules()
        if isinstance(module, supported)
    ]


def _calculate_sparsity(model: nn.Module) -> float:
    total = 0
    zeros = 0

    for parameter in model.parameters():
        values = parameter.detach()
        total += values.numel()
        zeros += torch.count_nonzero(values == 0).item()

    if total == 0:
        return 0.0

    return (zeros / total) * 100.0


def apply_pruning(state: WorkflowState) -> dict[str, Any]:
    """
    Apply L1 unstructured pruning to the current model.
    """

    model_path = state.get("model_path")

    if not model_path:
        return {
            "workflow_status": "failed",
            "errors": [
                *state.get("errors", []),
                "Pruning failed: model_path is missing.",
            ],
        }

    try:
        model = load_model(model_path)

        ratio = float(
            state.get("pruning_ratio", 0.30)
        )

        ratio = min(
            max(ratio, 0.05),
            0.90,
        )

        modules = _prunable_modules(model)

        if not modules:
            return {
                "workflow_status": "failed",
                "errors": [
                    *state.get("errors", []),
                    "No supported layers found for pruning.",
                ],
            }

        parameters_to_prune = [
            (module, "weight")
            for module in modules
            if hasattr(module, "weight")
        ]

        for module, parameter_name in parameters_to_prune:
            prune.l1_unstructured(
                module,
                name=parameter_name,
                amount=ratio,
            )

        # Remove pruning reparameterization.
        for module, parameter_name in parameters_to_prune:
            prune.remove(
                module,
                parameter_name,
            )

        output_dir = Path(
            state.get(
                "output_directory",
                "outputs",
            )
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir / "pruned_model.pt"
        )

        torch.save(
            model,
            output_path,
        )

        sparsity = _calculate_sparsity(model)

        logger.info(
            "Pruning completed: ratio=%s sparsity=%.2f%%",
            ratio,
            sparsity,
        )

        return {
            "model": model,
            "model_path": str(output_path),
            "pruned_model_path": str(output_path),
            "optimized_model_path": str(output_path),
            "optimized_parameters": parameter_count(model),
            "optimized_size_mb": model_size_mb(
                output_path
            ),
            "pruning_ratio": ratio,
            "sparsity": sparsity,
            "pruning_status": "success",
            "workflow_status": "pruned",
        }

    except Exception as exc:
        logger.exception("Pruning failed.")

        return {
            "workflow_status": "failed",
            "pruning_status": "failed",
            "errors": [
                *state.get("errors", []),
                f"Pruning failed: {exc}",
            ],
        }