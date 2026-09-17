from __future__ import annotations

import logging
from pathlib import Path

from torch import nn

from backend.models.schemas import WorkflowState
from backend.services.metrics import (
    layer_count,
    model_size_mb,
    parameter_count,
)
from backend.services.model_io import load_model

logger = logging.getLogger(__name__)


def analyze_model(
    state: WorkflowState,
) -> dict:
    """
    Analyze the uploaded PyTorch model
    and update workflow state.
    """

    model_path = state.get(
        "model_path"
    )

    if not model_path:
        return {
            "workflow_status": "failed",
            "errors": [
                *state.get("errors", []),
                "No model_path provided.",
            ],
        }

    try:
        model = load_model(
            model_path
        )

        total_params = parameter_count(
            model
        )

        trainable_params = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        modules = list(
            model.modules()
        )

        module_count = len(
            modules
        )

        supported_pruning_types = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
        )

        supported_quantization_types = (
            nn.Linear,
        )

        pruning_layers = sum(
            1
            for module in model.modules()
            if isinstance(
                module,
                supported_pruning_types,
            )
        )

        quantization_layers = sum(
            1
            for module in model.modules()
            if isinstance(
                module,
                supported_quantization_types,
            )
        )

        analysis = {
            "parameter_count": total_params,
            "trainable_parameters": trainable_params,
            "module_count": module_count,
            "layer_count": layer_count(model),
            "file_size_mb": model_size_mb(
                model_path
            ),
            "architecture_name": (
                model.__class__.__name__
            ),
            "supported_pruning_layers": (
                pruning_layers
            ),
            "supported_quantization_layers": (
                quantization_layers
            ),
        }

        logger.info(
            "Model analysis completed: %s",
            analysis,
        )

        return {
            "analysis": analysis,
            "analysis_summary": analysis,

            # Preserve original model
            "original_model_path": str(
                model_path
            ),

            "original_parameters": (
                total_params
            ),

            "original_size_mb": (
                analysis["file_size_mb"]
            ),

            "model_name": Path(
                model_path
            ).name,

            "model_type": (
                model.__class__.__name__
            ),

            "workflow_status": "analyzed",
        }

    except Exception as exc:
        logger.exception(
            "Model analysis failed."
        )

        return {
            "workflow_status": "failed",
            "errors": [
                *state.get(
                    "errors",
                    []
                ),
                f"Model analysis failed: {exc}",
            ],
        }