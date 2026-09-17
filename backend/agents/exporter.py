from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from backend.models.schemas import WorkflowState
from backend.services.model_io import load_model
from backend.utils.config import OUTPUT_DIR


def _contains_quantized_module(model: nn.Module) -> bool:
    """
    Detect dynamically quantized modules that may not be
    supported by the current ONNX exporter path.
    """

    for module in model.modules():
        module_module = module.__class__.__module__.lower()
        module_name = module.__class__.__name__.lower()

        if "quantized" in module_module:
            return True

        if (
            "dynamic" in module_name
            and "linear" in module_name
        ):
            return True

    return False


def _find_float_model_path(
    state: WorkflowState,
) -> str | None:
    """
    Find a float/pruned model that can be exported to ONNX.
    """

    candidates = [
        state.get("pruned_model_path"),
        state.get("pre_quantized_model_path"),
        state.get("original_model_path"),
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)

    return None


def export_onnx(
    state: WorkflowState,
) -> dict[str, Any]:

    try:
        current_model_path = state.get(
            "model_path"
        )

        if not current_model_path:
            raise RuntimeError(
                "Model path is missing."
            )

        model = load_model(
            current_model_path
        )

        model = model.cpu().eval()

        export_model = model
        export_source = current_model_path

        # Dynamic INT8 Linear operators may not be supported
        # by the current ONNX exporter.
        if _contains_quantized_module(model):

            fallback_path = _find_float_model_path(
                state
            )

            if fallback_path:
                export_model = load_model(
                    fallback_path
                )

                export_model = (
                    export_model
                    .cpu()
                    .eval()
                )

                export_source = fallback_path

            else:
                raise RuntimeError(
                    "ONNX export requires a float/pruned model "
                    "because the dynamically quantized model "
                    "contains unsupported quantized operators."
                )

        output_dir = Path(
            OUTPUT_DIR
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            output_dir
            / "optimized_model.onnx"
        )

        sample = torch.randn(
            1,
            3,
            32,
            32,
        )

        with torch.inference_mode():

            torch.onnx.export(
                export_model,
                sample,
                str(output_path),
                input_names=["input"],
                output_names=["output"],
                dynamo=False,
                opset_version=17,
            )

        # Validate ONNX structure.
        onnx_model = onnx.load(
            str(output_path)
        )

        onnx.checker.check_model(
            onnx_model
        )

        # Validate ONNX Runtime inference.
        session = ort.InferenceSession(
            str(output_path),
            providers=[
                "CPUExecutionProvider"
            ],
        )

        input_name = (
            session
            .get_inputs()[0]
            .name
        )

        outputs = session.run(
            None,
            {
                input_name: sample.numpy()
            },
        )

        if not outputs:
            raise RuntimeError(
                "ONNX Runtime returned no outputs."
            )

        return {
            "onnx_path": str(output_path),
            "onnx_validation_status": True,
            "onnx_runtime_status": True,
            "export_status": "success",
            "export_source": export_source,
            "error": None,
        }

    except Exception as exc:

        return {
            "onnx_path": None,
            "onnx_validation_status": False,
            "onnx_runtime_status": False,
            "export_status": "failed",
            "export_source": None,
            "error": str(exc),
        }