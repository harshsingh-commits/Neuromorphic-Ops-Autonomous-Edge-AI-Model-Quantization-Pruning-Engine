from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from backend.config import MODEL_DIRECTORY, OUTPUT_DIRECTORY
from backend.utils.file_utils import validate_safe_path


logger = logging.getLogger(__name__)


def _validate_model_file(path: str | Path) -> Path:
    """
    Validate model path and ensure it stays inside an approved
    model/output directory.
    """
    p = Path(path)

    # --------------------------------------------------------
    # Path traversal protection
    # --------------------------------------------------------
    try:
        validate_safe_path(MODEL_DIRECTORY, p)
    except ValueError:
        try:
            validate_safe_path(OUTPUT_DIRECTORY, p)
        except ValueError as exc:
            raise ValueError(
                "Security error: model path is outside the allowed "
                "model directories."
            ) from exc

    # --------------------------------------------------------
    # File validation
    # --------------------------------------------------------
    if not p.exists():
        raise FileNotFoundError(
            "Model file does not exist."
        )

    if not p.is_file():
        raise ValueError(
            "Model path must point to a regular file."
        )

    if p.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError(
            "Unsupported model file type. Only .pt and .pth files are allowed."
        )

    if p.stat().st_size <= 0:
        raise ValueError(
            "Model file is empty."
        )

    return p


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    """
    Extract a state_dict from supported safe checkpoint formats.

    Supported:
        1. Raw state_dict
        2. {"state_dict": ...}
        3. {"model_state_dict": ...}
        4. {"model_state": ...}

    This function does not instantiate arbitrary Python classes.
    """

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unsupported checkpoint format. "
            "Expected a PyTorch state_dict or a checkpoint dictionary."
        )

    # --------------------------------------------------------
    # Direct state_dict
    # --------------------------------------------------------
    direct_state_dict = checkpoint

    # Typical state_dict contains tensor values.
    if direct_state_dict and all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in direct_state_dict.items()
    ):
        return direct_state_dict

    # --------------------------------------------------------
    # Nested state_dict formats
    # --------------------------------------------------------
    for key in (
        "state_dict",
        "model_state_dict",
        "model_state",
    ):
        candidate = checkpoint.get(key)

        if isinstance(candidate, dict):
            if all(
                isinstance(k, str) and isinstance(v, torch.Tensor)
                for k, v in candidate.items()
            ):
                return candidate

    raise ValueError(
        "Checkpoint does not contain a valid state_dict."
    )


def load_pytorch_model(
    path: str | Path,
    model_class: nn.Module | None = None,
) -> nn.Module:
    """
    Safely load a PyTorch model from a state_dict-based checkpoint.

    Security model:
        - Validates filesystem location.
        - Accepts only .pt/.pth regular files.
        - Loads only with weights_only=True.
        - Never falls back to weights_only=False.
        - Never instantiates arbitrary classes from a checkpoint.
        - Requires a trusted model architecture to be supplied through
          model_class when the artifact contains only weights.

    Args:
        path:
            Path to the .pt/.pth model checkpoint.

        model_class:
            Optional already-instantiated trusted nn.Module.
            Required when loading a state_dict checkpoint.

    Returns:
        The trusted model architecture populated with the checkpoint weights.

    Raises:
        FileNotFoundError:
            When the model file does not exist.

        ValueError:
            When path validation or checkpoint validation fails.

        RuntimeError:
            When PyTorch cannot safely deserialize the checkpoint or
            the state_dict cannot be loaded into the supplied model.
    """

    p = _validate_model_file(path)

    logger.info(
        "Attempting secure PyTorch model load: %s",
        p,
    )

    # --------------------------------------------------------
    # SECURE DESERIALIZATION
    # --------------------------------------------------------
    #
    # weights_only=True restricts loading to supported safe
    # tensor/primitive/dictionary structures.
    #
    # IMPORTANT:
    # Do NOT fall back to weights_only=False for untrusted files.
    # --------------------------------------------------------
    try:
        checkpoint = torch.load(
            p,
            map_location="cpu",
            weights_only=True,
        )

    except Exception as exc:
        logger.warning(
            "Secure PyTorch checkpoint loading failed for %s: %s",
            p,
            type(exc).__name__,
        )

        raise RuntimeError(
            "Unable to safely load the PyTorch model checkpoint. "
            "The file may be corrupted, incompatible, or may contain "
            "unsupported serialized objects. Upload a state_dict-based "
            "PyTorch checkpoint."
        ) from exc

    # --------------------------------------------------------
    # FULL SERIALIZED MODEL REJECTION
    # --------------------------------------------------------
    #
    # A full nn.Module should not be accepted from an untrusted
    # checkpoint. The safe workflow is:
    #
    #   trusted architecture
    #        +
    #   safe state_dict
    #        ↓
    #      model
    # --------------------------------------------------------
    if isinstance(checkpoint, nn.Module):
        logger.error(
            "Unexpected serialized nn.Module detected in checkpoint: %s",
            p,
        )

        raise ValueError(
            "Full serialized PyTorch modules are not accepted by the "
            "secure loader. Save and upload the model state_dict instead."
        )

    # --------------------------------------------------------
    # EXTRACT STATE DICTIONARY
    # --------------------------------------------------------
    try:
        state_dict = _extract_state_dict(checkpoint)

    except ValueError:
        logger.warning(
            "Unsupported checkpoint structure detected: %s",
            p,
        )
        raise

    # --------------------------------------------------------
    # ARCHITECTURE REQUIRED
    # --------------------------------------------------------
    if model_class is None:
        raise ValueError(
            "Checkpoint contains model weights but no trusted model "
            "architecture was provided. Provide an initialized nn.Module "
            "and load the state_dict into it."
        )

    # --------------------------------------------------------
    # LOAD WEIGHTS INTO TRUSTED ARCHITECTURE
    # --------------------------------------------------------
    try:
        missing_keys, unexpected_keys = model_class.load_state_dict(
            state_dict,
            strict=False,
        )

    except Exception as exc:
        logger.exception(
            "Failed to load state_dict into trusted model architecture."
        )

        raise RuntimeError(
            "Failed to load model weights into the provided model "
            "architecture."
        ) from exc

    # --------------------------------------------------------
    # CHECK STATE-DICT COMPATIBILITY
    # --------------------------------------------------------
    if missing_keys:
        logger.warning(
            "Missing model keys detected: %d",
            len(missing_keys),
        )

    if unexpected_keys:
        logger.warning(
            "Unexpected checkpoint keys detected: %d",
            len(unexpected_keys),
        )

    # --------------------------------------------------------
    # CPU + EVAL MODE
    # --------------------------------------------------------
    model_class.to(torch.device("cpu"))
    model_class.eval()

    logger.info(
        "Secure PyTorch model loading completed successfully: %s",
        p,
    )

    return model_class