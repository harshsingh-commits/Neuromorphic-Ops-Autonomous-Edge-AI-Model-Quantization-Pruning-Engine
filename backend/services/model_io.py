from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
import torch
from torch import nn


class ModelIOError(RuntimeError):
    pass


def load_model(path: str | Path) -> nn.Module:
    try:
        loaded: Any = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ModelIOError(f"Could not load model: {exc}") from exc
    if isinstance(loaded, nn.Module):
        model = loaded
    elif isinstance(loaded, dict) and "model" in loaded and isinstance(loaded["model"], nn.Module):
        model = loaded["model"]
    else:
        raise ModelIOError("Checkpoint must contain a serialized torch.nn.Module or {'model': module}.")
    model.eval()
    return model


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model).cpu().eval()
