from pathlib import Path

import torch
import torch.nn as nn

from backend.agents.quantizer import quantize_model


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def test_dynamic_int8_quantization(tmp_path: Path) -> None:
    model = TinyModel().eval()

    output_path = tmp_path / "quantized_model.pt"

    result = quantize_model(model, output_path)

    assert result["quantization_type"] == "dynamic_int8"
    assert result["quantization_status"] == "success"
    assert result["supported_module_count"] == 2
    assert output_path.exists()