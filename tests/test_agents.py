from pathlib import Path
import torch
from torch import nn
from backend.agents.analyzer import analyze_model
from backend.agents.pruner import apply_pruning
from backend.agents.quantizer import apply_quantization
from backend.agents.evaluator import evaluate_accuracy, evaluate_latency, evaluate_memory
from backend.agents.exporter import export_onnx


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Flatten(), nn.Linear(4 * 30 * 30, 2))

    def forward(self, x):
        return self.features(x)


def make_model(tmp_path: Path) -> str:
    path = tmp_path / "model.pt"
    torch.save(TinyModel(), path)
    return str(path)


def test_analyzer(tmp_path):
    result = analyze_model({"model_path": make_model(tmp_path)})
    assert result["analysis"]["parameter_count"] > 0
    assert result["analysis"]["layer_count"] > 0


def test_pruning_and_quantization(tmp_path):
    path = make_model(tmp_path)
    state = {"model_path": path}
    pruned = apply_pruning(state)
    assert Path(pruned["model_path"]).exists()
    quantized = apply_quantization({"model_path": path})
    assert Path(quantized["model_path"]).exists()


def test_evaluation_and_onnx_export(tmp_path):
    path = make_model(tmp_path)
    state = {"model_path": path, "original_accuracy": 0.86, "iteration": 1}
    assert evaluate_accuracy(state)["optimized_accuracy"] > 0
    assert evaluate_latency(state)["latency_ms"] >= 0
    assert evaluate_memory(state)["memory_usage_mb"] > 0
    exported = export_onnx(state)
    assert Path(exported["onnx_path"]).exists()
