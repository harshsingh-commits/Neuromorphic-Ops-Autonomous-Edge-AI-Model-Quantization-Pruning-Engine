from __future__ import annotations

from pathlib import Path

import torch

from demo.demo_model import DemoCNN


def main():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "demo" / "demo_model.pt"

    model = DemoCNN()
    model.eval()

    sample = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():
        output = model(sample)

    print("Model created successfully")
    print(f"Input shape : {tuple(sample.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(
        f"Parameters  : "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    torch.save(model, model_path)

    print(f"\nSaved model: {model_path}")


if __name__ == "__main__":
    main()