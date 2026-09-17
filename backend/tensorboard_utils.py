from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path = "runs/neuromorphic_ops") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, Any],
    ) -> None:
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(
                    name,
                    value,
                    step,
                )

    def log_text(
        self,
        tag: str,
        text: str,
        step: int = 0,
    ) -> None:
        self.writer.add_text(
            tag,
            text,
            step,
        )

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()