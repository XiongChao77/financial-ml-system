from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TensorSplit(Dataset):
    """A time-ordered split shared by every training task."""

    X: torch.Tensor
    y: torch.Tensor
    returns: torch.Tensor

    def __post_init__(self) -> None:
        size = len(self.X)
        if len(self.y) != size or len(self.returns) != size:
            raise ValueError("X, y and returns must contain the same number of samples")

    @property
    def labels(self) -> np.ndarray:
        return self.y.detach().cpu().numpy()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index], self.returns[index]


@dataclass(frozen=True)
class DataSplits:
    train: TensorSplit
    validation: TensorSplit
    test: TensorSplit


@dataclass
class FitResult:
    selection_metric: str
    validation_score: float
    validation_metrics: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class PredictionResult:
    y_true: np.ndarray
    probabilities: np.ndarray
    loss: float | None = None

    @property
    def y_pred(self) -> np.ndarray:
        return self.probabilities.argmax(axis=1)
