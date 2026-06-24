import torch
import torch.nn as nn
import torch.nn.functional as F

from model.models.model_base import BaseTimeSeriesModel


class LogisticRegressionTS_V1(BaseTimeSeriesModel):
    """
    Logistic Regression for time-series window classification.

    Input:
        x: [B, T, F]
    """

    MODEL_TYPE = "logistic_regression"
    MODEL_VERSION = 1

    def __init__(
        self,
        input_size: int,
        n_classes: int = 3,
        window: int = 1,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"[LogisticRegressionTS_V1] Ignored kwargs: {list(kwargs.keys())}")

        self.input_size = int(input_size)
        self.n_classes = int(n_classes)
        self.window = int(window)

        linear_in = self.input_size * self.window

        self.classifier = nn.Linear(linear_in, self.n_classes)

    def forward(self, x: torch.Tensor, return_fused: bool = False):
        """
        x: [B, T, F]
        """

        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, T, F], got {tuple(x.shape)}")

        x = x.reshape(x.size(0), -1)

        logits = self.classifier(x)

        if return_fused:
            probs = F.softmax(logits, dim=1)
            return logits, probs

        return logits

    def export_meta(self, **extra) -> dict:
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "input_size": self.input_size,
            "n_classes": self.n_classes,
            "window": self.window,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        model = cls(
            input_size=meta.get("input_size", state.get("channel")),
            n_classes=len(meta["classes"]),
            window=meta.get("window", 1),
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)