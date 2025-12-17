import torch
import torch.nn as nn
from model.models.model_base import BaseTimeSeriesModel

"""
LSTM1D_V2 — ML-Robust / Regularized Model (System Version)

- BiLSTM backbone
- Strongly regularized MLP head
- BatchNorm + ReLU + Dropout
- Prioritizes stability and generalization over alpha tails
"""

class LSTM1D_V2(BaseTimeSeriesModel):
    """
    改进版双向 LSTM（V2）：
    - BatchNorm + MLP Head
    - 强正则化，适合 noisy / 小样本
    """

    MODEL_TYPE = "lstm"
    MODEL_VERSION = 2

    supports_lengths = False  # 与 V1 一致，只用 h_n

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_classes: int = 3,
        p_drop: float = 0.5,
        bidirectional: bool = True,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"[LSTM1D_V2] Ignored kwargs: {list(kwargs.keys())}")

        # ===== 保存架构参数（meta 用）=====
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.bidirectional = bidirectional
        self.p_drop = p_drop

        # ===== LSTM backbone =====
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size

        # ===== 强正则化分类头 =====
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),   # 🔥 稳定分布
            nn.ReLU(),
            nn.Dropout(p_drop),            # 🔥 强正则
            nn.Linear(hidden_size, n_classes),
        )

    # ============================================================
    # forward
    # ============================================================
    def forward(self, x):
        """
        x: [B, T, F]
        """
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            feat = h_n[-1]

        return self.classifier(feat)

    # ============================================================
    # meta / checkpoint
    # ============================================================
    def export_meta(self, **extra):
        """
        Export architecture-defining parameters.
        """
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "input_size": self.input_size,
            "lstm_hidden": self.hidden_size,
            "lstm_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "p_drop": self.p_drop,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        """
        Rebuild model from meta + checkpoint.
        推理阶段关闭 dropout。
        """
        model = cls(
            input_size=meta["input_size"],
            hidden_size=meta["lstm_hidden"],
            num_layers=meta["lstm_layers"],
            n_classes=len(meta["classes"]),
            bidirectional=meta["bidirectional"],
            p_drop=0.0,   # 🔥 推理 / 回测时关闭正则
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)
