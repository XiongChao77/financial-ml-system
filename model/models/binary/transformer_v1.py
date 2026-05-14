import torch
import torch.nn as nn
import math
from model.models.model_base import BaseTimeSeriesModel


# ============================================================
# Positional Encoding (component, not the model)
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d_model]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ============================================================
# Transformer1D — System-level model
# ============================================================
class Transformer1D_V1(BaseTimeSeriesModel):
    """
    Transformer Encoder for time-series classification.

    Input : [B, T, F]
    Output: [B, n_classes]
    """

    MODEL_TYPE = "transformer"
    MODEL_VERSION = 1

    supports_lengths = False  # Current implementation does not support variable lengths

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 5000,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"[Transformer1D_V1] Ignored kwargs: {list(kwargs.keys())}")

        # ===== Save architecture params (for meta) =====
        self.input_size = input_size
        self.n_classes = n_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_len = max_len

        # ===== input projection =====
        self.input_proj = nn.Linear(input_size, d_model)

        # ===== positional encoding =====
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        # ===== transformer encoder =====
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ===== classifier head =====
        self.fc = nn.Linear(d_model, n_classes)

    # ------------------------------------------------------------
    # forward
    # ------------------------------------------------------------
    def forward(self, x):
        """
        x: [B, T, F]
        """
        x = self.input_proj(x)          # [B, T, d_model]
        x = self.pos_encoder(x)         # add positional info
        x = self.transformer_encoder(x)

        # Global Average Pooling over time
        x = x.mean(dim=1)               # [B, d_model]
        return self.fc(x)

    # ------------------------------------------------------------
    # meta / checkpoint
    # ------------------------------------------------------------
    def export_meta(self, **extra):
        """
        Export architecture-defining parameters.
        """
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "input_size": self.input_size,
            "n_classes": self.n_classes,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "max_len": self.max_len,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        """
        Rebuild model from meta + checkpoint.
        In inference, dropout randomness is disabled (eval mode).
        """
        model = cls(
            input_size=meta["input_size"],
            n_classes=len(meta["classes"]),
            d_model=meta["d_model"],
            nhead=meta["nhead"],
            num_layers=meta["num_layers"],
            dim_feedforward=meta["dim_feedforward"],
            dropout=meta["dropout"],
            max_len=meta["max_len"],
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)
