import torch
import torch.nn as nn
from model.models.model_base import BaseTimeSeriesModel

class Transformer1D_V2(BaseTimeSeriesModel):
    """
    Transformer1D_V2
    - CLS Token
    - Learnable Positional Embedding
    - Pre-Norm + GELU
    - Designed for time-series classification / alpha modeling
    """

    MODEL_TYPE = "transformer"
    MODEL_VERSION = 2

    supports_lengths = False  # 当前实现不支持变长序列

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000,
        **kwargs,
    ):
        super().__init__()

        # ---------- 兼容未来参数 ----------
        if kwargs:
            print(f"[Transformer1D_V2] Ignored kwargs: {list(kwargs.keys())}")

        # ---------- 保存架构参数（用于 meta）----------
        self.input_size = input_size
        self.n_classes = n_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_p = dropout
        self.max_len = max_len

        # ---------- 特征投影 ----------
        self.input_proj = nn.Linear(input_size, d_model)

        # ---------- [CLS] Token ----------
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ---------- Learnable Positional Embedding ----------
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len + 1, d_model)
        )

        # ---------- Transformer Encoder ----------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-Norm
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)

        # ---------- Classification Head ----------
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    # =========================================================
    # forward
    # =========================================================
    def forward(self, x):
        """
        x: [B, T, F]
        """
        b, s, _ = x.shape

        # 1) input projection
        x = self.input_proj(x)  # [B, T, d_model]

        # 2) prepend CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, T+1, d_model]

        # 3) add positional embedding
        x = x + self.pos_embedding[:, : (s + 1)]
        x = self.dropout(x)

        # 4) transformer encoder
        x = self.transformer_encoder(x)

        # 5) CLS output
        cls_out = x[:, 0, :]  # [B, d_model]

        # 6) classifier
        return self.fc(cls_out)

    # =========================================================
    # meta / checkpoint
    # =========================================================
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
            "dropout": self.dropout_p,
            "max_len": self.max_len,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        """
        Rebuild model from meta + checkpoint.
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
