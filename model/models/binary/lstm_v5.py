import torch
import torch.nn as nn
from model.models.model_base import BaseTimeSeriesModel

"""
LSTM1D_V5 — Quant-Friendly / Tail-Preserving Regularized Model

Design goal
-----------
V3 is designed as a middle ground between:

- V1: simple, low-regularization, tail/ranking preserving
- V2: strongly regularized MLP head with BatchNorm + ReLU

For trading / alpha modeling, the classifier head should avoid destroying
strong-confidence tails too early, because ranking quality and extreme signals
can matter more than average classification accuracy.

Key choices
-----------
1. Same BiLSTM backbone as V1/V2.
2. Raw linear path is kept to preserve V1-style logits and signal amplitude.
3. A light normalized residual MLP is added as a correction path.
4. LayerNorm is used instead of BatchNorm:
   - independent of batch statistics
   - more stable for time-series / walk-forward inference
   - avoids batch-dependent score shifts
5. GELU is used instead of ReLU:
   - smoother nonlinearity
   - less aggressive information clipping
6. Dropout is kept moderate by default.

Forward structure
-----------------
    x [B, T, F]
        -> BiLSTM
        -> last hidden state feat [B, D]
        -> raw_head(feat)
        -> residual_scale * mlp_head(LayerNorm(feat))
        -> logits = raw_logits + residual_logits

Recommended use cases
---------------------
- Quant trading signal modeling
- Ranking / top-k / long-short selection
- When V1 backtest is good but validation is unstable
- When V2 validation is stable but backtest / CAGR is weaker
"""


class LSTM1D_V5(BaseTimeSeriesModel):
    """
    Bi-directional LSTM with a tail-preserving residual classifier head.

    Input:  x      [B, T, F]
    Output: logits [B, n_classes]
    """

    MODEL_TYPE = "lstm"
    MODEL_VERSION = 5

    supports_lengths = False  # Same as V1/V2: use h_n only

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_classes: int = 3,
        p_drop: float = 0.2,
        bidirectional: bool = True,
        residual_scale: float = 0.3,
        use_residual_mlp: bool = True,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"[LSTM1D_V5] Ignored kwargs: {list(kwargs.keys())}")

        # ===== Save architecture params for meta/checkpoint =====
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.bidirectional = bidirectional
        self.p_drop = p_drop
        self.residual_scale = residual_scale
        self.use_residual_mlp = use_residual_mlp

        # ===== LSTM backbone: same design as V1/V2 =====
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size

        # ===== V1-style raw path =====
        # This path directly maps LSTM features to logits and helps preserve
        # strong tail signals / ranking information.
        self.raw_dropout = nn.Dropout(p_drop)
        self.raw_head = nn.Linear(lstm_out_dim, n_classes)

        # ===== V3 residual correction path =====
        # This path adds mild nonlinear capacity, but does not replace the raw path.
        # LayerNorm is per-sample and does not depend on batch statistics.
        if use_residual_mlp:
            self.norm = nn.LayerNorm(lstm_out_dim)
            self.residual_head = nn.Sequential(
                nn.Linear(lstm_out_dim, hidden_size),
                nn.GELU(),
                nn.Dropout(p_drop),
                nn.Linear(hidden_size, n_classes),
            )
        else:
            self.norm = nn.Identity()
            self.residual_head = None

    # ============================================================
    # forward
    # ============================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F]
        """
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # h_n: [num_layers * 2, B, hidden_size]
            # h_n[-2]: last layer forward final state
            # h_n[-1]: last layer backward final state
            feat = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            feat = h_n[-1]

        # Tail-preserving raw logits, similar to V1.
        raw_logits = self.raw_head(self.raw_dropout(feat))

        # Optional normalized residual correction.
        if self.use_residual_mlp:
            residual_logits = self.residual_head(self.norm(feat))
            return raw_logits + self.residual_scale * residual_logits

        return raw_logits

    # ============================================================
    # meta / checkpoint support
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
            "n_classes": self.n_classes,
            "bidirectional": self.bidirectional,
            "p_drop": self.p_drop,
            "residual_scale": self.residual_scale,
            "use_residual_mlp": self.use_residual_mlp,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        """
        Rebuild model from meta + checkpoint.

        Dropout is disabled for inference/backtesting by constructing with p_drop=0.0.
        This is compatible because Dropout has no trainable parameters.
        """
        model = cls(
            input_size=meta.get("input_size", state.get("channel")),
            hidden_size=meta["lstm_hidden"],
            num_layers=meta["lstm_layers"],
            n_classes=len(meta.get("classes", range(meta.get("n_classes", 3)))),
            bidirectional=meta["bidirectional"],
            p_drop=0.0,
            residual_scale=meta.get("residual_scale", 0.3),
            use_residual_mlp=meta.get("use_residual_mlp", True),
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)
