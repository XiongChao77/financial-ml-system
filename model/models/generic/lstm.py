import torch
import torch.nn as nn
from model.models.model_base import BaseTimeSeriesModel
"""
LSTM1D (Version A) — Trading-Oriented / Alpha-Preserving Model

Design Philosophy
-----------------
This version is intentionally kept SIMPLE and LOW-REGULARIZATION.

Core idea:
    In quantitative trading, profitability is often driven by
    *strong tail predictions* rather than average classification accuracy.

Key Characteristics
-------------------
- Pure (Bi-)LSTM feature extractor
- Uses the final hidden state (h_n) directly for classification
- Minimal classifier head: Dropout + Linear
- No Batch Normalization, no extra non-linear projection

Why this matters in trading:
----------------------------
- Preserves the amplitude of strong signals (logit tails)
- Allows confident predictions to remain confident
- Often produces better ranking quality and stronger PnL
- Empirically observed to outperform more regularized models
  in backtesting, even if ML metrics are slightly worse

Recommended Use Cases
---------------------
- Alpha generation / signal modeling
- Ranking-based strategies (long-short, top-k selection)
- Small-to-medium datasets with strong engineered features
- When backtest PnL is the primary objective

Caution
-------
- Higher risk of overfitting in pure ML metrics
- Must be validated with strict walk-forward testing
- Should be combined with trading-side risk control
  (position sizing, clipping, turnover constraints)
"""


class LSTM1D_V1(BaseTimeSeriesModel):
    """
    Bi-directional LSTM supported version.
    Input: [B, T, F]   Output: logits [B, n_classes]
    """
    MODEL_TYPE = "lstm"
    MODEL_VERSION = 1
    def __init__(self, input_size, hidden_size=64, num_layers=2, n_classes=3, p_drop=0.3, bidirectional=True, **kwargs,):
        super().__init__()
        if len(kwargs) > 0:
            print(f"[BetterLSTM1D] Ignored kwargs: {list(kwargs.keys())}")
        # ===== Save architecture info (for meta) =====
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.p_drop = p_drop
        
        # 1. LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop if num_layers > 1 else 0,
            bidirectional=bidirectional  # Change 1: enable bi-directional
        )
        
        # 2. Dropout layer
        self.dropout = nn.Dropout(p_drop)
        
        # 3. Fully-connected classifier head
        # Change 2: bi-directional LSTM doubles the feature dimension
        fc_input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_dim, n_classes)

    def forward(self, x):                  # x: [B,T,F]
        # output: [B, T, num_directions * hidden_size]
        # h_n:    [num_layers * num_directions, B, hidden_size]
        output, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            # Change 3: hidden state handling for bi-directional
            # h_n shape: [num_layers * 2, B, hidden_size]
            # In this layout:
            # h_n[-2] is the last layer's forward final state
            # h_n[-1] is the last layer's backward final state
            
            # Concatenate forward/backward states to get a full-context vector
            last_hidden_state = torch.cat((h_n[-2], h_n[-1]), dim=1) # -> [B, hidden_size * 2]
        else:
            # Uni-directional: take the last layer state directly
            last_hidden_state = h_n[-1] # -> [B, hidden_size]
        
        out = self.dropout(last_hidden_state)
        return self.fc(out)
    
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
        Disable dropout for inference / backtesting.
        """
        model = cls(
            input_size=state["channel"],
            hidden_size=meta["lstm_hidden"],
            num_layers=meta["lstm_layers"],
            n_classes=len(meta["classes"]),
            bidirectional=meta["bidirectional"],
            p_drop=0.0,   # 🔥 Disable for inference
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)