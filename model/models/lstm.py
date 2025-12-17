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
    双向 LSTM 支持版本
    输入: [B, T, F]   输出: logits [B, n_classes]
    """
    MODEL_TYPE = "lstm"
    MODEL_VERSION = 1
    def __init__(self, input_size, hidden_size=64, num_layers=2, n_classes=3, p_drop=0.3, bidirectional=True, **kwargs,):
        super().__init__()
        if len(kwargs) > 0:
            print(f"[BetterLSTM1D] Ignored kwargs: {list(kwargs.keys())}")
        # ===== 保存架构信息（用于 meta）=====
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.p_drop = p_drop
        
        # 1. LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=p_drop if num_layers > 1 else 0,
            bidirectional=bidirectional  # 【修改点1】开启双向
        )
        
        # 2. Dropout 层
        self.dropout = nn.Dropout(p_drop)
        
        # 3. 全连接分类层
        # 【修改点2】如果是双向，LSTM 输出的特征维度会翻倍
        fc_input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_dim, n_classes)

    def forward(self, x):                  # x: [B,T,F]
        # output: [B, T, num_directions * hidden_size]
        # h_n:    [num_layers * num_directions, B, hidden_size]
        output, (h_n, c_n) = self.lstm(x)
        
        if self.bidirectional:
            # 【修改点3】双向时的隐藏状态处理
            # h_n 的形状是 [num_layers * 2, B, hidden_size]
            # 这种布局下：
            # h_n[-2] 是最后一层的前向 (Forward) 最终状态
            # h_n[-1] 是最后一层的后向 (Backward) 最终状态
            
            # 我们将这两个方向的状态拼接起来，得到包含完整上下文的向量
            last_hidden_state = torch.cat((h_n[-2], h_n[-1]), dim=1) # -> [B, hidden_size * 2]
        else:
            # 单向时，直接取最后一层的状态
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
        推理 / 回测时关闭 dropout。
        """
        model = cls(
            input_size=state["channel"],
            hidden_size=meta["lstm_hidden"],
            num_layers=meta["lstm_layers"],
            n_classes=len(meta["classes"]),
            bidirectional=meta["bidirectional"],
            p_drop=0.0,   # 🔥 推理时关闭
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)