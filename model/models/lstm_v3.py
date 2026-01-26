import torch,json
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.models.model_base import BaseTimeSeriesModel

class AttentionPooling(nn.Module):
    """
    Additive attention pooling over time.
    Given sequence features [B, T, D], produce pooled vector [B, D].
    Supports mask (True for valid positions).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False)
        )

    def forward(self, x, mask=None):
        # x: [B, T, D]
        logits = self.score(x).squeeze(-1)  # [B, T]
        if mask is not None:
            # mask: [B, T] with True for valid
            logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.softmax(logits, dim=1)  # [B, T]
        return torch.sum(x * w.unsqueeze(-1), dim=1)  # [B, D]


class LSTM1D_V3(BaseTimeSeriesModel):
    """
    Better LSTM for time-series classification / alpha modeling.

    Improvements over "take h_n only":
    - Supports variable length sequences via lengths + packing
    - Multiple readout strategies:
        * 'last'     : last-layer hidden state (like your Model A)
        * 'meanmax'  : concat(mean_pool, max_pool) over time (often very strong)
        * 'attn'     : attention pooling over time (learns which timesteps matter)
    - Uses LayerNorm (more stable than BatchNorm for non-stationary time series)
    - Keeps head light to preserve alpha tails (but still configurable)
    """
    MODEL_TYPE = "lstm"
    MODEL_VERSION = 3
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_classes: int = 3,
        bidirectional: bool = True,
        lstm_dropout: float = 0.2,     # dropout between LSTM layers
        head_dropout: float = 0.2,     # dropout in head
        readout: str = "meanmax",      # 'last' | 'meanmax' | 'attn'
        head: str = "linear",          # 'linear' | 'mlp'
        logit_clip: float | None = None,  # e.g., 10.0 to avoid crazy tails if needed
        **kwargs,
    ):
        super().__init__()
        assert readout in {"last", "meanmax", "attn"}
        assert head in {"linear", "mlp"}
        if len(kwargs) > 0:
            print(f"[BetterLSTM1D] Ignored kwargs: {list(kwargs.keys())}")
        if "p_drop" in kwargs:
            print("p_drop is deprecated in LSTM1D_V3. "
                "Use lstm_dropout and head_dropout instead.")
        # ---------------- store identity ----------------
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.readout = readout
        self.head_type = head
        self.logit_clip = logit_clip

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        # Readout-specific output dim
        if readout == "meanmax":
            feat_dim = out_dim * 2  # mean concat max
            self.attn_pool = None
        elif readout == "attn":
            feat_dim = out_dim
            self.attn_pool = AttentionPooling(out_dim)
        else:  # 'last'
            feat_dim = out_dim
            self.attn_pool = None

        # LayerNorm is usually safer than BatchNorm for time series
        self.norm = nn.LayerNorm(feat_dim)

        if head == "linear":
            # 任务 A: Trigger (是否有信号) -> [Hold, Action]
            self.head_trigger = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(feat_dim, 2)
            )
            # 任务 B: Direction (信号方向) -> [Short, Long]
            self.head_direction = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(feat_dim, 2)
            )
        else:
            mid = max(32, hidden_size)
            self.head_trigger = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(feat_dim, mid),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(mid, 2)
            )
            self.head_direction = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(feat_dim, mid),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(mid, 2)
            )

    @staticmethod
    def _make_mask(lengths: torch.Tensor, T: int):
        # lengths: [B] int64
        # returns mask: [B, T] True for valid positions
        device = lengths.device
        idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        return idx < lengths.unsqueeze(1)  # [B, T]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None, return_fused = False):
        """
        x: [B, T, F]
        lengths (optional): [B] actual lengths before padding.
                           If provided, packing is used and pooling ignores padding.
        """
        B, T, _ = x.shape

        if lengths is not None:
            # pack padded sequences for cleaner LSTM states
            lengths = lengths.to(torch.long).clamp(min=1, max=T)
            x_pack = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_pack, (h_n, _) = self.lstm(x_pack)
            out, _ = pad_packed_sequence(out_pack, batch_first=True, total_length=T)  # [B, T, D]
            mask = self._make_mask(lengths, T)  # [B, T]
        else:
            out, (h_n, _) = self.lstm(x)
            mask = None

        # ----- Readout -----
        if self.readout == "last":
            # last layer's final hidden state(s)
            if self.bidirectional:
                feat = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [B, 2H]
            else:
                feat = h_n[-1]  # [B, H]
        elif self.readout == "meanmax":
            # masked mean/max pool across time
            if mask is None:
                mean_pool = out.mean(dim=1)
                max_pool = out.max(dim=1).values
            else:
                out_masked = out.masked_fill(~mask.unsqueeze(-1), 0.0)
                denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
                mean_pool = out_masked.sum(dim=1) / denom

                out_for_max = out.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                max_pool = out_for_max.max(dim=1).values
            feat = torch.cat([mean_pool, max_pool], dim=1)
        else:  # 'attn'
            feat = self.attn_pool(out, mask=mask)

        feat = self.norm(feat)

        #  修改点 2：分别计算双头 Logits
        logits_trig = self.head_trigger(feat)    # [B, 2]
        logits_dir = self.head_direction(feat)  # [B, 2]

        if self.logit_clip is not None:
            logits_trig = torch.clamp(logits_trig, -self.logit_clip, self.logit_clip)
            logits_dir = torch.clamp(logits_dir, -self.logit_clip, self.logit_clip)

        #  修改点 3：增加概率融合逻辑 (Hierarchical Fusion)
        if return_fused:
            # 计算各头概率 (Softmax)
            p_trig = torch.softmax(logits_trig, dim=1) # [p_hold, p_act]
            p_dir = torch.softmax(logits_dir, dim=1)   # [p_short_in_act, p_long_in_act]
            
            # 合成 3 类概率
            # Class 1 (Neutral) = p_hold
            # Class 0 (Short)   = p_act * p_short_in_act
            # Class 2 (Long)    = p_act * p_long_in_act
            p_neutral = p_trig[:, 0]
            p_act     = p_trig[:, 1]
            p_short   = p_act * p_dir[:, 0]
            p_long    = p_act * p_dir[:, 1]
            
            # 拼接顺序: [Short(0), Neutral(1), Long(2)]
            fused_probs = torch.stack([p_short, p_neutral, p_long], dim=1)
            fused_preds = torch.argmax(fused_probs, dim=1)
            
            return fused_preds, fused_probs
        
        return logits_trig, logits_dir

    # ============================================================
    # meta / checkpoint interface
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
            "readout": self.readout,
            "head": self.head_type,
            "logit_clip": self.logit_clip,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        """
        Rebuild model from meta + state_dict.
        Dropout is disabled for inference/backtest.
        """
        model = cls(
            input_size=state["channel"],
            hidden_size=meta["lstm_hidden"],
            num_layers=meta["lstm_layers"],
            n_classes=len(meta["classes"]),
            bidirectional=meta["bidirectional"],
            readout=meta.get("readout", "meanmax"),
            head=meta.get("head", "linear"),
            logit_clip=meta.get("logit_clip", None),
            lstm_dropout=0.0,
            head_dropout=0.0,
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)