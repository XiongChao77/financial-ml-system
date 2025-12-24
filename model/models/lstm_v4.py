import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.models.model_base import BaseTimeSeriesModel


class LockedDropout(nn.Module):
    """
    Locked / Variational Dropout for sequences.
    Apply the same dropout mask across all timesteps (per sample),
    which is often better for RNN generalization than iid dropout over time.

    x: [B, T, D]
    """
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        # one mask per sample, shared over time
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1.0 - self.p)
        mask = mask / (1.0 - self.p)
        return x * mask


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, D]
        logits = self.score(x).squeeze(-1)  # [B, T]
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.softmax(logits, dim=1)  # [B, T]
        return torch.sum(x * w.unsqueeze(-1), dim=1)  # [B, D]


class LSTM1D_V4(BaseTimeSeriesModel):
    """
    LSTM1D_V4 — "Stage-Optimal" LSTM for time-series classification / alpha modeling.

    V3 baseline characteristics (kept):
    - variable length sequences via lengths + packing
    - readout: 'last' / 'meanmax' / 'attn'
    - LayerNorm + light head to reduce alpha dilution risk

    V4 upgrades (model-side only):
    1) p_drop compatibility alias (so old scripts don't silently lose dropout)
    2) Input LayerNorm (handles mixed-scale engineered features better)
    3) LockedDropout at input/output (RNN-friendly regularization)
    4) New readout: 'mix' = concat(last, mean_pool, max_pool)

    Notes:
    - For inference/backtest, build_from_meta disables dropout by default.
    - Extra kwargs are ignored to support "pass-all-params" factory usage.
    """
    MODEL_TYPE = "lstm"
    MODEL_VERSION = 4

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_classes: int = 3,
        bidirectional: bool = True,

        # ---- Dropout interface (compat & explicit) ----
        p_drop: float | None = 0.2,          # deprecated alias
        lstm_dropout: float | None = 0.2,    # dropout between LSTM layers
        head_dropout: float | None = None,    # dropout in head
        in_locked_p: float = 0.0,             # locked dropout on inputs
        out_locked_p: float = 0.0,            # locked dropout on LSTM outputs (before pooling)

        # ---- Input preprocessing ----
        input_norm: bool = True,              # LayerNorm on input features
        input_proj_dim: int | None = None,    # optional projection before LSTM

        # ---- Readout & head ----
        readout: str = "mix",                 # 'last' | 'meanmax' | 'attn' | 'mix'
        head: str = "linear",                 # 'linear' | 'mlp'
        logit_clip: float | None = None,

        **kwargs,
    ):
        super().__init__()

        # ---------- allow "pass all args" from factory ----------
        if len(kwargs) > 0:
            print(f"[LSTM1D_V4] Ignored kwargs: {list(kwargs.keys())}")

        # ---------- resolve dropout values ----------
        # precedence:
        #   1) explicit lstm_dropout/head_dropout if provided
        #   2) p_drop alias (if provided and explicit ones are None)
        #   3) defaults
        if lstm_dropout is None and head_dropout is None and p_drop is not None:
            lstm_dropout = float(p_drop)
            head_dropout = float(p_drop)
            # compatibility note (keep it as a hint, not an error)
            print("[LSTM1D_V4] p_drop is deprecated; mapped to lstm_dropout/head_dropout.")

        if lstm_dropout is None:
            lstm_dropout = 0.2
        if head_dropout is None:
            head_dropout = 0.2

        assert readout in {"last", "meanmax", "attn", "mix"}, f"Unknown readout={readout}"
        assert head in {"linear", "mlp"}, f"Unknown head={head}"

        # ---------- store identity / meta ----------
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.readout = str(readout)
        self.head_type = str(head)
        self.logit_clip = logit_clip

        # record training-time regularization (useful for reproducibility)
        self.lstm_dropout = float(lstm_dropout)
        self.head_dropout = float(head_dropout)
        self.in_locked_p = float(in_locked_p)
        self.out_locked_p = float(out_locked_p)

        self.input_norm_enabled = bool(input_norm)
        self.input_proj_dim = None if input_proj_dim is None else int(input_proj_dim)

        # ---------- input preprocessing ----------
        self.in_norm = nn.LayerNorm(self.input_size) if self.input_norm_enabled else nn.Identity()

        if self.input_proj_dim is not None and self.input_proj_dim != self.input_size:
            self.in_proj = nn.Linear(self.input_size, self.input_proj_dim)
            lstm_input_size = self.input_proj_dim
        else:
            self.in_proj = nn.Identity()
            lstm_input_size = self.input_size

        self.in_locked = LockedDropout(self.in_locked_p)
        self.out_locked = LockedDropout(self.out_locked_p)

        # ---------- LSTM backbone ----------
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.lstm_dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        out_dim = self.hidden_size * (2 if self.bidirectional else 1)

        # ---------- readout ----------
        if self.readout == "meanmax":
            feat_dim = out_dim * 2
            self.attn_pool = None
        elif self.readout == "attn":
            feat_dim = out_dim
            self.attn_pool = AttentionPooling(out_dim)
        elif self.readout == "mix":
            # last + mean + max
            feat_dim = out_dim * 3
            self.attn_pool = None
        else:  # 'last'
            feat_dim = out_dim
            self.attn_pool = None

        self.norm = nn.LayerNorm(feat_dim)

        # ---------- head ----------
        if self.head_type == "linear":
            self.classifier = nn.Sequential(
                nn.Dropout(self.head_dropout),
                nn.Linear(feat_dim, n_classes)
            )
        else:
            # modest MLP (watch for alpha dilution)
            mid = max(32, self.hidden_size)
            self.classifier = nn.Sequential(
                nn.Dropout(self.head_dropout),
                nn.Linear(feat_dim, mid),
                nn.GELU(),
                nn.Dropout(self.head_dropout),
                nn.Linear(mid, n_classes)
            )

    @staticmethod
    def _make_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
        """
        lengths: [B] int64
        returns mask: [B, T] True for valid positions
        """
        device = lengths.device
        idx = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        return idx < lengths.unsqueeze(1)

    def _readout_last(self, h_n: torch.Tensor) -> torch.Tensor:
        # h_n: [num_layers*num_directions, B, H]
        if self.bidirectional:
            return torch.cat((h_n[-2], h_n[-1]), dim=1)  # [B, 2H]
        return h_n[-1]  # [B, H]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, T, F]
        lengths (optional): [B] actual lengths before padding.
                           If provided, packing is used and pooling ignores padding.
        """
        B, T, F = x.shape
        assert F == self.input_size, f"Expected input_size={self.input_size}, got F={F}"

        # ----- input preprocessing -----
        x = self.in_norm(x)
        x = self.in_proj(x)
        x = self.in_locked(x)

        # ----- LSTM -----
        if lengths is not None:
            lengths = lengths.to(torch.long).clamp(min=1, max=T)
            x_pack = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_pack, (h_n, _) = self.lstm(x_pack)
            out, _ = pad_packed_sequence(out_pack, batch_first=True, total_length=T)  # [B, T, D]
            mask = self._make_mask(lengths, T)  # [B, T]
        else:
            out, (h_n, _) = self.lstm(x)
            mask = None

        out = self.out_locked(out)

        # ----- readout -----
        if self.readout == "last":
            feat = self._readout_last(h_n)

        elif self.readout == "attn":
            feat = self.attn_pool(out, mask=mask)

        elif self.readout in {"meanmax", "mix"}:
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

            if self.readout == "meanmax":
                feat = torch.cat([mean_pool, max_pool], dim=1)
            else:
                last_feat = self._readout_last(h_n)
                feat = torch.cat([last_feat, mean_pool, max_pool], dim=1)

        else:
            raise RuntimeError(f"Unhandled readout={self.readout}")

        feat = self.norm(feat)
        logits = self.classifier(feat)

        if self.logit_clip is not None:
            logits = torch.clamp(logits, -self.logit_clip, self.logit_clip)

        return logits

    # ============================================================
    # meta / checkpoint interface
    # ============================================================
    def export_meta(self, **extra) -> dict:
        """
        Export architecture-defining parameters.
        (Store more than V3 for reproducibility.)
        """
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,

            "input_size": self.input_size,
            "lstm_hidden": self.hidden_size,
            "lstm_layers": self.num_layers,
            "bidirectional": self.bidirectional,

            "readout": self.readout,
            "head": self.head_type,
            "logit_clip": self.logit_clip,

            # training-time regularization config
            "lstm_dropout": self.lstm_dropout,
            "head_dropout": self.head_dropout,
            "in_locked_p": self.in_locked_p,
            "out_locked_p": self.out_locked_p,

            # input preprocessing
            "input_norm": self.input_norm_enabled,
            "input_proj_dim": self.input_proj_dim,

            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        """
        Rebuild model from meta + state_dict.
        Dropout is disabled for inference/backtest (deterministic).
        """
        input_size = state.get("channel", meta.get("input_size"))

        model = cls(
            input_size=input_size,
            hidden_size=meta["lstm_hidden"],
            num_layers=meta["lstm_layers"],
            n_classes=len(meta["classes"]),
            bidirectional=meta["bidirectional"],
            readout=meta.get("readout", "meanmax"),
            head=meta.get("head", "linear"),
            logit_clip=meta.get("logit_clip", None),

            # keep deterministic in inference
            lstm_dropout=0.0,
            head_dropout=0.0,
            in_locked_p=0.0,
            out_locked_p=0.0,

            input_norm=meta.get("input_norm", True),
            input_proj_dim=meta.get("input_proj_dim", None),
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)
