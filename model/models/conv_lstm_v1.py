import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.models.model_base import BaseTimeSeriesModel


class LockedDropout(nn.Module):
    """Time-consistent dropout mask over T for each sample."""
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(keep)
        mask = mask / keep
        return x * mask


class AttentionPooling(nn.Module):
    """Additive attention pooling for [B,T,D] with optional mask [B,T]."""
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)  # [B,T]
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.softmax(logits, dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class ConvBlock(nn.Module):
    """
    Depthwise-separable 1D conv block with residual:
      x -> DWConv(dilation) -> PWConv -> GELU -> Dropout -> +res
    Using GroupNorm(1, C) as "LayerNorm over channels" for speed/stability.
    """
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation

        self.norm1 = nn.GroupNorm(1, channels)
        self.dw = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=pad, dilation=dilation, groups=channels, bias=False
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T]
        h = self.norm1(x)
        h = self.dw(h)
        h = self.pw(h)
        h = F.gelu(h)
        h = self.drop(h)
        return x + h


class ConvLSTM1D_V1(BaseTimeSeriesModel):
    """
    Conv + LSTM hybrid for time-series classification / alpha modeling.

    Pipeline:
      [B,T,F] -> Input LN -> Linear proj -> (Conv blocks on [B,C,T]) ->
      back to [B,T,C] -> optional mask -> packed LSTM -> readout -> head

    Readout:
      - 'last'   : last hidden
      - 'meanmax': mean + max pooling
      - 'attn'   : attention pooling
      - 'mix'    : last + mean + max  (recommended)
    """
    MODEL_TYPE = "conv_lstm"
    MODEL_VERSION = 1

    def __init__(
        self,
        input_size: int,
        n_classes: int = 3,

        # stem width
        d_model: int = 96,

        # conv stem
        conv_layers: int = 3,
        conv_kernel: int = 5,
        conv_dropout: float = 0.10,
        conv_dilations: tuple[int, ...] | None = None,  # default: (1,2,4,...)

        # LSTM
        hidden_size: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        lstm_dropout: float = 0.2,

        # regularization
        input_norm: bool = True,
        in_locked_p: float = 0.05,
        out_locked_p: float = 0.05,
        head_dropout: float = 0.2,

        # readout/head
        readout: str = "mix",        # 'last'|'meanmax'|'attn'|'mix'
        head: str = "linear",        # 'linear'|'mlp'
        logit_clip: float | None = None,

        # compatibility: allow pass-all params
        p_drop: float | None = None,  # alias: if provided and explicit dropouts not set
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"[ConvLSTM1D_V1] Ignored kwargs: {list(kwargs.keys())}")

        assert readout in {"last", "meanmax", "attn", "mix"}
        assert head in {"linear", "mlp"}

        # p_drop alias (optional)
        if p_drop is not None:
            # only apply when user didn't explicitly set these
            if conv_dropout == 0.10 and head_dropout == 0.2 and lstm_dropout == 0.2:
                conv_dropout = float(p_drop)
                head_dropout = float(p_drop)
                lstm_dropout = float(p_drop)
                print("[ConvLSTM1D_V1] p_drop is deprecated; mapped to conv/lstm/head dropout.")

        self.input_size = int(input_size)
        self.n_classes = int(n_classes)
        self.d_model = int(d_model)

        self.readout = readout
        self.head_type = head
        self.bidirectional = bool(bidirectional)
        self.logit_clip = logit_clip

        self.input_norm_enabled = bool(input_norm)
        self.in_locked_p = float(in_locked_p)
        self.out_locked_p = float(out_locked_p)

        # ---- input preprocessing ----
        self.in_norm = nn.LayerNorm(self.input_size) if self.input_norm_enabled else nn.Identity()
        self.proj = nn.Linear(self.input_size, self.d_model)
        self.in_locked = LockedDropout(self.in_locked_p)

        # ---- conv stem ----
        if conv_dilations is None:
            # e.g. 3 layers -> (1,2,4)
            conv_dilations = tuple(2 ** i for i in range(conv_layers))
        else:
            conv_layers = len(conv_dilations)

        self.conv_dilations = tuple(int(d) for d in conv_dilations)
        self.conv_kernel = int(conv_kernel)
        self.conv_dropout = float(conv_dropout)

        self.conv_blocks = nn.ModuleList([
            ConvBlock(self.d_model, kernel_size=self.conv_kernel, dilation=d, dropout=self.conv_dropout)
            for d in self.conv_dilations
        ])

        # ---- LSTM backbone ----
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.lstm_dropout = float(lstm_dropout)

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.lstm_dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        out_dim = self.hidden_size * (2 if self.bidirectional else 1)

        self.out_locked = LockedDropout(self.out_locked_p)

        # ---- readout ----
        if self.readout == "meanmax":
            feat_dim = out_dim * 2
            self.attn_pool = None
        elif self.readout == "attn":
            feat_dim = out_dim
            self.attn_pool = AttentionPooling(out_dim)
        elif self.readout == "mix":
            feat_dim = out_dim * 3
            self.attn_pool = None
        else:
            feat_dim = out_dim
            self.attn_pool = None

        self.norm = nn.LayerNorm(feat_dim)

        # ---- head ----
        self.head_dropout = float(head_dropout)
        if self.head_type == "linear":
            self.classifier = nn.Sequential(
                nn.Dropout(self.head_dropout),
                nn.Linear(feat_dim, self.n_classes),
            )
        else:
            mid = max(64, self.hidden_size)
            self.classifier = nn.Sequential(
                nn.Dropout(self.head_dropout),
                nn.Linear(feat_dim, mid),
                nn.GELU(),
                nn.Dropout(self.head_dropout),
                nn.Linear(mid, self.n_classes),
            )
        # ---- 修改为双头架构 ----
        # 任务 A: Trigger (判断是否有信号) -> 2类: 0(Neutral), 1(Action)
        self.head_trigger = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.Linear(feat_dim, 2)
        )
        
        # 任务 B: Direction (判断方向) -> 2类: 0(Short), 1(Long)
        self.head_direction = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.Linear(feat_dim, 2)
        )

    @staticmethod
    def _make_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
        idx = torch.arange(T, device=lengths.device).unsqueeze(0)
        return idx < lengths.unsqueeze(1)

    def _readout_last(self, h_n: torch.Tensor) -> torch.Tensor:
        # h_n: [num_layers*num_directions, B, H]
        if self.bidirectional:
            return torch.cat((h_n[-2], h_n[-1]), dim=1)
        return h_n[-1]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None, return_fused=False) -> torch.Tensor:
        """
        x: [B,T,F]
        lengths: [B] optional
        """
        B, T, F_in = x.shape
        assert F_in == self.input_size

        # mask for padded tokens (optional)
        if lengths is not None:
            lengths = lengths.to(torch.long).clamp(min=1, max=T)
            mask = self._make_mask(lengths, T)  # [B,T]
        else:
            mask = None

        # input norm + projection
        x = self.in_norm(x)
        x = self.proj(x)
        x = self.in_locked(x)  # [B,T,D]

        # conv stem on channels-first
        h = x.transpose(1, 2).contiguous()  # [B,D,T]
        for blk in self.conv_blocks:
            h = blk(h)
        h = h.transpose(1, 2).contiguous()  # [B,T,D]

        # if padding exists, zero-out padded timesteps before packing (reduce padding leakage)
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)

        # LSTM (pack if lengths provided)
        if lengths is not None:
            h_pack = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_pack, (h_n, _) = self.lstm(h_pack)
            out, _ = pad_packed_sequence(out_pack, batch_first=True, total_length=T)
        else:
            out, (h_n, _) = self.lstm(h)

        out = self.out_locked(out)

        # readout
        if self.readout == "last":
            feat = self._readout_last(h_n)

        elif self.readout == "attn":
            feat = self.attn_pool(out, mask=mask)

        elif self.readout in {"meanmax", "mix"}:
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
            raise RuntimeError(f"Unknown readout={self.readout}")

        feat = self.norm(feat)
        # 分别计算两个任务的 Logits
        logits_trig = self.head_trigger(feat)    # [B, 2]
        logits_dir = self.head_direction(feat)  # [B, 2]

        if self.logit_clip is not None:
            logits_trig = torch.clamp(logits_trig, -self.logit_clip, self.logit_clip)
            logits_dir = torch.clamp(logits_dir, -self.logit_clip, self.logit_clip)

        # 🌟 升级后的融合逻辑
        if return_fused:
            # 1. 计算各头概率 (Softmax)
            p_trig = torch.softmax(logits_trig, dim=1) # [p_hold, p_act]
            p_dir = torch.softmax(logits_dir, dim=1)   # [p_short_in_act, p_long_in_act]
            
            # 2. 合成 3 类概率 (Hierarchical Fusion)
            # p_neutral(1) = p_hold
            # p_short(0)   = p_act * p_short_in_act
            # p_long(2)    = p_act * p_long_in_act
            p_neutral = p_trig[:, 0]
            p_act     = p_trig[:, 1]
            p_short   = p_act * p_dir[:, 0]
            p_long    = p_act * p_dir[:, 1]
            
            # 拼接成 [B, 3] 顺序: [Short(0), Neutral(1), Long(2)]
            fused_probs = torch.stack([p_short, p_neutral, p_long], dim=1)
            
            # 3. 生成预测标签 (基于合成概率的 argmax 确保与概率对齐)
            fused_preds = torch.argmax(fused_probs, dim=1)
            
            return fused_preds, fused_probs # 🌟 返回元组
        
        return logits_trig, logits_dir

    # ---------- meta ----------
    def export_meta(self, **extra) -> dict:
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,

            "input_size": self.input_size,
            "n_classes": self.n_classes,

            "d_model": self.d_model,
            "conv_kernel": self.conv_kernel,
            "conv_dropout": self.conv_dropout,
            "conv_dilations": list(self.conv_dilations),

            "lstm_hidden": self.hidden_size,
            "lstm_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "lstm_dropout": self.lstm_dropout,

            "input_norm": self.input_norm_enabled,
            "in_locked_p": self.in_locked_p,
            "out_locked_p": self.out_locked_p,

            "readout": self.readout,
            "head": self.head_type,
            "head_dropout": self.head_dropout,
            "logit_clip": self.logit_clip,

            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        input_size = state.get("channel", meta.get("input_size"))

        model = cls(
            input_size=input_size,
            n_classes=len(meta["classes"]),

            d_model=meta.get("d_model", 96),
            conv_kernel=meta.get("conv_kernel", 5),
            conv_dropout=0.0,  # deterministic inference
            conv_dilations=tuple(meta.get("conv_dilations", [1, 2, 4])),

            hidden_size=meta.get("lstm_hidden", 64),
            num_layers=meta.get("lstm_layers", 2),
            bidirectional=meta.get("bidirectional", True),
            lstm_dropout=0.0,

            input_norm=meta.get("input_norm", True),
            in_locked_p=0.0,
            out_locked_p=0.0,

            readout=meta.get("readout", "mix"),
            head=meta.get("head", "linear"),
            head_dropout=0.0,
            logit_clip=meta.get("logit_clip", None),
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)
