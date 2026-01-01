import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.models.model_base import BaseTimeSeriesModel


# -------------------------
# Utils
# -------------------------
class LockedDropout(nn.Module):
    """
    Locked / Variational Dropout for sequences:
    Same dropout mask across all timesteps per sample.

    x: [B, T, D]
    """
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


class DropPath(nn.Module):
    """
    Stochastic Depth / DropPath.
    """
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        rnd = keep + torch.rand(shape, device=x.device, dtype=x.dtype)
        mask = torch.floor(rnd)  # 0/1
        return x * mask / keep


class LayerScale(nn.Module):
    """
    Per-channel residual scaling for stability in deep transformers.
    """
    def __init__(self, dim: int, init: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


def _get_alibi_slopes(n_heads: int):
    """
    Standard ALiBi slope generation (paper's reference implementation idea).
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** (-(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)

    closest_pow2 = 2 ** math.floor(math.log2(n_heads))
    slopes = get_slopes_power_of_2(closest_pow2)
    extra = _get_alibi_slopes(2 * closest_pow2)[0::2]
    return slopes + extra[: n_heads - closest_pow2]


class AttentionPooling(nn.Module):
    """
    Additive attention pooling over time.
    x: [B, T, D], mask: [B, T] (True valid)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)  # [B, T]
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        w = torch.softmax(logits, dim=1)  # [B, T]
        return torch.sum(x * w.unsqueeze(-1), dim=1)  # [B, D]


# -------------------------
# Core blocks
# -------------------------
class MultiheadSelfAttention(nn.Module):
    """
    Manual MHA to allow ALiBi additive biases cleanly.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_alibi: bool = True,
        alibi_mode: str = "abs",  # "abs" (bidirectional) or "causal"
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert alibi_mode in {"abs", "causal"}

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.use_alibi = bool(use_alibi)
        self.alibi_mode = alibi_mode

        if self.use_alibi:
            slopes = torch.tensor(_get_alibi_slopes(nhead), dtype=torch.float32)  # [H]
            self.register_buffer("alibi_slopes", slopes.view(1, nhead, 1, 1), persistent=False)

    def _alibi_bias(self, T: int, device, dtype):
        # bias: [1, H, T, T]
        pos = torch.arange(T, device=device)
        if self.alibi_mode == "causal":
            # distance i-j (only meaningful if you also use causal masking)
            dist = (pos.view(T, 1) - pos.view(1, T)).clamp(min=0).to(torch.float32)
        else:
            dist = (pos.view(T, 1) - pos.view(1, T)).abs().to(torch.float32)

        bias = -dist  # more distant => more negative
        bias = bias.view(1, 1, T, T)  # [1,1,T,T]
        bias = bias * self.alibi_slopes  # [1,H,T,T]
        return bias.to(dtype=dtype)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, T, D]
        key_mask: [B, T] True for valid tokens. (padding mask)
        """
        B, T, D = x.shape
        qkv = self.qkv(x)  # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, H, T, Hd]
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # scores: [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if self.use_alibi:
            scores = scores + self._alibi_bias(T, device=x.device, dtype=scores.dtype)

        if key_mask is not None:
            # mask invalid keys to -inf
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B, H, T, Hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FeedForward(nn.Module):
    """
    FFN with optional SwiGLU.
    """
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        ffn_type: str = "swiglu",  # "swiglu" | "gelu"
    ):
        super().__init__()
        assert ffn_type in {"swiglu", "gelu"}
        self.ffn_type = ffn_type
        self.drop = nn.Dropout(dropout)

        if ffn_type == "swiglu":
            # fc1 -> split -> silu(a)*b -> fc2
            self.fc1 = nn.Linear(d_model, 2 * dim_feedforward)
            self.fc2 = nn.Linear(dim_feedforward, d_model)
        else:
            self.fc1 = nn.Linear(d_model, dim_feedforward)
            self.fc2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == "swiglu":
            a, b = self.fc1(x).chunk(2, dim=-1)
            x = F.silu(a) * b
            x = self.drop(x)
            x = self.fc2(x)
            return x
        else:
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.drop(x)
            x = self.fc2(x)
            return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        attn_dropout: float,
        drop_path: float,
        use_alibi: bool,
        alibi_mode: str,
        ffn_type: str,
        layerscale_init: float | None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            nhead=nhead,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            use_alibi=use_alibi,
            alibi_mode=alibi_mode,
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            ffn_type=ffn_type,
        )
        self.drop_path2 = DropPath(drop_path)

        self.ls1 = LayerScale(d_model, layerscale_init) if layerscale_init is not None else nn.Identity()
        self.ls2 = LayerScale(d_model, layerscale_init) if layerscale_init is not None else nn.Identity()

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        # PreNorm
        a = self.attn(self.norm1(x), key_mask=key_mask)
        x = x + self.drop_path1(self.ls1(a))

        f = self.ffn(self.norm2(x))
        x = x + self.drop_path2(self.ls2(f))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1), :].to(dtype=x.dtype)


# -------------------------
# Transformer1D_V3
# -------------------------
class Transformer1D_V3(BaseTimeSeriesModel):
    """
    Transformer1D_V3 — stage-optimal exploration version (model-side only).

    - Input LayerNorm + optional projection
    - LockedDropout (sequence-friendly regularization)
    - ALiBi relative position bias (optional)
    - SwiGLU FFN (optional)
    - DropPath + LayerScale (optional)
    - Readout: 'cls' | 'meanmax' | 'attn' | 'mix'

    Input:  x [B, T, F], lengths optional (for padding mask)
    Output: logits [B, n_classes]
    """
    MODEL_TYPE = "transformer"
    MODEL_VERSION = 3

    def __init__(
        self,
        input_size: int,
        n_classes: int = 3,

        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,

        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        drop_path: float = 0.05,

        # input regularization
        input_norm: bool = True,
        in_locked_p: float = 0.05,

        # positional / bias
        use_alibi: bool = True,
        alibi_mode: str = "abs",          # "abs" | "causal"
        pos_encoding: str = "none",       # "none" | "learned" | "sin"
        max_len: int = 512,               # only for learned/sin

        # tokens / readout
        cls_token: bool = True,
        readout: str = "mix",             # "cls" | "meanmax" | "attn" | "mix"
        head: str = "linear",             # "linear" | "mlp"

        # ffn variant
        ffn_type: str = "swiglu",         # "swiglu" | "gelu"

        # stability
        layerscale_init: float | None = 1e-5,

        logit_clip: float | None = None,

        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"[Transformer1D_V3] Ignored kwargs: {list(kwargs.keys())}")

        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert pos_encoding in {"none", "learned", "sin"}
        assert readout in {"cls", "meanmax", "attn", "mix"}
        assert head in {"linear", "mlp"}
        assert ffn_type in {"swiglu", "gelu"}
        assert alibi_mode in {"abs", "causal"}

        # store meta
        self.input_size = int(input_size)
        self.n_classes = int(n_classes)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)

        self.dropout = float(dropout)
        self.attn_dropout = float(attn_dropout)
        self.drop_path = float(drop_path)

        self.input_norm_enabled = bool(input_norm)
        self.in_locked_p = float(in_locked_p)

        self.use_alibi = bool(use_alibi)
        self.alibi_mode = alibi_mode
        self.pos_encoding = pos_encoding
        self.max_len = int(max_len)

        self.cls_token_enabled = bool(cls_token)
        self.readout = readout
        self.head_type = head
        self.ffn_type = ffn_type

        self.layerscale_init = layerscale_init
        self.logit_clip = logit_clip

        # input stem
        self.in_norm = nn.LayerNorm(self.input_size) if self.input_norm_enabled else nn.Identity()
        self.input_proj = nn.Linear(self.input_size, self.d_model)
        self.in_locked = LockedDropout(self.in_locked_p)

        # cls token
        if self.cls_token_enabled:
            self.cls = nn.Parameter(torch.randn(1, 1, self.d_model))
        else:
            self.cls = None

        # absolute positional encoding (optional)
        if self.pos_encoding == "learned":
            extra = 1 if self.cls_token_enabled else 0
            self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len + extra, self.d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.pos_sin = None
        elif self.pos_encoding == "sin":
            extra = 1 if self.cls_token_enabled else 0
            self.pos_sin = SinusoidalPositionalEncoding(self.d_model, self.max_len + extra)
            self.pos_embed = None
        else:
            self.pos_embed = None
            self.pos_sin = None

        self.drop = nn.Dropout(self.dropout)

        # transformer blocks
        dpr = torch.linspace(0, self.drop_path, self.num_layers).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
                drop_path=float(dpr[i]),
                use_alibi=self.use_alibi,
                alibi_mode=self.alibi_mode,
                ffn_type=self.ffn_type,
                layerscale_init=self.layerscale_init,
            )
            for i in range(self.num_layers)
        ])

        # readout pooling
        if self.readout == "attn":
            self.attn_pool = AttentionPooling(self.d_model)
            feat_dim = self.d_model
        elif self.readout == "meanmax":
            self.attn_pool = None
            feat_dim = self.d_model * 2
        elif self.readout == "mix":
            self.attn_pool = None
            feat_dim = self.d_model * 3
        else:  # cls
            self.attn_pool = None
            feat_dim = self.d_model

        self.out_norm = nn.LayerNorm(feat_dim)

        mid = max(64, self.d_model)
        self.head_trigger = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(feat_dim, mid),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(mid, 2),
            )
        self.head_direction = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(feat_dim, mid),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(mid, 2),
        )

    @staticmethod
    def _make_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
        device = lengths.device
        idx = torch.arange(T, device=device).unsqueeze(0)
        return idx < lengths.unsqueeze(1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None, return_fused = False) -> torch.Tensor:
        """
        x: [B, T, F]
        lengths optional: [B]
        """
        B, T, F_in = x.shape
        assert F_in == self.input_size, f"Expected input_size={self.input_size}, got {F_in}"

        # mask for original tokens
        if lengths is not None:
            lengths = lengths.to(torch.long).clamp(min=1, max=T)
            mask_tok = self._make_mask(lengths, T)  # [B, T]
        else:
            mask_tok = None

        # input stem
        x = self.in_norm(x)
        x = self.input_proj(x)
        x = self.in_locked(x)

        # prepend CLS
        if self.cls_token_enabled:
            cls = self.cls.expand(B, -1, -1)  # [B,1,D]
            x = torch.cat([cls, x], dim=1)    # [B, T+1, D]
            if mask_tok is not None:
                mask = torch.cat([torch.ones(B, 1, device=x.device, dtype=torch.bool), mask_tok], dim=1)
            else:
                mask = None
        else:
            mask = mask_tok

        # positional
        if self.pos_encoding == "learned":
            if x.size(1) > self.pos_embed.size(1):
                raise ValueError(f"Sequence too long: {x.size(1)} > max_len({self.pos_embed.size(1)})")
            x = x + self.pos_embed[:, : x.size(1), :].to(dtype=x.dtype)
        elif self.pos_encoding == "sin":
            x = self.pos_sin(x)

        x = self.drop(x)

        # blocks
        for blk in self.blocks:
            x = blk(x, key_mask=mask)

        # split seq for pooling (exclude cls from pooling)
        if self.cls_token_enabled:
            cls_out = x[:, 0, :]
            seq = x[:, 1:, :]
            mask_seq = mask[:, 1:] if mask is not None else None
        else:
            cls_out = None
            seq = x
            mask_seq = mask

        # readout
        if self.readout == "cls":
            if cls_out is None:
                # fallback: last token
                feat = seq[:, -1, :] if mask_seq is None else seq[torch.arange(B, device=x.device), lengths - 1, :]
            else:
                feat = cls_out

        elif self.readout == "attn":
            feat = self.attn_pool(seq, mask=mask_seq)

        elif self.readout in {"meanmax", "mix"}:
            if mask_seq is None:
                mean_pool = seq.mean(dim=1)
                max_pool = seq.max(dim=1).values
            else:
                seq_masked = seq.masked_fill(~mask_seq.unsqueeze(-1), 0.0)
                denom = mask_seq.sum(dim=1).clamp(min=1).unsqueeze(-1)
                mean_pool = seq_masked.sum(dim=1) / denom

                seq_for_max = seq.masked_fill(~mask_seq.unsqueeze(-1), float("-inf"))
                max_pool = seq_for_max.max(dim=1).values

            if self.readout == "meanmax":
                feat = torch.cat([mean_pool, max_pool], dim=1)
            else:
                # mix = (cls or last) + mean + max
                if cls_out is not None:
                    anchor = cls_out
                else:
                    anchor = seq[:, -1, :] if mask_seq is None else seq[torch.arange(B, device=x.device), lengths - 1, :]
                feat = torch.cat([anchor, mean_pool, max_pool], dim=1)
        else:
            raise RuntimeError(f"Unknown readout={self.readout}")

        feat = self.out_norm(feat)
        # 🌟 修改点 2：分别计算双头 Logits
        logits_trig = self.head_trigger(feat)    # [B, 2]
        logits_dir = self.head_direction(feat)  # [B, 2]

        if self.logit_clip is not None:
            logits_trig = torch.clamp(logits_trig, -self.logit_clip, self.logit_clip)
            logits_dir = torch.clamp(logits_dir, -self.logit_clip, self.logit_clip)

        # 🌟 修改点 3：固化融合逻辑 (与 ConvLSTM_V1 保持同步)
        if return_fused:
            # 1. 计算各头概率 (Softmax)
            p_trig = torch.softmax(logits_trig, dim=1) # [p_hold, p_act]
            p_dir = torch.softmax(logits_dir, dim=1)   # [p_short_in_act, p_long_in_act]
            
            # 2. 合成 3 类概率
            # p_neutral(1) = p_hold
            # p_short(0)   = p_act * p_short_in_act
            # p_long(2)    = p_act * p_long_in_act
            p_neutral = p_trig[:, 0]
            p_act     = p_trig[:, 1]
            p_short   = p_act * p_dir[:, 0]
            p_long    = p_act * p_dir[:, 1]
            
            # 拼接成 [B, 3] 顺序: [Short(0), Neutral(1), Long(2)]
            fused_probs = torch.stack([p_short, p_neutral, p_long], dim=1)
            fused_preds = torch.argmax(fused_probs, dim=1)
            
            return fused_preds, fused_probs
        
        return logits_trig, logits_dir

    # -------------------------
    # meta / checkpoint
    # -------------------------
    def export_meta(self, **extra) -> dict:
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
            "attn_dropout": self.attn_dropout,
            "drop_path": self.drop_path,

            "input_norm": self.input_norm_enabled,
            "in_locked_p": self.in_locked_p,

            "use_alibi": self.use_alibi,
            "alibi_mode": self.alibi_mode,
            "pos_encoding": self.pos_encoding,
            "max_len": self.max_len,

            "cls_token": self.cls_token_enabled,
            "readout": self.readout,
            "head": self.head_type,
            "ffn_type": self.ffn_type,

            "layerscale_init": self.layerscale_init,
            "logit_clip": self.logit_clip,

            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        # inference/backtest: disable dropout-ish stuff for determinism
        input_size = state.get("channel", meta.get("input_size"))

        model = cls(
            input_size=input_size,
            n_classes=len(meta["classes"]),

            d_model=meta["d_model"],
            nhead=meta["nhead"],
            num_layers=meta["num_layers"],
            dim_feedforward=meta["dim_feedforward"],

            dropout=0.0,
            attn_dropout=0.0,
            drop_path=0.0,

            input_norm=meta.get("input_norm", True),
            in_locked_p=0.0,

            use_alibi=meta.get("use_alibi", True),
            alibi_mode=meta.get("alibi_mode", "abs"),
            pos_encoding=meta.get("pos_encoding", "none"),
            max_len=meta.get("max_len", 512),

            cls_token=meta.get("cls_token", True),
            readout=meta.get("readout", "mix"),
            head=meta.get("head", "linear"),
            ffn_type=meta.get("ffn_type", "swiglu"),

            layerscale_init=meta.get("layerscale_init", 1e-5),
            logit_clip=meta.get("logit_clip", None),
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)
