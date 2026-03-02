import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.models.model_base import BaseTimeSeriesModel

# ============================================================
# 基础组件（保持不变）：CausalConv1d, ECA1D, TimeDecayPool1D
# ============================================================
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=True):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation, bias=bias)

    def forward(self, x):
        y = super().forward(x)
        cut = (self.kernel_size[0] - 1) * self.dilation[0]
        return y[:, :, :-cut] if cut > 0 else y

class ECA1D(nn.Module):
    def __init__(self, channels, gamma=2.0, b=1.0, k_override=None):
        super().__init__()
        if k_override is None:
            k = int(abs((math.log2(channels) / gamma) + b))
            k = k if k % 2 == 1 else k + 1
            k = max(3, k)
        else:
            k = k_override if k_override % 2 == 1 else (k_override + 1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

    def forward(self, x):
        y = x.mean(dim=-1, keepdim=True)
        y = self.conv(y.transpose(1, 2))
        a = torch.sigmoid(y.transpose(1, 2))
        return x * a

class TimeDecayPool1D(nn.Module):
    def __init__(self, tau=16.0, learnable=False):
        super().__init__()
        if learnable:
            self.tau = nn.Parameter(torch.tensor(float(tau)))
        else:
            self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, x):
        B, C, T = x.shape
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        tau = self.tau
        w = torch.exp(-(T - 1 - t) / (tau + 1e-8))
        w = w / (w.sum() + 1e-12)
        return (x * w.view(1, 1, T)).sum(dim=-1)

# ============================================================
# CNN1D_V2 — Dual-Head Version
# ============================================================
class CNN1D_V1(BaseTimeSeriesModel):
    """
    Dual-head Causal CNN for time-series.
    Modified to match LSTM1D_V4's 2+2 architecture.
    """

    MODEL_TYPE = "cnn"
    MODEL_VERSION = 1 # 升级版本号以区分旧版单头 CNN

    supports_lengths = False

    def __init__(
        self,
        input_size: int,
        n_classes: int = 3, # 这里的 n_classes 主要用于 meta 兼容，实际输出为 2+2
        p_drop: float = 0.3,
        tau: float = 16.0,
        use_tpool: bool = False,
        logit_clip: float | None = None,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"[CNN1D_V2] Ignored kwargs: {list(kwargs.keys())}")

        self.input_size = input_size
        self.p_drop = p_drop
        self.tau = tau
        self.use_tpool = use_tpool
        self.logit_clip = logit_clip

        # ===== 骨干网络（CNN Backbone） =====
        self.conv_small = CausalConv1d(input_size, 64, kernel_size=5, bias=False)
        self.bn_small = nn.BatchNorm1d(64)

        self.conv_large = CausalConv1d(input_size, 64, kernel_size=21, bias=False)
        self.bn_large = nn.BatchNorm1d(64)

        self.eca_after_concat = ECA1D(channels=128)
        self.conv_post = CausalConv1d(128, 128, kernel_size=3, bias=False)
        self.bn_post = nn.BatchNorm1d(128)
        self.eca_after_post = ECA1D(channels=128)

        # ===== 池化与归一化 =====
        self.tpool = TimeDecayPool1D(tau=tau, learnable=False)
        self.norm = nn.LayerNorm(128) # 参考 LSTM 增加头前归一化

        # ===== 双头结构 (Trigger & Direction) =====
        self.head_trigger = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(128, 2)
        )
        self.head_direction = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(128, 2)
        )

    def forward(self, x, return_fused=False):
        """
        支持 2+2 输出和三分类融合输出。
        """
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)

        # 特征提取
        s = F.relu(self.bn_small(self.conv_small(x)))
        l = F.relu(self.bn_large(self.conv_large(x)))
        out = torch.cat([s, l], dim=1)

        out = self.eca_after_concat(out)
        out = F.relu(self.bn_post(self.conv_post(out)))
        out = self.eca_after_post(out)

        # 池化
        if self.use_tpool:
            feat = self.tpool(out)
        else:
            feat = out.mean(dim=-1)

        feat = self.norm(feat)

        # 双头计算
        logits_trig = self.head_trigger(feat)
        logits_dir = self.head_direction(feat)

        if self.logit_clip is not None:
            logits_trig = torch.clamp(logits_trig, -self.logit_clip, self.logit_clip)
            logits_dir = torch.clamp(logits_dir, -self.logit_clip, self.logit_clip)

        # 融合逻辑：将 2+2 转换为 3 分类 [Short(0), Neutral(1), Long(2)]
        if return_fused:
            p_trig = torch.softmax(logits_trig, dim=1) # [p_neutral, p_action]
            p_dir = torch.softmax(logits_dir, dim=1)   # [p_short, p_long]
            
            p_neu = p_trig[:, 0]
            p_act = p_trig[:, 1]
            p_s   = p_act * p_dir[:, 0]
            p_l   = p_act * p_dir[:, 1]
            
            fused_probs = torch.stack([p_s, p_neu, p_l], dim=1)
            fused_preds = torch.argmax(fused_probs, dim=1)
            return fused_preds, fused_probs
        
        return logits_trig, logits_dir

    def export_meta(self, **extra):
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "input_size": self.input_size,
            "p_drop": self.p_drop,
            "tau": self.tau,
            "use_tpool": self.use_tpool,
            "logit_clip": self.logit_clip,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        model = cls(
            input_size=meta["input_size"],
            p_drop=0.0,
            tau=meta["tau"],
            use_tpool=meta["use_tpool"],
            logit_clip=meta.get("logit_clip", None),
        )
        model.load_state_dict(state["state_dict"])
        return model.to(device)