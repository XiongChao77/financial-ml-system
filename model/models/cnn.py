import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.models.model_base import BaseTimeSeriesModel


# ============================================================
# 因果卷积（组件）
# ============================================================
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=True):
        pad = (kernel_size - 1) * dilation
        super().__init__(
            in_ch, out_ch, kernel_size,
            padding=pad, dilation=dilation, bias=bias
        )

    def forward(self, x):                  # x: [B,C,T]
        y = super().forward(x)
        cut = (self.kernel_size[0] - 1) * self.dilation[0]
        return y[:, :, :-cut] if cut > 0 else y


# ============================================================
# ECA: Efficient Channel Attention (1D)
# ============================================================
class ECA1D(nn.Module):
    def __init__(self, channels, gamma=2.0, b=1.0, k_override=None):
        super().__init__()
        if k_override is None:
            k = int(abs((math.log2(channels) / gamma) + b))
            k = k if k % 2 == 1 else k + 1
            k = max(3, k)
        else:
            k = k_override if k_override % 2 == 1 else (k_override + 1)

        self.conv = nn.Conv1d(
            1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False
        )

    def forward(self, x):                  # x: [B,C,T]
        y = x.mean(dim=-1, keepdim=True)   # [B,C,1]
        y = self.conv(y.transpose(1, 2))   # [B,1,C]
        a = torch.sigmoid(y.transpose(1, 2))
        return x * a


# ============================================================
# 时间衰减池化（组件）
# ============================================================
class TimeDecayPool1D(nn.Module):
    def __init__(self, tau=16.0, learnable=False):
        super().__init__()
        if learnable:
            self.tau = nn.Parameter(torch.tensor(float(tau)))
        else:
            self.register_buffer("tau", torch.tensor(float(tau)))
        self.learnable = learnable

    def forward(self, x):                  # x: [B,C,T]
        B, C, T = x.shape
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        tau = self.tau if self.learnable else self.tau.to(x.device)
        w = torch.exp(-(T - 1 - t) / (tau + 1e-8))
        w = w / (w.sum() + 1e-12)
        return (x * w.view(1, 1, T)).sum(dim=-1)  # [B,C]


# ============================================================
# CNN1D_V1 — System-level model
# ============================================================
class CNN1D_V1(BaseTimeSeriesModel):
    """
    Causal CNN for time-series classification / alpha modeling.

    Architecture:
    - Causal Inception (k=5 / 21)
    - Causal fusion conv
    - ECA channel attention
    - Time-aware pooling
    """

    MODEL_TYPE = "cnn"
    MODEL_VERSION = 1

    supports_lengths = False  # 固定窗口

    def __init__(
        self,
        input_size: int,
        n_classes: int = 3,
        p_drop: float = 0.3,
        tau: float = 16.0,
        use_tpool: bool = False,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"[CNN1D_V1] Ignored kwargs: {list(kwargs.keys())}")

        # ===== 保存架构参数（meta 用）=====
        self.input_size = input_size
        self.n_classes = n_classes
        self.p_drop = p_drop
        self.tau = tau
        self.use_tpool = use_tpool

        # ===== 并行因果卷积分支 =====
        self.conv_small = CausalConv1d(
            input_size, 64, kernel_size=5, dilation=1, bias=False
        )
        self.bn_small = nn.BatchNorm1d(64)

        self.conv_large = CausalConv1d(
            input_size, 64, kernel_size=21, dilation=1, bias=False
        )
        self.bn_large = nn.BatchNorm1d(64)

        # ===== concat 后的 ECA =====
        self.eca_after_concat = ECA1D(channels=128)

        # ===== 融合因果卷积 =====
        self.conv_post = CausalConv1d(
            128, 128, kernel_size=3, dilation=1, bias=False
        )
        self.bn_post = nn.BatchNorm1d(128)

        self.eca_after_post = ECA1D(channels=128)

        # ===== 时间池化 =====
        self.tpool = TimeDecayPool1D(tau=tau, learnable=False)

        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(128, n_classes)

    # ============================================================
    # forward
    # ============================================================
    def forward(self, x):                  # x: [B,T,F]
        x = x.transpose(1, 2)              # -> [B,F,T]

        s = F.relu(self.bn_small(self.conv_small(x)))
        l = F.relu(self.bn_large(self.conv_large(x)))
        out = torch.cat([s, l], dim=1)     # [B,128,T]

        out = self.eca_after_concat(out)
        out = F.relu(self.bn_post(self.conv_post(out)))
        out = self.eca_after_post(out)

        if self.use_tpool:
            out = self.tpool(out)          # [B,128]
        else:
            out = out.mean(dim=-1)         # GAP

        out = self.dropout(out)
        return self.fc(out)

    # ============================================================
    # meta / checkpoint
    # ============================================================
    def export_meta(self, **extra):
        """
        Export architecture-defining parameters.
        """
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "input_size": self.input_size,
            "n_classes": self.n_classes,
            "p_drop": self.p_drop,
            "tau": self.tau,
            "use_tpool": self.use_tpool,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        """
        Rebuild model from meta + checkpoint.
        推理时关闭 dropout。
        """
        model = cls(
            input_size=meta["input_size"],
            n_classes=len(meta["classes"]),
            p_drop=0.0,                 # 🔥 推理阶段关闭
            tau=meta["tau"],
            use_tpool=meta["use_tpool"],
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)
