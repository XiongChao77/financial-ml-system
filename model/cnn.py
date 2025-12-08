import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
# ---- 因果卷积：左填充，右裁剪，杜绝“看未来” ----
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=True):
        pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size,
                         padding=pad, dilation=dilation, bias=bias)
    def forward(self, x):                      # x: [B,C,T]
        y = super().forward(x)                 # 先做常规padding卷积
        cut = (self.kernel_size[0]-1) * self.dilation[0]
        return y[:, :, :-cut] if cut > 0 else y   # 右裁剪，保持长度 & 因果

# ---- ECA: 高效通道注意力（1D版，通道维上做1D卷积） ----
class ECA1D(nn.Module):
    def __init__(self, channels, gamma=2.0, b=1.0, k_override=None):
        super().__init__()
        if k_override is None:
            k = int(abs((math.log2(channels)/gamma) + b))
            k = k if k % 2 == 1 else k + 1
            k = max(3, k)
        else:
            k = k_override if k_override % 2 == 1 else (k_override + 1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)
    def forward(self, x):                  # x: [B,C,T]
        y = x.mean(dim=-1, keepdim=True)   # [B,C,1]
        y = self.conv(y.transpose(1,2))    # [B,1,C]
        a = torch.sigmoid(y.transpose(1,2))# [B,C,1]
        return x * a

# ---- 最近更重要：指数衰减加权池化（替代 mean） ----
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
        t = torch.arange(T, device=x.device, dtype=x.dtype)         # 0..T-1
        tau = self.tau if self.learnable else self.tau.to(x.device)
        w = torch.exp(-(T - 1 - t) / (tau + 1e-8))                  # 近端更大
        w = w / (w.sum() + 1e-12)
        return (x * w.view(1,1,T)).sum(dim=-1)                      # [B,C]

# ---- 整体模型：因果Inception -> 因果融合卷积 -> ECA -> 加权池化 -> FC ----
class CNN1D(nn.Module):
    """
    架构: Causal Inception(k=5/21) -> Causal conv3 -> ECA -> TimeDecayPool -> FC
    - 因果卷积：不利用未来信息
    - ECA：按通道自适应重标定
    - 最近更重要：时间指数衰减加权池化
    输入: [B, T, F]   输出: logits [B, n_classes]
    """
    def __init__(self, channel=9, n_classes=3, p_drop=0.3, tau=16.0):
        super().__init__()
        # 并行因果卷积分支（短/长核）
        self.conv_small = CausalConv1d(channel, 64, kernel_size=5, dilation=1, bias=False)
        self.bn_small   = nn.BatchNorm1d(64)
        self.conv_large = CausalConv1d(channel, 64, kernel_size=21, dilation=1, bias=False)
        self.bn_large   = nn.BatchNorm1d(64)

        # concat 后的 ECA（先做一次通道筛选）
        self.eca_after_concat = ECA1D(channels=128)

        # 融合卷积也使用因果版本，避免再次引入未来
        self.conv_post = CausalConv1d(128, 128, kernel_size=3, dilation=1, bias=False)
        self.bn_post   = nn.BatchNorm1d(128)

        # 融合后的第二次 ECA（可保留或去掉做对比）
        self.eca_after_post = ECA1D(channels=128)

        # 最近更重要的池化
        self.tpool = TimeDecayPool1D(tau=tau, learnable=False)

        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):                  # x: [B,T,F]
        x = x.transpose(1, 2)              # -> [B,F,T]

        s = F.relu(self.bn_small(self.conv_small(x)))   # [B,64,T]
        l = F.relu(self.bn_large(self.conv_large(x)))   # [B,64,T]
        out = torch.cat([s, l], dim=1)                  # [B,128,T]

        out = self.eca_after_concat(out)                # [B,128,T]
        out = F.relu(self.bn_post(self.conv_post(out))) # [B,128,T]
        out = self.eca_after_post(out)                  # [B,128,T]

        out = out.mean(dim=-1) #和tpool二选一
        # out = self.tpool(out)                           # [B,128] (最近更重要)
        out = self.dropout(out)
        return self.fc(out)                             # [B,n_classes]