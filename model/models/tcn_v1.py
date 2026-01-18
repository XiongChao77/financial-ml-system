import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from model.models.model_base import BaseTimeSeriesModel

class ChainedCausalConv(nn.Module):
    """
    因果卷积块：通过左侧填充确保输出 t 只依赖于 t 及之前的输入。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        # 计算因果填充量
        padding = (kernel_size - 1) * dilation
        
        # 第一层扩张卷积
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding) # 移除右侧多余填充，实现因果性
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二层扩张卷积
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # 残差连接：如果输入输出维度不同，使用 1x1 卷积对齐
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """辅助类：截断 Conv1d 的输出以实现因果卷积"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCN1D_V1(BaseTimeSeriesModel):
    """
    TCN1D_V1 — 针对 Alpha 建模优化的时域卷积网络
    架构：叠加多个残差扩张卷积块 + Readout 层 + 双头输出 (2+2)
    """
    MODEL_TYPE = "tcn"
    MODEL_VERSION = 1

    def __init__(
        self,
        input_size: int,
        n_classes: int = 3,
        num_channels: list = [64, 64, 64], # 每一层 TCN 的通道数
        kernel_size: int = 3,
        dropout: float = 0.2,
        readout: str = "mix", # 'last' | 'meanmax' | 'mix'
        head: str = "linear",
        logit_clip: float | None = None,
        **kwargs
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_channels = num_channels  # 修复在这里
        self.kernel_size = kernel_size    # 顺便把这个也存了，后面 meta 也要用
        self.readout = readout
        self.logit_clip = logit_clip

        # ---- TCN 主干网络 ----
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i # 扩张率随层数指数增长: 1, 2, 4, 8...
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [ChainedCausalConv(in_channels, out_channels, kernel_size, stride=1,
                                         dilation=dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)

        # ---- 特征维度计算 ----
        out_dim = num_channels[-1]
        if self.readout == "meanmax":
            feat_dim = out_dim * 2
        elif self.readout == "mix":
            feat_dim = out_dim * 3
        else:
            feat_dim = out_dim
        
        self.norm = nn.LayerNorm(feat_dim)

        # ---- 双头结构 (2+2) ----
        self.head_trigger = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2)
        )
        self.head_direction = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None, return_fused=False):
        # x shape: [B, T, F]
        B, T, F = x.shape
        
        # TCN 预期输入为 [B, C, T]
        x_in = x.transpose(1, 2).contiguous()
        
        # 通过 TCN 主干
        y = self.tcn(x_in) # [B, out_channels, T]
        y = y.transpose(1, 2).contiguous() # 转回 [B, T, C]

        # ---- Readout 层 ----
        if self.readout == "last":
            feat = y[:, -1, :]
        elif self.readout in {"meanmax", "mix"}:
            mean_pool = y.mean(dim=1)
            max_pool = y.max(dim=1).values
            if self.readout == "meanmax":
                feat = torch.cat([mean_pool, max_pool], dim=1)
            else: # mix
                feat = torch.cat([y[:, -1, :], mean_pool, max_pool], dim=1)
        
        feat = self.norm(feat)

        # ---- 双头输出 ----
        logits_trig = self.head_trigger(feat)
        logits_dir = self.head_direction(feat)

        if self.logit_clip is not None:
            logits_trig = torch.clamp(logits_trig, -self.logit_clip, self.logit_clip)
            logits_dir = torch.clamp(logits_dir, -self.logit_clip, self.logit_clip)

        if return_fused:
            p_trig = torch.softmax(logits_trig, dim=1)
            p_dir = torch.softmax(logits_dir, dim=1)
            
            p_neutral = p_trig[:, 0]
            p_act     = p_trig[:, 1]
            p_short   = p_act * p_dir[:, 0]
            p_long    = p_act * p_dir[:, 1]
            
            fused_probs = torch.stack([p_short, p_neutral, p_long], dim=1)
            fused_preds = torch.argmax(fused_probs, dim=1)
            return fused_preds, fused_probs

        return logits_trig, logits_dir

    def export_meta(self, **extra) -> dict:
        """
        导出模型架构参数，用于保存和后续加载。
        """
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "input_size": self.input_size,
            "num_channels": self.num_channels, # 记录 TCN 层结构
            "kernel_size": self.kernel_size,
            "readout": self.readout,
            "logit_clip": self.logit_clip,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        """
        从 Meta 数据中重建 TCN 模型。
        推理阶段 dropout 会被强制设为 0。
        """
        # 兼容性处理：从 state 中获取 input_size (针对动态特征增减的情况)
        input_size = state.get("channel", meta.get("input_size"))

        model = cls(
            input_size=input_size,
            num_channels=meta.get("num_channels", [64, 128, 256]),
            kernel_size=meta.get("kernel_size", 3),
            dropout=0.0,  # 推理模式关闭 dropout
            readout=meta.get("readout", "mix"),
            logit_clip=meta.get("logit_clip", None),
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)