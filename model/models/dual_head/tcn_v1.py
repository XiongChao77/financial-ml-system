import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from model.models.model_base import BaseTimeSeriesModel

class ChainedCausalConv(nn.Module):
    """
    Causal convolution block: left padding guarantees that output t only depends on inputs up to t.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        # Causal padding size
        padding = (kernel_size - 1) * dilation
        
        # First dilated convolution
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding) # trim the extra padding on the right to stay causal
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated convolution
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Residual connection: use a 1x1 convolution when the input and output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """Helper: trims the Conv1d output to make the convolution causal"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCN1D_V1(BaseTimeSeriesModel):
    """
    TCN1D_V1 - temporal convolutional network tuned for alpha modelling
    Architecture: several residual dilated convolution blocks + readout layer + dual head output (2+2)
    """
    MODEL_TYPE = "tcn"
    MODEL_VERSION = 1

    def __init__(
        self,
        input_size: int,
        num_channels: list = [64, 64, 64], # channels of every TCN layer
        kernel_size: int = 3,
        dropout: float = 0.2,
        readout: str = "mix", # 'last' | 'meanmax' | 'mix'
        head: str = "linear",
        logit_clip: float | None = None,
        **kwargs
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_channels = num_channels  # fixed here
        self.kernel_size = kernel_size    # stored too, the meta below needs it
        self.readout = readout
        self.logit_clip = logit_clip

        # ---- TCN backbone ----
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i # the dilation grows exponentially with depth: 1, 2, 4, 8...
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [ChainedCausalConv(in_channels, out_channels, kernel_size, stride=1,
                                         dilation=dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)

        # ---- feature dimensions ----
        out_dim = num_channels[-1]
        if self.readout == "meanmax":
            feat_dim = out_dim * 2
        elif self.readout == "mix":
            feat_dim = out_dim * 3
        else:
            feat_dim = out_dim
        
        self.norm = nn.LayerNorm(feat_dim)

        # ---- dual head (2+2) ----
        self.head_trigger = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2)
        )
        self.head_direction = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 2)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None):
        # x shape: [B, T, F]
        B, T, F = x.shape
        
        # The TCN expects [B, C, T]
        x_in = x.transpose(1, 2).contiguous()
        
        # Through the TCN backbone
        y = self.tcn(x_in) # [B, out_channels, T]
        y = y.transpose(1, 2).contiguous() # back to [B, T, C]

        # ---- readout layer ----
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

        # ---- dual head output ----
        logits_trig = self.head_trigger(feat)
        logits_dir = self.head_direction(feat)

        if self.logit_clip is not None:
            logits_trig = torch.clamp(logits_trig, -self.logit_clip, self.logit_clip)
            logits_dir = torch.clamp(logits_dir, -self.logit_clip, self.logit_clip)

        return logits_trig, logits_dir

    def export_meta(self, **extra) -> dict:
        """
        Export the architecture parameters, for saving and later loading.
        """
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "input_size": self.input_size,
            "num_channels": self.num_channels, # records the TCN layer structure
            "kernel_size": self.kernel_size,
            "readout": self.readout,
            "logit_clip": self.logit_clip,
            **extra,
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        """
        Rebuild the TCN model from the meta data.
        Dropout is forced to 0 for inference.
        """
        # Compatibility: read input_size from the state (for dynamically added/removed features)
        input_size = state.get("channel", meta.get("input_size"))

        model = cls(
            input_size=input_size,
            num_channels=meta.get("num_channels", [64, 128, 256]),
            kernel_size=meta.get("kernel_size", 3),
            dropout=0.0,  # dropout off in inference mode
            readout=meta.get("readout", "mix"),
            logit_clip=meta.get("logit_clip", None),
        )

        model.load_state_dict(state["state_dict"])
        return model.to(device)
