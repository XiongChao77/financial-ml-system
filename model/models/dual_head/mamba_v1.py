import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.models.model_base import BaseTimeSeriesModel
# from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    """
    High performance Mamba block (driven by the official CUDA kernels)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        #  Instantiate the official Mamba layer directly, it already contains:
        # In_proj -> Conv1d -> S6 (Selective Scan) -> Out_proj
    #     self.mamba = Mamba(
    #         d_model=d_model,    # input dimension
    #         d_state=d_state,    # SSM state dimension (N)
    #         d_conv=d_conv,      # local convolution kernel size
    #         expand=expand       # expansion factor (E)
    #     )

    # def forward(self, x):
    #     # x: [B, L, D]
    #     #  this line now calls the high performance CUDA kernel of the 5090
    #     return self.mamba(x)

class Mamba1D_V1(BaseTimeSeriesModel):
    MODEL_TYPE = "mamba"
    MODEL_VERSION = 1
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
        readout: str = "mix",
        logit_clip: float | None = None,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.readout = readout
        self.logit_clip = logit_clip

        # 1. Input projection
        self.embedding = nn.Linear(input_size, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        # 2. Mamba backbone (core of the 2+2 architecture)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mamba": MambaBlock(d_model=d_model, d_state=d_state, expand=expand),
                "norm": nn.RMSNorm(d_model) # RMSNorm for better numerical stability on the 5090
            })
            for _ in range(n_layers)
        ])

        # 3. Readout & Heads
        feat_dim = d_model * (3 if readout == "mix" else 2 if readout == "meanmax" else 1)

        self.head_trigger = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        self.head_direction = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor):
        # x: [B, L, F]
        x = self.embedding(x)
        x = self.norm_in(x)

        # Through the Mamba layers (residual connections)
        for layer in self.layers:
            #  this computation is fully parallel now
            x = x + layer["mamba"](layer["norm"](x))

        # Readout (your original logic kept)
        if self.readout == "last":
            feat = x[:, -1, :]
        elif self.readout == "meanmax":
            feat = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        else: # mix
            feat = torch.cat([x[:, -1, :], x.mean(dim=1), x.max(dim=1).values], dim=1)
        
        logits_trig = self.head_trigger(feat)
        logits_dir = self.head_direction(feat)

        if self.logit_clip:
            logits_trig = torch.clamp(logits_trig, -self.logit_clip, self.logit_clip)
            logits_dir = torch.clamp(logits_dir, -self.logit_clip, self.logit_clip)

        return logits_trig, logits_dir

    def export_meta(self, **extra) -> dict:
        return {
            "model_type": self.MODEL_TYPE,
            "model_version": self.MODEL_VERSION,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "readout": self.readout,
            **extra
        }

    @classmethod
    def build_from_meta(cls, meta: dict, state: dict, device):
        model = cls(
            input_size=state.get("channel", meta.get("input_size")),
            d_model=meta.get("d_model", 128),
            n_layers=meta.get("n_layers", 4),
            readout=meta.get("readout", "mix")
        )
        model.load_state_dict(state["state_dict"])
        return model.to(device)
