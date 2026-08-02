from typing import Tuple

import torch
import torch.nn as nn

from model.train_config import TrainTask

class FusionWrapper(nn.Module):
    """
    Inference-only wrapper.
    It does not save/load weights (handled by sub-models); it only chains models during forward.
    """
    def __init__(self, models_dict: dict[str, nn.Module], task_type: str):
        super().__init__()
        self.task_type = task_type
        # Use ModuleDict so sub-models switch together under eval(),
        # but we don't need to save this wrapper's state_dict
        self.models = nn.ModuleDict(models_dict)

    def forward(self, x: torch.Tensor, return_fused: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.task_type == TrainTask.TRIGGER_DIRECTION:
            return self._forward_trigger_direction(x)
        if self.task_type == TrainTask.LONG_SHORT_OVR:
            return self._forward_long_short_ovr(x)
        raise ValueError(f"Unknown pipeline task_type: {self.task_type}")

    def _forward_trigger_direction(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Trigger Inference
        logits_trig = self.models["trigger"](x)
        probs_trig = torch.softmax(logits_trig, dim=1) # [B, 2] (0:Neutral, 1:Action)
        
        # 2. Direction Inference
        logits_dir = self.models["direction"](x)
        probs_dir = torch.softmax(logits_dir, dim=1)   # [B, 2] (0:Short, 1:Long)
        
        # 3. Fusion Logic
        p_neutral = probs_trig[:, 0]
        p_action  = probs_trig[:, 1]
        
        p_short = p_action * probs_dir[:, 0]
        p_long  = p_action * probs_dir[:, 1]
        
        # Stack: [Short, Neutral, Long]
        fused_probs = torch.stack([p_short, p_neutral, p_long], dim=1)
        fused_logits = torch.log(fused_probs.clamp_min(1e-8))
        
        return fused_logits, fused_probs

    def _forward_long_short_ovr(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_long = self.models["long_ovr"](x)
        probs_long = torch.softmax(logits_long, dim=1)
        p_long = probs_long[:, 1]

        logits_short = self.models["short_ovr"](x)
        probs_short = torch.softmax(logits_short, dim=1)
        p_short = probs_short[:, 1]

        # One-sided evidence maps to its direction. No signal and conflicting
        # signals both map to neutral. These three terms already sum to one.
        p_short_only = p_short * (1.0 - p_long)
        p_long_only = p_long * (1.0 - p_short)
        p_neutral = (1.0 - p_short) * (1.0 - p_long) + p_short * p_long
        fused_probs = torch.stack((p_short_only, p_neutral, p_long_only), dim=1)
        fused_logits = torch.log(fused_probs.clamp_min(1e-8))
        return fused_logits, fused_probs
