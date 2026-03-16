import torch
import torch.nn as nn

class FusionWrapper(nn.Module):
    """
    Inference-only wrapper.
    It does not save/load weights (handled by sub-models); it only chains models during forward.
    """
    def __init__(self, models_dict, mode):
        super().__init__()
        self.mode = mode
        # Use ModuleDict so sub-models switch together under eval(),
        # but we don't need to save this wrapper's state_dict
        self.models = nn.ModuleDict(models_dict)

    def forward(self, x, return_fused=True):
        if self.mode == "trigger_direction":
            return self._forward_trigger_direction(x)
        elif self.mode == "long_short_ovr":
            return self._forward_exclusive_filter(x)
        else:
            raise ValueError(f"Unknown pipeline mode: {self.mode}")

    def _forward_trigger_direction(self, x):
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
        # Normalize
        fused_probs = fused_probs / (fused_probs.sum(dim=1, keepdim=True) + 1e-8)
        fused_logits = torch.log(fused_probs + 1e-8)
        
        return fused_logits, fused_probs

    def _forward_long_short_ovr(self, x):
        # Long Model (1=Long)
        logits_long = self.models["long_ovr"](x)
        score_long = logits_long[:, 1]

        # Short Model (1=Short)
        logits_short = self.models["short_ovr"](x)
        score_short = logits_short[:, 1]

        bias = 0  # Negative is more aggressive; positive is more conservative
        score_neutral = torch.full_like(score_long, bias)
        
        fused_logits = torch.stack([score_short, score_neutral, score_long], dim=1)
        fused_probs = torch.softmax(fused_logits, dim=1)
        
        return fused_logits, fused_probs
    
    def _forward_exclusive_filter(self, x):
        """
        Exclusive fusion logic: only when one side emits a signal and the other stays neutral,
        do we treat it as a directional signal.
        Adapted for binary OVR sub-models:
        - model_s: [0: Other/Neutral, 1: Short]
        - model_l: [0: Other/Neutral, 1: Long]
        """
        model_keys = list(self.models.keys())
        # 1. Auto-detect model roles (by key names)
        short_key = next(k for k in model_keys if "short" in k.lower())
        long_key = next(k for k in model_keys if "long" in k.lower())
        
        model_s = self.models[short_key]
        model_l = self.models[long_key]
        
        # 2. Get raw logits [B, 2]
        logits_s = model_s(x)
        logits_l = model_l(x)
        
        # 3. Whether a sub-model emits a non-neutral signal (argmax == 1)
        is_sig_s = (torch.argmax(logits_s, dim=1) == 1)
        is_sig_l = (torch.argmax(logits_l, dim=1) == 1)
        
        # 4. Exclusive masks (only one side can be signal; the other must be neutral/0)
        # Valid short: Short==1 and Long==0
        mask_exclusive_short = is_sig_s & (~is_sig_l)
        # Valid long: Long==1 and Short==0
        mask_exclusive_long  = is_sig_l & (~is_sig_s)

        # 5. Build 3-class logit space [Short, Neutral, Long]
        # Initialize scores: set non-exclusive signal classes to very low (-100.0) for hard filtering
        score_short = torch.full_like(logits_s[:, 0], -100.0) 
        score_long  = torch.full_like(logits_l[:, 0], -100.0)
        # Neutral pivot at 0
        score_neutral = torch.zeros_like(logits_s[:, 0])

        # Only when exclusive condition holds, fill in original signal-class (index 1) logit confidence
        score_short[mask_exclusive_short] = logits_s[mask_exclusive_short, 1]
        score_long[mask_exclusive_long]  = logits_l[mask_exclusive_long, 1]

        # 6. Stack and softmax to produce standard 3-class probs [B, 3]
        fused_logits = torch.stack([score_short, score_neutral, score_long], dim=1)
        fused_probs = torch.softmax(fused_logits, dim=1)
        
        return fused_logits, fused_probs