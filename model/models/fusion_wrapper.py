import torch
import torch.nn as nn
import os, sys
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..','..'))
from model.train_config import *

class FusionWrapper(nn.Module):
    """
    Inference-only wrapper.
    It does not save/load weights (handled by sub-models); it only chains models during forward.
    """
    def __init__(self, models_dict, task_type):
        super().__init__()
        self.task_type = task_type
        # Use ModuleDict so sub-models switch together under eval(),
        # but we don't need to save this wrapper's state_dict
        self.models = nn.ModuleDict(models_dict)

    def forward(self, x, return_fused=True):
        if self.task_type == "trigger_direction":
            return self._forward_trigger_direction(x)
        elif self.task_type == "long_short_ovr":
            return self._forward_exclusive_filter(x)
        elif self.task_type in [TrainTask.SINGLE_MODEL_LONG_OVR.name, TrainTask.SINGLE_MODEL_SHORT_OVR.name]:
            return self._forward_one_side_long_short_ovr(x)
        elif self.task_type == TrainTask.LONG_SHORT_OVR.name:
            return self._forward_long_short_ovr(x)
        else:
            raise ValueError(f"Unknown pipeline task_type: {self.task_type}")

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

    def _forward_one_side_long_short_ovr(self, x):
        # Long Model (1=Long)
        logits = self.models[self.task_type](x)
        probs = torch.softmax(logits, dim=1)

        if self.task_type == TrainTask.SINGLE_MODEL_LONG_OVR.name:
            probs_3 = torch.cat(
                [torch.zeros_like(probs[:, :1]), probs],
                dim=1
            )
        elif self.task_type == TrainTask.SINGLE_MODEL_SHORT_OVR.name:
            probs_3 = torch.cat(
                [probs[:, 1:2], probs[:, 0:1], torch.zeros_like(probs[:, :1])],
                dim=1
            )

        preds = torch.argmax(probs_3, dim=1)

        return preds, probs_3
    
    def _forward_long_short_ovr(self, x, mode="conservative"):
        logits_long = self.models["long_ovr"](x)
        probs_long = torch.softmax(logits_long, dim=1)
        p_long = probs_long[:, 1]

        logits_short = self.models["short_ovr"](x)
        probs_short = torch.softmax(logits_short, dim=1)
        p_short = probs_short[:, 1]

        if mode == "conservative":
            fused_logits = self._fuse_conservative(p_long, p_short)
        elif mode == "aggressive":
            fused_logits = self._fuse_aggressive(p_long, p_short)
        else:
            raise ValueError(f"Unknown mode: {mode}")

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

    def _fuse_conservative(self, p_long, p_short):
        """
        规则：
            long only  → long
            short only → short
            其他（都不触发 / 都触发） → neutral
        """
        pred_long = p_long > 0.5
        pred_short = p_short > 0.5

        B = p_long.size(0)
        device = p_long.device

        fused_logits = torch.zeros(B, 3, device=device)

        # 基础分数
        fused_logits[:, 0] = p_short   # short
        fused_logits[:, 2] = p_long    # long

        # neutral bias
        bias = 0.0
        fused_logits[:, 1] = bias

        # 只允许单边触发
        mask_long = pred_long & (~pred_short)
        mask_short = pred_short & (~pred_long)

        # 其余 → neutral
        neutral_mask = ~(mask_long | mask_short)

        # 清掉冲突或无信号的方向
        fused_logits[neutral_mask, 0] = 0
        fused_logits[neutral_mask, 2] = 0

        return fused_logits
    
    def _fuse_aggressive(self, p_long, p_short):
        """
        规则：
            任意一边触发 → 给方向
            两边都触发 → 概率大的胜出
        """
        pred_long = p_long > 0.5
        pred_short = p_short > 0.5

        B = p_long.size(0)
        device = p_long.device

        fused_logits = torch.zeros(B, 3, device=device)

        # 基础分数
        fused_logits[:, 0] = p_short
        fused_logits[:, 2] = p_long

        # neutral baseline
        bias = 0.0
        fused_logits[:, 1] = bias

        # 两边都触发
        both_mask = pred_long & pred_short

        # short 赢
        short_win = both_mask & (p_short > p_long)

        # long 赢
        long_win = both_mask & (p_long >= p_short)

        # 抑制输的一方
        fused_logits[short_win, 2] = 0
        fused_logits[long_win, 0] = 0

        return fused_logits