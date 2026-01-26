import torch
import torch.nn as nn

class FusionWrapper(nn.Module):
    """
    推理专用包装器：
    不负责保存/加载权重（由子模型自己负责），只负责在 forward 时把两个模型串起来。
    """
    def __init__(self, models_dict, mode):
        super().__init__()
        self.mode = mode
        # 使用 ModuleDict 确保子模型在 eval() 时能同步切换，
        # 但我们不需要保存这个 Wrapper 的 state_dict
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

        bias = 0  # 负值增加进攻性，正值增加保守性
        score_neutral = torch.full_like(score_long, bias)
        
        fused_logits = torch.stack([score_short, score_neutral, score_long], dim=1)
        fused_probs = torch.softmax(fused_logits, dim=1)
        
        return fused_logits, fused_probs
    
    def _forward_exclusive_filter(self, x):
        """
        互斥融合逻辑：只有一方给出信号 (Signal)，另一方为中性 (Other)，才是方向信号。
        适配二分类 OVR 子模型：
        - model_s: [0: Other/Neutral, 1: Short]
        - model_l: [0: Other/Neutral, 1: Long]
        """
        model_keys = list(self.models.keys())
        # 1. 自动识别模型角色 (基于键名)
        short_key = next(k for k in model_keys if "short" in k.lower())
        long_key = next(k for k in model_keys if "long" in k.lower())
        
        model_s = self.models[short_key]
        model_l = self.models[long_key]
        
        # 2. 获取原始 Logits [B, 2]
        logits_s = model_s(x)
        logits_l = model_l(x)
        
        # 3. 判定子模型是否给出非 Neutral 信号 (即 Argmax 为 1)
        is_sig_s = (torch.argmax(logits_s, dim=1) == 1)
        is_sig_l = (torch.argmax(logits_l, dim=1) == 1)
        
        # 4. 构造互斥掩码 (只有一方为信号，另一方必须为 Neutral/0)
        # 有效空头：Short 是 1，Long 必须是 0
        mask_exclusive_short = is_sig_s & (~is_sig_l)
        # 有效多头：Long 是 1，Short 必须是 0
        mask_exclusive_long  = is_sig_l & (~is_sig_s)

        # 5.  构造 3 分类 Logit 竞争空间 [Short, Neutral, Long]
        # 初始化分数：不满足互斥条件的信号类分数设为极低 (-100.0) 以实现硬过滤
        score_short = torch.full_like(logits_s[:, 0], -100.0) 
        score_long  = torch.full_like(logits_l[:, 0], -100.0)
        # Neutral 支点设为 0
        score_neutral = torch.zeros_like(logits_s[:, 0])

        # 仅在满足互斥信号时，填入子模型对信号类 (index 1) 的原始 Logit 信心
        score_short[mask_exclusive_short] = logits_s[mask_exclusive_short, 1]
        score_long[mask_exclusive_long]  = logits_l[mask_exclusive_long, 1]

        # 6. 堆叠并 Softmax，产出标准的 3 分类概率分布 [B, 3]
        # 这样 p_long = probs_all[:, 2] 就不再报错了
        fused_logits = torch.stack([score_short, score_neutral, score_long], dim=1)
        fused_probs = torch.softmax(fused_logits, dim=1)
        
        return fused_logits, fused_probs