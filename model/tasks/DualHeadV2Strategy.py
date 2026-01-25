
import torch,logging
from model.tasks.BaseTaskStrategy import BaseTaskStrategy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

class DualHeadV2Strategy(BaseTaskStrategy):
    """针对 V2 架构的双头融合策略"""
    def __init__(self, train_cfg, device, logger):
        self.cfg = train_cfg
        self.device = device
        self.logger = logger
        self.weights = {}
        self.criteria = {}

    def preprocess_labels(self, y):
        return y # V2 保持原始 [0, 1, 2]，内部拆分

    def prepare_resources(self, y_train_raw):
        """初始化权重和损失函数 (原 MTLManager 逻辑)"""
        # 1. 生成子任务标签用于计算权重
        y_trig = (y_train_raw != 1).astype(int) # 0: Neutral, 1: Action
        mask_dir = (y_train_raw != 1)
        y_dir = np.where(y_train_raw[mask_dir] == 2, 1, 0) # 0: Short, 1: Long

        # 2. 计算各层级平衡权重
        cw_main = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y_train_raw)
        cw_trig = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_trig)
        cw_dir = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_dir)

        # 3. 存储为 Tensor
        self.weights = {
            'main': torch.tensor(cw_main, dtype=torch.float32, device=self.device),
            'trig': torch.tensor(cw_trig, dtype=torch.float32, device=self.device),
            'dir': torch.tensor(cw_dir, dtype=torch.float32, device=self.device)
        }
        
        # 4. 初始化损失函数
        self.criteria = {
            'trig': nn.CrossEntropyLoss(weight=self.weights['trig']),
            'dir': nn.CrossEntropyLoss(weight=self.weights['dir'])
        }

        # 🌟 增加详细的权重打印逻辑
        self.logger.info(f"⚖️ [MTL Weights Prepared] Main (Short/Neu/Long): {cw_main} | Trig (Neu/Act): {cw_trig} | Dir (Short/Long): {cw_dir}")

    def compute_loss(self, model_output, targets):
        """计算复合损失并返回字典格式，以支持引擎解耦"""
        logits_trig, logits_dir = model_output
        t_trig = (targets != 1).long()
        t_dir = torch.where(targets == 2, 1, 0).long()
        act_mask = (targets != 1)

        # 1. 计算各子任务 Loss
        loss_trig = self.criteria['trig'](logits_trig, t_trig)
        loss_dir = torch.tensor(0.0, device=self.device)
        if act_mask.any():
            loss_dir = self.criteria['dir'](logits_dir[act_mask], t_dir[act_mask])

        # 2. 计算融合 Main Loss
        p_trig = torch.softmax(logits_trig, dim=1)
        p_dir = torch.softmax(logits_dir, dim=1)
        fused_probs = torch.stack([
            p_trig[:, 1] * p_dir[:, 0], 
            p_trig[:, 0],               
            p_trig[:, 1] * p_dir[:, 1]  
        ], dim=1)
        loss_main = F.nll_loss(torch.log(fused_probs + 1e-10), targets, weight=self.weights['main'])

        # 3. 对称性惩罚
        bias_loss = torch.abs(p_dir[act_mask, 0].mean() - p_dir[act_mask, 1].mean()) if act_mask.any() else torch.tensor(0.0, device=self.device)

        # 4. 计算总和
        total_loss = loss_main + \
                     (self.cfg.lambda_trig * loss_trig) + \
                     (self.cfg.lambda_dir * loss_dir) + \
                     (getattr(self.cfg, 'bias_lambda', 0.5) * bias_loss)
        
        # 🌟 核心修改：返回总 Loss 和包含子项的字典
        loss_dict = {
            "main": loss_main,
            "trig": loss_trig,
            "dir": loss_dir,
            "bias": bias_loss
        }
        
        return total_loss, loss_dict # 🌟 现在返回 2 个值了

    def get_predictions(self, model, xb):
        # 统一返回预测标签和概率分布
        return model(xb, return_fused=True)
    
    def get_model_out_dim(self):
        # 对于 V2 架构，虽然内部是双头，但在 Factory 构建时仍需指定逻辑输出维度
        return 3    
    
    def log_warmup_info(self, i, yb):
        """实现 3 分类的样本分布打印"""
        if i < 5:
            cnt_action = (yb != 1).sum().item()
            cnt_short = (yb == 0).sum().item()
            cnt_long  = (yb == 2).sum().item()
            self.logger.info(f"[Warmup Batch] action={cnt_action}, short={cnt_short}, long={cnt_long}")

    def call_model(self, model, xb, train_mode=True):
        # V2 模型需要 return_fused 参数
        return model(xb, return_fused=False if train_mode else True)