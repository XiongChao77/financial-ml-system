import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from model.tasks.BaseTaskStrategy import BaseTaskStrategy

class BinaryStrategy(BaseTaskStrategy):
    """
    通用二分类策略：支持方案四中的 Trigger 和 Direction 任务
    """
    def __init__(self, train_cfg, task_type="trigger", device="cuda", logger=None):
        self.cfg = train_cfg
        self.task_type = task_type # # trigger, direction, long-others, short-others
        self.device = device
        self.logger = logger
        self.weights = {}
        self.criterion = None

    def preprocess_labels(self, y: np.ndarray) -> np.ndarray:
        """
        根据任务类型将原始 [0:Short, 1:Neutral, 2:Long] 转换为二分类 [0, 1]
        """
        if self.task_type == "trigger":
            # 找动作：1(Neutral)->0, [0,2](Action)->1
            return (y != 1).astype(int)
        
        elif self.task_type == "direction":
            # 找方向：此时数据已过滤，0(Short)->0, 2(Long)->1
            return np.where(y == 2, 1, 0)
            
        elif self.task_type == "long-others":
            # 🌟 看多 vs 其他：2(Long)->1, [0,1](Short/Neutral)->0
            return (y == 2).astype(int)
            
        elif self.task_type == "short-others":
            # 🌟 看空 vs 其他：0(Short)->1, [1,2](Neutral/Long)->0
            return (y == 0).astype(int)
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def filter_data(self, X, y):
        """
        数据过滤逻辑
        """
        if self.task_type == "direction":
            # 仅方向任务需要剔除 Neutral 样本
            mask = (y != 1)
            return X[mask], y[mask]
        
        # 🌟 Trigger, long-others, short-others 均需要全量数据进行训练
        return X, y

    # def prepare_resources(self, y_train_raw: np.ndarray):
    #     """计算二分类平衡权重并初始化 Loss"""
    #     classes = np.unique(y_train_raw)
    #     cw = compute_class_weight("balanced", classes=classes, y=y_train_raw)
    #     self.weights['main'] = torch.tensor(cw, dtype=torch.float32, device=self.device)
    #     self.criterion = nn.CrossEntropyLoss(weight=self.weights['main'])
        
    #     self.logger.info(f"⚖️ [Binary Weights] Task: {self.task_type} | Weights: {cw}")
    def prepare_resources(self, y_train_raw: np.ndarray):
        """
        手动设置权重以增加少数类预测错误的惩罚
        """
        # 统计样本分布，仅用于日志
        counts = np.bincount(y_train_raw)
        
        # 🌟 核心逻辑：手动设置权重向量
        # 假设 0 是多数类 (Others/Neutral)，1 是少数类 (Signal/Action)
        # 我们给 0 类基础权重 1.0，给 1 类显式的倍数惩罚
        if self.task_type == "direction":
            weights = [1.0, 1]
        else:
            weights = [1.0, 3]
        
        # 转换为 Tensor
        self.weights['main'] = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # 初始化带手动权重的 Loss
        self.criterion = nn.CrossEntropyLoss(weight=self.weights['main'])
        
        if self.logger:
            self.logger.info(f"⚖️ [Custom Penalty] Task: {self.task_type}")
            self.logger.info(f"   - Distribution: {np.unique(y_train_raw, return_counts=True)}")

    def compute_loss(self, model_output, targets):
        """
        计算二分类 Loss
        model_output: [B, 2] Tensor
        """
        loss = self.criterion(model_output, targets)
        # 返回 (total_loss, loss_dict) 以兼容 Tracker
        return loss, {"main": loss}

    def get_predictions(self, model, xb):
        """标准二分类推理"""
        # 显式处理可能的元组返回（以防万一）
        output = model(xb)
        if isinstance(output, tuple):
            logits = output[0]
            probs = torch.softmax(logits, dim=1)
        else:
            logits = output
            probs = torch.softmax(logits, dim=1)
        
        preds = torch.argmax(logits, dim=1)
        return preds, probs

    def get_model_out_dim(self) -> int:
        return 2

    def log_warmup_info(self, i, yb):
        if i < 3:
            labels, counts = np.unique(yb.cpu().numpy(), return_counts=True)
            self.logger.info(f"[Warmup Binary] {self.task_type} - Labels: {labels}, Counts: {counts}")

    def call_model(self, model, xb, train_mode=True):
        # V1/标准模型不需要参数，直接调用
        return model(xb)