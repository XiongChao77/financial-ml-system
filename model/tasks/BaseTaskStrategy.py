from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np

class BaseTaskStrategy(ABC):
    @abstractmethod
    def preprocess_labels(self, y: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def prepare_resources(self, y_train_raw: np.ndarray): pass

    @abstractmethod
    def compute_loss(self, model_output, targets): pass

    @abstractmethod
    def get_predictions(self, model, xb): pass

    @abstractmethod
    def get_model_out_dim(self) -> int: pass

    def log_warmup_info(self, i, yb):
        """可选：打印任务特定的 Batch 分布信息"""
        pass # 默认不执行任何操作

    def filter_data(self, X, y):
        """可选：过滤数据集。默认返回全部。"""
        return X, y

    @abstractmethod
    def call_model(self, model, xb, train_mode=True):
        """根据任务类型决定如何调用 model 的 forward"""
        pass