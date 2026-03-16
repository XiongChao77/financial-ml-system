import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from model.tasks.BaseTaskStrategy import BaseTaskStrategy

class BinaryStrategy(BaseTaskStrategy):
    """
    Generic binary classification strategy: supports Trigger and Direction tasks.
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
        Convert raw [0:Short, 1:Neutral, 2:Long] into binary labels [0, 1] based on task_type.
        """
        if self.task_type == "trigger":
            # Trigger: 1(Neutral)->0, [0,2](Action)->1
            return (y != 1).astype(int)
        
        elif self.task_type == "direction":
            # Direction: after filtering, 0(Short)->0, 2(Long)->1
            return np.where(y == 2, 1, 0)
            
        elif self.task_type == "long-others":
            # Long vs others: 2(Long)->1, [0,1](Short/Neutral)->0
            return (y == 2).astype(int)
            
        elif self.task_type == "short-others":
            # Short vs others: 0(Short)->1, [1,2](Neutral/Long)->0
            return (y == 0).astype(int)
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def filter_data(self, X, y):
        """
        Data filtering logic.
        """
        if self.task_type == "direction":
            # Only direction task needs to drop Neutral samples
            mask = (y != 1)
            return X[mask], y[mask]
        
        # Trigger, long-others, short-others all train on full data
        return X, y

    # def prepare_resources(self, y_train_raw: np.ndarray):
    #     """Compute balanced weights for binary classification and initialize loss."""
    #     classes = np.unique(y_train_raw)
    #     cw = compute_class_weight("balanced", classes=classes, y=y_train_raw)
    #     self.weights['main'] = torch.tensor(cw, dtype=torch.float32, device=self.device)
    #     self.criterion = nn.CrossEntropyLoss(weight=self.weights['main'])
        
    #     self.logger.info(f"⚖️ [Binary Weights] Task: {self.task_type} | Weights: {cw}")
    def prepare_resources(self, y_train_raw: np.ndarray):
        """
        Manually set class weights to penalize minority-class mistakes more.
        """
        # Sample distribution (logging only)
        counts = np.bincount(y_train_raw)
        
        # Core logic: manual weight vector
        # Assume 0 is majority (Others/Neutral), 1 is minority (Signal/Action)
        # Give class 0 base weight 1.0 and class 1 an explicit multiplier penalty
        if self.task_type == "direction":
            weights = [1.0, 1]
        else:
            weights = [1.0, 3]
        
        # Convert to tensor
        self.weights['main'] = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Initialize weighted loss
        self.criterion = nn.CrossEntropyLoss(weight=self.weights['main'])
        
        if self.logger:
            self.logger.info(f"⚖️ [Custom Penalty] Task: {self.task_type}")
            self.logger.info(f"   - Distribution: {np.unique(y_train_raw, return_counts=True)}")

    def compute_loss(self, model_output, targets):
        """
        Compute binary classification loss.
        model_output: [B, 2] Tensor
        """
        loss = self.criterion(model_output, targets)
        # Return (total_loss, loss_dict) for tracker compatibility
        return loss, {"main": loss}

    def get_predictions(self, model, xb):
        """Standard binary inference."""
        # Handle possible tuple returns defensively
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
        # V1/standard models don't need extra args
        return model(xb)