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
        """Optional: print task-specific batch distribution info."""
        pass  # No-op by default

    def filter_data(self, X, y):
        """Optional: filter dataset. Default returns all."""
        return X, y

    @abstractmethod
    def call_model(self, model, xb, train_mode=True):
        """Call model.forward based on task type."""
        pass