from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from model.models.fusion import fuse_trigger_direction_logits
from model.train_config import TrainConfig, TrainTask
from model.training_types import TensorSplit


def _balanced_weights(labels: np.ndarray, classes: list[int], device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=max(classes) + 1)
    observed = sum(counts[class_id] for class_id in classes)
    weights = [
        observed / (len(classes) * counts[class_id]) if counts[class_id] else 0.0
        for class_id in classes
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _three_class_expected_cost(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    config: TrainConfig,
) -> torch.Tensor:
    flip = float(config.flip_penalty)
    miss = float(config.miss_penalty)
    false_trade = float(config.false_trade_penalty)
    cost_matrix = probabilities.new_tensor(
        [
            [0.0, miss, flip],
            [false_trade, 0.0, false_trade],
            [flip, miss, 0.0],
        ]
    )
    normalizer = cost_matrix.max().clamp_min(torch.finfo(probabilities.dtype).eps)
    normalized_cost = cost_matrix / normalizer
    return (normalized_cost[labels] * probabilities).sum(dim=1).mean()


class TaskStrategy(ABC):
    labels: list[int]
    output_dim: int

    def __init__(self, config: TrainConfig, device: torch.device):
        self.config = config
        self.device = device

    def dataset_split(self, split: TensorSplit) -> TensorSplit:
        return split

    def configure(self, train_labels: np.ndarray) -> None:
        pass

    def sampler(self, labels: np.ndarray) -> Optional[WeightedRandomSampler]:
        return None

    @abstractmethod
    def forward(self, model: torch.nn.Module, inputs: torch.Tensor) -> Any:
        pass

    @abstractmethod
    def loss(self, output: Any, labels: torch.Tensor, returns: torch.Tensor, epoch: int) -> torch.Tensor:
        pass

    @abstractmethod
    def probabilities(self, output: Any) -> torch.Tensor:
        pass


class DirectThreeClassTask(TaskStrategy):
    labels = [0, 1, 2]
    output_dim = 3

    def configure(self, train_labels: np.ndarray) -> None:
        sampling_ratio = self.config.minority_sampling_ratio
        self.class_weights = (
            _balanced_weights(train_labels, self.labels, self.device)
            if sampling_ratio is None
            else None
        )

    def sampler(self, labels: np.ndarray) -> Optional[WeightedRandomSampler]:
        minority_ratio = self.config.minority_sampling_ratio
        if minority_ratio is None or len(labels) == 0:
            return None

        counts = np.bincount(labels.astype(int), minlength=max(self.labels) + 1)
        missing = [class_id for class_id in self.labels if counts[class_id] == 0]
        if missing:
            raise ValueError(f"Cannot sample missing Direct task classes: {missing}")

        majority_class = max(self.labels, key=lambda class_id: counts[class_id])
        minority_classes = [
            class_id for class_id in self.labels if class_id != majority_class
        ]
        majority_ratio = 1.0 - minority_ratio * len(minority_classes)
        target_ratios = {
            class_id: (
                majority_ratio if class_id == majority_class else minority_ratio
            )
            for class_id in self.labels
        }
        sample_weights = np.asarray(
            [target_ratios[int(label)] / counts[int(label)] for label in labels],
            dtype=np.float64,
        )
        return WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def forward(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        return model(inputs)

    def loss(self, output: torch.Tensor, labels: torch.Tensor, returns: torch.Tensor, epoch: int) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            output,
            labels,
            weight=self.class_weights,
            label_smoothing=float(getattr(self.config, "label_smoothing", 0.0)),
        )
        expected_cost = _three_class_expected_cost(self.probabilities(output), labels, self.config)
        return ce_loss + self.config.lambda_cost * expected_cost

    def probabilities(self, output) -> torch.Tensor:
        return torch.softmax(output, dim=1)


class BinaryTask(TaskStrategy):
    labels = [0, 1]
    output_dim = 2

    def __init__(self, task_type: str, config: TrainConfig, device: torch.device):
        super().__init__(config, device)
        self.task_type = task_type

    def dataset_split(self, split: TensorSplit) -> TensorSplit:
        y = split.y
        if self.task_type == TrainTask.TRIGGER:
            return TensorSplit(split.X, (y != 1).long(), split.returns)
        if self.task_type == TrainTask.DIRECTION:
            mask = y != 1
            return TensorSplit(split.X[mask], (y[mask] == 2).long(), split.returns[mask])
        if self.task_type == TrainTask.LONG_OVR:
            return TensorSplit(split.X, (y == 2).long(), split.returns)
        if self.task_type == TrainTask.SHORT_OVR:
            return TensorSplit(split.X, (y == 0).long(), split.returns)
        raise ValueError(f"Unsupported binary task: {self.task_type}")

    def sampler(self, labels: np.ndarray) -> Optional[WeightedRandomSampler]:
        minority_ratio = self.config.minority_sampling_ratio
        if minority_ratio is None or len(labels) == 0:
            return None
        positives = max(int((labels == 1).sum()), 1)
        negatives = max(int((labels == 0).sum()), 1)
        weights = np.where(
            labels == 1,
            minority_ratio / positives,
            (1.0 - minority_ratio) / negatives,
        )
        return WeightedRandomSampler(weights, len(weights), replacement=True)

    def forward(self, model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        return model(inputs)

    def loss(self, output: torch.Tensor, labels: torch.Tensor, returns: torch.Tensor, epoch: int) -> torch.Tensor:
        probabilities = torch.softmax(output, dim=1)
        p_negative, p_positive = probabilities[:, 0], probabilities[:, 1]

        penalty = torch.zeros_like(p_negative)
        positive = labels == 1
        negative = labels == 0

        if self.task_type == TrainTask.TRIGGER:
            penalty[positive] += p_negative[positive] * self.config.miss_penalty
        elif self.task_type == TrainTask.DIRECTION:
            penalty[positive] += p_negative[positive] * self.config.flip_penalty
            penalty[negative] += p_positive[negative] * self.config.flip_penalty
        else:
            penalty[positive] += p_negative[positive] * self.config.miss_penalty

        weights = 1.0 + penalty
        weights = weights / (weights.mean() + 1e-8)
        sample_loss = F.cross_entropy(output, labels, reduction="none")
        return (sample_loss * weights).mean()

    def probabilities(self, output) -> torch.Tensor:
        return torch.softmax(output, dim=1)


class DualHeadThreeClassTask(TaskStrategy):
    labels = [0, 1, 2]
    output_dim = 3

    def configure(self, train_labels: np.ndarray) -> None:
        trigger_labels = (train_labels != 1).astype(int)
        direction_labels = (train_labels[train_labels != 1] == 2).astype(int)
        self.main_weights = _balanced_weights(train_labels, self.labels, self.device)
        self.trigger_weights = _balanced_weights(trigger_labels, [0, 1], self.device)
        self.direction_weights = _balanced_weights(direction_labels, [0, 1], self.device)

    def forward(self, model: torch.nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return model(inputs)

    @staticmethod
    def probabilities(output: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        trigger_logits, direction_logits = output
        return fuse_trigger_direction_logits(trigger_logits, direction_logits)

    def loss(
        self,
        output: tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        returns: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        trigger_logits, direction_logits = output
        trigger_targets = (labels != 1).long()
        direction_targets = (labels == 2).long()
        action_mask = labels != 1
        smoothing = float(getattr(self.config, "label_smoothing", 0.0))

        trigger_loss = F.cross_entropy(
            trigger_logits,
            trigger_targets,
            weight=self.trigger_weights,
            label_smoothing=smoothing,
        )

        direction_loss = trigger_logits.new_zeros(())
        if action_mask.any():
            per_sample = F.cross_entropy(
                direction_logits[action_mask],
                direction_targets[action_mask],
                weight=self.direction_weights,
                reduction="none",
                label_smoothing=smoothing,
            )
            direction_loss = per_sample.mean()

        probabilities = self.probabilities(output)
        main_loss = F.nll_loss(
            torch.log(probabilities.clamp_min(1e-8)),
            labels,
            weight=self.main_weights,
        )

        expected_cost = _three_class_expected_cost(
            probabilities, labels, self.config
        )

        return (
            trigger_loss
            + self.config.lambda_dir * direction_loss
            + self.config.lambda_main * main_loss
            + self.config.lambda_cost * expected_cost
        )


class FusedThreeClassTask(DirectThreeClassTask):
    """Evaluation-only task for a FusionWrapper."""

    def forward(self, model: torch.nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return model(inputs)

    def loss(
        self,
        output: tuple[torch.Tensor, torch.Tensor],
        labels: torch.Tensor,
        returns: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        probabilities = self.probabilities(output)
        return F.nll_loss(torch.log(probabilities.clamp_min(1e-8)), labels)

    def probabilities(self, output: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _, probabilities = output
        return probabilities


def build_task(task_type: str, config: TrainConfig, device: torch.device, model: Optional[torch.nn.Module] = None) -> TaskStrategy:
    if task_type == TrainTask.DIRECT_3CLASS:
        if model is not None and (
            hasattr(model, "head_trigger") or hasattr(model, "head_direction")
        ):
            raise TypeError(f"DIRECT_3CLASS requires a single-head model, model:{config.model_cfg.model_type} {config.model_cfg.model_version}")
        return DirectThreeClassTask(config, device)
    if task_type == TrainTask.DUAL_HEAD_3CLASS:
        if model is not None and not (
            hasattr(model, "head_trigger") and hasattr(model, "head_direction")
        ):
            raise TypeError(f"DUAL_HEAD_3CLASS requires a model with trigger and direction heads, model:{config.model_cfg.model_type} {config.model_cfg.model_version}")
        return DualHeadThreeClassTask(config, device)
    if task_type in TrainTask.BINARY_TASKS:
        return BinaryTask(task_type, config, device)
    raise ValueError(f"Unsupported training task: {task_type}")
