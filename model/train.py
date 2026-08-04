#!/usr/bin/env python3
from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from data_process import common
from model.artifacts import save_fusion_run, save_single_run
from model.data_loader import TimeSeriesWindowDataset
from model.evaluator import evaluate_classification, log_evaluation_report
from model.model_factory import ModelFactory
from model.tasks.strategies import build_task
from model.train_config import (
    DataConfig,
    SingleModelConfig,
    TrainConfig,
    TrainTask,
    feature_direction_map,
)
from model.tasks.strategies import TaskStrategy
from model.trainer import build_trainer
from model.training_types import DataSplits, TensorSplit


@dataclass(frozen=True)
class PreparedData:
    splits: DataSplits
    feature_names: list[str]
    requested_features: list[str]
    feature_count: int
    prep_output_dir: str


@dataclass
class SingleRun:
    task_type: str
    model: torch.nn.Module
    report: dict
    save_dir: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_feature_direction(
    inputs: torch.Tensor,
    feature_names: list[str],
    direction_map: dict[str, int],
    logger: logging.Logger,
) -> None:
    indices = [
        index
        for index, feature_name in enumerate(feature_names)
        if direction_map.get(feature_name, 1) == -1
    ]
    if indices:
        inputs[:, :, indices] *= -1
        logger.info("Flipped %d features according to ic_direction", len(indices))


def split_by_time(
    dataset: TimeSeriesWindowDataset,
    data_config: DataConfig,
) -> DataSplits:
    """Split once on the original timeline; every task derives from these splits."""
    sample_count = len(dataset)
    train_end = int(sample_count * data_config.train_ratio)
    validation_end = int(sample_count * (data_config.train_ratio + data_config.val_ratio))

    purge = 0
    if data_config.purge_overlap:
        purge = max(math.ceil(dataset.seq_len / max(dataset.stride, 1)) - 1, 0)

    validation_start = train_end + purge
    test_start = validation_end + purge
    if not (
        0 < train_end <= validation_start < validation_end <= test_start < sample_count
    ):
        raise ValueError(
            "Dataset is too small for the requested train/validation/test split "
            f"and purge gap ({purge} samples)"
        )

    def make(start: int, end: int) -> TensorSplit:
        return TensorSplit(
            dataset.X[start:end],
            dataset.y[start:end],
            dataset.returns[start:end],
        )

    return DataSplits(
        train=make(0, train_end),
        validation=make(validation_start, validation_end),
        test=make(test_start, sample_count),
    )


def prepare_data(
    *,
    config: TrainConfig,
    prep_output_dir: str,
    cache_dir: str,
    logger: logging.Logger,
) -> PreparedData:
    frame = common.load_train_df_from_dir(prep_output_dir)
    preparation = common.load_pre_params_from_dir(prep_output_dir)
    interval_ms = common.get_interval_ms(preparation.interval)
    requested_features = list(config.feature_conf_list)

    dataset = TimeSeriesWindowDataset(
        df=frame,
        kline_interval_ms=interval_ms,
        feature_cols=requested_features,
        label_col=config.data_cfg.label_col,
        seq_len=config.model_cfg.seq_len,
        stride=config.stride,
        cache_path=os.path.join(cache_dir, "train_cache.pt"),
        use_cache=config.use_cache,
        show_feature_distribution=True,
    )
    directions = {
        feature_name: feature_direction_map.get(feature_name, 1)
        for feature_name in requested_features
    }
    apply_feature_direction(dataset.X, dataset.feature_names, directions, logger)

    splits = split_by_time(dataset, config.data_cfg)
    logger.info(
        "Prepared %d samples: train=%d validation=%d test=%d, features=%d",
        len(dataset),
        len(splits.train),
        len(splits.validation),
        len(splits.test),
        dataset.feature_count,
    )
    return PreparedData(
        splits=splits,
        feature_names=list(dataset.feature_names),
        requested_features=requested_features,
        feature_count=dataset.feature_count,
        prep_output_dir=prep_output_dir,
    )


def _output_dim(task_type: str) -> int:
    return 2 if task_type in TrainTask.BINARY_TASKS else 3


def _build_model(config: TrainConfig, task_type: str, input_size: int, device: torch.device) -> torch.nn.Module:
    model_parameters = asdict(config.model_cfg)
    return ModelFactory.build_for_training(
        device=device,
        input_size=input_size,
        n_classes=_output_dim(task_type),
        **model_parameters,
    )


def _dataset_splits(task:TaskStrategy, raw: DataSplits) -> DataSplits:
    transformed = DataSplits(
        train=task.dataset_split(raw.train),
        validation=task.dataset_split(raw.validation),
        test=task.dataset_split(raw.test),
    )
    for name, split in (
        ("train", transformed.train),
        ("validation", transformed.validation),
        ("test", transformed.test),
    ):
        if not len(split):
            raise ValueError(f"{name} split is empty after task transformation")
    missing_train_labels = set(task.labels) - set(transformed.train.labels.tolist())
    if missing_train_labels:
        raise ValueError(
            f"Training split is missing labels required by the task: {sorted(missing_train_labels)}"
        )
    return transformed


def train_single(
    *,
    task_type: str,
    config: TrainConfig,
    prepared: PreparedData,
    save_dir: str,
    device: torch.device,
    logger: logging.Logger,
) -> SingleRun:
    if task_type in TrainTask.COMBO_TASKS:
        raise ValueError(f"{task_type} is a combo task, not a single-model task")

    set_seed(config.seed)
    model = _build_model(config, task_type, prepared.feature_count, device)
    task = build_task(task_type, config, device, model=model)
    splits = _dataset_splits(task, prepared.splits)
    trainer = build_trainer(model, config, device, logger)

    fit_result = trainer.fit(model, task, splits.train, splits.validation)
    prediction = trainer.predict(model, task, splits.test)
    metrics = evaluate_classification(prediction, task.labels)
    report = save_single_run(
        model=model,
        task_type=task_type,
        fit_result=fit_result,
        metrics=metrics,
        save_dir=save_dir,
        prep_output_dir=prepared.prep_output_dir,
        feature_names=prepared.feature_names,
        feature_list=prepared.requested_features,
        label_col=config.data_cfg.label_col,
        seq_len=config.model_cfg.seq_len,
    )
    logger.info(
        "Completed %s: validation_%s=%.4f test_macro_f1=%.4f",
        task_type,
        fit_result.selection_metric,
        fit_result.validation_score,
        metrics["macro_f1"],
    )
    model_label = "Best_F1" if fit_result.selection_metric == "macro_f1" else "Best_Loss"
    log_evaluation_report(
        logger,
        task_type=task_type,
        metrics=metrics,
        save_dir=save_dir,
        model_label=model_label,
    )
    return SingleRun(task_type=task_type, model=model, report=report, save_dir=save_dir)


def fusion_trigger_dir(logger: logging.Logger, trigger_dir: str, direction_dir: str, fusion_dir: str) -> dict:
    """Fuse two independently trained models (trigger + direction) into one artifact dir.

    Both models must already be trained (e.g. via `train()` with `TrainTask.TRIGGER`
    and `TrainTask.DIRECTION`). This does not retrain or evaluate anything; it only
    records where each checkpoint lives so inference code can load and fuse them.
    """
    description = save_fusion_run(
        task_type=TrainTask.TRIGGER_DIRECTION,
        role_directories={"trigger": trigger_dir, "direction": direction_dir},
        fusion_dir=fusion_dir,
    )
    logger.info("Fused TRIGGER_DIRECTION: trigger=%s direction=%s -> %s", trigger_dir, direction_dir, fusion_dir)
    return description


def fusion_long_short_ovr(logger: logging.Logger, long_ovr_dir: str, short_ovr_dir: str, fusion_dir: str) -> dict:
    """Fuse two independently trained models (long OvR + short OvR) into one artifact dir.

    Both models must already be trained (e.g. via `train()` with `TrainTask.LONG_OVR`
    and `TrainTask.SHORT_OVR`). This does not retrain or evaluate anything; it only
    records where each checkpoint lives so inference code can load and fuse them.
    """
    description = save_fusion_run(
        task_type=TrainTask.LONG_SHORT_OVR,
        role_directories={"long_ovr": long_ovr_dir, "short_ovr": short_ovr_dir},
        fusion_dir=fusion_dir,
    )
    logger.info("Fused LONG_SHORT_OVR: long_ovr=%s short_ovr=%s -> %s", long_ovr_dir, short_ovr_dir, fusion_dir)
    return description


def train(
    *,
    config: TrainConfig,
    prep_output_dir: str = common.DATA_OUT_DIR,
    save_dir: str = common.TRAIN_OUT_DIR,
    logger: logging.Logger | None = None,
) -> dict:
    """Single public training entry point: trains exactly one model for one TrainConfig.

    To combine two single-head models (trigger+direction or long/short OvR), train each
    with its own TrainConfig via this function, then fuse the saved checkpoints with
    fusion_trigger_dir / fusion_long_short_ovr.
    """
    logger = logger or logging.getLogger("train")
    task_type = config.train_task
    if task_type in TrainTask.COMBO_TASKS:
        raise ValueError(
            f"{task_type} trains two models jointly and is no longer supported here; "
            "train each sub-task separately with train() and fuse the saved checkpoints "
            "with fusion_trigger_dir/fusion_long_short_ovr"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    prepared = prepare_data(
        config=config,
        prep_output_dir=prep_output_dir,
        cache_dir=save_dir,
        logger=logger,
    )
    return train_single(
        task_type=task_type,
        config=config,
        prepared=prepared,
        save_dir=save_dir,
        device=device,
        logger=logger,
    ).report


if __name__ == "__main__":
    session_logger, _ = common.setup_session_logger(sub_folder="train", file_level=logging.DEBUG)
    train(
        config=SingleModelConfig,
        save_dir=os.path.join(common.TRAIN_OUT_DIR, SingleModelConfig.train_task),
        logger=session_logger,
    )
