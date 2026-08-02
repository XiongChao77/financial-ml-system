from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import torch

from model.evaluator import confusion_frame
from model.training_types import FitResult


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_single_run(
    *,
    model: torch.nn.Module,
    task_type: str,
    fit_result: FitResult,
    metrics: dict,
    save_dir: str,
    feature_names: list[str],
    feature_list: list[str],
    label_col: str,
    seq_len: int,
) -> dict:
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.pt")
    meta_path = os.path.join(save_dir, "meta.json")
    metrics_path = os.path.join(save_dir, "metrics.json")
    history_path = os.path.join(save_dir, "history.json")

    state_source = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "state_dict": state_source.state_dict(),
            "feature_list": feature_list,
            "classes": metrics["labels"],
            "channel": len(feature_names),
            "seq_len": seq_len,
            "feature_cols": feature_names,
            "label_col": label_col,
            "selection_metric": fit_result.selection_metric,
            "validation_score": fit_result.validation_score,
            "task_type": task_type,
        },
        model_path,
    )

    meta = model.export_meta(
        feature_cols=feature_names,
        label_col=label_col,
        classes=metrics["labels"],
        seq_len=seq_len,
        task_type=task_type,
    )
    _write_json(meta_path, meta)
    metrics_report = {
        "task_type": task_type,
        "selection": {
            "metric": fit_result.selection_metric,
            "validation_score": fit_result.validation_score,
            "validation_metrics": fit_result.validation_metrics,
        },
        "test": metrics,
    }
    _write_json(metrics_path, metrics_report)
    _write_json(history_path, fit_result.history)
    confusion_frame(metrics).to_csv(os.path.join(save_dir, "confusion_matrix.csv"))

    description = {
        "task_type": task_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {"main": {"model": "model.pt", "meta": "meta.json"}},
        "metrics": "metrics.json",
        "history": "history.json",
    }
    _write_json(os.path.join(save_dir, "task_description.json"), description)
    return metrics_report


def save_fusion_run(
    *,
    task_type: str,
    role_directories: dict[str, str],
    fusion_dir: str,
) -> dict:
    """Point a fusion artifact dir at two independently trained single-model runs.

    Unlike save_single_run, this does not train or evaluate anything: it just
    records, per role, the (relative) path to an existing save_single_run
    output directory so inference-time code (ModelHandler/FusionWrapper) can
    load both checkpoints and fuse them.
    """
    os.makedirs(fusion_dir, exist_ok=True)
    description = {
        "task_type": task_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            role: os.path.relpath(os.path.abspath(directory), os.path.abspath(fusion_dir))
            for role, directory in role_directories.items()
        },
    }
    _write_json(os.path.join(fusion_dir, "task_description.json"), description)
    return description
