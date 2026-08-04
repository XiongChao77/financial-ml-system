#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os

# ---------------------------------------------------------------------------
# Important: the BLAS thread environment variables (OMP/MKL etc.) must be set before numpy / torch
# is imported for the first time to be 100% effective (an OpenMP runtime usually reads them once,
# when the first parallel region is entered, and freezes them afterwards). So they are set before "import torch".
# The actual thread count comes from the --torch-threads command line argument; a conservative default is used
# here and main() sets it again through torch.set_num_threads once the real arguments are known.
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

import batch_train
from data_process import common
from data_process.utils import param_hash
from trade.runner import backtest_runner
from model.train import fusion_trigger_dir, TrainTask
from model import model_loader
from model import data_loader

TRAIN_REPORTS_FILE = "train_reports.jsonl"
SIM_REPORTS_FILE = "sim_reports.jsonl"
SELECTED_MODELS_FILE = "selected_models.json"
SELECTED_MODELS_CSV = "selected_models_summary.csv"

# Note: this global cache is now only valid "inside one process, during one batch".
# In multiprocess mode every subprocess has its own memory, so the cache cannot be shared across processes.
# To reuse the cache as much as possible, run_backtests packs the tasks of the same (pre_key, train_compatibility)
# into one batch and hands it to a single subprocess to run sequentially.
QUICK_DS_CACHE = {}


@dataclass
class ModelRef:
    task_type: str
    task_hash: str
    model_type: str
    model_version: int
    score: float
    score_source: str
    pre_key: str
    train_compatibility: str
    save_dir: str
    pre_params: Dict[str, Any]
    train_params: Dict[str, Any]
    metrics: Dict[str, Any]


@dataclass
class FusionTask:
    fusion_hash: str
    pre_key: str
    train_compatibility: str
    trigger: ModelRef
    direction: ModelRef
    fusion_dir: str = ""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def setup_logger(exp_dir: str) -> logging.Logger:
    os.makedirs(exp_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    file_handler = logging.FileHandler(
        os.path.join(exp_dir, "experiment.log"),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return logging.getLogger("batch_simulation")


def score_trigger(metrics: Dict[str, Any]) -> Tuple[float, str]:
    best = metrics["Best_F1"]
    pos = best["per_class"]["1"]

    pos_f1 = pos["f1"]
    pos_recall = pos["recall"]
    pos_precision = pos["precision"]
    mcc = best["mcc"]

    score = pos_f1

    return score, "pos_f1"


def score_direction(metrics: Dict[str, Any]) -> Tuple[float, str]:
    best = metrics["Best_F1"]

    score = best["macro_f1"]

    return score, "macro_f1"


def calc_score(task_type: str, metrics: Dict[str, Any]) -> Tuple[float, str]:
    if task_type == TrainTask.SINGLE_MODEL_TRIGGER:
        return score_trigger(metrics)

    if task_type == TrainTask.SINGLE_MODEL_DIR:
        return score_direction(metrics)

    raise ValueError(f"Unsupported task_type: {task_type}")


def build_model_registry_from_reports(
    logger: logging.Logger,
    train_exp_dir: str,
) -> Dict[Tuple[str, str], Dict[str, List[ModelRef]]]:
    reports_path = os.path.join(train_exp_dir, TRAIN_REPORTS_FILE)
    reports = load_jsonl(reports_path)

    registry = defaultdict(lambda: defaultdict(list))

    for r in reports:
        task_type = r["task_type"]
        metrics = r["metrics"]
        pre_params = r["pre_params"]
        train_params = r["train_params"]

        score, score_source = calc_score(task_type, metrics)

        pre_key = param_hash(pre_params)
        train_compatibility = train_params["train_compatibility"]

        ref = ModelRef(
            task_type=task_type,
            task_hash=r["task_hash"],
            model_type=r["model_type"],
            model_version=r["model_version"],
            score=score,
            score_source=score_source,
            pre_key=pre_key,
            train_compatibility=train_compatibility,
            save_dir=r["save_dir"],
            pre_params=pre_params,
            train_params=train_params,
            metrics=metrics,
        )

        registry[(pre_key, train_compatibility)][task_type].append(ref)

    logger.info(f"Loaded groups: {len(registry)}")
    return registry


def make_fusion_hash(trigger: ModelRef, direction: ModelRef) -> str:
    payload = {
        "pre_key": trigger.pre_key,
        "train_compatibility": trigger.train_compatibility,
        "trigger_hash": trigger.task_hash,
        "dir_hash": direction.task_hash,
    }

    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def select_fusion_pairs(
    logger: logging.Logger,
    registry: Dict[Tuple[str, str], Dict[str, List[ModelRef]]],
) -> List[FusionTask]:

    def select_representative_models(
        models: List[ModelRef],
        top_k: int = 5,
        mid_k: int = 5,
    ) -> List[ModelRef]:
        selected: Dict[str, ModelRef] = {}

        groups: Dict[Tuple[str, int], List[ModelRef]] = {}

        for m in models:
            key = (m.model_type, m.model_version)
            groups.setdefault(key, []).append(m)

        for _, group in groups.items():
            group = sorted(group, key=lambda x: x.score, reverse=True)
            n = len(group)

            top_models = group[:top_k]

            mid = n // 2
            half = mid_k // 2
            start = max(0, mid - half)
            end = min(n, start + mid_k)

            start = max(0, end - mid_k)

            mid_models = group[start:end]

            for m in top_models + mid_models:
                selected[m.task_hash] = m

        return list(selected.values())

    fusion_tasks = []

    for (pre_key, compatibility), task_map in registry.items():
        triggers = task_map[TrainTask.SINGLE_MODEL_TRIGGER]
        dirs = task_map[TrainTask.SINGLE_MODEL_DIR]

        logger.info(
            f"pre={pre_key}, compat={compatibility}: "
            f"selected triggers={len(triggers)}, selected dirs={len(dirs)}"
        )

        for trigger_model in triggers:
            for dir_model in dirs:
                fusion_hash = make_fusion_hash(trigger_model, dir_model)

                fusion_tasks.append(
                    FusionTask(
                        fusion_hash=fusion_hash,
                        pre_key=pre_key,
                        train_compatibility=compatibility,
                        trigger=trigger_model,
                        direction=dir_model,
                    )
                )

    fusion_tasks.sort(
        key=lambda x: x.trigger.score + x.direction.score,
        reverse=True,
    )

    logger.info(f"Selected fusion pairs: {len(fusion_tasks)}")

    for i, task in enumerate(fusion_tasks[:20], start=1):
        logger.info(
            f"[{i}] pre={task.pre_key}, compat={task.train_compatibility}, "
            f"trigger={task.trigger.model_type}v{task.trigger.model_version}, "
            f"trigger_score={task.trigger.score:.4f}, "
            f"dir={task.direction.model_type}v{task.direction.model_version}, "
            f"dir_score={task.direction.score:.4f}"
        )

    return fusion_tasks


def infer_prep_output_dir(save_dir: str) -> str:
    return str(Path(save_dir).parents[1])


def build_train_cfg(task: FusionTask):
    train_cfg = batch_train._config_from_dict_train(
        copy.deepcopy(task.trigger.train_params)
    )

    if hasattr(train_cfg, "__post_init__"):
        train_cfg.__post_init__()

    return train_cfg


def load_pred_df_for_quick_eval(
    logger: logging.Logger,
    prep_output_dir: str,
    fusion_dir: str,
    pre_para: common.BaseDefine,
    device: str,
    task: FusionTask,
):
    if isinstance(device, str):
        device = torch.device(device)

    interval_ms = common.get_interval_ms(pre_para.interval)

    handler = model_loader.ModelHandler(
        tarin_out_path=fusion_dir,
        device=device,
    )

    global QUICK_DS_CACHE
    df = common.load_test_df_from_dir(prep_output_dir)

    cache_key = (task.pre_key, task.train_compatibility)
    if cache_key not in QUICK_DS_CACHE:
        QUICK_DS_CACHE[cache_key] = data_loader.TimeSeriesWindowDataset(
            df=df,
            kline_interval_ms=interval_ms,
            feature_cols=handler.feature_cols,
            label_col=handler.label_col,
            seq_len=handler.seq_len,
            is_live=False,
        )
    ds = QUICK_DS_CACHE[cache_key]

    df_with_pred, model_stats = handler.predict_with_ds(
        ds,
        df,
        is_live=False,
        diff_thresh=None,
    )

    first_valid_idx = df_with_pred["pred"].first_valid_index()

    if first_valid_idx is None:
        logger.warning("QuickEval: no valid predictions.")
        return None, model_stats

    df_with_pred = df_with_pred.loc[first_valid_idx:].copy()

    logger.info(
        f"QuickEval range: "
        f"{df_with_pred['open_time_date_utc'].min()} "
        f"to {df_with_pred['open_time_date_utc'].max()}"
    )

    return df_with_pred, model_stats


def calc_fixed_horizon_signal_avg_return(
    df_with_pred,
    horizon: int,
    fee_per_trade_list=(0.0, 0.005)
):
    close = df_with_pred["close"].to_numpy(dtype=np.float64)
    pred = df_with_pred["pred"].to_numpy()

    n = len(df_with_pred)
    valid_n = n - horizon

    if valid_n <= 0:
        return {
            f"{fee:g}": {
                "signal_count": 0,
                "signal_avg_return": 0.0,
                "signal_median_return": 0.0,
                "signal_win_rate": 0.0,
                "long_count": 0,
                "short_count": 0,
                "fee_per_trade": float(fee),
                "horizon": int(horizon),
            }
            for fee in fee_per_trade_list
        }

    pred = pred[:valid_n]
    entry_close = close[:valid_n]
    exit_close = close[horizon:]

    signal_mask = np.isin(pred, [common.Signal.NEGATIVE, common.Signal.POSITIVE])

    if signal_mask.sum() == 0:
        return {
            f"{fee:g}": {
                "signal_count": 0,
                "signal_avg_return": 0.0,
                "signal_median_return": 0.0,
                "signal_win_rate": 0.0,
                "long_count": 0,
                "short_count": 0,
                "fee_per_trade": float(fee),
                "horizon": int(horizon),
            }
            for fee in fee_per_trade_list
        }

    signal_pred = pred[signal_mask]
    entry = entry_close[signal_mask]
    exit_ = exit_close[signal_mask]

    raw_ret = np.zeros_like(entry, dtype=np.float64)

    long_mask = signal_pred == common.Signal.POSITIVE
    short_mask = signal_pred == common.Signal.NEGATIVE

    raw_ret[long_mask] = exit_[long_mask] / entry[long_mask] - 1.0
    raw_ret[short_mask] = entry[short_mask] / exit_[short_mask] - 1.0

    out = {}

    for fee_per_trade in fee_per_trade_list:
        net_ret = raw_ret - 2.0 * float(fee_per_trade)
        fee_key = f"{fee_per_trade:g}"

        out[fee_key] = {
            "signal_count": int(len(net_ret)),
            "signal_avg_return": round(float(np.mean(net_ret)), 6),
            "signal_median_return": round(float(np.median(net_ret)), 6),
            "signal_win_rate": round(float(np.mean(net_ret > 0)), 4),
            "long_count": int(np.sum(long_mask)),
            "short_count": int(np.sum(short_mask)),
            "fee_per_trade": float(fee_per_trade),
            "horizon": int(horizon),
        }

    return out


def quick_signal_eval(
    logger: logging.Logger,
    pre_para,
    prep_output_dir,
    task: FusionTask,
    fusion_dir: str,
    device: str,
    fee_per_trade_list=(0.0,)
):
    df_with_pred, model_stats = load_pred_df_for_quick_eval(
        logger=logger,
        prep_output_dir=prep_output_dir,
        fusion_dir=fusion_dir,
        pre_para=pre_para,
        device=device,
        task=task
    )

    horizon = int(pre_para.predict_num)

    signal_return = calc_fixed_horizon_signal_avg_return(
        df_with_pred=df_with_pred,
        horizon=horizon,
        fee_per_trade_list=fee_per_trade_list,
    )

    zero_fee_key = f"{0.0:g}"
    passed = signal_return[zero_fee_key]["signal_avg_return"] > 0

    logger.info(
        f"QuickEval fusion={task.fusion_hash} | "
        f"fee=0 avg={signal_return[zero_fee_key]['signal_avg_return']:.6f}, "
        f"count={signal_return[zero_fee_key]['signal_count']}, "
        f"passed={passed}"
    )

    return signal_return


def run_one_backtest(
    logger: logging.Logger,
    sim_exp_dir: str,
    task: FusionTask,
    device: str,
) -> Dict[str, Any]:
    t0 = time.time()

    fusion_dir = os.path.join(
        sim_exp_dir,
        "fusion",
        f"pre_{task.pre_key}",
        f"compat_{task.train_compatibility}",
        f"fusion_{task.fusion_hash}",
    )

    os.makedirs(fusion_dir, exist_ok=True)
    task.fusion_dir = fusion_dir

    logger.info(
        f"Fusion: trigger={task.trigger.task_hash}, "
        f"dir={task.direction.task_hash}, fusion={task.fusion_hash}"
    )

    fusion_trigger_dir(
        logger,
        task.trigger.save_dir,
        task.direction.save_dir,
        fusion_dir,
    )

    pre_para = common.BaseDefine(**task.trigger.pre_params)
    train_cfg = build_train_cfg(task)
    prep_output_dir = infer_prep_output_dir(task.trigger.save_dir)

    fee_per_trade_list = (0.0, 0.005)
    signal_return = quick_signal_eval(
        pre_para=pre_para,
        prep_output_dir=prep_output_dir,
        logger=logger,
        task=task,
        fusion_dir=fusion_dir,
        device=device,
        fee_per_trade_list=fee_per_trade_list,
    )

    zero_fee_key = f"{0.0:g}"
    passed = signal_return[zero_fee_key]["signal_avg_return"] > 0
    if True:
        simulation_task: List[Any] = []
        hold_range = [36, 40, 44, 100, 1000]
        if pre_para.predict_num not in hold_range:
            hold_range.append(pre_para.predict_num)
        for i in hold_range:
            min_hold_bars = i
            for (atr_sl, atr_tp_mult) in [(6, 100)]:
                sim_para = backtest_runner.StrategyPara(
                    allow_long=True, allow_short=True, min_hold_bars=min_hold_bars,
                    commission_pct=0.05, init_equity=10000.0, prob_thresh=None,
                    atr_sl_long_mult=atr_sl, atr_sl_short_mult=atr_sl,
                    atr_tp_mult=atr_tp_mult, risk_per_trade_pct=0.1, max_daily_loss_pct=0.025,
                )
                simulation_task.append(sim_para)
        logger.info(f" {simulation_task} task for each simulation_task ")

    sim_result = {}
    for sim_task in simulation_task:
        sim_d = asdict(sim_task)
        sim_h = param_hash(sim_d)
        sim_result[sim_h] = {'forward': {}, 'short': {}, 'long': {}}
        for period in ['forward', 'short', 'long']:
            sim_result[sim_h][period] = backtest_runner.main(
                logger,
                para=sim_task,
                train_cfg=train_cfg,
                prep_output_dir=prep_output_dir,
                train_output_dir=fusion_dir,
                device=device,
                period=period,
            )["statistics"][1]

    elapsed = time.time() - t0

    return {
        "fusion_hash": task.fusion_hash,
        "pre_key": task.pre_key,
        "train_compatibility": task.train_compatibility,
        "fusion_dir": fusion_dir,
        "prep_output_dir": prep_output_dir,
        "device": device,
        "elapsed_sec": elapsed,
        "trigger": asdict(task.trigger),
        "direction": asdict(task.direction),
        "simulation": sim_result,
        "signal_return": signal_return,
    }


def save_selected_models(sim_exp_dir: str, fusion_tasks: List[FusionTask]) -> None:
    json_path = os.path.join(sim_exp_dir, SELECTED_MODELS_FILE)
    csv_path = os.path.join(sim_exp_dir, SELECTED_MODELS_CSV)

    rows = []

    for rank, task in enumerate(fusion_tasks, start=1):
        rows.append(
            {
                "rank": rank,
                "fusion_hash": task.fusion_hash,
                "pre_key": task.pre_key,
                "train_compatibility": task.train_compatibility,
                "trigger": asdict(task.trigger),
                "direction": asdict(task.direction),
            }
        )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fields = [
            "rank",
            "fusion_hash",
            "pre_key",
            "train_compatibility",
            "trigger_hash",
            "trigger_model",
            "trigger_score",
            "dir_hash",
            "dir_model",
            "dir_score",
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for rank, task in enumerate(fusion_tasks, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "fusion_hash": task.fusion_hash,
                    "pre_key": task.pre_key,
                    "train_compatibility": task.train_compatibility,
                    "trigger_hash": task.trigger.task_hash,
                    "trigger_model": f"{task.trigger.model_type}v{task.trigger.model_version}",
                    "trigger_score": task.trigger.score,
                    "dir_hash": task.direction.task_hash,
                    "dir_model": f"{task.direction.model_type}v{task.direction.model_version}",
                    "dir_score": task.direction.score,
                }
            )


def load_done_fusion_hashes(reports_path: str) -> set:
    if not os.path.exists(reports_path):
        return set()

    done = set()

    for r in load_jsonl(reports_path):
        if "fusion_hash" in r:
            done.add(str(r["fusion_hash"]))

    return done


# ---------------------------------------------------------------------------
# Multiprocessing part
# ---------------------------------------------------------------------------

def _pool_initializer(torch_threads: int) -> None:
    """
    ProcessPoolExecutor initializer, executed exactly once when a subprocess starts.

    - torch.set_num_threads / set_num_interop_threads control the torch thread count directly;
      this API does not depend on when the environment variables were set and always works in a subprocess.
    - The OMP/MKL environment variables are set as well, as a fallback for other libraries that may use BLAS
      (numpy for instance) -- under fork this can still take effect as long as the native library has not
      been used in this process yet.
    """
    if torch_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(torch_threads)
        os.environ["MKL_NUM_THREADS"] = str(torch_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(torch_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(torch_threads)

        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def _make_worker_logger(sim_exp_dir: str, name: str) -> logging.Logger:
    worker_log_dir = os.path.join(sim_exp_dir, "worker_logs")
    os.makedirs(worker_log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.propagate = False

    log_path = os.path.join(worker_log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


def run_batch_backtests_worker(
    sim_exp_dir: str,
    tasks: List[FusionTask],
    device: str,
) -> List[Dict[str, Any]]:
    """
    Run a batch of FusionTasks sequentially inside one subprocess.

    Compared with "one subprocess per task", the tasks of one
    (pre_key, train_compatibility) group are packed into a single batch
    and processed sequentially by the same subprocess:

    1. QUICK_DS_CACHE is a process private global, so when several tasks in a batch share the same
       (pre_key, train_compatibility), the later ones reuse the TimeSeriesWindowDataset built by the
       earlier ones and avoid rebuilding it.

    2. Fewer process create/destroy cycles (when a batch holds many short tasks).

    Exceptions are caught per task, so one failing task does not lose the rest of the batch
    results.
    """
    results: List[Dict[str, Any]] = []
    pid = os.getpid()

    logger = _make_worker_logger(sim_exp_dir, f"worker_{pid}")
    logger.info(f"Worker(pid={pid}) start, batch_size={len(tasks)}")

    for task in tasks:
        try:
            result = run_one_backtest(
                logger=logger,
                sim_exp_dir=sim_exp_dir,
                task=task,
                device=device,
            )
            result["status"] = "ok"
        except Exception as e:
            result = {
                "status": "error",
                "fusion_hash": task.fusion_hash,
                "pre_key": task.pre_key,
                "train_compatibility": task.train_compatibility,
                "device": device,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
            logger.error(f"Task failed fusion_hash={task.fusion_hash}: {e!r}")

        results.append(result)

    logger.info(f"Worker(pid={pid}) done, batch_size={len(tasks)}")
    return results


def _group_tasks_into_batches(
    tasks: List[FusionTask],
    workers: int,
) -> List[List[FusionTask]]:
    """
    Group pending_tasks by (pre_key, train_compatibility) so that the tasks of one group are
    run sequentially by the same subprocess and QUICK_DS_CACHE can be reused.

    When there are fewer groups than workers (say only 1-2 pre_keys), the parallelism would be
    limited by the group count; oversized groups are therefore split into several chunks based on
    the worker count, so "one group holds every task while the other workers idle" cannot happen.
    """
    groups: Dict[Tuple[str, str], List[FusionTask]] = defaultdict(list)
    for t in tasks:
        groups[(t.pre_key, t.train_compatibility)].append(t)

    batches: List[List[FusionTask]] = []

    for _, group_tasks in groups.items():
        if workers <= 1 or len(group_tasks) <= max(1, len(group_tasks) // workers or 1):
            batches.append(group_tasks)
            continue

        # Split the larger groups into at most `workers` chunks, the cache is still reused inside a chunk
        chunk_size = max(1, (len(group_tasks) + workers - 1) // workers)
        for i in range(0, len(group_tasks), chunk_size):
            batches.append(group_tasks[i:i + chunk_size])

    return batches


def run_backtests(
    logger: logging.Logger,
    sim_exp_dir: str,
    fusion_tasks: List[FusionTask],
    max_backtests: int,
    period: str,
    device: str,
    workers: int = 1,
    torch_threads: int = 1,
) -> None:
    reports_path = os.path.join(sim_exp_dir, SIM_REPORTS_FILE)

    done_fusion_hashes = load_done_fusion_hashes(reports_path)

    if max_backtests > 0:
        fusion_tasks = fusion_tasks[:max_backtests]

    pending_tasks = []
    skipped = 0

    for task in fusion_tasks:
        if task.fusion_hash in done_fusion_hashes:
            skipped += 1
            continue
        pending_tasks.append(task)

    total = len(fusion_tasks)
    pending = len(pending_tasks)

    logger.info(
        f"Backtest tasks: total={total}, pending={pending}, "
        f"already_done={len(done_fusion_hashes)}, skipped={skipped}, "
        f"workers={workers}, torch_threads={torch_threads}, device={device}"
    )

    if pending == 0:
        logger.info("No pending backtest tasks.")
        return

    # Serial mode, handy for debugging
    if workers <= 1:
        completed = 0
        failed = 0

        for i, task in enumerate(pending_tasks, start=1):
            logger.info(f"Backtest [{i}/{pending}] fusion_hash={task.fusion_hash}")

            try:
                result = run_one_backtest(
                    logger=logger,
                    sim_exp_dir=sim_exp_dir,
                    task=task,
                    device=device,
                )
                result["status"] = "ok"
                completed += 1
            except Exception as e:
                result = {
                    "status": "error",
                    "fusion_hash": task.fusion_hash,
                    "pre_key": task.pre_key,
                    "train_compatibility": task.train_compatibility,
                    "device": device,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
                failed += 1

            append_jsonl(reports_path, result)
            done_fusion_hashes.add(task.fusion_hash)

        logger.info(
            f"Backtest finished: completed={completed}, failed={failed}, "
            f"skipped={skipped}, total={total}"
        )
        return

    # -----------------------------------------------------------------
    # Parallel mode: a process pool where every subprocess uses only torch_threads threads (usually 1),
    # letting the OS schedule the real parallelism across the workers processes and avoiding the
    # "processes x threads" oversubscription. device is fixed to cpu, so the default fork start method is
    # used everywhere; spawn is not needed (fork is faster and there is no CUDA context involved).
    # -----------------------------------------------------------------
    completed = 0
    failed = 0

    batches = _group_tasks_into_batches(pending_tasks, workers)

    logger.info(
        f"Split {pending} pending tasks into {len(batches)} batches "
        f"(grouped by pre_key/train_compatibility for cache reuse)"
    )

    ctx = mp.get_context("fork")  # CPU-only, fork explicitly: faster and no CUDA involved

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_pool_initializer,
        initargs=(torch_threads,),
    ) as venue:
        future_to_batch = {
            venue.submit(
                run_batch_backtests_worker,
                sim_exp_dir,
                batch,
                device,
            ): batch
            for batch in batches
        }

        batch_idx = 0
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            batch_idx += 1

            try:
                batch_results = future.result()
            except Exception as e:
                # An exception at the subprocess/batch level (e.g. the subprocess was killed):
                # mark every task of the batch as failed, so nothing is silently dropped.
                batch_results = [
                    {
                        "status": "error",
                        "fusion_hash": t.fusion_hash,
                        "pre_key": t.pre_key,
                        "train_compatibility": t.train_compatibility,
                        "device": device,
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    }
                    for t in batch
                ]

            # The results are written in the main process, naturally serialized, so no extra lock is needed
            for result in batch_results:
                append_jsonl(reports_path, result)

                if result.get("status") == "ok":
                    completed += 1
                    done_fusion_hashes.add(result["fusion_hash"])
                    logger.info(
                        f"Done fusion_hash={result['fusion_hash']}, "
                        f"elapsed={result.get('elapsed_sec', 0):.1f}s "
                        f"(batch {batch_idx}/{len(batches)})"
                    )
                else:
                    failed += 1
                    logger.error(
                        f"Failed fusion_hash={result.get('fusion_hash')}: "
                        f"{result.get('error')} (batch {batch_idx}/{len(batches)})"
                    )

    logger.info(
        f"Backtest finished: completed={completed}, failed={failed}, "
        f"skipped={skipped}, total={total}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--simulation",
        default="/home/chao/work/quant_output/batch_train/DOGEUSDT_15m/2026-07-11/03_29_32",
    )
    parser.add_argument("--max-backtests", type=int, default=0)
    parser.add_argument("--period", type=str, default="short")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--torch-threads", type=int, default=2)

    args = parser.parse_args()

    if args.device != "cpu":
        raise ValueError(
            f"This script is configured for CPU-only multiprocessing "
            f"(fork start method, no CUDA context handling). "
            f"Got --device={args.device!r}; use 'cpu'."
        )

    train_exp_dir = args.simulation

    sim_exp_dir = os.path.join(train_exp_dir, 'batch_simulation')

    logger = setup_logger(sim_exp_dir)

    logger.info(f"batch_train dir: {train_exp_dir}")
    logger.info(f"batch_simulation dir: {sim_exp_dir}")

    registry = build_model_registry_from_reports(
        logger=logger,
        train_exp_dir=train_exp_dir,
    )

    fusion_tasks = select_fusion_pairs(
        logger=logger,
        registry=registry,
    )

    save_selected_models(
        sim_exp_dir=sim_exp_dir,
        fusion_tasks=fusion_tasks,
    )

    run_backtests(
        logger=logger,
        sim_exp_dir=sim_exp_dir,
        fusion_tasks=fusion_tasks,
        max_backtests=args.max_backtests,
        period=args.period,
        device=args.device,
        workers=args.workers,
        torch_threads=args.torch_threads,
    )

    logger.info("batch_simulation completed.")


if __name__ == "__main__":
    main()