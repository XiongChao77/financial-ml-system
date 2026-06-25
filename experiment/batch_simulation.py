#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os,torch
import sys
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

import batch_train
from data_process import common
from data_process.utils import param_hash
from trade.bt import simulation
from model.train import fusion_trigger_dir,TrainTask
from model import model_loader
from model import data_loader

REPORTS_FILE = "reports.jsonl"
SELECTED_MODELS_FILE = "selected_models.json"
SELECTED_MODELS_CSV = "selected_models_summary.csv"

SYMBOL = "DOGEUSDT"
INTERVAL = "15m"

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

    # score = (
    #     0.40 * pos_f1
    #     + 0.30 * pos_recall
    #     + 0.20 * pos_precision
    #     + 0.10 * mcc
    # )
    score = pos_f1

    return score, "pos_f1"#"0.40*pos_f1 + 0.30*pos_recall + 0.20*pos_precision + 0.10*mcc"


def score_direction(metrics: Dict[str, Any]) -> Tuple[float, str]:
    best = metrics["Best_F1"]

    # score = (
    #     0.45 * best["mcc"]
    #     + 0.30 * best["macro_f1"]
    #     + 0.20 * best["balanced_accuracy"]
    #     + 0.05 * best["accuracy"]
    # )
    score = best["macro_f1"]

    return score, "macro_f1"#"0.45*mcc + 0.30*macro_f1 + 0.20*balanced_accuracy + 0.05*accuracy"


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
    reports_path = os.path.join(train_exp_dir, REPORTS_FILE)
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
        """
        每个模型架构内选择：
        1. score 最高的 top_k
        2. score 位于中间附近的 mid_k

        分组依据：
            model_type + model_version
        """
        selected: Dict[str, ModelRef] = {}

        groups: Dict[Tuple[str, int], List[ModelRef]] = {}

        for m in models:
            key = (m.model_type, m.model_version)
            groups.setdefault(key, []).append(m)

        for _, group in groups.items():
            group = sorted(group, key=lambda x: x.score, reverse=True)
            n = len(group)

            # top k
            top_models = group[:top_k]

            # middle k
            mid = n // 2
            half = mid_k // 2
            start = max(0, mid - half)
            end = min(n, start + mid_k)

            # 如果靠近尾部导致数量不足，往前补
            start = max(0, end - mid_k)

            mid_models = group[start:end]

            for m in top_models + mid_models:
                selected[m.task_hash] = m

        return list(selected.values())
    
    fusion_tasks = []

    for (pre_key, compatibility), task_map in registry.items():
        triggers = task_map[TrainTask.SINGLE_MODEL_TRIGGER]
        dirs = task_map[TrainTask.SINGLE_MODEL_DIR]

        # triggers = select_representative_models(
        #     triggers,
        #     top_k=3,
        #     mid_k=2,
        # )

        # dirs = select_representative_models(
        #     dirs,
        #     top_k=3,
        #     mid_k=2,
        # )

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
        # best_trigger = max(triggers, key=lambda x: x.score)
        # best_dir = max(dirs, key=lambda x: x.score)

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
    # save_dir = .../pre_xxx/SINGLE_MODEL_DIR/train_xxx
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
    if cache_key not in QUICK_DS_CACHE :
        QUICK_DS_CACHE[cache_key] = data_loader.TimeSeriesWindowDataset(
            df=df,
            kline_interval_ms=interval_ms,
            feature_cols=handler.feature_cols,
            label_col=handler.label_col,
            window=handler.window,
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

    signal_mask = np.isin(pred,[common.Signal.NEGATIVE,common.Signal.POSITIVE,],)

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
    fee_per_trade_list=(0.0)
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

    fee_per_trade_list=(0.0, 0.005)
    signal_return = quick_signal_eval(
        pre_para= pre_para,
        prep_output_dir = prep_output_dir,
        logger=logger,
        task=task,
        fusion_dir=fusion_dir,
        device=device,
        fee_per_trade_list=fee_per_trade_list,
    )

    sim_result = {}
    zero_fee_key = f"{0.0:g}"
    passed = signal_return[zero_fee_key]["signal_avg_return"] > 0
    if True:
        for period in ['forward', 'short', 'long']:
            sim_result[period] = simulation.main(
                logger,
                para=simulation.StrategyPara(),
                pre_para=pre_para,
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

def load_done_fusion_hashes(reports_path: str) -> set[str]:
    if not os.path.exists(reports_path):
        return set()

    done = set()

    for r in load_jsonl(reports_path):
        if "fusion_hash" in r:
            done.add(str(r["fusion_hash"]))

    return done

def run_one_backtest_worker(
    sim_exp_dir: str,
    task: FusionTask,
    device: str,
    torch_threads: int = 1,
) -> Dict[str, Any]:
    """
    子进程执行单个回测任务。

    注意：
    1. 不传 logger 对象，因为 logger 不适合跨进程传递。
    2. 每个子进程自己创建 logger。
    3. reports.jsonl 由主进程统一写，避免并发写文件冲突。
    """

    try:
        # 限制每个进程内部的 PyTorch CPU 线程数
        # 否则 workers * torch内部线程 会导致 CPU 过度竞争
        if torch_threads > 0:
            torch.set_num_threads(torch_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

        worker_log_dir = os.path.join(sim_exp_dir, "worker_logs")
        os.makedirs(worker_log_dir, exist_ok=True)

        logger = logging.getLogger(f"worker_{os.getpid()}_{task.fusion_hash}")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        logger.propagate = False

        log_path = os.path.join(worker_log_dir, f"{task.fusion_hash}.log")
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(file_handler)

        logger.info(f"Worker start fusion_hash={task.fusion_hash}")

        result = run_one_backtest(
            logger=logger,
            sim_exp_dir=sim_exp_dir,
            task=task,
            device=device,
        )

        result["status"] = "ok"
        return result

    except Exception as e:
        return {
            "status": "error",
            "fusion_hash": task.fusion_hash,
            "pre_key": task.pre_key,
            "train_compatibility": task.train_compatibility,
            "device": device,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }
    
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
    reports_path = os.path.join(sim_exp_dir, REPORTS_FILE)

    done_fusion_hashes = load_done_fusion_hashes(reports_path)

    if max_backtests > 0:
        fusion_tasks = fusion_tasks[:max_backtests]

    # 过滤已经完成的任务
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

    # 串行模式，方便 debug
    if workers <= 1:
        completed = 0
        failed = 0

        for i, task in enumerate(pending_tasks, start=1):
            logger.info(f"Backtest [{i}/{pending}] fusion_hash={task.fusion_hash}")

            result = run_one_backtest(
                logger=logger,
                sim_exp_dir=sim_exp_dir,
                task=task,
                device=device,
            )

            result["status"] = "ok"
            append_jsonl(reports_path, result)

            done_fusion_hashes.add(task.fusion_hash)
            completed += 1

        logger.info(
            f"Backtest finished: completed={completed}, failed={failed}, "
            f"skipped={skipped}, total={total}"
        )
        return

    # 并行模式
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {}

        for task in pending_tasks:
            future = executor.submit(
                run_one_backtest_worker,
                sim_exp_dir,
                task,
                device,
                torch_threads,
            )
            future_to_task[future] = task

        for idx, future in enumerate(as_completed(future_to_task), start=1):
            task = future_to_task[future]

            try:
                result = future.result()
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

            # 只有主进程写 reports.jsonl，避免并发写冲突
            append_jsonl(reports_path, result)

            if result.get("status") == "ok":
                completed += 1
                done_fusion_hashes.add(task.fusion_hash)

                logger.info(
                    f"Done [{idx}/{pending}] fusion_hash={task.fusion_hash}, "
                    f"elapsed={result.get('elapsed_sec', 0):.1f}s"
                )
            else:
                failed += 1
                logger.error(
                    f"Failed [{idx}/{pending}] fusion_hash={task.fusion_hash}: "
                    f"{result.get('error')}"
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
        default="/home/chao/work/quant_output/batch_train/DOGEUSDT_30m/2026-06-25/04_09_15",
    )
    parser.add_argument("--max-backtests", type=int, default=0)
    parser.add_argument("--period", type=str, default="short")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=2)

    args = parser.parse_args()

    train_exp_dir = args.simulation

    sim_exp_dir = os.path.join(train_exp_dir,'batch_simulation')

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