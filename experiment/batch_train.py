#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch experiment runner (prep -> train ) with resume support.

Key design goals
- Deterministic task spec (tasks_spec.json) and stable param hashing
- Resume by skipping reports already present in reports.jsonl
- Simple process model: prep workers; train runs in main process
- Clean separation: path/layout, spec I/O, worker loops, main orchestration
"""

from __future__ import annotations

import argparse,shutil
import copy
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from queue import Empty
from typing import Any, Dict, Iterable, List, Optional, Tuple,Set
from collections import defaultdict
import numpy as np

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

from model import train_config
from data_process import common, preparation
from data_process.utils import (
    json_safe,
    load_selected_configs,
    param_hash,
)

# NOTE: train are imported lazily inside the process that needs them.
#       This avoids CUDA / heavy imports in workers.

TASKS_SPEC_FILE = "tasks_spec.json"
REPORTS_FILE = "reports.jsonl"
SELECTED_FILE = "selected_configs.jsonl"
MAX_PREP = 1
MAX_TRAIN = 2  # max concurrent train processes (each train runs in its own process)
SYMBOL: str = "DOGEUSDT"    #ETHUSDT DOGEUSDT
INTERVAL: str = "15m"
# -----------------------------------------------------------------------------
# Path layout helpers
# -----------------------------------------------------------------------------
def _batch_temp_dir(exp_dir: str) -> str:
    """
    Put all intermediate artifacts under TEMPORARY_DIR so persistence stays clean.
    """
    if exp_dir.startswith(common.PERSISTENCE_DIR):
        rel = os.path.relpath(exp_dir, common.PERSISTENCE_DIR)
        return os.path.join(common.PERSISTENCE_DIR,"train" , rel)
    base = os.path.basename(exp_dir.rstrip(os.sep)) or "run"
    return os.path.join(common.PERSISTENCE_DIR, "train" ,"batch_temp", base)

def _prep_output_dir(temp_dir: str, pre_h: str) -> str:
    return os.path.join(temp_dir, f"pre_{pre_h}")


def _train_output_dir(temp_dir: str, pre_h: str, task_type:str, tr_h: str) -> str:
    return os.path.join(temp_dir, f"pre_{pre_h}", str(task_type), f"train_{tr_h}")

def calc_train_params_hash(*, common, train, algo="sha1", length=8):
    import hashlib
    """
    Compute a stable hash for a parameter snapshot.
    """
    payload = {
        "common": asdict(common),
        "train": asdict(train),
    }

    # canonical JSON: sorted keys, no extra whitespace
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    h = hashlib.new(algo, s.encode("utf-8")).hexdigest()

    return h[:length] if length else h
# -----------------------------------------------------------------------------
# Spec build / load
# -----------------------------------------------------------------------------
def build_task_spec(
    preparation_task: List[Any],
    training_task: Dict[Any],
) -> Dict[str, Any]:
    """
    Build a tree spec:
      pre_hash -> {params, train: train_hash -> {params, task_hash,...]}}
    NOTE: prep_output_dir/save_dir are NOT written to spec; they are derived from hash layout.
    """
    spec: Dict[str, Any] = {}
    for pre in preparation_task:
        pre_d = asdict(pre)
        pre_d.pop("prep_output_dir", None)
        pre_h = param_hash(pre_d)

        node_pre = spec.setdefault(pre_h, {"params": json_safe(pre_d), "train": {}})

        for task_type, tran_cfg in training_task.items():
            node_pre["train"][task_type] = {}
            for tr in tran_cfg:
                tr_d = asdict(tr)
                tr_d.pop("save_dir", None)
                tr_h = param_hash(tr_d)
                task_hash = calc_train_params_hash(common=pre, train= tr)
                node_pre["train"][task_type][tr_h] = {"params": json_safe(tr_d),"task_hash":task_hash}
    return spec


def _count_spec_tasks(task_spec):
    n_prep = len(task_spec)
    n_train = sum(
        len(tr_nodes)
        for pre_node in task_spec.values()
        for tr_nodes in pre_node["train"].values()
    )
    return n_prep, n_train


def load_done_set(reports_path: str) -> set[str]:
    """
    Read reports.jsonl and collect completed params.hash.
    """
    done: Dict[str, set[str]] = {}
    if not os.path.exists(reports_path):
        raise RuntimeError(f"{reports_path}")
    with open(reports_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_type = str(d.get("task_type"))
            task_hash = str(d.get("task_hash"))
            if task_type not in done:
                done[task_type] = set()
            task_hash = d.get("task_hash")
            done[task_type].add(task_hash)
    return done

def make_model_cfg(d):
    model_type = d.get("model_type")

    for cls in train_config.BaseModelConfig.__subclasses__():
        obj = cls()
        if obj.model_type == model_type:
            valid_keys = {f.name for f in fields(cls)}
            kwargs = {k: v for k, v in d.items() if k in valid_keys}
            return cls(**kwargs)

    raise ValueError(f"Unknown model_type: {model_type}")

def _config_from_dict_train(train_params: Dict[str, Any]):
    """
    Restore TrainConfig from dict stored in task spec.
    Intentionally ignores nested model_cfg/data_cfg dicts in spec (those fields are dataclasses).
    """
    import model.train_2head as train

    t_cfg = train.TrainConfig()
    for k, v in (train_params or {}).items():
        if k == "model_cfg" and isinstance(v, dict):
            t_cfg.model_cfg = make_model_cfg(v)
        elif k == "data_cfg" and isinstance(v, dict):
            t_cfg.data_cfg = train.DataConfig(**v)
        elif hasattr(t_cfg, k):
            setattr(t_cfg, k, v)
    return t_cfg


def filter_pending_from_spec(task_spec: Dict[str, Any], done_set: set[str]) -> Dict[str, Any]:
    """
    Filter sim leaf tasks that are already present in reports.jsonl.
    """


    pending: Dict[str, Any] = {}
    for pre_h, pre_node in task_spec.items():
        pre_params = pre_node["params"]
        train_pending: Dict[str, Any] = {}

        for task_type, tr_nodes in pre_node["train"].items():
            for tr_h, tr_node in tr_nodes.items():
                train_params = tr_node["params"]
                task_hash = tr_node["task_hash"]
                if task_type not in done_set:
                    train_pending[task_type] = tr_nodes
                    break
                else:
                    if task_hash not in done_set[task_type]:
                        train_pending.setdefault(task_type, {})[tr_h] = tr_node

        if train_pending:
            pending[pre_h] = {"params": pre_params, "train": train_pending}

    return pending

def load_json_file(json_file: str):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Tasks spec not found: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        task_spec = json.load(f)
    return task_spec

def load_pending_tasks(exp_dir: str, done_set: set[str]) -> Tuple[Dict[str, Any], Tuple[int, int, int]]:
    tasks_spec_path = os.path.join(exp_dir, TASKS_SPEC_FILE)
    task_spec = load_json_file(tasks_spec_path)
    total_counts = _count_spec_tasks(task_spec)
    pending = filter_pending_from_spec(task_spec, done_set)
    return pending, total_counts


def _create_output_dirs(task_spec: Dict[str, Any], temp_dir: str) -> None:
    """
    Create prep/train output dirs for all pending tasks.
    """
    for pre_h, pre_node in task_spec.items():
        os.makedirs(_prep_output_dir(temp_dir, pre_h), exist_ok=True)
        for task_type, tr_nodes in pre_node["train"].items():
            for tr_h, tr_node in tr_nodes.items():
                os.makedirs(_train_output_dir(temp_dir, pre_h, task_type, tr_h), exist_ok=True)

# -----------------------------------------------------------------------------
# Worker logging
# -----------------------------------------------------------------------------
def _worker_logger(log_file: str) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s %(message)s"))
    root.addHandler(fh)
    return root

SWEEPABLE_TYPES = (int, float, str, bool, type(None))


def collect_from_any(
    obj: Any,
    out: Dict[str, Set[Any]],
    prefix: str = "",
):
    if isinstance(obj, SWEEPABLE_TYPES):
        out[prefix].add(obj)
        return

    if is_dataclass(obj):
        for f in fields(obj):
            value = getattr(obj, f.name)
            key = f"{prefix}.{f.name}" if prefix else f.name
            collect_from_any(value, out, key)
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            collect_from_any(v, out, key)
        return

def log_param_sweep(logger, sweep):
    logger.info("📌 Experiment parameter sweep:")

    for stage in ["pre", "train"]:
        if not sweep[stage]:
            continue
        logger.info(f"  [{stage}]")
        for k, v in sweep[stage].items():
            logger.info(f"    {k}: {v}")

def collect_param_sweep(task_spec):
    sweep = {
        "pre": defaultdict(set),
        "train": defaultdict(set),
    }

    for pre_node in task_spec.values():
        # pre params
        collect_from_any(pre_node["params"], sweep["pre"])

        for task_type,tr_nodes in pre_node["train"].items():
            for tr_node in tr_nodes.values():
                collect_from_any(tr_node["params"], sweep["train"])
            break

    def finalize(d):
        return {
            k: sorted(v, key=lambda x: (x is None, str(type(x)), str(x)))
            for k, v in d.items()
            if len(v) > 1
        }

    return {
        "pre": finalize(sweep["pre"]),
        "train": finalize(sweep["train"])
    }

def construct_task_doge():
    import model.train_2head as train
    preparation_task: List[Any] = []

    for pn in [16]:#[4,6,8,12,16,20,24,28,32,36]: #[10,12,14,16,18]
        for vol_multiplier in [1.7,1.8,1.9,2.0]:#1.8,1.9,2
            for vol_ewma_span in [80]:
                preparation_task.append(common.BaseDefine(
                        vol_ewma_span = vol_ewma_span,
                        predict_num=pn,
                        vol_multiplier_long=vol_multiplier,
                        stop_multiplier_rate_long=None,
                        vol_multiplier_short=vol_multiplier,
                        stop_multiplier_rate_short=None,
                        symbol=SYMBOL,   #ETHUSDT
                        interval=INTERVAL,
                        trading_type= 'um',
                        version=0
                    ))

    training_task: Dict[str,List[train.TrainConfig]] = {train.TrainTask.SINGLE_MODEL_DIR:[],train.TrainTask.SINGLE_MODEL_TRIGGER:[]}

    # for flip_penalty in np.arange(0.5, 2.5, 0.1).round(1):
    #     for miss_penalty in np.arange(0.2, 2.5, 0.1).round(1):
    # for lambda_dir in np.arange(0.1, 0.7, 0.1).round(1):
    #     for lambda_cost in np.arange(0.1, 0.7, 0.1).round(1):
    #         for stride in [8]: #2,4,8
    #             for bestf1 in [True]:
    #                 for loss_fun_version_v in [4]:
    #                     training_task.append(train.TrainConfig(use_cache = False,epochs = 100, batch_size=256,best_f1=bestf1,loss_fun_version = loss_fun_version_v,
    #                                                 flip_penalty = float(1.3),miss_penalty = float(1.7),false_trade = 1,
    #                                                 stride = stride, patience = 8,lambda_main = 0.7,lambda_dir = lambda_dir,lambda_cost = lambda_cost))
    to_remove_1 = ["open", "high",'low' ]
    to_remove_2 = to_remove_1 + ['close','volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume']
    to_remove_3 = to_remove_1 + ['close','volume','taker_buy_base_volume','taker_buy_quote_volume']
    to_remove_4 = to_remove_1 + ['close','taker_buy_base_volume','taker_buy_quote_volume']
    to_remove_5 = to_remove_1 + ['taker_buy_base_volume','taker_buy_quote_volume']
    feature_conf_list_1 = [f for f in train.feature_conf_list if f not in to_remove_1]
    feature_conf_list_2 = [f for f in train.feature_conf_list if f not in to_remove_2]
    feature_conf_list_3 = [f for f in train.feature_conf_list if f not in to_remove_3]
    feature_conf_list_4 = [f for f in train.feature_conf_list if f not in to_remove_4]
    feature_conf_list_5 = [f for f in train.feature_conf_list if f not in to_remove_5]
    for seq_len in [96]:#in range(3*16,11*16,16): #12,16,24,32
        for stride in [2]: #2,4,8
            for featrue_conf in [train.feature_conf_list]:
                # compatibility seq_len_stride_featrue_conf
                for model_cfg in [train_config.LogisticConfig(model_version= 1,seq_len=seq_len),
                                  train_config.ConvLSTMConfig(model_version= 1,seq_len=seq_len),
                                  train_config.LSTMConfig(model_version= 1,seq_len=seq_len),
                                  train_config.LSTMConfig(model_version= 2,seq_len=seq_len),
                                  train_config.TransformerConfig(model_version= 1,seq_len=seq_len),
                                  train_config.TransformerConfig(model_version= 2,seq_len=seq_len)]:
                    for miss_penalty in [2]:#np.arange(0.5,5, 0.5).round(1):#in np.arange(0.3, 2.1, 0.2).round(1):
                        train_conf = train.TrainConfig(use_cache = False,epochs = 50, batch_size=256,
                                                        feature_conf_list= featrue_conf, model_cfg = model_cfg,
                                                        miss_penalty = float(miss_penalty),stride = stride, patience = 8)
                        training_task[train.TrainTask.SINGLE_MODEL_TRIGGER].append(train_conf)

                    train_conf = train.TrainConfig(use_cache = False,epochs = 50, batch_size=256,
                                                    feature_conf_list= featrue_conf, model_cfg = model_cfg,
                                                    flip_penalty = float(1),stride = stride, patience = 8)
                    training_task[train.TrainTask.SINGLE_MODEL_DIR].append(train_conf)
                        
    return preparation_task, training_task

def construct_task_eth():
    import model.train_2head as train
    preparation_task: List[Any] = []

    for cn in [12,16]:#list(range(56, 224, 8)): #[4,8,12,56,64,72,80,88,96,108,116,124,132,144,156,168,176,188]
        for pn in [4,8,16,24]:#[4,6,8,12,16,20,24,28,32,36]: #[10,12,14,16,18]
            for vol_multiplier in [1.8,1.9,2,2.1]:#1.8,1.9,2
                for vol_ewma_span in [88]:
                    preparation_task.append(common.BaseDefine(
                            vol_ewma_span = vol_ewma_span,
                            seq_len=cn,
                            predict_num=pn,
                            vol_multiplier_long=vol_multiplier,
                            stop_multiplier_rate_long=0.2,
                            vol_multiplier_short=vol_multiplier,
                            stop_multiplier_rate_short=0.2,
                            symbol=SYMBOL,   #ETHUSDT
                            interval=INTERVAL,
                            trading_type= 'um',
                            version=0
                        ))

    training_task: List[train.TrainConfig] = []

    for false_trade in [1]:
        for flip_penalty in np.arange(0.9, 1.7, 0.1).round(1):# np.arange(0.2, 2.1, 0.1).round(1):
            for miss_penalty in np.arange(0.7, 1.2, 0.1).round(1):#in np.arange(0.3, 2.1, 0.2).round(1):
                for stride in [4,8]: #2,4,8
                    for bestf1 in [True]:
                        for loss_fun_version_v in [2]:
                            training_task.append(train.TrainConfig(use_cache = False,epochs = 100, batch_size=256,best_f1=bestf1,loss_fun_version = loss_fun_version_v,
                                                        flip_penalty = float(flip_penalty),miss_penalty = float(miss_penalty),false_trade = 1,
                                                        stride = stride, patience = 8,lambda_main = 0.7,lambda_dir = 0.7,lambda_cost = 0.4,mag_alpha = 0))
    return preparation_task, training_task

def create_task_spec(logger, exp_dir,done_set: set[str]):
    if SYMBOL == "DOGEUSDT":
        preparation_task, training_task = construct_task_doge()
    elif SYMBOL == "ETHUSDT":
        preparation_task, training_task = construct_task_eth()
    else:
        raise RuntimeError(f"no construct for {SYMBOL} yet")

    task_spec = build_task_spec(preparation_task, training_task)
    # task_spec is already ready
    sweep = collect_param_sweep(task_spec)
    log_param_sweep(logger, sweep)

    tasks_spec_path = os.path.join(exp_dir, TASKS_SPEC_FILE)
    with open(tasks_spec_path, "w", encoding="utf-8") as f:
        json.dump(task_spec, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"📄 Tasks spec saved: {tasks_spec_path}")

    (n_prep_total, n_train_total) = _count_spec_tasks(task_spec)
    total_all = n_prep_total + n_train_total 
    logger.info(f"📊 Total: {total_all} (prep={n_prep_total}, train={n_train_total})")
    if done_set:
        task_spec = filter_pending_from_spec(task_spec, done_set)
        n_prep, n_train = _count_spec_tasks(task_spec)
        total_pending = n_prep + n_train
        logger.info(f"📊 Pending: {total_pending} (prep={n_prep}, train={n_train}), done: {total_all - total_pending}")
    return task_spec

# -----------------------------------------------------------------------------
# Worker loops
# -----------------------------------------------------------------------------
def _worker_prep(worker_log_file: str, task_queue: mp.Queue, train_queue: mp.Queue, temp_dir: str):
    logger = _worker_logger(worker_log_file)
    while True:
        try:
            msg = task_queue.get(timeout=0.5)
        except Empty:
            continue
        if msg is None:
            break

        task_spec = msg
        for pre_h,pre_task in task_spec.items():
            para = common.BaseDefine(**pre_task["params"])
            t0 = time.time()
            try:
                prep_dir = _prep_output_dir(temp_dir, pre_h)
                preparation.main(logger, para=para, prep_output_dir=prep_dir)
            except Exception:
                logger.exception(f"Prep failed: {pre_h}")
                # still notify main so it can terminate early
                train_queue.put(("prep_failed", pre_h, time.time() - t0, []))
                continue

            elapsed = time.time() - t0

            # Build train items for this prep hash (pass only json-safe dicts across processes)
            train_items = []
            for task_type, tr_nodes in pre_task.get("train", {}).items():
                for tr_h, tr_node in tr_nodes.items():
                    save_dir = _train_output_dir(temp_dir, pre_h, task_type, tr_h)
                    train_items.append(
                        {
                            "pre_h": pre_h,
                            "tr_h": tr_h,
                            "pre_params": copy.deepcopy(pre_task["params"]),
                            "train_params": copy.deepcopy(tr_node["params"]),
                            "task_hash": copy.deepcopy(tr_node["task_hash"]),
                            "task_type": task_type,
                            "prep_output_dir": prep_dir,
                            "save_dir": save_dir,
                        }
                    )

            train_queue.put(("prep_done", pre_h, elapsed, train_items))


def _train_task(
    worker_log_file: str,
    item: Dict[str, Any],
    train_result_queue: mp.Queue,
):
    """Run a single train in its own process."""
    logger = _worker_logger(worker_log_file)
    import model.train_2head as train

    pre_h = item["pre_h"]
    tr_h = item["tr_h"]
    pre_params = item["pre_params"]
    train_params = item["train_params"]
    task_hash = item["task_hash"]
    task_type = item["task_type"]
    prep_output_dir = item["prep_output_dir"]
    save_dir = item["save_dir"]

    t0 = time.time()
    try:
        pre_para = common.BaseDefine(**pre_params)
        t_cfg:train.TrainConfig = _config_from_dict_train(train_params)
        result = train.main(logger, train_task=task_type ,train_cfg=t_cfg, prep_output_dir=prep_output_dir, save_dir=save_dir,experiment=True)

        train_result_queue.put(("train_done", {'task_type':task_type, 'task_hash':task_hash,
                                               'model_type':t_cfg.model_cfg.model_type,'model_version':t_cfg.model_cfg.model_version,
                                               'metrics':result,'pre_params':pre_params,'train_params':train_params,'save_dir':save_dir}
                                               , pre_h, tr_h, time.time() - t0))
    except Exception:
        logger.exception(f"Train failed: {pre_h}/{tr_h}")
        train_result_queue.put(("train_failed", None, pre_h, tr_h, time.time() - t0))

def _send_none_to_workers(q: mp.Queue, n: int) -> None:
    for _ in range(n):
        q.put(None)

# -----------------------------------------------------------------------------
# ETA helper
# -----------------------------------------------------------------------------
def _make_eta_fn(n_prep: int, n_train: int, stats: Dict[str, Any]):
    def phase_eta(total: int, count: int, elapsed: float, workers: int) -> Optional[float]:
        if total <= 0 or count >= total:
            return 0.0
        if count <= 0:
            return None
        return (elapsed / count) * (total - count) / max(1, workers)

    def fmt(seconds: Optional[float]) -> str:
        if seconds is None:
            return "—"
        if seconds <= 0:
            return "0h"
        hours = seconds / 3600
        return f"{seconds:.0f}s" if hours < 0.1 else f"{hours:.2f}h"

    def eta_msg() -> str:
        prep_eta = phase_eta(n_prep, stats["preparation"]["count"], stats["preparation"]["time"], MAX_PREP)
        train_eta = phase_eta(n_train, stats["train"]["count"], stats["train"]["time"], 1)

        parts = []
        if n_prep > 0:
            parts.append(f"prep:{fmt(prep_eta)}")
        if n_train > 0:
            parts.append(f"train:{fmt(train_eta)}")

        if not parts:
            return ""

        total_eta = None
        if (n_prep == 0 or stats["preparation"]["count"] > 0) and (n_train == 0 or stats["train"]["count"] > 0):
            total_eta = (prep_eta or 0) + (train_eta or 0)

        msg = "[ETA] " + ", ".join(parts)
        if total_eta is not None and total_eta > 0:
            msg += f" | total ~{fmt(total_eta)}"
        return msg

    return eta_msg


# -----------------------------------------------------------------------------
# CLI / entry
# -----------------------------------------------------------------------------
def _setup_root_logger(exp_dir: str) -> logging.Logger:
    log_file_path = os.path.join(exp_dir, "experiment.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logger = logging.getLogger("batch")
    logger.setLevel(logging.INFO)
    return logger

def main():
    parser = argparse.ArgumentParser(description="Batch experiments: prep -> train(with resume)")
    parser.add_argument("-r", "--resume", type=str, help="Resume experiment from specified directory name under PERSISTENCE_DIR")

    args = parser.parse_args()

    # ---------------- resolve exp_dir ----------------
    if args.resume:
        exp_dir = os.path.join(common.PERSISTENCE_DIR, args.resume)
        if not os.path.exists(exp_dir):
            print(f"❌ Error: Resume directory not found: {exp_dir}")
            return
    else:
        exp_dir = common.create_experiment_dir(
            os.path.join(common.PERSISTENCE_DIR, "batch_train"),
            SYMBOL,
            INTERVAL,
        )

    logger = _setup_root_logger(exp_dir)
    common.get_git_info(logger)

    begin_time = time.time()
    reports_path = os.path.join(exp_dir, REPORTS_FILE)

    temp_dir = os.path.join(exp_dir,'train')
    os.makedirs(temp_dir, exist_ok=True)

    # ---------------- build/load spec ----------------
    if args.resume:
        done_set = load_done_set(reports_path)
        task_spec, (n_prep_total, n_train_total) = load_pending_tasks(exp_dir, done_set)
        n_prep, n_train = _count_spec_tasks(task_spec)
        total_all = n_prep_total + n_train_total
        total_pending = n_prep + n_train
        logger.info(f"📥 Loaded from {exp_dir}")
        logger.info(f"📊 Total: {total_all} (prep={n_prep_total}, train={n_train_total})")
        logger.info(f"📊 Pending: {total_pending} (prep={n_prep}, train={n_train}), done: {total_all - total_pending}")
    else:
        task_spec = create_task_spec(logger, exp_dir, None)
    if not task_spec:
        logger.info("✅ No pending tasks.")
        return

    logger.info(f"🚀 Pipeline: MAX_PREP={MAX_PREP}, train={MAX_TRAIN}")

    # ---------------- queues & workers ----------------
    prep_task_queue: mp.Queue = mp.Manager().Queue()
    train_task_queue: mp.Queue = mp.Manager().Queue()
    train_result_queue: mp.Queue = mp.Manager().Queue()

    # start workers
    prep_workers = []
    for i in range(MAX_PREP):
        worker_log = os.path.join(exp_dir, f"prep_{i}.log")
        p = mp.Process(target=_worker_prep, args=(worker_log, prep_task_queue, train_task_queue, temp_dir))
        p.start()
        prep_workers.append(p)

    run_task_spec(
        task_spec,
        temp_dir,
        exp_dir,
        prep_task_queue,
        train_task_queue,
        train_result_queue,
        logger,
        prep_workers,
        reports_path,
    )
    
    
    _send_none_to_workers(prep_task_queue, MAX_PREP)

    logger.info("\n" + "=" * 40)
    elapsed = time.time() - begin_time

    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(f"✅ Completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    logger.info("=" * 40)

def run_task_spec(
    task_spec,
    temp_dir,
    exp_dir,
    prep_task_queue,
    train_task_queue,
    train_result_queue,
    logger,
    prep_workers,
    reports_path,
):
    stats = {"preparation": {"time": 0.0, "count": 0}, "train": {"time": 0.0, "count": 0}}
    _create_output_dirs(task_spec, temp_dir)
    prep_task_queue.put(task_spec)

    # ETA printer
    n_prep, n_train = _count_spec_tasks(task_spec)

    pending_train_items: List[Dict[str, Any]] = []
    train_procs: List[mp.Process] = []
    train_idx = 0

    def _reap_train_procs():
        nonlocal train_procs
        alive = []
        for p in train_procs:
            if p.is_alive():
                alive.append(p)
            else:
                p.join(timeout=0)
        train_procs = alive

    # -----------------------------------------------------------------------------
    # Result handling
    # -----------------------------------------------------------------------------
    def _drain_train_results():
        while True:
            try:
                msg = train_result_queue.get_nowait()
                # (("train_done", result, pre_h, tr_h, time.time() - t0))
            except Empty:
                break
            if not msg:
                continue
            typ, result, pre_h, tr_h, elapsed = msg
            if typ == "train_failed":
                logger.error(f"❌ Train failed for {pre_h}/{tr_h}, aborting.")
                raise RuntimeError("train_failed")

            stats["train"]["time"] += float(elapsed)
            stats["train"]["count"] += 1
            common.append_jsonl(reports_path, result)
            logger.info(f"  {pre_h}/{tr_h}  Train done in {elapsed:.2f}s")

    def _try_start_train_procs():
        nonlocal train_idx
        _reap_train_procs()
        while pending_train_items and len(train_procs) < MAX_TRAIN:
            item = pending_train_items.pop(0)
            worker_log = os.path.join(exp_dir, f"train_{train_idx%MAX_TRAIN}.log")
            p = mp.Process(target=_train_task, args=(worker_log, item, train_result_queue))
            p.start()
            train_procs.append(p)
            train_idx += 1

    # main loop: consume prep_done -> spawn train processes
    try:
        while stats["preparation"]["count"] < n_prep or stats["train"]["count"] < n_train:
            # consume prep->train messages
            try:
                msg = train_task_queue.get(timeout=0.2)
            except Empty:
                msg = None

            if msg is not None:
                typ, pre_h, elapsed, train_items = msg
                if typ == "prep_failed":
                    logger.error(f"❌ Prep failed for {pre_h}, aborting.")
                    break

                stats["preparation"]["time"] += float(elapsed)
                stats["preparation"]["count"] += 1
                logger.info(f"  {pre_h}  Prep done in {elapsed:.2f}s")
                pending_train_items.extend(train_items or [])

            _try_start_train_procs()
            _drain_train_results()

        # final drains
        _drain_train_results()
    except RuntimeError:
        pass
    finally:
        # best-effort shutdown
        for p in train_procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for p in prep_workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
