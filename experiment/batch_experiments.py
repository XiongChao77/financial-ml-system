#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch experiment runner (prep -> train -> sim) with resume support.

Key design goals
- Deterministic task spec (tasks_spec.json) and stable param hashing
- Resume by skipping reports already present in reports.jsonl
- Simple process model: prep workers + sim workers; train runs in main process
- Clean separation: path/layout, spec I/O, worker loops, main orchestration
"""

from __future__ import annotations

import argparse,shutil
import copy
import hashlib
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
    calc_params_hash,
    json_safe,
    load_selected_configs,
    param_hash,
)

# NOTE: train/simulation are imported lazily inside the process that needs them.
#       This avoids CUDA / heavy imports in workers.

TASKS_SPEC_FILE = "tasks_spec.json"
REPORTS_FILE = "reports.jsonl"
TRAIN_REPORTS_FILE = "train_reports.jsonl"
SELECTED_FILE = "selected_configs.jsonl"
MAX_PREP = 1
MAX_TRAIN = 4  # max concurrent train processes (each train runs in its own process)
MAX_SIM = 4
SYMBOL: str = "DOGEUSDT"    #ETHUSDT DOGEUSDT
INTERVAL: str = "30m"

# 训练模式开关：
# - train_config.TrainTask.SINGLE_MODEL_3CLASS（或 SINGLE_MODEL_LONG_OVR 等其它单模型任务）
#   -> 走现有的单模型 prep->train->sim 流水线（construct_task_doge）
# - train_config.TrainTask.TRIGGER_DIR / LONG_SHORT_OVR
#   -> combo_model 模式：分别 sweep 两个子模型角色（construct_task_doge_combo），
#      训练完成后自动按 (pre_key, train_compatibility) 分组两两 fuse 并回测
TRAIN_MODE: str = train_config.TrainTask.DIRECT_3CLASS
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


def _train_output_dir(temp_dir: str, pre_h: str, tr_h: str) -> str:
    return os.path.join(temp_dir, f"pre_{pre_h}", f"train_{tr_h}")


def _sim_output_dir(temp_dir: str, pre_h: str, tr_h: str, sim_h: str) -> str:
    return os.path.join(temp_dir, f"pre_{pre_h}", f"train_{tr_h}", f"sim_{sim_h}")


# -----------------------------------------------------------------------------
# Spec build / load
# -----------------------------------------------------------------------------
def build_task_spec(
    preparation_task: List[Any],
    training_task: List[Any],
    simulation_task: List[Any],
) -> Dict[str, Any]:
    """
    Build a tree spec:
      pre_hash -> {params, train: train_hash -> {params, sim_tasks:[{hash, params}, ...]}}
    NOTE: prep_output_dir/save_dir are NOT written to spec; they are derived from hash layout.
    """
    spec: Dict[str, Any] = {}
    for pre in preparation_task:
        pre_d = asdict(pre)
        pre_d.pop("prep_output_dir", None)
        pre_h = param_hash(pre_d)

        node_pre = spec.setdefault(pre_h, {"params": json_safe(pre_d), "train": {}})

        for tr in training_task:
            tr_d = asdict(tr)
            tr_d.pop("save_dir", None)
            tr_h = param_hash(tr_d)

            node_tr = node_pre["train"].setdefault(tr_h, {"params": json_safe(tr_d), "sim_tasks": []})

            # de-dup sim tasks by hash
            existing = {s["hash"] for s in node_tr["sim_tasks"]}
            for sim in simulation_task:
                sim_d = asdict(sim)
                sim_h = param_hash(sim_d)
                if sim_h in existing:
                    continue
                node_tr["sim_tasks"].append({"hash": sim_h, "params": json_safe(sim_d)})
                existing.add(sim_h)
    return spec


def _count_spec_tasks(task_spec: Dict[str, Any]) -> Tuple[int, int, int]:
    n_prep = len(task_spec)
    n_train = sum(len(n["train"]) for n in task_spec.values())
    n_sim = sum(len(tr["sim_tasks"]) for pre in task_spec.values() for tr in pre["train"].values())
    return n_prep, n_train, n_sim


def load_done_set(reports_path: str) -> set[str]:
    """
    Read reports.jsonl and collect completed params.hash.
    """
    done: set[str] = set()
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
            h = (((d or {}).get("short").get("params") or {}).get("hash"))
            if isinstance(h, str) and h:
                done.add(h)
    return done


def _config_from_dict_train(train_params: Dict[str, Any]):
    """
    Restore TrainConfig from dict stored in task spec.
    Intentionally ignores nested model_cfg/data_cfg dicts in spec (those fields are dataclasses).
    """
    import model.train as train

    t_cfg = train.TrainConfig()
    for k, v in (train_params or {}).items():
        if isinstance(v, dict) and k in ("model_cfg", "data_cfg"):
            continue
        if hasattr(t_cfg, k):
            setattr(t_cfg, k, v)
    return t_cfg


def filter_pending_from_spec(task_spec: Dict[str, Any], done_set: set[str]) -> Dict[str, Any]:
    """
    Filter sim leaf tasks that are already present in reports.jsonl.
    """
    from trade.bt import simulation

    pending: Dict[str, Any] = {}
    for pre_h, pre_node in task_spec.items():
        pre_params = pre_node["params"]
        train_pending: Dict[str, Any] = {}

        for tr_h, tr_node in pre_node["train"].items():
            train_params = tr_node["params"]
            sim_pending = []
            for sim in tr_node.get("sim_tasks", []):
                task_hash = calc_params_hash(
                    strategy=simulation.StrategyPara(**sim["params"]),
                    common=common.BaseDefine(**pre_params),
                    train=_config_from_dict_train(train_params),
                )
                if task_hash not in done_set:
                    sim_pending.append(sim)

            if sim_pending:
                train_pending[tr_h] = {"params": train_params, "sim_tasks": sim_pending}

        if train_pending:
            pending[pre_h] = {"params": pre_params, "train": train_pending}

    return pending


def load_pending_tasks(exp_dir: str, done_set: set[str]) -> Tuple[Dict[str, Any], Tuple[int, int, int]]:
    """
    Load tasks_spec.json then filter already-finished tasks based on reports.jsonl.
    """
    tasks_spec_path = os.path.join(exp_dir, TASKS_SPEC_FILE)
    if not os.path.exists(tasks_spec_path):
        raise FileNotFoundError(f"Tasks spec not found: {tasks_spec_path}")
    with open(tasks_spec_path, "r", encoding="utf-8") as f:
        task_spec = json.load(f)
    total_counts = _count_spec_tasks(task_spec)
    pending = filter_pending_from_spec(task_spec, done_set)
    return pending, total_counts


def _create_output_dirs(task_spec: Dict[str, Any], temp_dir: str) -> None:
    """
    Create prep/train/sim output dirs for all pending tasks.
    """
    for pre_h, pre_node in task_spec.items():
        os.makedirs(_prep_output_dir(temp_dir, pre_h), exist_ok=True)
        for tr_h, tr_node in pre_node["train"].items():
            os.makedirs(_train_output_dir(temp_dir, pre_h, tr_h), exist_ok=True)
            for sim_task in tr_node.get("sim_tasks", []):
                sim_h = sim_task.get("hash")
                if sim_h:
                    os.makedirs(_sim_output_dir(temp_dir, pre_h, tr_h, sim_h), exist_ok=True)


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

    for stage in ["pre", "train", "sim"]:
        if not sweep[stage]:
            continue
        logger.info(f"  [{stage}]")
        for k, v in sweep[stage].items():
            logger.info(f"    {k}: {v}")

def collect_param_sweep(task_spec):
    sweep = {
        "pre": defaultdict(set),
        "train": defaultdict(set),
        "sim": defaultdict(set),
    }

    for pre_node in task_spec.values():
        # pre params
        collect_from_any(pre_node["params"], sweep["pre"])

        for tr_node in pre_node["train"].values():
            collect_from_any(tr_node["params"], sweep["train"])

            for sim in tr_node.get("sim_tasks", []):
                collect_from_any(sim["params"], sweep["sim"])

    def finalize(d):
        return {
            k: sorted(v)
            for k, v in d.items()
            if len(v) > 1
        }

    return {
        "pre": finalize(sweep["pre"]),
        "train": finalize(sweep["train"]),
        "sim": finalize(sweep["sim"]),
    }

def construct_task_doge():
    import model.train as train
    from trade.bt import simulation
    preparation_task: List[Any] = []
    for pn in [16,24,32]:#[4,6,8,12,16,20,24,28,32,36]: #[10,12,14,16,18]
        for vol_multiplier in [1.8]:#1.8,1.9,2
            for vol_ewma_span in [80]:
                preparation_task.append(common.BaseDefine(
                        market_category = "Cryptocurrency", data_source = "binance_public_data",
                        vol_ewma_span = vol_ewma_span, predict_num=pn,
                        vol_multiplier_long=vol_multiplier, stop_multiplier_rate_long=None, vol_multiplier_short=vol_multiplier, stop_multiplier_rate_short=None,
                        symbol=SYMBOL,  interval=INTERVAL, trading_type= 'um', label_type = 'FTHL', version=0 ))

    training_task: List[train.TrainConfig] = []

    to_remove_1 = ["open", "high",'low' ]
    to_remove_2 = to_remove_1 + ['close','volume','number_of_trades','taker_buy_base_volume','taker_buy_quote_volume']
    to_remove_3 = to_remove_1 + ['close','volume','taker_buy_base_volume','taker_buy_quote_volume']
    to_remove_4 = to_remove_1 + ['close','taker_buy_base_volume','taker_buy_quote_volume']
    to_remove_5 = to_remove_1 + ['taker_buy_base_volume','taker_buy_quote_volume']
    feature_conf_list_1 = [f for f in train_config.feature_conf_list if f not in to_remove_1]
    feature_conf_list_2 = [f for f in train_config.feature_conf_list if f not in to_remove_2]
    feature_conf_list_3 = [f for f in train_config.feature_conf_list if f not in to_remove_3]
    feature_conf_list_4 = [f for f in train_config.feature_conf_list if f not in to_remove_4]
    feature_conf_list_5 = [f for f in train_config.feature_conf_list if f not in to_remove_5]
    for false_trade_penalty in [1]:
        for seq_len in [12,16,24,96,256]: #12,16,24,32
            for flip_penalty in np.arange(0.8, 1.7, 1).round(1):# np.arange(0.2, 2.1, 0.1).round(1):
                for miss_penalty in np.arange(0.5,2, 0.1).round(1):#in np.arange(0.3, 2.1, 0.2).round(1):
                    for stride in [2]: #2,4,8
                        for bestf1 in [True]:
                            for featrue_conf in [train_config.feature_conf_list]:
                                train_conf = train.TrainConfig( use_cache = False,epochs = 100, batch_size=256,
                                                                    feature_conf_list= featrue_conf,
                                                            flip_penalty = float(flip_penalty),miss_penalty = float(miss_penalty),false_trade_penalty = 1,
                                                            stride = stride, patience = 8)
                                train_conf.model_cfg = train_config.LogisticConfig(seq_len = seq_len,model_version= 1)
                                # 单模型 sweep：这一整批配置统一按 TRAIN_MODE 指定的任务类型训练
                                # (默认 SINGLE_MODEL_3CLASS；也可以设成 SINGLE_MODEL_LONG_OVR 等)
                                train_conf.train_task = TRAIN_MODE
                                training_task.append(train_conf)

    simulation_task: List[Any] = []

    for i in [12,16,24]: #16,24,30,32,36,40,44,48
        holdbar = i
        for (atr_sl_mult_long, atr_sl_mult_short) in [(6,5),(5,4)]: #(6,5),(5,4)
            simulation_task.append(simulation.StrategyPara(allow_long=True,allow_short=True,holdbar=holdbar,commission=0.05,init_equity=10000.0,thresh=None,
                                            atr_sl_mult_long=atr_sl_mult_long,atr_sl_mult_short=atr_sl_mult_short,atr_tp=0.99,trade_risk=0.4,max_daily_loss_pct=0.04))
    return preparation_task, training_task, simulation_task

def construct_task_doge_combo():
    """
    combo_model(TRIGGER_DIR) 版本的任务构造：分别为 trigger / direction 两个子任务
    角色构造各自的超参 sweep（两个独立的循环，对应 Q&A 里"两种架构分成两个
    construct 函数"的约定——这里在单个 SYMBOL 下用两段循环各自打标签，
    而不是复用 construct_task_doge 那一套单模型三分类的循环）。

    两个角色的 TrainConfig 都放进同一个 flat training_task 列表里，只是通过
    .train_task 字段打上不同标签（SINGLE_MODEL_TRIGGER / SINGLE_MODEL_DIR）。
    训练完成后由 run_combo_fusion_and_backtest 按 (pre_key, train_compatibility)
    分组，把同组下所有 trigger x direction 两两 fuse 成组合模型再回测。
    """
    import model.train as train
    from trade.bt import simulation
    preparation_task: List[Any] = []
    for pn in [16, 24, 32]:
        for vol_multiplier in [1.8, 1.9]:
            for vol_ewma_span in [80]:
                preparation_task.append(common.BaseDefine(
                        market_category = "Cryptocurrency", data_source = "binance_public_data",
                        vol_ewma_span = vol_ewma_span, predict_num=pn,
                        vol_multiplier_long=vol_multiplier, stop_multiplier_rate_long=None, vol_multiplier_short=vol_multiplier, stop_multiplier_rate_short=None,
                        symbol=SYMBOL,  interval=INTERVAL, trading_type= 'um', label_type = 'FTHL', version=0 ))

    training_task: List[train.TrainConfig] = []

    seq_len = 24
    stride = 2
    # --- trigger 子任务 sweep ---
    for pos_ratio in np.arange(0.2, 0.3, 0.02).round(2):
        for miss_penalty in np.linspace(1 / pos_ratio / 3 * 2, 1 / pos_ratio * 4, 4).round(2):
            trig_cfg = train.TrainConfig(
                use_cache=False, epochs=20, batch_size=256,
                model_cfg=train_config.LogisticConfig(model_version=1, seq_len=seq_len),
                pos_ratio=pos_ratio, miss_penalty=float(miss_penalty), stride=stride, patience=5,
            )
            trig_cfg.train_task = train_config.TrainTask.SINGLE_MODEL_TRIGGER
            training_task.append(trig_cfg)

    # --- direction 子任务 sweep ---
    for model_cfg in [train_config.TransformerConfig(model_version=1, seq_len=seq_len)]:
        dir_cfg = train.TrainConfig(
            use_cache=False, epochs=20, batch_size=256,
            model_cfg=model_cfg, pos_ratio=0.5, stride=stride, patience=5,
        )
        dir_cfg.train_task = train_config.TrainTask.SINGLE_MODEL_DIR
        training_task.append(dir_cfg)

    simulation_task: List[Any] = []
    for i in [4, 8, 12, 16, 24]:
        holdbar = i
        for (atr_sl_mult_long, atr_sl_mult_short) in [(6, 5), (5, 4)]:
            simulation_task.append(simulation.StrategyPara(
                allow_long=True, allow_short=True, holdbar=holdbar, commission=0.05, init_equity=10000.0, thresh=None,
                atr_sl_mult_long=atr_sl_mult_long, atr_sl_mult_short=atr_sl_mult_short, atr_tp=0.99, trade_risk=0.4, max_daily_loss_pct=0.04))
    return preparation_task, training_task, simulation_task

def construct_task_eth():
    import model.train as train
    from trade.bt import simulation
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

    for false_trade_penalty in [1]:
        for flip_penalty in np.arange(0.9, 1.7, 0.1).round(1):# np.arange(0.2, 2.1, 0.1).round(1):
            for miss_penalty in np.arange(0.7, 1.2, 0.1).round(1):#in np.arange(0.3, 2.1, 0.2).round(1):
                for stride in [4,8]: #2,4,8
                    for bestf1 in [True]:
                        for loss_fun_version_v in [2]:
                            training_task.append(train.TrainConfig(use_cache = False,epochs = 100, batch_size=256,
                                                        flip_penalty = float(flip_penalty),miss_penalty = float(miss_penalty),false_trade_penalty = 1,
                                                        stride = stride, patience = 8,lambda_main = 0.7,lambda_dir = 0.7,lambda_cost = 0.4,mag_alpha = 0))

    simulation_task: List[Any] = []

    for i in [30,32,36,38,40,44]: #16,24,30,32,36,40,44,48
        holdbar = i
        for (atr_sl_mult_long, atr_sl_mult_short) in [(6,6),(5,4)]: #(6,5),(5,4)
            simulation_task.append(simulation.StrategyPara(allow_long=True,allow_short=True,holdbar=holdbar,commission=0.05,init_equity=10000.0,thresh=None,
                                            atr_sl_mult_long=atr_sl_mult_long,atr_sl_mult_short=atr_sl_mult_short,trade_risk=0.1,max_daily_loss_pct=0.025))
    return preparation_task, training_task, simulation_task

def create_task_spec(logger, exp_dir,done_set: set[str]):
    is_combo = TRAIN_MODE in train_config.COMBO_SUB_TASKS

    if is_combo:
        if SYMBOL == "DOGEUSDT":
            preparation_task, training_task, simulation_task = construct_task_doge_combo()
        else:
            raise RuntimeError(f"no combo construct for {SYMBOL} yet")
    elif SYMBOL == "DOGEUSDT":
        preparation_task, training_task, simulation_task = construct_task_doge()
    elif SYMBOL == "ETHUSDT":
        preparation_task, training_task, simulation_task = construct_task_eth()
    else:
        raise RuntimeError(f"no construct for {SYMBOL} yet")

    # combo_model 模式下，子模型自己不是可交易的策略，不在这里挂 sim_tasks——
    # 真正的回测任务是训练完成后针对两两 fuse 出来的组合模型单独生成的，
    # 见 run_combo_fusion_and_backtest。这里把真实的 simulation_task 列表
    # 原样返回给调用方，供 fuse 阶段使用。
    task_spec = build_task_spec(preparation_task, training_task, [] if is_combo else simulation_task)
    # task_spec is already ready
    sweep = collect_param_sweep(task_spec)
    log_param_sweep(logger, sweep)

    tasks_spec_path = os.path.join(exp_dir, TASKS_SPEC_FILE)
    with open(tasks_spec_path, "w", encoding="utf-8") as f:
        json.dump(task_spec, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"📄 Tasks spec saved: {tasks_spec_path}")

    (n_prep_total, n_train_total, n_sim_total) = _count_spec_tasks(task_spec)
    total_all = n_prep_total + n_train_total + n_sim_total
    logger.info(f"📊 Total: {total_all} (prep={n_prep_total}, train={n_train_total}, sim={n_sim_total})")
    if done_set:
        task_spec = filter_pending_from_spec(task_spec, done_set)
        n_prep, n_train, n_sim = _count_spec_tasks(task_spec)
        total_pending = n_prep + n_train + n_sim
        logger.info(f"📊 Pending: {total_pending} (prep={n_prep}, train={n_train}, sim={n_sim}), done: {total_all - total_pending}")
    return task_spec, (simulation_task if is_combo else None)

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
            for tr_h, tr_node in pre_task.get("train", {}).items():
                save_dir = _train_output_dir(temp_dir, pre_h, tr_h)
                train_items.append(
                    {
                        "pre_h": pre_h,
                        "tr_h": tr_h,
                        "pre_params": copy.deepcopy(pre_task["params"]),
                        "train_params": copy.deepcopy(tr_node["params"]),
                        "sim_tasks": copy.deepcopy(tr_node.get("sim_tasks", [])),
                        "prep_output_dir": prep_dir,
                        "save_dir": save_dir,
                    }
                )

            train_queue.put(("prep_done", pre_h, elapsed, train_items))


def _train_task(
    worker_log_file: str,
    item: Dict[str, Any],
    sim_task_queue: mp.Queue,
    train_result_queue: mp.Queue,
):
    """Run a single train in its own process, then enqueue sims."""
    logger = _worker_logger(worker_log_file)
    import model.train as train

    pre_h = item["pre_h"]
    tr_h = item["tr_h"]
    pre_params = item["pre_params"]
    train_params = item["train_params"]
    sim_tasks = item.get("sim_tasks", [])
    prep_output_dir = item["prep_output_dir"]
    save_dir = item["save_dir"]

    t0 = time.time()
    try:
        pre_para = common.BaseDefine(**pre_params)
        t_cfg = _config_from_dict_train(train_params)
        # 单模型模式(DIRECT_3CLASS/LONG_OVR/...)和 combo_model 子模型模式共用同一个
        # worker：具体训练哪种任务由 TrainConfig 自己的 train_task 字段决定（construct_task_doge*
        # 在构造每一行配置时已经打好标签），这里不再写死某一个任务类型。
        result = train.train(logger, config=t_cfg, prep_output_dir=prep_output_dir, save_dir=save_dir)

        # IMPORTANT: enqueue sims BEFORE reporting train_done (so main can safely send None after last train_done)
        for sim in sim_tasks:
            sim_task_queue.put((pre_h, pre_params, tr_h, train_params, sim, save_dir, {}))
        # 训练报告（task_type/metrics/save_dir 等）单独落一份 train_reports.jsonl。
        # 单模型模式下这只是额外的存档；combo_model 模式下这是后续 fuse 阶段
        # 用来按 (pre_key, train_compatibility) 分组、两两配对子模型的唯一数据源。
        train_report = {
            "pre_h": pre_h,
            "tr_h": tr_h,
            "task_type": t_cfg.train_task,
            "model_type": t_cfg.model_cfg.model_type,
            "model_version": t_cfg.model_cfg.model_version,
            "train_compatibility": t_cfg.train_compatibility,
            "metrics": result,
            "pre_params": pre_params,
            "train_params": train_params,
            "save_dir": save_dir,
        }
        train_result_queue.put(("train_done", pre_h, tr_h, time.time() - t0, train_report))
    except Exception:
        logger.exception(f"Train failed: {pre_h}/{tr_h}")
        train_result_queue.put(("train_failed", pre_h, tr_h, time.time() - t0, None))

def _worker_sim(worker_log_file: str, task_queue: mp.Queue, result_queue: mp.Queue, reports_path: str, temp_dir: str):
    logger = _worker_logger(worker_log_file)

    from trade.bt import simulation

    while True:
        try:
            msg = task_queue.get(timeout=0.5)
        except Empty:
            continue
        if msg is None:
            break

        # train_output_dir 由生产者（单模型 _train_task 或 combo_model 的
        # run_combo_fusion_and_backtest）显式给出，不再由 worker 自己根据 (pre_h, tr_h)
        # 反推路径——这样同一套 worker 池既能跑单模型的训练产出目录，也能跑
        # combo_model fuse 出来的组合模型目录，回测阶段两者完全一样。
        pre_h, pre_params, tr_h, train_params, sim, train_output_dir, extra_report_fields = msg
        sim_h = sim['hash']
        s_para = simulation.StrategyPara(**sim["params"])
        pre_para = common.BaseDefine(**pre_params)
        t_cfg = _config_from_dict_train(train_params)
        prep_dir = _prep_output_dir(temp_dir, pre_h)

        t0 = time.time()
        try:
            report_stat = None
            report = {'short':{}, 'long':{}, 'forward': {}, 'pass':False, **(extra_report_fields or {})}
            report['short'] = simulation.main( logger, para=s_para, pre_para=pre_para, train_cfg=t_cfg, prep_output_dir=prep_dir,
                                train_output_dir=train_output_dir, device="cpu", period='short' )["statistics"][1]
            if report['short']["performance"]["cagr"] > 0 :
                report['forward'] = simulation.main( logger, para=s_para, pre_para=pre_para, train_cfg=t_cfg, prep_output_dir=prep_dir,
                                    train_output_dir=train_output_dir, device="cpu", period='forward' )["statistics"][1]
                if report['forward']["performance"]["cagr"] > -1000 :
                    report['long'] = simulation.main( logger, para=s_para, pre_para=pre_para, train_cfg=t_cfg, prep_output_dir=prep_dir,
                                        train_output_dir=train_output_dir, device="cpu", period='long' )["statistics"][1]
                    if report['long']["performance"]["cagr"] > 0 :
                        report['pass'] = True
                else:
                    logger.info(f"Sim {pre_h}/{tr_h}/{sim_h} skip long test due to forward period performance cagr:{report['forward']['performance']['cagr']}")
            else:
                logger.info(f"Sim {pre_h}/{tr_h}/{sim_h} skip long test due to short period performance cagr:{report['short']['performance']['cagr']}")
            report_stat = report
        except Exception:
            logger.exception(f"Sim failed: {pre_h}/{tr_h}/{sim_h}")
            report_stat = None

        elapsed = time.time() - t0
        result_queue.put(("sim_done", pre_h, tr_h, sim_h, elapsed, report_stat, reports_path, train_output_dir))


# -----------------------------------------------------------------------------
# Result handling
# -----------------------------------------------------------------------------
def _drain_sim_results(sim_result_queue: mp.Queue, stats: Dict[str, Any], logger: logging.Logger, eta_msg,pending_sim_hashes: Dict[Tuple[str, str], Set[str]],temp_dir: str, valid:bool,task_spec) -> None:
    while True:
        try:
            msg = sim_result_queue.get_nowait()
        except Empty:
            break

        if not msg:
            continue
        typ = msg[0]
        if typ != "sim_done":
            continue

        _, pre_h, tr_h, sim_h, elapsed, report_stat, rp, train_dir = msg
        stats["simulation"]["time"] += elapsed
        stats["simulation"]["count"] += 1

        if report_stat is not None:
            common.append_jsonl(rp, report_stat)
            if valid == True:
                strategy_hash = report_stat['short']['params']['hash']
                target_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments",'valid_train_out', strategy_hash)
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.copytree(train_dir, target_dir)
                logger.info(f"🚀 Successfully moved artifacts: {tr_h} -> {strategy_hash} {target_dir}")
        # 2. Hash-based cleanup and deletion logic
        train_key = (pre_h, tr_h)
        if train_key in pending_sim_hashes:
            # Remove current finished sim_h from the pending set
            pending_sim_hashes[train_key].discard(sim_h)
            
            # If all sim tasks for this train task have been removed
            if not pending_sim_hashes[train_key]:
                if os.path.exists(train_dir):
                    try:
                        if valid == False:
                            shutil.rmtree(train_dir)
                            logger.info(f"🧹 All sims finished for Train {tr_h}. Deleted: {train_dir}")
                    except Exception as e:
                        logger.error(f"❌ Failed to handle {train_dir}: {e}")

        logger.info(f"    Sim {pre_h}/{tr_h}/{sim_h} done in {elapsed:.2f}s")
        em = eta_msg()
        if em:
            logger.info(f"    {em}")


def _send_none_to_workers(q: mp.Queue, n: int) -> None:
    for _ in range(n):
        q.put(None)


# -----------------------------------------------------------------------------
# Reporting: compare old/new (valid mode)
# -----------------------------------------------------------------------------
def compare_old_new_reports(old_reports_path: str, new_reports_path: str, output_dir: str, logger: logging.Logger):
    """
    Enhanced comparison between old (selected_configs) and new (reports) files.
    Supports CAGR precision comparison for \"short\", \"long\" and \"forward\" periods (rounded to 1 decimal place).
    """
    logger.info("\n" + "=" * 40)
    logger.info("📊 Starting Multi-Period Comparison...")

    periods = ["short", "long", "forward"]

    # --- 1. Define internal loader to avoid redundant I/O ---
    def load_records_by_hash(path: str, is_selected_config: bool = False) -> Dict[str, Dict[str, Any]]:
        """Load records from file and index them by hash, preserving all period information."""
        data_map = {}
        if not os.path.exists(path):
            return data_map
        
        # Support both selected_configs (list) and reports (jsonl)
        try:
            if is_selected_config:
                # Assume load_selected_configs is the existing helper
                records = load_selected_configs(path) 
            else:
                with open(path, "r", encoding="utf-8") as f:
                    records = [json.loads(line.strip()) for line in f if line.strip()]
        except Exception as e:
            logger.error(f"❌ Failed to load {path}: {e}")
            return data_map
        for record in records:
            h = record["short"].get("params", {}).get("hash")
            data_map[h] = record
        return data_map

    # --- 2. Load data ---
    old_data = load_records_by_hash(old_reports_path, is_selected_config=True)
    new_data = load_records_by_hash(new_reports_path, is_selected_config=False)

    logger.info(f"📥 Loaded {len(old_data)} old records and {len(new_data)} new records")

    # --- 3. Core comparison logic ---
    compare_results = []
    hashes_only_in_old = []
    hashes_only_in_new = list(set(new_data.keys()) - set(old_data.keys()))

    for h, old_record in old_data.items():
        if h not in new_data:
            hashes_only_in_old.append(h)
            continue

        new_record = new_data[h]
        # Initialize comparison entry
        comparison_entry = {
            "hash": h,
            "verify_all_passed": True,
            "period_details": {}
        }

        # Iterate over three periods
        for p in periods:
            old_p = old_record.get(p)
            new_p = new_record.get(p)

            # Case A: both reports contain this period
            if old_p and new_p:
                old_cagr = old_p.get("performance", {}).get("cagr")
                new_cagr = new_p.get("performance", {}).get("cagr")

                # Only compare when both CAGR values are numeric
                if isinstance(old_cagr, (int, float)) and isinstance(new_cagr, (int, float)):
                    # Compare with one decimal place (e.g., 0.1556 represents 15.6%)
                    v1 = round(old_cagr, 1)
                    v2 = round(new_cagr, 1)
                    is_match = (v1 == v2)
                    
                    if not is_match:
                        comparison_entry["verify_all_passed"] = False
                    
                    comparison_entry["period_details"][p] = {
                        "status": "match" if is_match else "mismatch",
                        "old_cagr": v1,
                        "new_cagr": v2
                    }
                else:
                    comparison_entry["period_details"][p] = {"status": "missing_performance_data"}
            
            # Case B: one side is missing this period
            elif old_p or new_p:
                comparison_entry["period_details"][p] = {"status": "period_not_in_both"}
            # If the period should exist in the strategy but is missing, mark as failed
                comparison_entry["verify_all_passed"] = False

        # Keep params in results for easier inspection
        comparison_entry["params"] = old_record.get("short", {}).get("params") or old_record.get("long", {}).get("params")
        compare_results.append(comparison_entry)

    # --- 4. Save and summarize ---
    if not compare_results:
        logger.warning("⚠️ No matching hashes found to compare.")
        return None, 0, len(hashes_only_in_old), len(hashes_only_in_new)

    output_path = os.path.join(output_dir, "compare_reports.jsonl")
    failed_count = sum(1 for r in compare_results if not r["verify_all_passed"])

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in compare_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"✅ Comparison finished. Result saved to: {output_path}")
    logger.info(f"📊 Matched: {len(compare_results)} | Failed: {failed_count}")
    logger.info(f"ℹ️  Only in old: {len(hashes_only_in_old)} | Only in new: {len(hashes_only_in_new)}")

    return output_path, len(compare_results), len(hashes_only_in_old), len(hashes_only_in_new)

# -----------------------------------------------------------------------------
# ETA helper
# -----------------------------------------------------------------------------
def _make_eta_fn(n_prep: int, n_train: int, n_sim: int, stats: Dict[str, Any]):
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
        sim_eta = phase_eta(n_sim, stats["simulation"]["count"], stats["simulation"]["time"], MAX_SIM)

        parts = []
        if n_prep > 0:
            parts.append(f"prep:{fmt(prep_eta)}")
        if n_train > 0:
            parts.append(f"train:{fmt(train_eta)}")
        if n_sim > 0:
            parts.append(f"sim:{fmt(sim_eta)}")

        if not parts:
            return ""

        total_eta = None
        if (n_prep == 0 or stats["preparation"]["count"] > 0) and (n_train == 0 or stats["train"]["count"] > 0) and (n_sim == 0 or stats["simulation"]["count"] > 0):
            total_eta = (prep_eta or 0) + (train_eta or 0) + (sim_eta or 0)

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

def train_and_cross_test(logger:logging.Logger,output_dir,task_spec: Dict[str, Any] = {}):
    from trade.bt import simulation
    #data prepare
    results = {}
    for pre_h, pre_node in task_spec.items():
        pre_params = pre_node["params"]
        pre_para = common.BaseDefine(**pre_params)
        prep_output_dir = os.path.join(output_dir,'prep',f'{pre_para.symbol}_{pre_para.interval}')
        if not os.path.exists(prep_output_dir):
            preparation.main(logger, para=pre_para, prep_output_dir=prep_output_dir)
            time.sleep(1)
        original_symbol = pre_para.symbol
        original_interval = pre_para.interval
        for tr_h, tr_node in pre_node["train"].items():
            train_params = tr_node["params"]
            t_cfg = _config_from_dict_train(train_params)
            for sim_task in tr_node['sim_tasks']:
                hash_value =  sim_task['hash']
                strategy_hash = sim_task['strategy_hash']
                train_save_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments",'valid_train_out', strategy_hash)
                if not os.path.exists(train_save_dir):
                    raise RuntimeError(f" {train_save_dir} not exist,run valid first!")
                sim_params = sim_task['params']
                sim_para=simulation.StrategyPara(**sim_params)
                results[strategy_hash] = {'orignal_symbol': f'{pre_para.symbol}_{pre_para.interval}','CAGR':{}}
                for symbol in ["DOGEUSDT","ETHUSDT", "BTCUSDT"]:   #BTCUSDT ETHUSDT DOGEUSDT
                    if symbol != original_symbol:
                        t_pre_para = common.BaseDefine(**pre_params)
                        t_pre_para.symbol = symbol
                        t_pre_para.interval = original_interval
                        sim_prep_output_dir = os.path.join(output_dir,'prep',f'{t_pre_para.symbol}_{t_pre_para.interval}')
                        if not os.path.exists(sim_prep_output_dir):
                            preparation.main(logger, para=t_pre_para, prep_output_dir=sim_prep_output_dir)
                            time.sleep(1)
                        result = simulation.main( logger, para=sim_para, pre_para=t_pre_para, train_cfg=t_cfg, prep_output_dir=sim_prep_output_dir,
                                                                train_output_dir=train_save_dir, device="cpu", period='long' )["statistics"][1]
                        results[strategy_hash][f'{t_pre_para.symbol}_{t_pre_para.interval}'] = result
                        results[strategy_hash]['CAGR'][f'{t_pre_para.symbol}_{t_pre_para.interval}'] = result['performance']['cagr']
                    else:
                        for interval in ["15m","30m","1h"]:
                            t_pre_para = common.BaseDefine(**pre_params)
                            t_pre_para.symbol = original_symbol
                            t_pre_para.interval = interval
                            sim_prep_output_dir = os.path.join(output_dir,'prep',f'{t_pre_para.symbol}_{t_pre_para.interval}')
                            if not os.path.exists(sim_prep_output_dir):
                                preparation.main(logger, para=t_pre_para, prep_output_dir=sim_prep_output_dir)
                                time.sleep(1)
                            result = simulation.main( logger, para=sim_para, pre_para=t_pre_para, train_cfg=t_cfg, prep_output_dir=sim_prep_output_dir,
                                                                    train_output_dir=train_save_dir, device="cpu", period='long' )["statistics"][1]
                            results[strategy_hash][f'{t_pre_para.symbol}_{t_pre_para.interval}'] = result
                            results[strategy_hash]['CAGR'][f'{t_pre_para.symbol}_{t_pre_para.interval}'] = result['performance']['cagr']
    output_path = os.path.join(output_dir, "cross_test_reports.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s_hash, data in results.items():
            record = {"strategy_hash": s_hash}
            record.update(data) 
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    logger.info(f"Successfully saved {len(results)} cross test records to {output_path}")

def run_combo_fusion_and_backtest(
    logger: logging.Logger,
    train_reports_path: str,
    temp_dir: str,
    simulation_task: List[Any],
    reports_path: str,
    sim_task_queue: mp.Queue,
    sim_result_queue: mp.Queue,
    valid: bool = False,
):
    """
    combo_model 模式训练完成后的 fuse + 回测阶段：

    1. 从 train_reports_path（每训练完一个子模型就追加一行，见 _train_task /
       run_task_spec 的 _drain_train_results）读回所有已完成的子模型记录。
    2. 按 (pre_h, train_compatibility) 分组——只有同一份数据准备(pre_h)、且
       seq_len/stride/特征集完全一致(train_compatibility)的两个子模型才允许
       配对，否则两个子模型消费的输入窗口对不上，融合没有意义。这个约束和
       batch_simulation.py 里 build_model_registry_from_reports/select_fusion_pairs
       的思路一致。
    3. 组内按 TRAIN_MODE 对应的两个角色(COMBO_SUB_TASKS[TRAIN_MODE])做笛卡尔积
       配对：M 个 role_1 子模型 x N 个 role_2 子模型 = M*N 个组合。
    4. 每一对调用 fusion_trigger_dir / fusion_long_short_ovr 生成组合模型的
       task_description.json，再用 simulation_task 跑 short/long/forward 回测，
       写入 reports_path。回测报告里的 model_metrics(融合后 3 类整体指标)和
       sub_model_metrics(两个子模型各自在回测数据上的指标)由 simulation.main
       ->ModelHandler.evaluate_sub_models 自动附带，这里不用手工计算。
    """
    import model.train as train

    role_1, role_2 = train_config.COMBO_SUB_TASKS[TRAIN_MODE]

    if not os.path.exists(train_reports_path):
        logger.warning(f"No train reports found at {train_reports_path}, nothing to fuse.")
        return

    records = []
    with open(train_reports_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    groups: Dict[Tuple[str, str], Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[(r["pre_h"], r["train_compatibility"])][r["task_type"]].append(r)

    fusion_pairs = []
    for (pre_h, compat), by_role in groups.items():
        role_1_items = by_role.get(role_1, [])
        role_2_items = by_role.get(role_2, [])
        logger.info(
            f"[combo fuse] pre={pre_h}, compat={compat}: "
            f"{role_1}={len(role_1_items)}, {role_2}={len(role_2_items)} -> "
            f"{len(role_1_items) * len(role_2_items)} pairs"
        )
        for r1 in role_1_items:
            for r2 in role_2_items:
                fusion_pairs.append((pre_h, compat, r1, r2))

    logger.info(f"[combo fuse] total fusion pairs: {len(fusion_pairs)}")

    # 粗粒度的 resume-safety：整个 fusion_hash（不细到单个 sim_task）已经出现
    # 在 reports_path 里就跳过——不是完整的 --resume 支持，只是避免重复脚本被
    # 误跑两次时把同一批回测重复写一遍。
    done_fusion_hashes = set()
    if os.path.exists(reports_path):
        with open(reports_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fh = d.get("fusion_hash")
                if fh:
                    done_fusion_hashes.add(fh)

    pending_sim_hashes: Dict[Tuple[str, str], Set[str]] = {}
    n_sim_total = 0

    for pre_h, compat, r1, r2 in fusion_pairs:
        payload = {
            "pre_h": pre_h, "compat": compat,
            f"{role_1}_tr_h": r1["tr_h"], f"{role_2}_tr_h": r2["tr_h"],
        }
        fusion_hash = hashlib.sha1(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]

        if fusion_hash in done_fusion_hashes:
            logger.info(f"[combo fuse] skip fusion_hash={fusion_hash} (already in reports)")
            continue

        fusion_dir = os.path.join(temp_dir, f"pre_{pre_h}", "fusion", f"compat_{compat}", f"fusion_{fusion_hash}")
        os.makedirs(fusion_dir, exist_ok=True)

        logger.info(f"[combo fuse] fusing {role_1}={r1['tr_h']} + {role_2}={r2['tr_h']} -> {fusion_hash}")
        if TRAIN_MODE == train_config.TrainTask.TRIGGER_DIR:
            train.fusion_trigger_dir(logger, r1["save_dir"], r2["save_dir"], fusion_dir)
        else:
            train.fusion_long_short_ovr(logger, r1["save_dir"], r2["save_dir"], fusion_dir)

        # 回测阶段和单模型模式完全一样(融合完的目录对 simulation.main 而言就是
        # 一个普通的 train_output_dir)，所以这里不再自己顺序调用 simulation.main，
        # 而是把每个 sim_task 塞进和单模型模式共用的 sim_task_queue，由已经在跑的
        # MAX_SIM 个 _worker_sim 并行处理——combo 模式下这几个 worker 进程本来就
        # 是空闲的(单模型子任务不挂 sim_tasks)，现在正好用上。
        extra_report_fields = {
            "fusion_hash": fusion_hash,
            "train_compatibility": compat,
            role_1: {"tr_h": r1["tr_h"], "metrics": r1["metrics"], "train_params": r1["train_params"]},
            role_2: {"tr_h": r2["tr_h"], "metrics": r2["metrics"], "train_params": r2["train_params"]},
        }
        for sim_task in simulation_task:
            sim_d = json_safe(asdict(sim_task))
            sim_h = param_hash(sim_d)
            pending_sim_hashes.setdefault((pre_h, fusion_hash), set()).add(sim_h)
            sim_task_queue.put((
                pre_h, r1["pre_params"], fusion_hash, r1["train_params"],
                {"hash": sim_h, "params": sim_d}, fusion_dir, extra_report_fields,
            ))
            n_sim_total += 1

    logger.info(f"[combo fuse] enqueued {n_sim_total} backtest tasks across {len(fusion_pairs)} fusion pairs")

    if n_sim_total == 0:
        return

    # 复用单模型模式同一套 _drain_sim_results：写 reports_path、以及所有 sim
    # 跑完后删掉对应的 fusion_dir（和单模型训练目录的清理逻辑一致），只是这里
    # 没有 prep/train 阶段要一起协调，用一个简单的轮询循环等它跑完就够了。
    stats = {"simulation": {"time": 0.0, "count": 0}}
    def _no_eta():
        return ""
    while stats["simulation"]["count"] < n_sim_total:
        _drain_sim_results(sim_result_queue, stats, logger, _no_eta, pending_sim_hashes, temp_dir, valid, {})
        if stats["simulation"]["count"] < n_sim_total:
            time.sleep(0.5)
    logger.info(f"[combo fuse] all {n_sim_total} backtest tasks done.")

def main():
    parser = argparse.ArgumentParser(description="Batch experiments: prep -> train -> sim (with resume)")
    parser.add_argument("-p", "--prep", action="store_true", help="Execute data preparation stage")
    parser.add_argument("-t", "--train", action="store_true", help="Execute model training stage")
    parser.add_argument("-s", "--sim", action="store_true", help="Execute backtest simulation stage")
    parser.add_argument("-n", "--new", action="store_true",  help="new train")
    parser.add_argument("-a", "--add", type=str, help="add more to exist expirement")
    parser.add_argument("-v", "--valid", action="store_true", default=False, help="Rerun selected_configs.jsonl then compare")
    parser.add_argument("-r", "--resume", type=str, help="Resume experiment from specified directory name under PERSISTENCE_DIR")
    parser.add_argument("-c", "--cross_test", action="store_true", default=False, help="crosss test")
    parser.add_argument("-l", "--load", action="store_true", default=False, help="load condidate configs for verification,befor applying to market")

    args = parser.parse_args()
    run_all = args.new

    if TRAIN_MODE in train_config.COMBO_SUB_TASKS and (
        args.resume or args.add or args.valid or args.cross_test or args.load
    ):
        print(
            "❌ combo_model 模式 (TRAIN_MODE=TRIGGER_DIR/LONG_SHORT_OVR) 暂不支持 "
            "--resume/--add/--valid/--cross_test/--load，这些流程假设的是单模型模式下 "
            "sim_tasks 直接挂在训练节点上的 spec 形状。请用不带这些参数的全新实验跑一遍。"
        )
        return

    # ---------------- resolve exp_dir ----------------
    if args.resume:
        exp_dir = os.path.join(common.PERSISTENCE_DIR, args.resume)
        if not os.path.exists(exp_dir):
            print(f"❌ Error: Resume directory not found: {exp_dir}")
            return
    elif args.add:
        exp_dir = os.path.join(common.PERSISTENCE_DIR, args.add)
        if not os.path.exists(exp_dir):
            print(f"❌ Error: add directory not found: {exp_dir}")
            return 
    elif args.valid:
        selected_configs = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs", SELECTED_FILE)
        exp_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs")
        os.makedirs(exp_dir, exist_ok=True)
        if not os.path.exists(selected_configs):
            print(f"❌ Error: valid file not found: {selected_configs}")
            return
    elif args.cross_test:
        selected_configs = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs", SELECTED_FILE)
        exp_dir = os.path.join(common.TEMPORARY_DIR, "batch_experiments", "selected_configs", "cross_test")
        os.makedirs(exp_dir, exist_ok=True)
        if not os.path.exists(selected_configs):
            print(f"❌ Error: select file not found: {selected_configs}")
            return
    elif args.load:
        selected_configs = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs", SELECTED_FILE)
        records = common.load_selected_configs(selected_configs)  # just to validate file and format
        from trade.bt import simulation
        import model.train as train
        exp_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "load_configs")
        os.makedirs(exp_dir, exist_ok=True)
        logger = _setup_root_logger(exp_dir)
        common.get_git_info(logger)
        begin_time = time.time()
        results = []
        for r in records:
            report = {'short':{}, 'long':{}}
            params = r["short"] if "short" in r else r
            sim_para =simulation.StrategyPara(**params["params"]["strategy"])
            pre_para =common.BaseDefine(**params["params"]["common"])
            train_para=_config_from_dict_train(params["params"]["train"])
            load_prep_output_dir = os.path.join(common.TEMPORARY_DIR, "batch_experiments", "load_configs",'prep',f'{pre_para.symbol}_{pre_para.interval}')
            strategy_hash = params["params"]['hash']
            #prepare train output for market
            train_save_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments",'valid_train_out', strategy_hash)
            if not os.path.exists(train_save_dir):
                logger.info(f"skip {strategy_hash}, tarin data not found {train_save_dir}")
                continue
            preparation.main(logger, para=pre_para,prep_output_dir = load_prep_output_dir)
            last_cagr = 0
            for trade_risk in [0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                result = {}
                sim_para.trade_risk = trade_risk
                result[strategy_hash] = {trade_risk:{'cagr':{}}}
                short_result = simulation.main(logger, pre_para=pre_para, para=sim_para, train_cfg=train_para, prep_output_dir =load_prep_output_dir,train_output_dir= train_save_dir,period='short')["statistics"][1]
                long_result = simulation.main(logger, pre_para=pre_para, para=sim_para, train_cfg=train_para, prep_output_dir =load_prep_output_dir,train_output_dir= train_save_dir,period='long')["statistics"][1]
                forward_result = simulation.main(logger, pre_para=pre_para, para=sim_para, train_cfg=train_para,prep_output_dir =load_prep_output_dir,train_output_dir= train_save_dir, period='forward')["statistics"][1]
                result[strategy_hash][trade_risk]['cagr']['short'] = short_result['performance']['cagr']
                result[strategy_hash][trade_risk]['cagr']['long'] = long_result['performance']['cagr']
                result[strategy_hash][trade_risk]['cagr']['forward'] = forward_result['performance']['cagr']
                result[strategy_hash][trade_risk]['short'] = short_result
                result[strategy_hash][trade_risk]['long'] = long_result
                result[strategy_hash][trade_risk]['forward'] = forward_result
                results.append(result)
                if long_result['performance']['cagr'] < last_cagr:
                    break
                last_cagr = long_result['performance']['cagr']
        output_path = os.path.join(exp_dir, 'trade_risk_test' , "loaded_reports.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok= True)
        with open(output_path, "w", encoding="utf-8") as f:
            for report in results:
                f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")
        logger.info(f"✅ Completed in {time.time() - begin_time:.2f}s , saved to {output_path}")
        exit(0)
    else:
        exp_dir = common.create_experiment_dir(
            os.path.join(common.PERSISTENCE_DIR, "batch_experiments"),
            SYMBOL,
            INTERVAL,
        )

    logger = _setup_root_logger(exp_dir)
    common.get_git_info(logger)

    begin_time = time.time()
    reports_path = os.path.join(exp_dir, REPORTS_FILE)
    train_reports_path = os.path.join(exp_dir, TRAIN_REPORTS_FILE)

    temp_dir = _batch_temp_dir(exp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    combo_simulation_task = None

    # ---------------- build/load spec ----------------
    if args.resume:
        done_set = load_done_set(reports_path)
        task_spec, (n_prep_total, n_train_total, n_sim_total) = load_pending_tasks(exp_dir, done_set)
        n_prep, n_train, n_sim = _count_spec_tasks(task_spec)
        total_all = n_prep_total + n_train_total + n_sim_total
        total_pending = n_prep + n_train + n_sim
        logger.info(f"📥 Loaded from {exp_dir}")
        logger.info(f"📊 Total: {total_all} (prep={n_prep_total}, train={n_train_total}, sim={n_sim_total})")
        logger.info(f"📊 Pending: {total_pending} (prep={n_prep}, train={n_train}, sim={n_sim}), done: {total_all - total_pending}")
    elif args.add:
        done_set = load_done_set(reports_path)
        task_spec, combo_simulation_task = create_task_spec(logger, exp_dir, done_set)
    elif args.valid:
        valid_save_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments",'valid_train_out')
        shutil.rmtree(valid_save_dir,ignore_errors=True)
        selected_configs = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs", SELECTED_FILE)
        task_spec = _load_task_from_configs(selected_configs)
        n_prep, n_train, n_sim = _count_spec_tasks(task_spec)
        logger.info(f"📥 Loaded from {selected_configs}")
        logger.info(f"📊 Pending: prep={n_prep}, train={n_train}, sim={n_sim}")
    elif args.cross_test:
        task_spec = _load_task_from_configs(selected_configs)
        n_prep, n_train, n_sim = _count_spec_tasks(task_spec)
        logger.info(f"📥 Loaded from {selected_configs}")
        logger.info(f"📊 Pending: prep={n_prep}, train={n_train}, sim={n_sim}")
        train_and_cross_test(logger,exp_dir,task_spec)
        exit()
    else:
        task_spec, combo_simulation_task = create_task_spec(logger, exp_dir, None)
    if not task_spec:
        logger.info("✅ No pending tasks.")
        return

    logger.info(f"🚀 Pipeline: MAX_PREP={MAX_PREP}, train={MAX_TRAIN}, MAX_SIM={MAX_SIM}")

    # ---------------- queues & workers ----------------
    prep_task_queue: mp.Queue = mp.Manager().Queue()
    train_task_queue: mp.Queue = mp.Manager().Queue()
    train_result_queue: mp.Queue = mp.Manager().Queue()
    sim_task_queue: mp.Queue = mp.Manager().Queue()
    sim_result_queue: mp.Queue = mp.Manager().Queue()

    # start workers
    prep_workers = []
    for i in range(MAX_PREP):
        worker_log = os.path.join(exp_dir, f"prep_{i}.log")
        p = mp.Process(target=_worker_prep, args=(worker_log, prep_task_queue, train_task_queue, temp_dir))
        p.start()
        prep_workers.append(p)

    is_combo = TRAIN_MODE in train_config.COMBO_SUB_TASKS

    # combo_model 模式下，训练阶段的子模型不挂 sim_tasks，这里的 sim_task_queue/
    # sim_workers 在训练期间根本用不上——而且 run_task_spec 一旦训练全部完成就会
    # 自动往 sim_task_queue 里塞 None 把 sim worker 关掉（单模型模式下这是对的，
    # 因为那时候该跑的 sim 早就在训练过程中挂上队列了；combo 模式下这时候还
    # 一个回测任务都没有）。所以 combo 模式这里不提前起 sim worker，等子模型
    # 全部训练完、fuse 阶段真正有回测任务要跑时再起一批新的（见下面
    # run_combo_fusion_and_backtest 调用处），避免被提前关掉。
    sim_workers = []
    if not is_combo:
        for i in range(MAX_SIM):
            worker_log = os.path.join(exp_dir, f"sim_{i}.log")
            p = mp.Process(target=_worker_sim, args=(worker_log, sim_task_queue, sim_result_queue, reports_path, temp_dir))
            p.start()
            sim_workers.append(p)
    run_task_spec(
        task_spec,
        temp_dir,
        exp_dir,
        prep_task_queue,
        train_task_queue,
        train_result_queue,
        sim_task_queue,
        sim_result_queue,
        logger,
        prep_workers,
        sim_workers,
        valid= args.valid,
        train_reports_path=train_reports_path,
    )


    _send_none_to_workers(prep_task_queue, MAX_PREP)
    if not is_combo:
        _send_none_to_workers(sim_task_queue, MAX_SIM)

    if args.valid:
        selected_configs = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs", SELECTED_FILE)
        compare_old_new_reports(selected_configs, reports_path, exp_dir, logger)

    # combo_model 模式：子模型全部训练完成后，按 (pre_key, train_compatibility)
    # 分组两两 fuse 成组合模型并回测，结果同样写入 reports_path。
    # 用一套全新的 queue + sim worker（而不是复用上面训练阶段那套），避免和
    # run_task_spec 训练完就自动发的 None 关闭信号产生竞争/串号。
    if is_combo:
        combo_sim_task_queue: mp.Queue = mp.Manager().Queue()
        combo_sim_result_queue: mp.Queue = mp.Manager().Queue()
        combo_sim_workers = []
        for i in range(MAX_SIM):
            worker_log = os.path.join(exp_dir, f"combo_sim_{i}.log")
            p = mp.Process(target=_worker_sim, args=(worker_log, combo_sim_task_queue, combo_sim_result_queue, reports_path, temp_dir))
            p.start()
            combo_sim_workers.append(p)

        try:
            run_combo_fusion_and_backtest(
                logger=logger,
                train_reports_path=train_reports_path,
                temp_dir=temp_dir,
                simulation_task=combo_simulation_task or [],
                reports_path=reports_path,
                sim_task_queue=combo_sim_task_queue,
                sim_result_queue=combo_sim_result_queue,
                valid=args.valid,
            )
        finally:
            _send_none_to_workers(combo_sim_task_queue, MAX_SIM)
            for p in combo_sim_workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

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
    sim_task_queue,
    sim_result_queue,
    logger,
    prep_workers,
    sim_workers,
    valid = False,
    train_reports_path = None,
):
    stats = {"preparation": {"time": 0.0, "count": 0}, "train": {"time": 0.0, "count": 0}, "simulation": {"time": 0.0, "count": 0}}
    # key: (pre_h, tr_h), value: set of sim_h
    pending_sim_hashes = {}
    for pre_h, pre_node in task_spec.items():
        for tr_h, tr_node in pre_node["train"].items():
            sim_ids = {sim["hash"] for sim in tr_node.get("sim_tasks", [])}
            if sim_ids:
                pending_sim_hashes[(pre_h, tr_h)] = sim_ids
    _create_output_dirs(task_spec, temp_dir)
    prep_task_queue.put(task_spec)

    # ETA printer
    n_prep, n_train, n_sim = _count_spec_tasks(task_spec)
    eta_msg = _make_eta_fn(n_prep, n_train, n_sim, stats)

    # If no train stage exists, we should stop sim workers after we enqueue all sims.
    sim_nones_sent = (n_train == 0)

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

    def _drain_train_results():
        nonlocal sim_nones_sent
        while True:
            try:
                msg = train_result_queue.get_nowait()
            except Empty:
                break
            if not msg:
                continue
            typ, pre_h, tr_h, elapsed, train_report = msg
            if typ == "train_failed":
                logger.error(f"❌ Train failed for {pre_h}/{tr_h}, aborting.")
                raise RuntimeError("train_failed")

            stats["train"]["time"] += float(elapsed)
            stats["train"]["count"] += 1
            logger.info(f"  {pre_h}/{tr_h}  Train done in {elapsed:.2f}s")
            if train_reports_path and train_report is not None:
                common.append_jsonl(train_reports_path, train_report)

            # safe to stop sim workers only after ALL train_done received
            if stats["train"]["count"] >= n_train and not sim_nones_sent:
                _send_none_to_workers(sim_task_queue, MAX_SIM)
                sim_nones_sent = True

    def _try_start_train_procs():
        nonlocal train_idx
        _reap_train_procs()
        while pending_train_items and len(train_procs) < MAX_TRAIN:
            item = pending_train_items.pop(0)
            worker_log = os.path.join(exp_dir, f"train_{train_idx%MAX_TRAIN}.log")
            p = mp.Process(target=_train_task, args=(worker_log, item, sim_task_queue, train_result_queue))
            p.start()
            train_procs.append(p)
            train_idx += 1

    # main loop: consume prep_done -> spawn train processes -> enqueue sims; also drain sim results
    try:
        while stats["preparation"]["count"] < n_prep or stats["train"]["count"] < n_train or stats["simulation"]["count"] < n_sim:
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
            _drain_sim_results(sim_result_queue, stats, logger, eta_msg, pending_sim_hashes, temp_dir, valid, task_spec)

        # final drains
        _drain_train_results()
        _drain_sim_results(sim_result_queue, stats, logger, eta_msg, pending_sim_hashes, temp_dir, valid, task_spec)
    except RuntimeError:
        pass
    finally:
        # best-effort shutdown
        for p in train_procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for p in prep_workers + sim_workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

def _load_task_from_configs(path: str) -> Dict[str, Any]:
    """
    Build a task spec tree from selected_configs.jsonl.
    Each record is a full report with {"params": {"strategy":.., "common":.., "train":.., "hash":..}}
    """
    task_spec: Dict[str, Any] = {}
    records = load_selected_configs(path)
    for r in records:
        params = common.recursive_get(r, "params")

        r_hash = params['hash']
        pre_conf =  common.recursive_get(params, "common")
        tr_conf  =   common.recursive_get(params, "train")
        sim_conf =  common.recursive_get(params, "strategy")

        if not pre_conf or not tr_conf or not sim_conf:
            continue

        if "prep_output_dir" in pre_conf or "save_dir" in tr_conf:
            raise ValueError("Unexpected prep_output_dir/save_dir in config params")

        pre_h = param_hash(pre_conf)
        tr_h = param_hash(tr_conf)
        sim_h = param_hash(sim_conf)

        if pre_h in task_spec:
            print(f"⚠️  Warning: duplicate prep config hash {pre_h} in {path}")
        node_pre = task_spec.setdefault(pre_h, {"params": json_safe(pre_conf), "train": {}})
        node_tr = node_pre["train"].setdefault(tr_h, {"params": json_safe(tr_conf), "sim_tasks": []})

        existing = {s["hash"] for s in node_tr["sim_tasks"]}
        if sim_h not in existing:
            node_tr["sim_tasks"].append({"hash": sim_h, "params": json_safe(sim_conf),"strategy_hash":r_hash})

    return task_spec

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
