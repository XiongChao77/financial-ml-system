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
try:
    import psutil  # optional
except ImportError:  # pragma: no cover
    psutil = None

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

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
SELECTED_FILE = "selected_configs.jsonl"
MAX_PREP = 1
MAX_TRAIN = 4  # max concurrent train processes (each train runs in its own process)
MAX_SIM = 4
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
    import model.train_2head as train

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

def log_param_sweep(logger, sweep):
    logger.info("📌 Experiment parameter sweep:")

    for stage in ["pre", "train", "sim"]:
        if not sweep[stage]:
            continue
        logger.info(f"  [{stage}]")
        for k, v in sweep[stage].items():
            logger.info(f"    {k}: {v}")

def create_task_spec(logger, exp_dir,done_set: set[str]):

    import model.train_2head as train
    from trade.bt import simulation

    preparation_task: List[Any] = []

    for cn in [12]:#list(range(56, 224, 8)): #[4,8,12,56,64,72,80,88,96,108,116,124,132,144,156,168,176,188]
        for pn in [4,8,24,40]:#[4,6,8,12,16,20,24,28,32,36]: #[10,12,14,16,18]
            for vol_multiplier in [1.7,1.8,1.9,2]:#1.8,1.9,2
                for vol_ewma_span in [80,88]:
                    preparation_task.append(common.BaseDefine(
                            vol_ewma_span = vol_ewma_span,
                            candlestick_num=cn,
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
    for false_trade in [1]:
        for flip_penalty in np.arange(1, 1.6, 0.1).round(1):# np.arange(0.2, 2.1, 0.1).round(1):
            for miss_penalty in np.arange(0.8, 1.1, 0.1).round(1):#in np.arange(0.3, 2.1, 0.2).round(1):
                for stride in [4,8]: #2,4,8
                    for bestf1 in [True]:
                        for loss_fun_version_v in [2]:
                            training_task.append(train.TrainConfig(use_cache = False,epochs = 100, batch_size=256,best_f1=bestf1,loss_fun_version = loss_fun_version_v,
                                                        flip_penalty = float(flip_penalty),miss_penalty = float(miss_penalty),false_trade = 1,
                                                        stride = stride, patience = 8,lambda_main = 0.7,lambda_dir = 0.7,lambda_cost = 0.4,mag_alpha = 0))

    simulation_task: List[Any] = []

    for i in [30,32,36,38]: #16,24,30,32,36,40,44,48
        holdbar = i
        for (atr_sl_mult_long, atr_sl_mult_short) in [(8,6),(6,6)]: #(6,5),(5,4)
            simulation_task.append(simulation.StrategyPara(allow_long=True,allow_short=True,holdbar=holdbar,commission=0.05,cash=10000.0,thresh=None,stop_loss_long=0.03,
                                            stop_loss_short=0.015,atr_sl_mult_long=atr_sl_mult_long,atr_sl_mult_short=atr_sl_mult_short,take_profit=0.99,trade_risk=0.4,max_daily_loss_pct=0.04))
    task_spec = build_task_spec(preparation_task, training_task, simulation_task)
    # task_spec 已经 ready
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
    import model.train_2head as train

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
        train.main(logger, train_cfg=t_cfg, pre_para=pre_para, prep_output_dir=prep_output_dir, save_dir=save_dir,experiment=True)

        # IMPORTANT: enqueue sims BEFORE reporting train_done (so main can safely send None after last train_done)
        for sim in sim_tasks:
            sim_task_queue.put((pre_h, pre_params, tr_h, train_params, sim))
        train_result_queue.put(("train_done", pre_h, tr_h, time.time() - t0))
    except Exception:
        logger.exception(f"Train failed: {pre_h}/{tr_h}")
        train_result_queue.put(("train_failed", pre_h, tr_h, time.time() - t0))

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

        pre_h, pre_params, tr_h, train_params, sim = msg
        sim_h = sim['hash']
        s_para = simulation.StrategyPara(**sim["params"])
        pre_para = common.BaseDefine(**pre_params)
        t_cfg = _config_from_dict_train(train_params)
        prep_dir = _prep_output_dir(temp_dir, pre_h)
        train_output_dir = _train_output_dir(temp_dir, pre_h, tr_h)
        
        t0 = time.time()
        try:
            report_stat = None
            report = {'short':{}, 'long':{}, 'forward': {}, 'pass':False}
            report['short'] = simulation.main( logger, para=s_para, pre_para=pre_para, train_cfg=t_cfg, prep_output_dir=prep_dir,
                                train_output_dir=train_output_dir, device="cpu", period='short' )["statistics"][1]
            if report['short']["performance"]["cagr"] > 0.2 :
                report['forward'] = simulation.main( logger, para=s_para, pre_para=pre_para, train_cfg=t_cfg, prep_output_dir=prep_dir,
                                    train_output_dir=train_output_dir, device="cpu", period='forward' )["statistics"][1]
                if report['forward']["performance"]["cagr"] > 0.2 :
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
        result_queue.put(("sim_done", pre_h, tr_h, sim_h, elapsed, report_stat, reports_path))


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

        _, pre_h, tr_h, sim_h, elapsed, report_stat, rp = msg
        stats["simulation"]["time"] += elapsed
        stats["simulation"]["count"] += 1

        train_dir = _train_output_dir(temp_dir, pre_h, tr_h)
        if report_stat is not None:
            common.append_jsonl(rp, report_stat)
            if valid == True:
                strategy_hash = report_stat['short']['params']['hash']
                target_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments",'valid_train_out', strategy_hash)
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.copytree(train_dir, target_dir)
                logger.info(f"🚀 Successfully moved artifacts: {tr_h} -> {strategy_hash} {target_dir}")
        # 2. 基于哈希的核销与清理逻辑
        train_key = (pre_h, tr_h)
        if train_key in pending_sim_hashes:
            # 从待办集合中移除当前完成的 sim_h
            pending_sim_hashes[train_key].discard(sim_h)
            
            # 如果该训练任务对应的所有模拟任务都已从集合中移除
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
    完善版：对比 old (selected_configs) 和 new (reports) 报告。
    支持 "short", "long", "forward" 三个周期的 CAGR 精度对比（保留一位小数）。
    """
    logger.info("\n" + "=" * 40)
    logger.info("📊 Starting Multi-Period Comparison...")

    periods = ["short", "long", "forward"]

    # --- 1. 定义内部加载函数，避免冗余 I/O ---
    def load_records_by_hash(path: str, is_selected_config: bool = False) -> Dict[str, Dict[str, Any]]:
        """将文件中的记录按 hash 索引，保留所有周期的信息"""
        data_map = {}
        if not os.path.exists(path):
            return data_map
        
        # 兼容 selected_configs (list) 和 reports (jsonl)
        try:
            if is_selected_config:
                # 假设 load_selected_configs 是你已有的函数
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

    # --- 2. 加载数据 ---
    old_data = load_records_by_hash(old_reports_path, is_selected_config=True)
    new_data = load_records_by_hash(new_reports_path, is_selected_config=False)

    logger.info(f"📥 Loaded {len(old_data)} old records and {len(new_data)} new records")

    # --- 3. 核心对比逻辑 ---
    compare_results = []
    hashes_only_in_old = []
    hashes_only_in_new = list(set(new_data.keys()) - set(old_data.keys()))

    for h, old_record in old_data.items():
        if h not in new_data:
            hashes_only_in_old.append(h)
            continue

        new_record = new_data[h]
        # 初始化对比条目
        comparison_entry = {
            "hash": h,
            "verify_all_passed": True,
            "period_details": {}
        }

        # 遍历三个周期
        for p in periods:
            old_p = old_record.get(p)
            new_p = new_record.get(p)

            # 情况 A: 两个报告中都有这个周期的内容
            if old_p and new_p:
                old_cagr = old_p.get("performance", {}).get("cagr")
                new_cagr = new_p.get("performance", {}).get("cagr")

                # 只有当两个 CAGR 都是数值时才比较
                if isinstance(old_cagr, (int, float)) and isinstance(new_cagr, (int, float)):
                    # 保留一位小数比较 (假设 cagr 为 0.1556 代表 15.6%)
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
            
            # 情况 B: 其中一方缺失该周期
            elif old_p or new_p:
                comparison_entry["period_details"][p] = {"status": "period_not_in_both"}
                # 如果这个周期在策略中本该存在却缺失，标记失败
                comparison_entry["verify_all_passed"] = False

        # 将 params 保留一份在结果里方便回溯
        comparison_entry["params"] = old_record.get("short", {}).get("params") or old_record.get("long", {}).get("params")
        compare_results.append(comparison_entry)

    # --- 4. 保存与统计 ---
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
    logger.info(f"ℹ️  Only in Old: {len(hashes_only_in_old)} | Only in New: {len(hashes_only_in_new)}")

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
        import model.train_2head as train
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

    temp_dir = _batch_temp_dir(exp_dir)
    os.makedirs(temp_dir, exist_ok=True)

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
        task_spec = create_task_spec(logger, exp_dir, done_set)
    elif args.valid:
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
        task_spec = create_task_spec(logger, exp_dir, None)
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

    sim_workers = []
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
        valid= args.valid
    )
    
    
    _send_none_to_workers(prep_task_queue, MAX_PREP)
    _send_none_to_workers(sim_task_queue, MAX_SIM)

    if args.valid:
        selected_configs = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs", SELECTED_FILE)
        compare_old_new_reports(selected_configs, reports_path, exp_dir, logger)

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
    valid = False
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
            typ, pre_h, tr_h, elapsed = msg
            if typ == "train_failed":
                logger.error(f"❌ Train failed for {pre_h}/{tr_h}, aborting.")
                raise RuntimeError("train_failed")

            stats["train"]["time"] += float(elapsed)
            stats["train"]["count"] += 1
            logger.info(f"  {pre_h}/{tr_h}  Train done in {elapsed:.2f}s")

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
