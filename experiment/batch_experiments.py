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

import argparse
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
MAX_TRAIN = 2  # max concurrent train processes (each train runs in its own process)
MAX_SIM = 4
# -----------------------------------------------------------------------------
# Path layout helpers
# -----------------------------------------------------------------------------
def _batch_temp_dir(exp_dir: str) -> str:
    """
    Put all intermediate artifacts under TEMPORARY_DIR so persistence stays clean.
    """
    if exp_dir.startswith(common.PERSISTENCE_DIR):
        rel = os.path.relpath(exp_dir, common.PERSISTENCE_DIR)
        return os.path.join(common.TEMPORARY_DIR, rel)
    base = os.path.basename(exp_dir.rstrip(os.sep)) or "run"
    return os.path.join(common.TEMPORARY_DIR, "batch_resume", base)


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
        return done
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
            if report['short']["performance"]["cagr"] > 0.3 :
                report['long'] = simulation.main( logger, para=s_para, pre_para=pre_para, train_cfg=t_cfg, prep_output_dir=prep_dir,
                                    train_output_dir=train_output_dir, device="cpu", period='long' )["statistics"][1]
                if report['long']["performance"]["cagr"] > 0.1 :
                    report['pass'] = True
                    report['forward'] = simulation.main( logger, para=s_para, pre_para=pre_para, train_cfg=t_cfg, prep_output_dir=prep_dir,
                                        train_output_dir=train_output_dir, device="cpu", period='forward' )["statistics"][1]
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
def _drain_sim_results(sim_result_queue: mp.Queue, stats: Dict[str, Any], logger: logging.Logger, eta_msg) -> None:
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
        if report_stat is not None:
            common.append_jsonl(rp, report_stat)

        logger.info(f"    Sim {pre_h}/{tr_h}/{sim_h} done in {elapsed:.2f}s")
        em = eta_msg()
        if em:
            logger.info(f"    {em}")


def _run_train_and_dispatch_sim(
    pre_h: str,
    tr_h: str,
    t_cfg: Any,
    pre_para: Any,
    prep_output_dir: str,
    save_dir: str,
    tr_node: Dict[str, Any],
    sim_task_queue: mp.Queue,
    stats: Dict[str, Any],
    n_train: int,
    sim_nones_sent: bool,
    logger: logging.Logger,
) -> bool:
    """
    Run a train task in main process, then enqueue sim tasks for workers.
    """
    from trade.bt import simulation
    import model.train_2head as train

    t0 = time.time()
    train.main(logger, train_cfg=t_cfg, pre_para=pre_para, prep_output_dir=prep_output_dir, save_dir=save_dir,experiment=True)
    el = time.time() - t0

    stats["train"]["time"] += el
    stats["train"]["count"] += 1
    logger.info(f"  {pre_h}/{tr_h}  Train done in {el:.2f}s")

    for sim in tr_node.get("sim_tasks", []):
            sim_task_queue.put((pre_h,pre_para, tr_h, t_cfg, sim))
    # when all train tasks finished, we can stop sim workers once queue drained
    if stats["train"]["count"] >= n_train and not sim_nones_sent:
        for _ in range(MAX_SIM):
            sim_task_queue.put(None)
        return True

    return sim_nones_sent


def _send_none_to_workers(q: mp.Queue, n: int) -> None:
    for _ in range(n):
        q.put(None)


# -----------------------------------------------------------------------------
# Reporting: compare old/new (valid mode)
# -----------------------------------------------------------------------------
def compare_old_new_reports(old_reports_path: str, new_reports_path: str, output_dir: str, logger: logging.Logger):
    """
    Compare selected_configs.jsonl (old) vs reports.jsonl (new), by params.hash.
    """
    logger.info("\n" + "=" * 40)
    logger.info("📊 Comparing old and new results...")

    old_reports: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(old_reports_path):
        for record in load_selected_configs(old_reports_path):
            params = record.get("params", {}) or {}
            h = params.get("hash")
            if isinstance(h, str) and h:
                old_reports[h] = {"params": params, "performance": record.get("performance", {}) or {}}
        logger.info(f"📥 Loaded {len(old_reports)} old reports from {old_reports_path}")
    else:
        logger.warning(f"⚠️  Old reports file not found: {old_reports_path}")

    new_reports: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(new_reports_path):
        with open(new_reports_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                params = record.get("params", {}) or {}
                h = params.get("hash")
                if isinstance(h, str) and h:
                    new_reports[h] = {"params": params, "performance": record.get("performance", {}) or {}}
        logger.info(f"📥 Loaded {len(new_reports)} new reports from {new_reports_path}")
    else:
        logger.warning(f"⚠️  New reports file not found: {new_reports_path}")

    compare_reports = []
    only_in_old = []
    only_in_new = []

    for h, old_r in old_reports.items():
        if h in new_reports:
            compare_reports.append({
                "params": old_r["params"],
                "performance": old_r["performance"],
                "performance_r": new_reports[h]["performance"],
            })
        else:
            only_in_old.append(h)

    for h in new_reports:
        if h not in old_reports:
            only_in_new.append(h)

    if not compare_reports:
        logger.warning("⚠️  No matching configurations found for comparison")
        return None, 0, len(only_in_old), len(only_in_new)

    compare_file_path = os.path.join(output_dir, "compare_reports.jsonl")
    with open(compare_file_path, "w", encoding="utf-8") as f:
        for report in compare_reports:
            f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")

    logger.info(f"📄 Compare reports saved: {compare_file_path} ({len(compare_reports)} matched)")
    if only_in_old:
        logger.info(f"ℹ️  {len(only_in_old)} only in old (not rerun)")
    if only_in_new:
        logger.info(f"ℹ️  {len(only_in_new)} only in new")

    return compare_file_path, len(compare_reports), len(only_in_old), len(only_in_new)


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

def main():
    parser = argparse.ArgumentParser(description="Batch experiments: prep -> train -> sim (with resume)")
    parser.add_argument("-p", "--prep", action="store_true", help="Execute data preparation stage")
    parser.add_argument("-t", "--train", action="store_true", help="Execute model training stage")
    parser.add_argument("-s", "--sim", action="store_true", help="Execute backtest simulation stage")
    parser.add_argument("-a", "--all", choices=["fast", "full", "all"], default="all",  help="Two-stage training: fast / full / all")
    parser.add_argument("-v", "--valid", action="store_true", default=False, help="Rerun selected_configs.jsonl then compare")
    parser.add_argument("-r", "--resume", type=str, help="Resume experiment from specified directory name under PERSISTENCE_DIR")
    parser.add_argument("-l", "--load", type=str, help="load condidate configs for verification,befor applying to market")

    args = parser.parse_args()
    run_all = args.all

    # ---------------- resolve exp_dir ----------------
    if args.resume:
        exp_dir = os.path.join(common.PERSISTENCE_DIR, args.resume)
        if not os.path.exists(exp_dir):
            print(f"❌ Error: Resume directory not found: {exp_dir}")
            return
    elif args.valid:
        selected_configs = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs", SELECTED_FILE)
        exp_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs")
        os.makedirs(exp_dir, exist_ok=True)
        if not os.path.exists(selected_configs):
            print(f"❌ Error: valid file not found: {selected_configs}")
            return
    elif args.load:
        load_file = os.path.join(common.PERSISTENCE_DIR, args.load)
        records = common.load_selected_configs(load_file)  # just to validate file and format
        from trade.bt import simulation
        import model.train_2head as train
        exp_dir = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "load_configs")
        os.makedirs(exp_dir, exist_ok=True)
        logger = _setup_root_logger(exp_dir)
        common.get_git_info(logger)
        begin_time = time.time()
        report_list = []
        for r in records:
            report = {'short':{}, 'long':{}}
            params = r["short"] if "short" in r else r
            sim_para =simulation.StrategyPara(**params["params"]["strategy"])
            pre_para =common.BaseDefine(**params["params"]["common"])
            train_para=_config_from_dict_train(params["params"]["train"])
            
            # preparation.main(logger, para=pre_para)
            # train_para.use_cache = False
            # train.main(logger, train_cfg=train_para, pre_para=pre_para,experiment=True)
            sim_para.max_daily_loss_pct = 0.03
            sim_para.trade_risk = 0.3
            report['short'] = simulation.main(logger, pre_para=pre_para, para=sim_para, train_cfg=train_para, period='short')["statistics"][1]
            report['long'] = simulation.main(logger, pre_para=pre_para, para=sim_para, train_cfg=train_para, period='long')["statistics"][1]
            sim_para.trade_risk = 0.4
            report['short'] = simulation.main(logger, pre_para=pre_para, para=sim_para, train_cfg=train_para, period='short')["statistics"][1]
            report['long'] = simulation.main(logger, pre_para=pre_para, para=sim_para, train_cfg=train_para, period='long')["statistics"][1]
            sim_para.trade_risk = 0.5
            report['short'] = simulation.main(logger, pre_para=pre_para, para=sim_para, train_cfg=train_para, period='short')["statistics"][1]
            report['long'] = simulation.main(logger, pre_para=pre_para, para=sim_para, train_cfg=train_para, period='long')["statistics"][1]
            sim_para.trade_risk = 0.6
            report['short'] = simulation.main(logger, pre_para=pre_para, para=sim_para, train_cfg=train_para, period='short')["statistics"][1]
            report['long'] = simulation.main(logger, pre_para=pre_para, para=sim_para, train_cfg=train_para, period='long')["statistics"][1]
            report_list.append(report)
        output_path = os.path.join(exp_dir, "loaded_reports.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for report in report_list:
                f.write(json.dumps(report, ensure_ascii=False, default=str) + "\n")
        logger.info(f"✅ Completed in {time.time() - begin_time:.2f}s")
        exit(0)
    else:
        exp_dir = common.create_experiment_dir(
            os.path.join(common.PERSISTENCE_DIR, "batch_experiments"),
            common.BaseDefine.symbol,
            common.BaseDefine.interval,
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

    elif args.valid:
        selected_configs = os.path.join(common.PERSISTENCE_DIR, "batch_experiments", "selected_configs", SELECTED_FILE)
        task_spec = _load_task_from_configs(selected_configs)
        n_prep, n_train, n_sim = _count_spec_tasks(task_spec)
        logger.info(f"📥 Loaded from {selected_configs}")
        logger.info(f"📊 Pending: prep={n_prep}, train={n_train}, sim={n_sim}")
    else:
        import model.train_2head as train
        from trade.bt import simulation

        preparation_task: List[Any] = []
        if args.prep or run_all:
            for cn in [120]: #[96,120]
                for pn in [16]: #[10,12,14,16,18]
                    for vol_multiplier in [2]:
                        item = common.BaseDefine()
                        item.candlestick_num = cn
                        item.predict_num = pn
                        item.vol_multiplier_long = vol_multiplier
                        item.vol_multiplier_short = vol_multiplier
                        preparation_task.append(item)
        else:
            preparation_task.append(common.BaseDefine())

        training_task: List[train.TrainConfig] = []
        if args.train or run_all:
            # for flip_penalty in np.arange(0.5, 2.5, 0.1).round(1):
            #     for miss_penalty in np.arange(0.2, 2.5, 0.1).round(1):
            for flip_penalty in np.arange(0.2, 2.1, 0.1).round(1):
                for miss_penalty in np.arange(0.2, 2, 0.1).round(1):
                    t_cfg = train.TrainConfig(use_cache = False)
                    t_cfg.flip_penalty = float(flip_penalty)
                    t_cfg.miss_penalty = float(miss_penalty)
                    training_task.append(t_cfg)
        else:
            training_task.append(train.TrainConfig())

        simulation_task: List[Any] = []
        if args.sim or run_all:
            for holdbar in [20, 24 ,28, 32,36]:
                s_cfg = simulation.StrategyPara()
                s_cfg.holdbar = holdbar
                s_cfg.atr_sl_mult_long = 100    #for model test
                s_cfg.atr_sl_mult_short = 100
                s_cfg.max_daily_loss_pct = 0.9
                simulation_task.append(s_cfg)
        else:
            simulation_task.append(simulation.StrategyPara())
        task_spec = build_task_spec(preparation_task, training_task, simulation_task)
        # task_spec 已经 ready
        sweep = collect_param_sweep(task_spec)
        log_param_sweep(logger, sweep)

        n_prep, n_train, n_sim = _count_spec_tasks(task_spec)
        logger.info(f"📊 Pending: prep={n_prep}, train={n_train}, sim={n_sim} (new run)")

    if not task_spec:
        logger.info("✅ No pending tasks.")
        return

    logger.info(f"🚀 Pipeline: MAX_PREP={MAX_PREP}, train={MAX_TRAIN}, MAX_SIM={MAX_SIM}")

    # ---------------- queues & workers ----------------
    prep_task_queue: mp.Queue = mp.Queue()
    train_task_queue: mp.Queue = mp.Queue()
    train_result_queue: mp.Queue = mp.Queue()
    sim_task_queue: mp.Queue = mp.Queue()
    sim_result_queue: mp.Queue = mp.Queue()

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
):
    stats = {"preparation": {"time": 0.0, "count": 0}, "train": {"time": 0.0, "count": 0}, "simulation": {"time": 0.0, "count": 0}}
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
            worker_log = os.path.join(exp_dir, f"train_{train_idx}.log")
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
            _drain_sim_results(sim_result_queue, stats, logger, eta_msg)

        # final drains
        _drain_train_results()
        _drain_sim_results(sim_result_queue, stats, logger, eta_msg)
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
            node_tr["sim_tasks"].append({"hash": sim_h, "params": json_safe(sim_conf)})

    return task_spec


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
