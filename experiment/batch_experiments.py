import os, sys, time, logging, argparse, copy, json, hashlib
from datetime import datetime, timedelta
from dataclasses import asdict
from itertools import product
from queue import Empty
import numpy as np
import multiprocessing
from multiprocessing import Process

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
from data_process import preparation, common
from data_process.utils import json_safe, calc_params_hash, param_hash

# train / simulation 不在顶层 import；prep/sim 子进程不使用 CUDA，使用 fork 启动
TASKS_SPEC_FILE = "tasks_spec.json"
REPORTS_FILE = "reports.jsonl"


def build_task_spec(preparation_task, training_task, simulation_task):
    """构建与 tasks_spec.json 同构的 pre -> train -> sim_tasks 嵌套结构。"""
    spec = {}
    for i, pre in enumerate(preparation_task):
        pre_d = asdict(pre)
        pre_h = param_hash(pre_d)
        if pre_h not in spec:
            spec[pre_h] = {"params": json_safe(pre_d), "train": {}}
        for j, tr in enumerate(training_task):
            tr_d = asdict(tr)
            tr_h = param_hash(tr_d)
            if tr_h not in spec[pre_h]["train"]:
                spec[pre_h]["train"][tr_h] = {"params": json_safe(tr_d), "sim_tasks": []}
            for k, sim in enumerate(simulation_task):
                sim_d = asdict(sim)
                sim_h = param_hash(sim_d)
                spec[pre_h]["train"][tr_h]["sim_tasks"].append({"hash": sim_h, "params": json_safe(sim_d)})
    return spec


def load_done_set(reports_path):
    """从 reports.jsonl 中读取已完成的 params.hash。"""
    done = set()
    if not os.path.exists(reports_path):
        return done
    with open(reports_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("params") and "hash" in d["params"]:
                    done.add(d["params"]["hash"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


def _config_from_dict_train(train_params):
    """从 task_spec 的 train params 字典恢复 TrainConfig。"""
    import model.train_2head as train
    t_cfg = train.TrainConfig()
    for k, v in train_params.items():
        if isinstance(v, dict) and k in ("model_cfg", "data_cfg"):
            continue
        if hasattr(t_cfg, k):
            setattr(t_cfg, k, v)
    return t_cfg


def filter_pending_from_spec(task_spec, done_set):
    """
    从 task_spec（与 tasks_spec.json 同构）中过滤掉 reports.jsonl 已完成的部分。
    返回仅含待完成任务的 pending_task_spec，结构同 task_spec。
    """
    from trade.bt import simulation
    pending = {}
    for pre_h, pre_node in task_spec.items():
        pre_params = pre_node["params"]
        train_pending = {}
        for tr_h, tr_node in pre_node["train"].items():
            train_params = tr_node["params"]
            sim_pending = []
            for sim in tr_node["sim_tasks"]:
                task_hash = calc_params_hash(
                    strategy=simulation.StrategyPara(**sim["params"]),
                    common=common.CommonDefine(**pre_params),
                    train=_config_from_dict_train(train_params),
                )
                if task_hash not in done_set:
                    sim_pending.append(sim)
            if sim_pending:
                train_pending[tr_h] = {"params": train_params, "sim_tasks": sim_pending}
        if train_pending:
            pending[pre_h] = {"params": pre_params, "train": train_pending}
    return pending


def _batch_temp_dir(exp_dir):
    """与 exp_dir 对应的临时目录，用于 prep/train 产物，放在 TEMPORARY_DIR 下。"""
    if exp_dir.startswith(common.PERSISTENCE_DIR):
        rel = os.path.relpath(exp_dir, common.PERSISTENCE_DIR)
        return os.path.join(common.TEMPORARY_DIR, rel)
    return os.path.join(common.TEMPORARY_DIR, "batch_resume", os.path.basename(exp_dir.rstrip(os.sep)) or "run")


def spec_to_pipeline(pending_spec, exp_dir, temp_dir):
    """
    将 pending_task_spec（与 tasks_spec.json 同构）转换为流水线所需的平面结构。
    prep/train 产出目录使用 temp_dir；log/报告等仍用 exp_dir（PERSISTENCE_DIR）。
    返回: (prep_items, train_items, sim_items, train_by_prep, sim_by_train)
    """
    from trade.bt import simulation
    prep_items = []
    train_items = []
    sim_items = []
    train_by_prep = {}
    sim_by_train = {}

    for pre_h, pre_node in pending_spec.items():
        pre_para = common.CommonDefine(**pre_node["params"])
        pre_para.prep_output_dir = os.path.join(temp_dir, "preparation", f"pre_{pre_h}")
        prep_items.append((pre_h, pre_para))

        for tr_h, tr_node in pre_node["train"].items():
            t_cfg = _config_from_dict_train(tr_node["params"])
            t_cfg.save_dir = os.path.join(temp_dir, "training", f"pre_{pre_h}_train_{tr_h}")
            pre_para_cp = copy.deepcopy(pre_para)
            tr_item = (pre_h, tr_h, t_cfg, pre_para_cp)
            train_items.append(tr_item)
            train_by_prep.setdefault(pre_h, []).append(tr_item)

            for sim in tr_node["sim_tasks"]:
                sim_h, sim_params = sim["hash"], sim["params"]
                pre_para_cp2 = copy.deepcopy(pre_para)
                t_cfg_cp = copy.deepcopy(t_cfg)
                s_para = simulation.StrategyPara(**sim_params)
                sim_item = (pre_h, tr_h, sim_h, pre_para_cp2, t_cfg_cp, s_para)
                sim_items.append(sim_item)
                sim_by_train.setdefault((pre_h, tr_h), []).append(sim_item)

    return prep_items, train_items, sim_items, train_by_prep, sim_by_train


def load_pending_tasks(exp_dir, done_set):
    """
    从 tasks_spec.json 直接加载，过滤掉 reports.jsonl 已完成部分，得到待完成任务。
    返回: (prep_items, train_items, sim_items, train_by_prep, sim_by_train)
    """
    tasks_spec_path = os.path.join(exp_dir, TASKS_SPEC_FILE)
    if not os.path.exists(tasks_spec_path):
        raise FileNotFoundError(f"Tasks spec not found: {tasks_spec_path}")
    with open(tasks_spec_path, "r", encoding="utf-8") as f:
        task_spec = json.load(f)
    pending_spec = filter_pending_from_spec(task_spec, done_set)
    temp_dir = _batch_temp_dir(exp_dir)
    return spec_to_pipeline(pending_spec, exp_dir, temp_dir)


# ---------- 辅助函数 ----------
def _process_sim_results(sim_result_queue, stats, logger, eta_msg):
    """排空 sim_result_queue，处理 sim 子进程返回的结果。"""
    while True:
        try:
            msg = sim_result_queue.get_nowait()
        except Empty:
            break
        typ = msg[0]
        if typ == "sim_done":
            _, pre_h, tr_h, sim_h, elapsed, report_stat, rp = msg
            stats["simulation"]["time"] += elapsed
            stats["simulation"]["count"] += 1
            if report_stat is not None:
                common.append_jsonl(rp, report_stat)
            logger.info(f"    Sim {pre_h}/{tr_h}/{sim_h} done in {elapsed:.2f}s")
            em = eta_msg()
            if em:
                logger.info(f"    {em}")


def _run_train_and_dispatch_sim(pre_h, tr_h, t_cfg, pre_para, sim_by_train, sim_task_queue, 
                                  stats, n_train, max_sim, sim_nones_sent, logger):
    """执行 train 任务并投递对应的 sim 任务。返回更新后的 sim_nones_sent。"""
    t0 = time.time()
    import model.train_2head as train
    train.main(logger, train_cfg=t_cfg, pre_para=pre_para)
    el = time.time() - t0
    stats["train"]["time"] += el
    stats["train"]["count"] += 1
    logger.info(f"    Train {pre_h}/{tr_h} done in {el:.2f}s")
    for (_, _, sim_h, pre_para_sim, t_cfg_sim, s_para) in sim_by_train.get((pre_h, tr_h), []):
        sim_task_queue.put((pre_h, tr_h, sim_h, pre_para_sim, t_cfg_sim, s_para))
    if stats["train"]["count"] >= n_train and not sim_nones_sent:
        for _ in range(max_sim):
            sim_task_queue.put(None)
        return True
    return sim_nones_sent


def _send_none_to_workers(queue, count):
    """向队列发送 count 个 None，用于通知 workers 结束。"""
    for _ in range(count):
        queue.put(None)


def _create_output_dirs(prep_items, train_items, temp_dir):
    """创建 prep 和 train 的输出目录。"""
    for pre_h, _ in prep_items:
        os.makedirs(os.path.join(temp_dir, "preparation", f"pre_{pre_h}"), exist_ok=True)
    for pre_h, tr_h, t_cfg, _ in train_items:
        os.makedirs(t_cfg.save_dir, exist_ok=True)


# ---------- Worker 进程（各自写独立 log 文件） ----------
def _worker_logger(log_file):
    """子进程内配置 root logger，仅写当前进程的 log 文件。"""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    root.addHandler(fh)
    return root


def _worker_prep(worker_log_file, task_queue, train_queue, train_by_prep):
    _worker_logger(worker_log_file)
    while True:
        try:
            msg = task_queue.get(timeout=0.5)
            if msg is None:
                break
        except Empty:
            continue
        pre_h, p_dict = msg
        # prep_output_dir 已在 spec_to_pipeline 中设为 temp_dir 下路径，不再覆盖
        para = common.CommonDefine(**p_dict)
        t0 = time.time()
        preparation.main(logging.getLogger(), para=para)
        elapsed = time.time() - t0
        train_queue.put(("prep_done", pre_h, elapsed, train_by_prep.get(pre_h, [])))


def _worker_sim(worker_log_file, task_queue, result_queue, reports_path):
    from trade.bt import simulation
    _worker_logger(worker_log_file)
    while True:
        try:
            msg = task_queue.get(timeout=0.5)
            if msg is None:
                break
        except Empty:
            continue
        pre_h, tr_h, sim_h, pre_para, t_cfg, s_para = msg
        t0 = time.time()
        s_para.holdbar = pre_para.predict_num
        report = simulation.main(logging.getLogger(), para=s_para, pre_para=pre_para, train_cfg=t_cfg)
        elapsed = time.time() - t0
        result_queue.put(("sim_done", pre_h, tr_h, sim_h, elapsed, report["statistics"], reports_path))


def main():
    import model.train_2head as train
    from trade.bt import simulation

    parser = argparse.ArgumentParser(description="Quant Trading Pipeline Control")
    parser.add_argument('-p', '--prep', action='store_true', help='Execute data preparation stage')
    parser.add_argument('-t', '--train', action='store_true', help='Execute model training stage')
    parser.add_argument('-s', '--sim', action='store_true', help='Execute backtest simulation stage')
    parser.add_argument('-a', '--all', action='store_true', default=True, help='Execute all stages')
    parser.add_argument('-l', '--load', action='store_true', help='Load unfinished tasks from existing exp_dir')
    parser.add_argument('-r', '--resume', type=str, help='Resume experiment from specified directory')
    parser.add_argument('--max-prep', type=int, default=1, help='Max concurrent preparation tasks')
    parser.add_argument('--max-sim', type=int, default=1, help='Max concurrent simulation tasks')

    args = parser.parse_args()

    if args.resume:
        exp_dir = args.resume
        if not os.path.exists(exp_dir):
            print(f"❌ Error: Resume directory {exp_dir} not found.")
            return
    else:
        exp_dir = common.create_experiment_dir(
            os.path.join(common.PERSISTENCE_DIR, 'batch_experiments'),
            common.CommonDefine.symbol,
            common.CommonDefine.interval
        )

    log_file_path = os.path.join(exp_dir, 'experiment.log')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s"))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger("batch")
    logger.setLevel(logging.INFO)
    common.get_git_info(logger)

    run_all = args.all
    begin_time = time.time()
    stats = {"preparation": {"time": 0, "count": 0}, "train": {"time": 0, "count": 0}, "simulation": {"time": 0, "count": 0}}
    reports_path = os.path.join(exp_dir, REPORTS_FILE)
    temp_dir = _batch_temp_dir(exp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    if args.load:
        if not args.resume:
            logger.error("❌ --load requires --resume <exp_dir>")
            return
        done_set = load_done_set(reports_path)
        prep_items, train_items, sim_items, train_by_prep, sim_by_train = load_pending_tasks(exp_dir, done_set)
        logger.info(f"📥 Loaded pending from {exp_dir}; prep={len(prep_items)}, train={len(train_items)}, sim={len(sim_items)}")
    else:
        preparation_task = []
        if args.prep or run_all:
            for cn in [96]:
                for pn in range(10,31,1):
                    for vol_multiplier in [1.9]:
                        item = common.CommonDefine()
                        item.candlestick_num = cn
                        item.predict_num = pn
                        item.vol_multiplier_long = vol_multiplier
                        item.vol_multiplier_short = vol_multiplier
                        preparation_task.append(item)
        else:
            preparation_task.append(common.CommonDefine())

        training_task = []
        if args.train or run_all:
            for flip_penalty in np.arange(3, 3.5, 0.1).round(1):
                for miss_penalty in np.arange(3, 3.5, 0.1).round(1):
                    t_cfg = train.TrainConfig()
                    t_cfg.use_cache = True
                    t_cfg.flip_penalty = float(flip_penalty)
                    t_cfg.miss_penalty = float(miss_penalty)
                    training_task.append(t_cfg)
        else:
            training_task.append(train.TrainConfig())

        simulation_task = []
        if args.sim or run_all:
            s_cfg = simulation.StrategyPara(device='cpu')
            s_cfg.holdbar = 20
            simulation_task.append(s_cfg)
        else:
            simulation_task.append(simulation.StrategyPara(device='cpu'))

        task_spec = build_task_spec(preparation_task, training_task, simulation_task)
        tasks_spec_path = os.path.join(exp_dir, TASKS_SPEC_FILE)
        with open(tasks_spec_path, "w", encoding="utf-8") as f:
            json.dump(task_spec, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"📄 Tasks spec saved: {tasks_spec_path}")

        prep_items, train_items, sim_items, train_by_prep, sim_by_train = spec_to_pipeline(task_spec, exp_dir, temp_dir)
        logger.info(f"📊 Pending: prep={len(prep_items)}, train={len(train_items)}, sim={len(sim_items)}")

    if not prep_items and not train_items and not sim_items:
        logger.info("✅ No pending tasks.")
        return

    _create_output_dirs(prep_items, train_items, temp_dir)

    max_prep = max(1, args.max_prep)
    max_sim = max(1, args.max_sim)
    logger.info(f"🚀 Process+Queue pipeline: max_prep={max_prep}, train=main, max_sim={max_sim}")

    try:
        if hasattr(multiprocessing, "get_all_start_methods") and "fork" in multiprocessing.get_all_start_methods():
            multiprocessing.set_start_method("fork", force=True)  # 子进程不用 CUDA，fork 避免 spawn 重复导入
    except RuntimeError:
        pass

    prep_task_queue = multiprocessing.Queue()
    sim_task_queue = multiprocessing.Queue()
    train_done_queue = multiprocessing.Queue()
    sim_result_queue = multiprocessing.Queue()

    n_prep, n_train, n_sim = len(prep_items), len(train_items), len(sim_items)

    def eta_msg():
        def phase_eta(total, count, elapsed, max_w):
            if total == 0 or count >= total:
                return 0.0
            if count == 0:
                return None
            return (elapsed / count) * (total - count) / max(1, max_w)
        
        def format_hours(seconds):
            """将秒数转换为小时格式显示。"""
            if seconds is None:
                return "—"
            if seconds == 0:
                return "0h"
            hours = seconds / 3600
            if hours < 0.1:
                return f"{seconds:.0f}s"  # 小于 0.1 小时时仍显示秒
            return f"{hours:.2f}h"

        prep_eta = phase_eta(n_prep, stats["preparation"]["count"], stats["preparation"]["time"], max_prep)
        train_eta = phase_eta(n_train, stats["train"]["count"], stats["train"]["time"], 1)  # train 在主进程执行
        sim_eta = phase_eta(n_sim, stats["simulation"]["count"], stats["simulation"]["time"], max_sim)
        total_eta = None
        if (n_prep == 0 or stats["preparation"]["count"] > 0) and (n_train == 0 or stats["train"]["count"] > 0) and (n_sim == 0 or stats["simulation"]["count"] > 0):
            total_eta = (prep_eta or 0) + (train_eta or 0) + (sim_eta or 0)
        parts = []
        if n_prep > 0:
            parts.append(f"prep:{format_hours(prep_eta)}")
        if n_train > 0:
            parts.append(f"train:{format_hours(train_eta)}")
        if n_sim > 0:
            parts.append(f"sim:{format_hours(sim_eta)}")
        if parts:
            msg = "[ETA] " + ", ".join(parts)
            if total_eta is not None and total_eta > 0:
                msg += f" | total ~{format_hours(total_eta)}"
            return msg
        return ""

    prep_workers = []
    for i in range(max_prep):
        worker_log = os.path.join(exp_dir, f"prep_{i}.log")
        p = Process(target=_worker_prep, args=(worker_log, prep_task_queue, train_done_queue, train_by_prep))
        p.start()
        prep_workers.append(p)

    sim_workers = []
    for i in range(max_sim):
        worker_log = os.path.join(exp_dir, f"sim_{i}.log")
        p = Process(target=_worker_sim, args=(worker_log, sim_task_queue, sim_result_queue, reports_path))
        p.start()
        sim_workers.append(p)

    for (pre_h, pre_para) in prep_items:
        prep_task_queue.put((pre_h, asdict(pre_para)))
    _send_none_to_workers(prep_task_queue, max_prep)

    # train 在主进程执行；无 prep 时先跑完所有 train 并直接投递 sim 任务
    if n_prep == 0 and n_train > 0:
        for (pre_h, tr_h, t_cfg, pre_para) in train_items:
            sim_nones_sent = _run_train_and_dispatch_sim(
                pre_h, tr_h, t_cfg, pre_para, sim_by_train, sim_task_queue,
                stats, n_train, max_sim, sim_nones_sent, logger
            )
            _process_sim_results(sim_result_queue, stats, logger, eta_msg)
        # 所有 train 完成后，发送 None 给 sim workers
        _send_none_to_workers(sim_task_queue, max_sim)
        sim_nones_sent = True
    else:
        sim_nones_sent = n_train == 0

    if n_train == 0 and n_sim > 0:
        for (pre_h, tr_h, sim_h, pre_para, t_cfg, s_para) in sim_items:
            sim_task_queue.put((pre_h, tr_h, sim_h, pre_para, t_cfg, s_para))
        _send_none_to_workers(sim_task_queue, max_sim)
    sim_nones_sent = n_train == 0

    while stats["preparation"]["count"] < n_prep or stats["train"]["count"] < n_train or stats["simulation"]["count"] < n_sim:
        try:
            msg = train_done_queue.get(timeout=0.2)
        except Empty:
            msg = None
        if msg is not None:
            typ, pre_h, elapsed, next_trains = msg
            stats["preparation"]["time"] += elapsed
            stats["preparation"]["count"] += 1
            logger.info(f"    Prep {pre_h} done in {elapsed:.2f}s")
            for (_, tr_h, t_cfg, pre_para) in next_trains:
                sim_nones_sent = _run_train_and_dispatch_sim(
                    pre_h, tr_h, t_cfg, pre_para, sim_by_train, sim_task_queue,
                    stats, n_train, max_sim, sim_nones_sent, logger
                )
                # 每次执行完 train_task 后立即检查队列，处理 sim 子进程返回的结果
                _process_sim_results(sim_result_queue, stats, logger, eta_msg)

        # 也检查队列（处理可能在其他时机完成的 sim 任务）
        _process_sim_results(sim_result_queue, stats, logger, eta_msg)

    for p in prep_workers + sim_workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    logger.info("\n" + "=" * 40)
    logger.info(f"✅ All tasks completed in {time.time() - begin_time:.2f}s")
    logger.info("=" * 40)


if __name__ == "__main__":
    main()
