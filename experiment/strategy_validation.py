#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import asdict
from queue import Empty
from typing import Any, Dict, List, Tuple

current_work_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_work_dir, ".."))

from data_process import common, preparation
from data_process.utils import TaskIdentity, config_from_dict_train, load_selected_configs
from model import train_config
from trade.runner.config import ExperimentContext

SELECTED_FILE = "selected_configs.jsonl"

CROSS_TEST_SYMBOLS = ("DOGEUSDT", "ETHUSDT", "BTCUSDT")
CROSS_TEST_INTERVALS = ("15m", "30m", "1h")

try:
    experiment_tasks = importlib.import_module("experiment.task_constructors")
except ModuleNotFoundError as exc:
    if exc.name != "experiment.task_constructors":
        raise
    experiment_tasks = importlib.import_module("experiment.task_constructors_example")

MAX_PREP = experiment_tasks.MAX_PREP
MAX_TRAIN = experiment_tasks.MAX_TRAIN
MAX_SIM = experiment_tasks.MAX_SIM
INFERENCE_BATCH_SIZE = experiment_tasks.INFERENCE_BATCH_SIZE


class CrossTestError(RuntimeError):
    pass


def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    formatter = logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s %(message)s")

    file_handler = logging.FileHandler(
        os.path.join(output_dir, "cross_test.log"),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logger = logging.getLogger("cross_test")
    logger.setLevel(logging.INFO)
    return logger


def worker_logger(log_file: str) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    handler = logging.FileHandler(
        log_file,
        mode="a",
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(processName)s] %(levelname)s %(message)s"))
    root.addHandler(handler)
    return root


def cross_test_targets(
    original_symbol: str,
    original_interval: str,
) -> List[Tuple[str, str]]:
    targets = [(symbol, original_interval) for symbol in CROSS_TEST_SYMBOLS if symbol != original_symbol]
    targets.extend((original_symbol, interval) for interval in CROSS_TEST_INTERVALS if interval != original_interval)
    return list(dict.fromkeys(targets))


def target_key(symbol: str, interval: str) -> str:
    return f"{symbol}_{interval}"


def prep_output_dir(
    output_dir: str,
    strategy_hash: str,
    symbol: str,
    interval: str,
) -> str:
    return os.path.join(
        output_dir,
        "prep",
        strategy_hash,
        target_key(symbol, interval),
    )


def retrain_output_dir(
    output_dir: str,
    strategy_hash: str,
    symbol: str,
    interval: str,
) -> str:
    return os.path.join(
        output_dir,
        "retrained_models",
        strategy_hash,
        target_key(symbol, interval),
    )


def backtest_output_dir(
    output_dir: str,
    mode: str,
    strategy_hash: str,
    symbol: str,
    interval: str,
) -> str:
    return os.path.join(
        output_dir,
        "backtests",
        mode,
        strategy_hash,
        target_key(symbol, interval),
    )


def validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    params = common.recursive_get(record, "params")
    if not params:
        raise ValueError("Selected config record has no params")

    required = ("hash", "common", "train", "strategy", "broker")
    missing = [field for field in required if common.recursive_get(params, field) is None]
    if missing:
        raise ValueError(f"Selected config record is missing required params: {missing}")
    return params


def validate_preparation(
    pre_para: common.BaseDefine,
    output_dir: str,
) -> None:
    expected_hash = TaskIdentity.prep_hash_for(asdict(pre_para))
    manifest = common.load_data_manifest_from_dir(output_dir)

    actual_hash = manifest.get("configuration_hash")
    if actual_hash != expected_hash:
        raise RuntimeError("Preparation configuration mismatch: " f"path={output_dir}, expected={expected_hash}, actual={actual_hash}")

    stored_pre_para = common.load_pre_params_from_dir(output_dir)
    stored_hash = TaskIdentity.prep_hash_for(asdict(stored_pre_para))
    if stored_hash != expected_hash:
        raise RuntimeError("Preparation metadata mismatch: " f"path={output_dir}, expected={expected_hash}, actual={stored_hash}")

    source_path = common.market_data_path(pre_para)
    expected_source = {
        "filename": os.path.basename(source_path),
        "size_bytes": os.path.getsize(source_path),
        "sha256": common.sha256_file(source_path),
    }
    actual_source = manifest.get("source") or {}

    mismatches = {key: (expected_value, actual_source.get(key)) for key, expected_value in expected_source.items() if actual_source.get(key) != expected_value}
    if mismatches:
        raise RuntimeError("Preparation source mismatch: " f"path={output_dir}, mismatches={mismatches}")

    required_paths = (
        common.get_train_data_path_in_dir(output_dir),
        common.get_test_data_path_in_dir(output_dir),
    )
    missing_paths = [path for path in required_paths if not os.path.isfile(path)]
    if missing_paths:
        raise RuntimeError(f"Preparation is incomplete: missing={missing_paths}")


def build_jobs(
    records: List[Dict[str, Any]],
    output_dir: str,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    for record in records:
        params = validate_record(record)
        strategy_hash = params["hash"]
        original_pre_para = common.BaseDefine(**params["common"])

        for symbol, interval in cross_test_targets(
            original_pre_para.symbol,
            original_pre_para.interval,
        ):
            target_pre_para = copy.deepcopy(original_pre_para)
            target_pre_para.symbol = symbol
            target_pre_para.interval = interval

            jobs.append(
                {
                    "strategy_hash": strategy_hash,
                    "params": copy.deepcopy(params),
                    "target_symbol": symbol,
                    "target_interval": interval,
                    "target_pre_params": asdict(target_pre_para),
                    "prep_dir": prep_output_dir(
                        output_dir,
                        strategy_hash,
                        symbol,
                        interval,
                    ),
                    "retrain_dir": retrain_output_dir(
                        output_dir,
                        strategy_hash,
                        symbol,
                        interval,
                    ),
                }
            )

    return jobs


def prep_worker(
    worker_log_file: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    logger = worker_logger(worker_log_file)

    while True:
        try:
            job = task_queue.get(timeout=0.5)
        except Empty:
            continue

        if job is None:
            return

        t0 = time.time()
        try:
            pre_para = common.BaseDefine(**job["target_pre_params"])
            prep_dir = job["prep_dir"]

            manifest_path = common.get_data_manifest_path_in_dir(prep_dir)
            if not os.path.isfile(manifest_path):
                preparation.main(
                    logger,
                    para=pre_para,
                    prep_output_dir=prep_dir,
                )

            validate_preparation(pre_para, prep_dir)

            result_queue.put(
                (
                    "prep_done",
                    job,
                    time.time() - t0,
                )
            )
        except Exception:
            result_queue.put(
                (
                    "prep_failed",
                    job,
                    time.time() - t0,
                    traceback.format_exc(),
                )
            )
            return


def original_sim_worker(
    worker_log_file: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: str,
    experiment_context: ExperimentContext,
) -> None:
    logger = worker_logger(worker_log_file)

    from trade.runner import backtest_runner

    while True:
        try:
            job = task_queue.get(timeout=0.5)
        except Empty:
            continue

        if job is None:
            return

        t0 = time.time()
        try:
            params = job["params"]
            strategy_hash = job["strategy_hash"]
            symbol = job["target_symbol"]
            interval = job["target_interval"]

            train_cfg = config_from_dict_train(params["train"])
            strategy_config = backtest_runner.strategy_config_from_dict(params["strategy"])
            broker_config = backtest_runner.BrokerConfig(**params["broker"])

            original_train_dir = os.path.join(
                common.PERSISTENCE_DIR,
                "batch_experiments",
                "valid_train_out",
                strategy_hash,
            )
            if not os.path.isdir(original_train_dir):
                raise FileNotFoundError(f"Original training artifact is missing: {original_train_dir}")

            data_config = backtest_runner.ModelDataConfig(
                prep_output_dir=job["prep_dir"],
                train_output_dir=original_train_dir,
                device="cpu",
                use_prediction_cache=True,
            )

            backtest_runner.precompute_prediction_cache(
                logger,
                data_config,
                train_cfg,
                "long",
                inference_batch_size=INFERENCE_BATCH_SIZE,
            )

            runner_config = backtest_runner.RunnerConfig(
                strategy_config=strategy_config,
                broker_config=broker_config,
                save_dir=backtest_output_dir(
                    output_dir,
                    "original_model",
                    strategy_hash,
                    symbol,
                    interval,
                ),
                data_config=data_config,
                experiment_context=experiment_context,
            )

            report = backtest_runner.main(
                logger,
                runner_config,
                "long",
            )["report"]

            validate_report_market(
                report,
                symbol,
                interval,
            )

            result_queue.put(
                (
                    "original_done",
                    job,
                    time.time() - t0,
                    report,
                )
            )
        except Exception:
            result_queue.put(
                (
                    "original_failed",
                    job,
                    time.time() - t0,
                    traceback.format_exc(),
                )
            )
            return


def retrain_worker(
    worker_log_file: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    logger = worker_logger(worker_log_file)

    import model.train as train

    while True:
        try:
            job = task_queue.get(timeout=0.5)
        except Empty:
            continue

        if job is None:
            return

        t0 = time.time()
        try:
            train_cfg = config_from_dict_train(job["params"]["train"])
            save_dir = job["retrain_dir"]
            os.makedirs(save_dir, exist_ok=True)

            train_metrics = train.train(
                logger=logger,
                config=train_cfg,
                prep_output_dir=job["prep_dir"],
                save_dir=save_dir,
            )

            result_queue.put(
                (
                    "retrain_done",
                    job,
                    time.time() - t0,
                    train_metrics,
                )
            )
        except Exception:
            result_queue.put(
                (
                    "retrain_failed",
                    job,
                    time.time() - t0,
                    traceback.format_exc(),
                )
            )
            return


def retrained_sim_worker(
    worker_log_file: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    output_dir: str,
    experiment_context: ExperimentContext,
) -> None:
    logger = worker_logger(worker_log_file)

    from trade.runner import backtest_runner

    while True:
        try:
            payload = task_queue.get(timeout=0.5)
        except Empty:
            continue

        if payload is None:
            return

        job, train_metrics = payload
        t0 = time.time()

        try:
            params = job["params"]
            strategy_hash = job["strategy_hash"]
            symbol = job["target_symbol"]
            interval = job["target_interval"]

            train_cfg = config_from_dict_train(params["train"])
            strategy_config = backtest_runner.strategy_config_from_dict(params["strategy"])
            broker_config = backtest_runner.BrokerConfig(**params["broker"])

            data_config = backtest_runner.ModelDataConfig(
                prep_output_dir=job["prep_dir"],
                train_output_dir=job["retrain_dir"],
                device="cpu",
                use_prediction_cache=True,
            )

            backtest_runner.precompute_prediction_cache(
                logger,
                data_config,
                train_cfg,
                "long",
                inference_batch_size=INFERENCE_BATCH_SIZE,
            )

            runner_config = backtest_runner.RunnerConfig(
                strategy_config=strategy_config,
                broker_config=broker_config,
                save_dir=backtest_output_dir(
                    output_dir,
                    "retrained_model",
                    strategy_hash,
                    symbol,
                    interval,
                ),
                data_config=data_config,
                experiment_context=experiment_context,
            )

            report = backtest_runner.main(
                logger,
                runner_config,
                "long",
            )["report"]

            validate_report_market(
                report,
                symbol,
                interval,
            )

            result_queue.put(
                (
                    "retrained_done",
                    job,
                    time.time() - t0,
                    train_metrics,
                    report,
                )
            )
        except Exception:
            result_queue.put(
                (
                    "retrained_failed",
                    job,
                    time.time() - t0,
                    traceback.format_exc(),
                )
            )
            return


def validate_report_market(
    report: Dict[str, Any],
    expected_symbol: str,
    expected_interval: str,
) -> None:
    report_common = report["params"]["common"]

    if report_common.get("symbol") != expected_symbol or report_common.get("interval") != expected_interval:
        raise RuntimeError("Cross-test report market mismatch: " f"expected={expected_symbol}_{expected_interval}, " f"report_common={report_common}")


def send_none(queue: mp.Queue, count: int) -> None:
    for _ in range(count):
        queue.put(None)


def ensure_workers_alive(
    stage_name: str,
    workers: List[mp.Process],
) -> None:
    for process in workers:
        if process.exitcode not in (None, 0):
            raise CrossTestError(f"{stage_name} worker {process.name} exited " f"with code {process.exitcode}")


def terminate_workers(workers: List[mp.Process]) -> None:
    for process in workers:
        if process.is_alive():
            process.terminate()

    for process in workers:
        process.join(timeout=5)


def job_id(job: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        job["strategy_hash"],
        job["target_symbol"],
        job["target_interval"],
    )


def run_parallel_pipeline(
    logger: logging.Logger,
    jobs: List[Dict[str, Any]],
    output_dir: str,
    experiment_context: ExperimentContext,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    manager = mp.Manager()

    prep_queue = manager.Queue()
    prep_result_queue = manager.Queue()

    original_sim_queue = manager.Queue()
    original_result_queue = manager.Queue()

    retrain_queue = manager.Queue()
    retrain_result_queue = manager.Queue()

    retrained_sim_queue = manager.Queue()
    retrained_result_queue = manager.Queue()

    prep_workers: List[mp.Process] = []
    original_sim_workers: List[mp.Process] = []
    retrain_workers: List[mp.Process] = []
    retrained_sim_workers: List[mp.Process] = []

    for index in range(MAX_PREP):
        process = mp.Process(
            target=prep_worker,
            name=f"CrossPrep-{index}",
            args=(
                os.path.join(output_dir, f"prep_{index}.log"),
                prep_queue,
                prep_result_queue,
            ),
        )
        process.start()
        prep_workers.append(process)

    for index in range(MAX_SIM):
        process = mp.Process(
            target=original_sim_worker,
            name=f"CrossOriginalSim-{index}",
            args=(
                os.path.join(output_dir, f"original_sim_{index}.log"),
                original_sim_queue,
                original_result_queue,
                output_dir,
                experiment_context,
            ),
        )
        process.start()
        original_sim_workers.append(process)

    for index in range(MAX_TRAIN):
        process = mp.Process(
            target=retrain_worker,
            name=f"CrossRetrain-{index}",
            args=(
                os.path.join(output_dir, f"retrain_{index}.log"),
                retrain_queue,
                retrain_result_queue,
            ),
        )
        process.start()
        retrain_workers.append(process)

    for index in range(MAX_SIM):
        process = mp.Process(
            target=retrained_sim_worker,
            name=f"CrossRetrainedSim-{index}",
            args=(
                os.path.join(output_dir, f"retrained_sim_{index}.log"),
                retrained_sim_queue,
                retrained_result_queue,
                output_dir,
                experiment_context,
            ),
        )
        process.start()
        retrained_sim_workers.append(process)

    all_workers = prep_workers + original_sim_workers + retrain_workers + retrained_sim_workers

    results: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    total_jobs = len(jobs)

    prep_done = 0
    original_done = 0
    retrain_done = 0
    retrained_done = 0

    for job in jobs:
        results[job_id(job)] = {
            "job": job,
            "original_model": None,
            "retrained_model": None,
        }
        prep_queue.put(job)

    try:
        while retrained_done < total_jobs or original_done < total_jobs:
            ensure_workers_alive("prep", prep_workers)
            ensure_workers_alive(
                "original simulation",
                original_sim_workers,
            )
            ensure_workers_alive("retrain", retrain_workers)
            ensure_workers_alive(
                "retrained simulation",
                retrained_sim_workers,
            )

            try:
                message = prep_result_queue.get(timeout=0.1)
            except Empty:
                message = None

            if message is not None:
                kind, job, elapsed, *payload = message

                if kind == "prep_failed":
                    raise CrossTestError("Preparation failed for " f"{job_id(job)} after {elapsed:.2f}s:\n" f"{payload[0]}")

                if kind != "prep_done":
                    raise CrossTestError(f"Unexpected prep result: {message!r}")

                prep_done += 1
                logger.info(
                    "Prep %d/%d done: %s in %.2fs",
                    prep_done,
                    total_jobs,
                    job_id(job),
                    elapsed,
                )

                original_sim_queue.put(job)
                retrain_queue.put(job)

            while True:
                try:
                    message = original_result_queue.get_nowait()
                except Empty:
                    break

                kind, job, elapsed, *payload = message
                if kind == "original_failed":
                    raise CrossTestError("Original-model simulation failed for " f"{job_id(job)} after {elapsed:.2f}s:\n" f"{payload[0]}")

                if kind != "original_done":
                    raise CrossTestError(f"Unexpected original-model result: {message!r}")

                report = payload[0]
                results[job_id(job)]["original_model"] = {
                    "cagr": report["results"]["long"]["performance"]["cagr"],
                    "report": report,
                }
                original_done += 1

                logger.info(
                    "Original model %d/%d done: %s in %.2fs",
                    original_done,
                    total_jobs,
                    job_id(job),
                    elapsed,
                )

            while True:
                try:
                    message = retrain_result_queue.get_nowait()
                except Empty:
                    break

                kind, job, elapsed, *payload = message
                if kind == "retrain_failed":
                    raise CrossTestError("Retraining failed for " f"{job_id(job)} after {elapsed:.2f}s:\n" f"{payload[0]}")

                if kind != "retrain_done":
                    raise CrossTestError(f"Unexpected retrain result: {message!r}")

                train_metrics = payload[0]
                retrain_done += 1

                logger.info(
                    "Retrain %d/%d done: %s in %.2fs",
                    retrain_done,
                    total_jobs,
                    job_id(job),
                    elapsed,
                )

                retrained_sim_queue.put((job, train_metrics))

            while True:
                try:
                    message = retrained_result_queue.get_nowait()
                except Empty:
                    break

                kind, job, elapsed, *payload = message
                if kind == "retrained_failed":
                    raise CrossTestError("Retrained-model simulation failed for " f"{job_id(job)} after {elapsed:.2f}s:\n" f"{payload[0]}")

                if kind != "retrained_done":
                    raise CrossTestError(f"Unexpected retrained-model result: {message!r}")

                train_metrics, report = payload
                results[job_id(job)]["retrained_model"] = {
                    "cagr": report["results"]["long"]["performance"]["cagr"],
                    "train_metrics": train_metrics,
                    "report": report,
                }
                retrained_done += 1

                logger.info(
                    "Retrained model %d/%d done: %s in %.2fs",
                    retrained_done,
                    total_jobs,
                    job_id(job),
                    elapsed,
                )

        return results

    finally:
        send_none(prep_queue, MAX_PREP)
        send_none(original_sim_queue, MAX_SIM)
        send_none(retrain_queue, MAX_TRAIN)
        send_none(retrained_sim_queue, MAX_SIM)
        terminate_workers(all_workers)


def aggregate_results(
    records: List[Dict[str, Any]],
    flat_results: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aggregated: List[Dict[str, Any]] = []

    for record in records:
        params = validate_record(record)
        strategy_hash = params["hash"]
        original_common = params["common"]

        strategy_result: Dict[str, Any] = {
            "strategy_hash": strategy_hash,
            "original_symbol": original_common["symbol"],
            "original_interval": original_common["interval"],
            "targets": {},
        }

        for (
            result_strategy_hash,
            symbol,
            interval,
        ), result in flat_results.items():
            if result_strategy_hash != strategy_hash:
                continue

            key = target_key(symbol, interval)
            strategy_result["targets"][key] = {
                "symbol": symbol,
                "interval": interval,
                "original_model": result["original_model"],
                "retrained_model": result["retrained_model"],
            }

        aggregated.append(strategy_result)

    return aggregated


def write_jsonl(
    path: str,
    records: List[Dict[str, Any]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel cross-test runner. For every selected strategy and "
            "target market, test both the original trained model and a "
            "model retrained on the target market with the same TrainConfig."
        )
    )
    parser.add_argument(
        "--selected-configs",
        default=os.path.join(
            common.PERSISTENCE_DIR,
            "batch_experiments",
            "selected_configs",
            SELECTED_FILE,
        ),
        help="Path to selected_configs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory",
    )
    parser.add_argument(
        "--check-git-clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a clean Git working tree",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.selected_configs):
        raise FileNotFoundError(f"Selected config file not found: {args.selected_configs}")

    records = load_selected_configs(args.selected_configs)
    if not records:
        raise RuntimeError(f"No selected configs found: {args.selected_configs}")

    first_params = validate_record(records[0])
    first_common = common.BaseDefine(**first_params["common"])

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = common.create_experiment_dir(
            os.path.join(
                common.PERSISTENCE_DIR,
                "batch_experiments",
                "cross_test",
            ),
            first_common.symbol,
            first_common.interval,
        )

    logger = setup_logger(output_dir)

    experiment_context = ExperimentContext(git_commit=common.git_revision(require_clean=args.check_git_clean))

    jobs = build_jobs(records, output_dir)

    logger.info(
        "Cross-test jobs=%d | MAX_PREP=%d | MAX_TRAIN=%d | MAX_SIM=%d",
        len(jobs),
        MAX_PREP,
        MAX_TRAIN,
        MAX_SIM,
    )
    logger.info("Output directory: %s", output_dir)
    logger.info("Git commit: %s", experiment_context.git_commit)

    begin_time = time.time()

    flat_results = run_parallel_pipeline(
        logger=logger,
        jobs=jobs,
        output_dir=output_dir,
        experiment_context=experiment_context,
    )

    aggregated = aggregate_results(
        records,
        flat_results,
    )

    output_path = os.path.join(
        output_dir,
        "cross_test_reports.jsonl",
    )
    write_jsonl(output_path, aggregated)

    current_commit = common.git_revision(require_clean=args.check_git_clean)
    if current_commit != experiment_context.git_commit:
        raise RuntimeError("Git state changed while cross test was running: " f"started={experiment_context.git_commit}, " f"current={current_commit}")

    logger.info(
        "Completed in %.2fs. Reports saved to %s",
        time.time() - begin_time,
        output_path,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
