from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os, sys, logging

current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..','..'))
from data_process import common

from trade.runner import backtest_runner
from data_process import common
from model import train_config

app = FastAPI()
logger, _= common.setup_session_logger(sub_folder='backend',console_level= logging.INFO, file_level = logging.DEBUG)

if False:
    report_file = r'/home/chao/work/quant_output/batch_train/DOGEUSDT_30m/2026-06-28/19_15_16/batch_simulation/report_view/selected_configs.jsonl'
    report = common.load_reports(report_file)
    simulation_result = report[0]['raw'].get("simulation", report)
    forward = simulation_result.get("forward", report)
    sim_params = forward['params']['strategy']
    pre_params = forward['params']['common']
    fusion_dir = common.recursive_get(report, 'fusion_dir')
    prep_output_dir = common.recursive_get(report, 'prep_output_dir')
    strategy_config = backtest_runner.strategy_config_from_dict(
        forward["params"]["strategy"],
    )
    broker_config = backtest_runner.BrokerConfig(**forward["params"]["broker"])
    runner_config = backtest_runner.RunnerConfig(
        strategy_config=strategy_config,
        broker_config=broker_config,
        save_dir=common.TEMPORARY_DIR,
        data_config=backtest_runner.ModelDataConfig(
            atr_ref_bars=strategy_config.min_hold_bars,
            prep_output_dir=prep_output_dir,
            train_output_dir=fusion_dir,
            device="cpu",
            period="long",
        ),
    )
    result = backtest_runner.main(logger, runner_config)
else:
    train_output_dir = os.path.join(common.TRAIN_OUT_DIR, train_config.TrainTask.DIRECT_3CLASS)
    strategy_config = backtest_runner.MlStrategyConfig(
        min_hold_bars=48,
        atr_sl_long_mult=6,
        atr_sl_short_mult=6,
        atr_tp_mult=100,
        max_daily_loss_pct=0.04,
    )
    broker_config = backtest_runner.BrokerConfig()
    runner_config = backtest_runner.RunnerConfig(
        strategy_config=strategy_config,
        broker_config=broker_config,
        save_dir=common.TEMPORARY_DIR,
        data_config=backtest_runner.ModelDataConfig(
            atr_ref_bars=strategy_config.min_hold_bars,
            prep_output_dir=common.DATA_OUT_DIR,
            train_output_dir=train_output_dir,
            period="long",
        ),
    )
    result = backtest_runner.main(logger, runner_config)

candles = result["candles"]
statistics = result["statistics"][0]  # full report

# Allow cross-domain access (required for front-end and back-end separation)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/run_backtest")
def run_backtest():
    return result
