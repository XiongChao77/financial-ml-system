from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os, sys, logging

current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..','..'))
from data_process import common

from trade.bt import simulation
from data_process import common

app = FastAPI()
logger, _= common.setup_session_logger(sub_folder='backend',console_level= logging.INFO, file_level = logging.DEBUG)

# report_file = r'/home/chao/work/quant_output/batch_train/DOGEUSDT_30m/2026-06-25/04_09_15/batch_simulation/report_view/selected_configs.jsonl'
# report = common.load_reports(report_file)
# simulation_result = report[0]['raw'].get("simulation", report)
# short = simulation_result.get("short", report)
# sim_params = short['params']['strategy']
# pre_params = short['params']['common']
# train_params = short['params']['train']
# fusion_dir = common.recursive_get(report, 'fusion_dir')
# prep_output_dir = common.recursive_get(report, 'prep_output_dir')
# result = simulation.main(
#                 logger,
#                 para=simulation.StrategyPara(**sim_params),
#                 pre_para=common.BaseDefine(**pre_params),
#                 train_cfg=common.config_from_dict_train(train_params),
#                 prep_output_dir=prep_output_dir,
#                 train_output_dir=fusion_dir,
#                 device='cpu',
#                 period='long',
#             )

result = simulation.main(logger)

candles = result["candles"]
markers = result["markers"]
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
