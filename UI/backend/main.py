from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os, sys, logging

from trade.bt import simulation
from data_process import common

app = FastAPI()
logger, _= common.setup_session_logger(sub_folder='backend',console_level= logging.INFO, file_level = logging.DEBUG)

result = simulation.main(logger)
candles = result["candles"]
markers = result["markers"]
statistics = result["statistics"][0]  # full report

# 允许跨域（前后端分离必需）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/run_backtest")
def run_backtest():
    return result
