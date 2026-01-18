from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os, sys

from trade import simulation

app = FastAPI()
result = simulation.main()
candles = result["candles"]
markers = result["markers"]
statistics = result["statistics"]

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
