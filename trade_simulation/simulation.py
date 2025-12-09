from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
import os, sys, time, json
import backtrader as bt
import backtrader.analyzers as btanalyzers
from datetime import timezone
import pandas as pd
import numpy as np
import logging

# --- 新增：引入 sklearn 计算模型指标 ---
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

# 引入自定义模块
from data_process.common import *
from model.train import CNN1D
from model.data_loader import TimeSeriesWindowDataset
from trade_simulation import cus_analyzer, cus_comminfo, model_loader

_cash = 10000
commission = 0.001  # 0.001 # 0.1%

log_file = os.path.join(TEMPORARY_DIR, "backtest.log")
logger = setup_logger(log_name='trade' ,log_path= log_file)

class TradeResult:
    def __init__(self) -> None:
        self.times = 0

# --- DataFeed 扩展 ---
class PandasDataWithPred(bt.feeds.PandasData):
    lines = (
        "pred",
        "conf",
    )
    params = (
        ("pred", -1),
        ("conf", -1),
    )

# --- Strategy ---
class MyStrategy(bt.Strategy):
    params = dict(
        holdbar=1,
        trade_risk=0.1,  # 每次加仓 10% 总资金
        max_layers=6,  # 最大加仓层数
        allow_short=True,
        allow_long=True,
        thresh=None,  # 置信度阈值
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.bar_executed = None
        self.held_bars = 0
        self.dir = 0  # 当前持仓方向: 1(多), -1(空), 0(无)
        self.layers = 0  # 当前加仓层数
        self.trade_logs = []

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                pass
                # logger.debug(
                #     f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}"
                # )
            elif order.issell():
                pass
                # logger.debug(
                #     f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}"
                # )
            self.bar_executed = len(self)

            # 记录交易日志 (修复 UTC 时间戳问题)
            dt = self.data.datetime.datetime()
            dt_utc = dt.replace(tzinfo=timezone.utc)
            record = {
                "dt": int(dt_utc.timestamp()),
                "price": order.executed.price,
                "size": order.executed.size,
                "is_buy": order.isbuy(),
            }
            self.trade_logs.append(record)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"Order Canceled/Margin/Rejected: {order.getstatusname()}")

    def stop(self):
        value = self.broker.getvalue()
        logger.info(f"End Value: {value:.2f}")
        # UI
        self.cerebro.trade_logs = self.trade_logs

    def next(self):
        # 获取预测结果
        pred = self.data.pred[0]
        conf = self.data.conf[0]

        # 1. 数据有效性检查
        if np.isnan(pred) or np.isnan(conf):
            return

        # 2. 置信度过滤
        if self.params.thresh is not None and conf < self.params.thresh:
            # 如果置信度不够，我们可以选择“保持不动”或者“视为震荡”
            # 这里简单处理：视为震荡信号(target_dir=0)，如果不持仓则不开，如果持仓则可能平仓
            pred = 1  # 强制视为震荡/观望

        pred = int(pred)

        # 3. 映射信号到方向
        # 假设: 0=空(Short), 1=震荡(Neutral), 2=多(Long)
        target_dir = 0
        if pred == 2 and self.params.allow_long:
            target_dir = 1
        elif pred == 0 and self.params.allow_short:
            target_dir = -1
        else:
            target_dir = 0  # 震荡或不允许的方向

        # 更新持仓时间计数
        if self.position:
            self.held_bars += 1
        else:
            self.held_bars = 0
            self.dir = 0
            self.layers = 0

        # === 4. 核心交易逻辑优化 ===

        # 情况 A: 当前无持仓
        if not self.position:
            if target_dir != 0:
                # 只有明确的多/空信号才开仓
                self.dir = target_dir
                self.layers = 1
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.order_target_percent(target=target_pct)
            return

        # 情况 B: 当前有持仓 (self.dir != 0)

        # B-1: 信号变为震荡 (Label 1) -> 立即平仓
        if target_dir == 0:
            if self.held_bars >= self.params.holdbar:
                self.close()
                self.dir = 0
                self.layers = 0
            return

        # B-2: 信号反转 (多转空 或 空转多) -> 反手 (Reverse)
        if target_dir != self.dir:
            if self.held_bars >= self.params.holdbar:
                # 记录新方向
                self.dir = target_dir
                self.layers = 1  # 重置层数为1
                # 直接计算反向的目标仓位 (例如从 +0.05 变成 -0.05)
                # Backtrader 会自动平掉旧仓位并开新仓位
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.order_target_percent(target=target_pct)
            return

        # B-3: 信号同向 (同向预测) -> 加仓 (Pyramiding)
        if target_dir == self.dir:
            if self.layers < self.params.max_layers:
                self.layers += 1
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.order_target_percent(target=target_pct)
                logger.debug(
                    f"Pyramiding: Layer {self.layers}, Target {target_pct:.2%}"
                )
            return

        # 强制最后平仓
        if len(self.data) - 1 == len(self) - 1 and self.position:
            self.close()

class Parameters:
    def __init__(self):
        self.allow_short = True
        self.allow_long = False
        self.thresh: float = 0.4

def main():
    args = Parameters()
    logger.info(
        f"Backtest settings: Short={args.allow_short}, Long={args.allow_long}, Thresh={args.thresh}"
    )

    # 1. 数据加载
    data_path = test_data_path
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    # 直接读取 CSV，假设其中已包含所有特征列和时间列
    df = pd.read_csv(data_path)
    # 【关键】检查时间列是否存在
    if "open_time_utc" not in df.columns:
        logger.error("CRITICAL: 'open_time_utc' column missing.")
        sys.exit(1)
    # 【关键】解析时间列
    # 不再调用 attach_attr，避免重复计算和潜在的数据修改
    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"], utc=True)

    # -----------------------------------------------------------
    # 2. 封装的模型预测 (一行代码搞定加载和推理)
    # -----------------------------------------------------------
    try:
        # 初始化处理类
        handler = model_loader.ModelHandler()
        # 执行预测，获取结果和指标
        df_with_pred, model_stats = handler.predict(df)
        # 过滤掉没有预测结果的前面部分数据（用于 Backtrader）
        df_backtest = df_with_pred.dropna(subset=["pred"]).copy()
        logger.info(
            f"Backtest range: {df_backtest['open_time_utc'].min()} to {df_backtest['open_time_utc'].max()}"
        )

    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 4. Backtrader 执行
    cerebro = bt.Cerebro(runonce=False)
    cerebro.addstrategy(
        MyStrategy,
        holdbar=4,
        allow_short=args.allow_short,
        allow_long=args.allow_long,
        thresh=args.thresh,
    )

    data = PandasDataWithPred(
        dataname=df_with_pred,
        datetime="open_time_utc",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
        nocase=True,
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(_cash)
    cerebro.broker.addcommissioninfo(
        cus_comminfo.CommInfo_Cryptocurrency(commission=commission)
    )
    # cerebro.broker.set_coc(True)  #

    cerebro.addanalyzer(
        btanalyzers.SharpeRatio,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        compression=1,
    )
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(cus_analyzer.CusAnalyzer, _name="customize")

    logger.info("Starting Backtest...")
    results = cerebro.run()
    strat = results[0]

    # 5. 结果统计
    # UI
    # 封装统计数据 (合并回测数据和模型指标)
    statistics = generate_backtest_report(strat, model_stats, save_path=os.path.join(TEMPORARY_DIR,'full_backtest_report.json'))

    trade_logs = cerebro.trade_logs

    # ========== 转 K线 JSON ==========
    candles = df[["open_time_utc", "open", "high", "low", "close"]].copy()
    candles.rename(columns={"open_time_utc": "time"}, inplace=True)
    candles["time"] = candles["time"].apply(lambda dt: int(dt.timestamp()))
    candles_json = candles.to_dict(orient="records")

    markers = []
    for t in trade_logs:
        markers.append(
            {
                "time": t["dt"],
                "price": t["price"],
                "size": t["size"],
                "position": "aboveBar" if t["is_buy"] else "belowBar",
                "color": "green" if t["is_buy"] else "red",
                "shape": "arrowUp" if t["is_buy"] else "arrowDown",
                "text": ("BUY" if t["is_buy"] else "SELL") + f" @ {t['price']:.2f}",
            }
        )

    return {"candles": candles_json, "markers": markers, "statistics": statistics}


def safe_get(d, keys, default=0):
    """从多层 dict 中取值，避免 KeyError"""
    cur = d
    for k in keys:
        cur = cur.get(k, {})
    return cur if cur != {} else default


def generate_backtest_report(
    strat, model_stats, save_path
):
    """
    功能：
    1. 读取 analyzers（customize / dd / sharpe / trades）
    2. 打印 summary（保持你现在的格式）
    3. 保存完整信息到 JSON 文件
    4. 返回前端 UI 需要的 statistics（保持你现在的格式）
    """

    # ========== 1. 提取 analyzers ==========
    perf = strat.analyzers.customize.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()

    # ----- 基础数值 -----
    gross_return = perf.get("gross_return", 0)
    cagr = perf.get("cagr", 0)
    start_value = perf.get("start_value", 0)
    end_value = perf.get("end_value", 0)

    # ----- Sharpe -----
    sr = sharpe.get("sharperatio", 0.0) or 0.0

    # ----- 最大回撤 -----
    maxdd_pct = dd.get("max", {}).get("drawdown", 0.0)
    maxdd_amt = dd.get("max", {}).get("moneydown", 0.0)
    maxdd_len = dd.get("max", {}).get("len", 0)

    # ----- 交易统计 -----
    total_trades = safe_get(trades, ["total", "closed"], 0)
    total_won = safe_get(trades, ["won", "total"], 0)
    win_rate = (total_won / total_trades * 100) if total_trades > 0 else 0.0

    # ========== 2. 打印 summary ==========
    summary = (
        f"SUMMARY | GrossRet: {gross_return*100:.2f}% | CAGR: {cagr*100:.2f}% | "
        f"Sharpe: {sr:.3f} | MaxDD: {maxdd_pct:.2f}% ({maxdd_amt:.0f}) | "
        f"Trades: {total_trades} | WinRate: {win_rate:.2f}% commission: {commission*100:.2f}%"
    )
    logger.debug(summary)

    # ========== 3. 构造完整 JSON report ==========
    full_report = {
        # 基础资金曲线
        "start_value": start_value,
        "end_value": end_value,
        "gross_return": gross_return,
        "cagr": cagr,
        "sharpe": sr,
        # 回撤
        "max_drawdown_pct": maxdd_pct,
        "max_drawdown_amount": maxdd_amt,
        "max_drawdown_duration": maxdd_len,
        # 交易统计
        "total_trades": total_trades,
        "total_won": total_won,
        "win_rate": win_rate,
        # 暴露信息来自 CusAnalyzer
        "avg_pos_ratio": perf.get("avg_pos_ratio", 0),
        "std_pos_ratio": perf.get("std_pos_ratio", 0),
        "p95_pos_ratio": perf.get("p95_pos_ratio", 0),
        "max_pos_ratio": perf.get("max_pos_ratio", 0),
        # 各 Analyzer 原始数据
        "trade_analyzer_raw": trades,
        "drawdown_raw": dd,
        "sharpe_raw": sharpe,
        # 模型预测指标（从外部传入）
        "model_metrics": model_stats or {},
    }

    # 写入 JSON 文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4)

    # ========== 4. 返回前端所需 statistics ==========
    ui_stats = {
        "gross_return": f"{gross_return*100:.2f}%",
        "cagr": f"{cagr*100:.2f}%",
        "sharpe": f"{sr:.3f}",
        "max_drawdown": f"{maxdd_pct:.2f}%",
        "total_trades": total_trades,
        "win_rate": f"{win_rate:.2f}%",
        "start_value": f"{start_value:.2f}",
        "end_value": f"{end_value:.2f}",
        "equity_curve": perf.get("equity_curve", []),
        "drawdown_curve": perf.get("drawdown_curve", []),
        "return_curve": perf.get("return_curve", []),
        "rolling_return_30": perf.get("rolling_return_30", []),
        **(model_stats or {}),
    }
    return ui_stats


if __name__ == "__main__":
    main()
