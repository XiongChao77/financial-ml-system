from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
import os, sys, time, json
import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
import logging

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..",'..'))

# 引入自定义模块
from data_process.common import *
from model import model_loader
from trade.bt import cus_analyzer, cus_comminfo, result_analyze
from trade.bt.bt_trade_turtle import TurtleStrategy
from trade.bt.bt_trade_ma import MaCrossoverStrategy
from trade.bt.bt_trade_rules import RulesStrategy
log_file = os.path.join(TEMPORARY_DIR, 'trade_log_ftmo')

class TradeResult:
    def __init__(self) -> None:
        self.times = 0

# --- DataFeed 扩展 ---
class PandasDataWithPred(bt.feeds.PandasData):
    lines = (
        "pred",
        "pred_prob",
        "threshold",       # 动态止盈阈值
        "stop_threshold",  # 动态止损阈值
    )
    params = (
        ("pred", -1),
        ("pred_prob", -1),
        ("threshold", -1),      # 自动匹配列名
        ("stop_threshold", -1), # 自动匹配列名
    )

class Parameters:
    def __init__(self):
        self.allow_short = True
        self.allow_long = True
        self.thresh: float =None#None#0.5#None#0.45
        self.commission = 0.1   # 0.1 = 0.1%  .can't be 0
        self.cash = 10000
        self.stop_loss = 2  # should be 1-10   stop_loss = self.data.stop_threshold[0]*self.params.stop_loss
        self.take_profit = 0.99 #止盈. 0 - n倍
        self.position_ratio = 0.1     #0-1


def main(logger:logging.Logger):
    args = Parameters()
    logger.info(
        f"Backtest settings: Short={args.allow_short}, Long={args.allow_long}, Thresh={args.thresh}, commission={args.commission}"
    )

    # 1. 数据加载
    symbol = 'BTCUSDT'  #BTCUSDT  ETHUSDT  DOGEUSDT SOLUSDT BNBUSDT TRXUSDT XRPUSDT  SUIUSDT ADAUSDT PEPEUSDT", "AAVEUSDT", "DOTUSDT
    interval = '4h'
    origin_data_path = os.path.join(PROJECT_DATA_DIR, f"{symbol}_{interval}.csv")
    data_path = origin_data_path
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    # 直接读取 CSV，假设其中已包含所有特征列和时间列
    df = pd.read_csv(data_path, encoding="utf-8")
    # 【关键】检查时间列是否存在
    if "open_time_date_utc" not in df.columns:
        logger.error("CRITICAL: 'open_time_date_utc' column missing.")
        sys.exit(1)
    df["open_time_date_utc"] = pd.to_datetime(df["open_time_date_utc"], utc=True)

    # 4. Backtrader 执行
    cerebro = bt.Cerebro(runonce=False,cheat_on_open=True)
    select =1
    if select ==0:
        cerebro.addstrategy(
            TurtleStrategy,
            entry_period=15, # System 1
            exit_period=10,
            risk_per_unit=0.01, # 1% per unit
            max_daily_loss_pct = 0.1,
            upper_limit = 0.6,
            unit_pct_scale = 0.5,
        )
    if select==1:
        # simulation_typical.py 
        cerebro.addstrategy(
            RulesStrategy,
        )

    data = PandasDataWithPred(
        dataname=df,
        datetime="open_time_date_utc",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
        nocase=True,
        fromdate=datetime(2022, 1, 1),
        # todate=datetime(2020, 1, 1),
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(args.cash)
    # cerebro.broker.set_checksubmit(True)
    cerebro.broker.addcommissioninfo(
        cus_comminfo.CommInfo_Cryptocurrency(commission=args.commission, leverage =5)
    )
    # cerebro.broker.set_coc(True)  #

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, compression=1, factor=365)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns' , tann=365)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(cus_analyzer.CusAnalyzer, _name="customize")

    logger.info("Starting Backtest...")
    results = cerebro.run()
    strat = results[0]
    statistics = generate_backtest_report(logger, strat, commission=args.commission, save_path=os.path.join(TEMPORARY_DIR,'full_backtest_report.json'))

def safe_get(d, keys, default=0):
    """从多层 dict 中取值，避免 KeyError"""
    cur = d
    for k in keys:
        cur = cur.get(k, {})
    return cur if cur != {} else default


def generate_backtest_report(logger, strat, save_path, commission):
    """
    保留原有逻辑，仅新增 Top-N 风险统计与 Robust Max Loss 逻辑
    """
    # ========== 1. 提取 analyzers (保持原样) ==========
    perf = strat.analyzers.customize.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()

    # ----- 基础数值 (保持原样) -----
    start_value = strat.broker.startingcash
    end_value = strat.broker.getvalue()
    gross_return = (end_value - start_value) / start_value
    ret_analyzer = strat.analyzers.returns.get_analysis()
    cagr = ret_analyzer.get('rnorm', 0.0)
    sr = sharpe.get("sharperatio", 0.0) or 0.0

    # ----- 最大回撤 (保持原样) -----
    maxdd_pct = dd.get("max", {}).get("drawdown", 0.0)
    maxdd_amt = dd.get("max", {}).get("moneydown", 0.0)
    maxdd_len = dd.get("max", {}).get("len", 0)
    calmar = (cagr*100 / abs(maxdd_pct)) if maxdd_pct > 0 else 0.0
    
    # --- 读取日内回撤数据 ---
    max_daily_dd = perf.get('max_daily_dd', 0.0) 
    max_daily_date = perf.get('max_daily_dd_date', 'N/A')
    violation_days = perf.get('daily_dd_violation_days', 0)
    max_violation_days = perf.get('daily_dd_max_violation_days', 0)
    max_3_violation_days = perf.get('daily_dd_max_3_violation_days', 0)

    # ============================================================
    # === 【新增：鲁棒风险统计逻辑】仅修改此块以支持 Top-N 展示 ===
    # ============================================================
    daily_losses = perf.get('daily_returns_list', []) # 需要 CusAnalyzer 配合暴露此列表
    top_10_str = "N/A"
    robust_max_loss = 0.0
    
    if daily_losses:
        # 筛选负收益并排序 (最惨的排在前面)
        sorted_losses = sorted([l for l in daily_losses if l < 0])
        top_5_losses = sorted_losses[:20]
        top_10_str = " | ".join([f"{l*100:.2f}%" for l in top_5_losses])
        
        # 计算 Robust Max Loss: 剔除第1名离群值，取 2-5 名均值
        if len(top_5_losses) > 1:
            robust_max_loss = sum(top_5_losses[1:]) / len(top_5_losses[1:])
        else:
            robust_max_loss = top_5_losses[0] if top_5_losses else 0.0
    # ============================================================

    # 1. 获取全局最低净值 (保持原样)
    global_min_equity = perf.get('global_min_equity', 0.0)
    start_cash = strat.broker.startingcash
    dist_to_start_pct = (global_min_equity - start_cash) / start_cash
    
    logger.info(f"FTMO LINE | Dist to Start: {dist_to_start_pct*100:.2f}% (Limit: -10%)")
    if dist_to_start_pct < -0.10:
        logger.warning("❌ FAILED: 账户曾经跌破初始本金的 10%！")

    # 【修改日志输出】增加鲁棒指标显示
    logger.info(f"RISK(Daily)| Top 5 Losses: [{top_10_str}]")
    logger.info(f"RISK(Daily)| Robust Max Loss (Avg 2nd-5th): {robust_max_loss*100:.2f}%")
    logger.info(f"RISK(Daily)| Worst Day: {max_daily_dd*100:.2f}% ({max_daily_date}) | >4% Days: {violation_days} | >5% Days: {max_violation_days}")
    logger.info(f">3% Days: {max_3_violation_days}")

    if max_daily_dd < -0.05:
        logger.warning("❌ 严重警告：单日回撤已触发 FTMO 5% 违规红线！")

    # ----- 交易统计 (保持原逻辑：包含你原本的 avg_pct_gross 计算公式) -----
    total_trades = safe_get(trades, ["total", "closed"], 0)
    total_won = safe_get(trades, ["won", "total"], 0)
    total_lost = safe_get(trades, ["lost", "total"], 0)
    win_rate = (total_won / total_trades * 100) if total_trades > 0 else 0.0

    gross_won_total = safe_get(trades, ["won", "pnl", "total"], 0.0)
    gross_lost_total = abs(safe_get(trades, ["lost", "pnl", "total"], 0.0))
    profit_factor = (gross_won_total / gross_lost_total) if gross_lost_total != 0 else 0.0

    avg_pnl_net = safe_get(trades, ["pnl", "net", "average"], 0.0)
    avg_pnl_gross = safe_get(trades, ["pnl", "gross", "average"], 0.0)
    avg_cost = avg_pnl_gross - avg_pnl_net

    if len(strat.datas) > 0 and len(strat.datas[0]) > 0:
        t_start = bt.num2date(strat.datas[0].datetime.array[0])
        t_end = bt.num2date(strat.datas[0].datetime.array[-1])
        duration = t_end - t_start
        total_days = max(duration.days + (duration.seconds / 86400), 1)
    else:
        total_days = 1
    daily_trades = total_trades / total_days

    # 保持你原本的收益率计算逻辑
    avg_pct_gross = avg_pnl_gross /(avg_cost/2) * commission if avg_cost != 0 else 0
    avg_pct_net = avg_pnl_net / (avg_cost/2) * commission if avg_cost != 0 else 0

    long_total = safe_get(trades, ["long", "total"], 0)
    long_won   = safe_get(trades, ["long", "won"], 0)
    long_pnl_total = safe_get(trades, ["long", "pnl", "total"], 0.0)
    long_win_rate = (long_won / long_total * 100) if long_total > 0 else 0.0
    short_total = safe_get(trades, ["short", "total"], 0)
    short_won   = safe_get(trades, ["short", "won"], 0)
    short_pnl_total = safe_get(trades, ["short", "pnl", "total"], 0.0)
    short_win_rate = (short_won / short_total * 100) if short_total > 0 else 0.0

    avg_pos = perf.get("avg_pos_ratio", 0)
    max_pos = perf.get("max_pos_ratio", 0)
    p95_pos = perf.get("p95_pos_ratio", 0)

    # summary 输出 (保持原样)
    logger.info("-" * 80)
    logger.info(f"Time    | {bt.num2date(strat.datas[0].datetime.array[0])} --> {bt.num2date(strat.datas[0].datetime.array[-1])} | end value {end_value} ")
    logger.info(f"SUMMARY | GrossRet: {gross_return*100:.2f}% | CAGR: {cagr*100:.2f}% | "
                f"Sharpe: {sr:.3f} | MaxDD: {maxdd_pct:.2f}% ({maxdd_amt:.0f}) | Calmar: {calmar:.2f}")
    logger.info(f"EXPOSURE| Avg Pos: {avg_pos*100:.2f}% | Max Pos: {max_pos*100:.2f}% | P95 Pos: {p95_pos*100:.2f}% | Position: {Parameters().position_ratio}")
    logger.info(f"TRADES  | Total: {total_trades} | Freq: {daily_trades:.2f} trades/day | WinRate: {win_rate:.2f}% | Commission: {commission}%")
    logger.info(f"PNL($)  | Avg Gross: {avg_pnl_gross:.2f}({avg_pct_gross:.3f}%) | Avg Net: {avg_pnl_net:.2f}({avg_pct_net:.3f}%) (Cost: {avg_cost:.2f}/trade)")
    logger.info(f"DETAILS | Long: {long_pnl_total} Winrate: {long_win_rate:.1f}% | Short: {short_pnl_total} Winrate: {short_win_rate:.1f}%")
    logger.info("-" * 80)

    # 构造 JSON (保持原结构，仅注入 robust_max_loss)
    full_report = {
        "gross_return": gross_return,
        "profit_factor": profit_factor,
        "robust_max_loss": robust_max_loss, # 新增
        "avg_pct_gross": avg_pct_gross,
        "avg_pct_net": avg_pct_net,
        "trade_analyzer_raw": trades,
        "start_value": start_value,
        "end_value": end_value,
        "cagr": cagr,
        "sharpe": sr,
        "max_drawdown_pct": maxdd_pct,
        "max_drawdown_amount": maxdd_amt,
        "max_drawdown_duration": maxdd_len,
        "total_trades": total_trades,
        "total_won": total_won,
        "win_rate": win_rate,
        "avg_pos_ratio": perf.get("avg_pos_ratio", 0),
        "std_pos_ratio": perf.get("std_pos_ratio", 0),
        "p95_pos_ratio": perf.get("p95_pos_ratio", 0),
        "max_pos_ratio": perf.get("max_pos_ratio", 0),
        "drawdown_raw": dd,
    }
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, default=str)

    ui_stats = {
        "gross_return": f"{gross_return*100:.2f}%",
        "robust_max_loss": f"{robust_max_loss*100:.2f}%", # 新增
        "profit_factor": f"{profit_factor:.2f}",
        "avg_trade_gross": f"{avg_pct_gross:.3f}%",
        "avg_trade_net": f"{avg_pct_net:.3f}%",
        "daily_frequency": f"{daily_trades:.2f} 次/天",
        "avg_pnl_net": f"${avg_pnl_net:.2f}",
        "cagr": f"{cagr*100:.2f}%",
        "sharpe": f"{sr:.3f}",
        "max_drawdown": f"{maxdd_pct:.2f}%",
        "total_trades": total_trades,
        "win_rate": f"{win_rate:.2f}%",
        "start_value": f"{start_value:.2f}",
        "end_value": f"{end_value:.2f}",
    }
    return ui_stats

if __name__ == "__main__":
    logger, _= setup_session_logger(sub_folder='simulation_typical', console_level= logging.INFO, file_level = logging.DEBUG)
    start_time = time.time()
    main(logger)
    end_time = time.time()
    run_time = end_time - start_time
    logger.info(f": run_time: {run_time:.4f} s")