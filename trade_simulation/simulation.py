from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
import os, sys, time, json
import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
import logging

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

# 引入自定义模块
from data_process.common import *
from model.train import CNN1D
from model.data_loader import TimeSeriesWindowDataset
from trade_simulation import cus_analyzer, cus_comminfo, model_loader
from trade_simulation.strategy.ftmo import FtmoStrategy
from trade_simulation.strategy.simpe import  SimpleStrategy

log_file = os.path.join(TEMPORARY_DIR, "backtest.log")
logger = setup_logger(log_name='trade' ,log_path= log_file, console_level =logging.INFO)

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

class Parameters:
    def __init__(self):
        self.allow_short = True
        self.allow_long = True
        self.thresh: float =None# None#0.5#None#0.45
        self.commission = 0.1   # 0.1 = 0.1%
        self.cash = 10000
        self.stop_loss = 0.9  #1% stop loss
        self.take_profit = 0.9


def main():
    args = Parameters()
    logger.record(
        f"Backtest settings: Short={args.allow_short}, Long={args.allow_long}, Thresh={args.thresh}, commission={args.commission}"
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
        df_with_pred = df_with_pred.dropna(subset=["pred"]).copy()
        logger.record(
            f"Backtest range: {df_with_pred['open_time_utc'].min()} to {df_with_pred['open_time_utc'].max()}"
        )

    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 4. Backtrader 执行
    cerebro = bt.Cerebro(runonce=False,cheat_on_open=True)
    cerebro.addstrategy(
        FtmoStrategy,
        holdbar=4,
        allow_short=args.allow_short,
        allow_long=args.allow_long,
        thresh=args.thresh,
        stop_loss=args.stop_loss,
        take_profit = args.take_profit,
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
    cerebro.broker.setcash(args.cash)
    cerebro.broker.addcommissioninfo(
        cus_comminfo.CommInfo_Cryptocurrency(commission=args.commission)
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

    logger.record("Starting Backtest...")
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


def generate_backtest_report(strat, model_stats, save_path):
    """
    修复版报告生成器：
    1. 修正 Profit Factor 计算公式 (Gross Won / Gross Lost)
    2. 修正 PnL% 抓取逻辑 (处理 Backtrader 列表套列表结构)
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
    sr = sharpe.get("sharperatio", 0.0) or 0.0

    # ----- 最大回撤 -----
    maxdd_pct = dd.get("max", {}).get("drawdown", 0.0)
    maxdd_amt = dd.get("max", {}).get("moneydown", 0.0)

    # ----- 交易统计 (总体) -----
    total_trades = safe_get(trades, ["total", "closed"], 0)
    total_won = safe_get(trades, ["won", "total"], 0)
    total_lost = safe_get(trades, ["lost", "total"], 0)
    win_rate = (total_won / total_trades * 100) if total_trades > 0 else 0.0

    # 【修正1】Profit Factor 正确计算
    # PF = 总盈利金额 / 总亏损金额绝对值
    gross_won_total = safe_get(trades, ["won", "pnl", "total"], 0.0)
    gross_lost_total = abs(safe_get(trades, ["lost", "pnl", "total"], 0.0))
    profit_factor = (gross_won_total / gross_lost_total) if gross_lost_total != 0 else 0.0

    # ----- 原始绝对值 PnL -----
    avg_pnl_net = safe_get(trades, ["pnl", "net", "average"], 0.0)  # 扣费后
    avg_pnl_gross = safe_get(trades, ["pnl", "gross", "average"], 0.0) # 扣费前
    avg_cost = avg_pnl_gross - avg_pnl_net

    # ============================================================
    # === 【修正2】计算 单笔收益率百分比 (鲁棒遍历) ===
    # ============================================================
    
    # 1. 计算日均频率
    if len(strat.datas) > 0 and len(strat.datas[0]) > 0:
        t_start = bt.num2date(strat.datas[0].datetime.array[0])
        t_end = bt.num2date(strat.datas[0].datetime.array[-1])
        duration = t_end - t_start
        total_days = max(duration.days + (duration.seconds / 86400), 1)
    else:
        total_days = 1
    daily_trades = total_trades / total_days

    # 2. 遍历交易记录
    pct_gross_list = []
    pct_net_list = []
    
    # Backtrader 的 trades['closed'] 是一个列表，列表的每个元素对应一个 data feed
    # 比如 trades['closed'][0] 是第一个币种的所有交易列表
    closed_container = trades.get('closed', [])
    
    all_trades_objects = []
    
    # 扁平化处理：不管是一层列表还是两层列表，都摊平
    if isinstance(closed_container, list):
        for item in closed_container:
            if isinstance(item, list):
                all_trades_objects.extend(item) # 正常的 Backtrader 结构
            else:
                all_trades_objects.append(item) # 防御性代码
    elif isinstance(closed_container, dict):
        for key in closed_container:
            all_trades_objects.extend(closed_container[key])

    for tr in all_trades_objects:
        # 确保数据存在
        if 'price' in tr and 'size' in tr:
            entry_price = tr['price']
            size = abs(tr['size'])
            entry_value = entry_price * size
            
            if entry_value > 0:
                pct_gross_list.append(tr['pnl'] / entry_value)
                pct_net_list.append(tr['pnlcomm'] / entry_value)

    # 计算平均值 (*100 转为百分比)
    avg_pct_gross = (sum(pct_gross_list) / len(pct_gross_list) * 100) if pct_gross_list else 0.0
    avg_pct_net = (sum(pct_net_list) / len(pct_net_list) * 100) if pct_net_list else 0.0

    # ============================================================

    # 多空统计
    long_total = safe_get(trades, ["long", "total"], 0)
    long_win_rate = (safe_get(trades, ["long", "won"], 0) / long_total * 100) if long_total > 0 else 0.0
    short_total = safe_get(trades, ["short", "total"], 0)
    short_win_rate = (safe_get(trades, ["short", "won"], 0) / short_total * 100) if short_total > 0 else 0.0

    # Log 输出
    logger.info("-" * 80)
    summary = (
        f"SUMMARY | GrossRet: {gross_return*100:.2f}% | CAGR: {cagr*100:.2f}% | "
        f"Sharpe: {sr:.3f} | MaxDD: {maxdd_pct:.2f}% | PF: {profit_factor:.2f}"
    )
    logger.info(summary)
    logger.info(f"TRADES  | Total: {total_trades} | Freq: {daily_trades:.2f} trades/day | WinRate: {win_rate:.2f}%")
    logger.info(f"PNL($)  | Avg Gross: {avg_pnl_gross:.2f} | Avg Net: {avg_pnl_net:.2f} (Cost: {avg_cost:.2f}/trade)")
    logger.info(f"PNL(%)  | Avg Gross: {avg_pct_gross:.3f}% | Avg Net: {avg_pct_net:.3f}%")
    logger.info(f"DETAILS | Long WR: {long_win_rate:.1f}% | Short WR: {short_win_rate:.1f}%")
    logger.info("-" * 80)

    # 构造 JSON (略微精简，保持原有结构)
    full_report = {
        "gross_return": gross_return,
        "profit_factor": profit_factor,
        "avg_pct_gross": avg_pct_gross,
        "avg_pct_net": avg_pct_net,
        "trade_analyzer_raw": trades,
        # ... 其他字段保持不变 ...
    }
    
    # 写入文件
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, default=str)

    # 返回 UI 数据
    ui_stats = {
        "gross_return": f"{gross_return*100:.2f}%",
        "profit_factor": f"{profit_factor:.2f}",
        "avg_trade_gross": f"{avg_pct_gross:.3f}%",
        "avg_trade_net": f"{avg_pct_net:.3f}%",
        "daily_frequency": f"{daily_trades:.2f} 次/天",
        "avg_pnl_net": f"${avg_pnl_net:.2f}",
        # ... 其他字段 ...
    }
    return ui_stats


if __name__ == "__main__":
    main()
