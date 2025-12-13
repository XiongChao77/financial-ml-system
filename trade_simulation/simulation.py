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
from trade_simulation import cus_analyzer, cus_comminfo, model_loader, result_analyze
from trade_simulation.strategy.ftmo import FtmoStrategy
from trade_simulation.strategy.simpe import  SimpleStrategy

log_file = os.path.join(TEMPORARY_DIR, "backtest.log")
logger = setup_logger(log_name='trade' ,log_path= log_file, console_level =logging.DEBUG)

class TradeResult:
    def __init__(self) -> None:
        self.times = 0

# --- DataFeed 扩展 ---
class PandasDataWithPred(bt.feeds.PandasData):
    lines = (
        "pred",
        "conf",
        "threshold",       # 动态止盈阈值
        "stop_threshold",  # 动态止损阈值
    )
    params = (
        ("pred", -1),
        ("conf", -1),
        ("threshold", -1),      # 自动匹配列名
        ("stop_threshold", -1), # 自动匹配列名
    )

class Parameters:
    def __init__(self):
        self.allow_short = True
        self.allow_long = True
        self.thresh: float =0.4#None#0.5#None#0.45
        self.commission = 0.05   # 0.1 = 0.1%
        self.cash = 10000
        self.stop_loss = 2  # should be 1-10   stop_loss = self.data.stop_threshold[0]*self.params.stop_loss
        self.take_profit = 0.99
        self.position_ratio = 0.6     #0-1


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

        # B. 【关键修改】调用封装函数计算动态阈值
        # 这会自动在 df_with_pred 中生成 'threshold' 和 'stop_threshold' 两列
        # 且使用的参数 (VOL_MULTIPLIER 等) 默认与 common.py 中定义的训练参数完全一致
        df_with_pred = calculate_thresholds(df_with_pred)

        # C. 过滤无效数据 (过滤掉 pred 为空 或 阈值为 NaN 的行)
        df_with_pred = df_with_pred.dropna(subset=["pred", "threshold", "stop_threshold"]).copy()

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
        position_ratio = args.position_ratio,
    )

    data = PandasDataWithPred(
        dataname=df_with_pred,
        datetime="open_time_utc",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        threshold="threshold",
        stop_threshold="stop_threshold",
        openinterest=-1,
        nocase=True,
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(args.cash)
    cerebro.broker.addcommissioninfo(
        cus_comminfo.CommInfo_Cryptocurrency(commission=args.commission)
    )
    # cerebro.broker.set_coc(True)  #

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, compression=1, factor=365)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns' , tann=365)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(cus_analyzer.CusAnalyzer, _name="customize")

    logger.record("Starting Backtest...")
    results = cerebro.run()
    strat = results[0]
    result_analyze.analyze_pnl_distribution(strat.closed_pnl)
    # result_analyze.analyze_trade_dependency(strat.closed_pnl)
    # 5. 结果统计
    # UI
    # 封装统计数据 (合并回测数据和模型指标)
    statistics = generate_backtest_report(strat, model_stats, commission=args.commission, save_path=os.path.join(TEMPORARY_DIR,'full_backtest_report.json'))

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


def generate_backtest_report(strat, model_stats, save_path, commission):
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
    start_value = strat.broker.startingcash  # 初始资金
    end_value = strat.broker.getvalue()      # 最终资金
    gross_return = (end_value - start_value) / start_value
    ret_analyzer = strat.analyzers.returns.get_analysis()
    cagr = ret_analyzer.get('rnorm', 0.0)
    sr = sharpe.get("sharperatio", 0.0) or 0.0

    # ----- 最大回撤 -----
    maxdd_pct = dd.get("max", {}).get("drawdown", 0.0)
    maxdd_amt = dd.get("max", {}).get("moneydown", 0.0)
    maxdd_len = dd.get("max", {}).get("len", 0)

    # --- 读取日内回撤数据 ---
    max_daily_dd = perf.get('max_daily_dd', 0.0) # 例如 -0.045
    max_daily_date = perf.get('max_daily_dd_date', 'N/A')
    violation_days = perf.get('daily_dd_violation_days', 0)
    # 1. 获取全局最低净值
    global_min_equity = perf.get('global_min_equity', 0.0)
    # 2. 计算距离初始资金的跌幅 (FTMO Max Loss)
    start_cash = strat.broker.startingcash
    dist_to_start_pct = (global_min_equity - start_cash) / start_cash
    # 3. 打印日志
    logger.info(f"FTMO LINE | Min Equity: ${global_min_equity:.2f}")
    logger.info(f"          | Dist to Start: {dist_to_start_pct*100:.2f}% (Limit: -10%)")

    if dist_to_start_pct < -0.10:
        logger.warning("❌ FAILED: 账户曾经跌破初始本金的 10%！")
    else:
        logger.info("✅ PASSED: 总资金风控通过")

    # 在日志中打印
    logger.info(f"RISK(Daily)| Worst Day: {max_daily_dd*100:.2f}% ({max_daily_date}) | >4% Days: {violation_days}")

    if max_daily_dd < -0.05:
        logger.warning("❌ 严重警告：单日回撤已触发 FTMO 5% 违规红线！")

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

    # 计算平均值 (*100 转为百分比)
    avg_pct_gross = avg_pnl_gross /(avg_cost/2) * commission
    avg_pct_net = avg_pnl_net / (avg_cost/2) * commission

    # ============================================================
    # --- 1. 多头统计 (Long) ---
    long_total = safe_get(trades, ["long", "total"], 0)
    long_won   = safe_get(trades, ["long", "won"], 0)   # 获胜次数
    # 多头总盈亏 (金额)
    long_pnl_total = safe_get(trades, ["long", "pnl", "total"], 0.0)
    # 多头胜率
    long_win_rate = (long_won / long_total * 100) if long_total > 0 else 0.0
    # --- 2. 空头统计 (Short) ---
    short_total = safe_get(trades, ["short", "total"], 0)
    short_won   = safe_get(trades, ["short", "won"], 0)
    # 空头总盈亏 (金额)
    short_pnl_total = safe_get(trades, ["short", "pnl", "total"], 0.0)
    # 空头胜率
    short_win_rate = (short_won / short_total * 100) if short_total > 0 else 0.0


    # summary 输出
    logger.info("-" * 80)
    logger.info(f"SUMMARY | GrossRet: {gross_return*100:.2f}% | CAGR: {cagr*100:.2f}% | "
                f"Sharpe: {sr:.3f} | MaxDD: {maxdd_pct:.2f}% ({maxdd_amt:.0f}) | commission: {commission:.2f}%")
    logger.info(f"TRADES  | Total: {total_trades} | Freq: {daily_trades:.2f} trades/day | WinRate: {win_rate:.2f}%")
    logger.info(f"PNL($)  | Avg Gross: {avg_pnl_gross:.2f}({avg_pct_gross:.3f}%) | Avg Net: {avg_pnl_net:.2f}({avg_pct_net:.3f}%) (Cost: {avg_cost:.2f}/trade)")
    logger.info(f"DETAILS | Long: {long_pnl_total} Winrate: {long_win_rate:.1f}% | Short: {short_pnl_total} Winrate: {short_win_rate:.1f}%")
    logger.info("-" * 80)

    # 构造 JSON (略微精简，保持原有结构)
    full_report = {
        "gross_return": gross_return,
        "profit_factor": profit_factor,
        "avg_pct_gross": avg_pct_gross,
        "avg_pct_net": avg_pct_net,
        # 各 Analyzer 原始数据
        "trade_analyzer_raw": trades,
        # 基础资金曲线
        "start_value": start_value,
        "end_value": end_value,
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
        "drawdown_raw": dd,
        # 模型预测指标（从外部传入）
        "model_metrics": model_stats or {},
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
        "cagr": f"{cagr*100:.2f}%",
        "sharpe": f"{sr:.3f}",
        "max_drawdown": f"{maxdd_pct:.2f}%",
        "total_trades": total_trades,
        "win_rate": f"{win_rate:.2f}%",
        "start_value": f"{start_value:.2f}",
        "end_value": f"{end_value:.2f}",
        **(model_stats or {}),
    }
    return ui_stats

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    run_time = end_time - start_time
    logger.info(f": run_time: {run_time:.4f} s")