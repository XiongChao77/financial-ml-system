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
from model import train_2head 
from trade.bt.bt_trade_ml import FtmoStrategy
log_file = os.path.join(TEMPORARY_DIR, 'trade_log_ftmo')

class TradeResult:
    def __init__(self) -> None:
        self.times = 0

# --- DataFeed 扩展 ---
class PandasDataWithPred(bt.feeds.PandasData):
    lines = (
        "pred",
        "pred_prob",
        'label',
        "atr",
        "slow_atr",
        "vol_regime",
    )
    params = (
        ("pred", -1),
        ("pred_prob", -1),
        ("atr", -1),      # 自动匹配列名
        ("slow_atr", -1),      # 自动匹配列名
        ("vol_regime", -1),      # 自动匹配列名
        ("label", -1),
    )

def log_parameters(params_obj, logger):
    """
    自适应获取参数并按 4 个一组格式化打印
    """
    # 1. 过滤掉 Python 内置的 __xx__ 属性和方法 (callable)
    # 这样无论参数定义在 __init__ 内还是类级别都能获取到
    all_keys = [k for k in dir(params_obj) 
                if not k.startswith('__') and not callable(getattr(params_obj, k))]
    
    # 2. 按照名称排序（可选，方便在日志中快速定位）
    # all_keys.sort() 

    items_per_line = 4
    
    for i in range(0, len(all_keys), items_per_line):
        chunk_keys = all_keys[i : i + items_per_line]
        
        # 3. 构造 "key: value" 字符串组
        # 使用 getattr 安全获取值
        para_parts = []
        for k in chunk_keys:
            val = getattr(params_obj, k)
            para_parts.append(f"{k}: {val}")
            
        para_str = " | ".join(para_parts)
        
        # 4. 打印，确保 "Para" 后面的空格与你的 SUMMARY/EXPOSURE 对齐
        logger(f"Para    | {para_str}")

class Parameters:
    allow_short = True
    allow_long = True
    holdbar = CommonDefine.PREDICT_NUM#CommonDefine.PREDICT_NUM
    commission = 0.05   # 0.1 = 0.1%  .can't be 0
    cash = 10000
    thresh: float =None#0.5#None#0.45
    stop_loss_long = 0.03  # 0-1
    stop_loss_short = 0.015  # 0-1
    atr_sl_mult_long = 8 # 5
    atr_sl_mult_short = 4.5 #2.5
    take_profit = 0.99 #止盈. 0 - n倍
    trade_risk = 0.8    #0-leverage
    max_daily_loss_pct = 0.03

def main(logger:logging.Logger):
    logger.info(
        f"Backtest settings: Short={Parameters.allow_short}, Long={Parameters.allow_long}, Thresh={Parameters.thresh}, commission={Parameters.commission}"
    )

    # 1. 数据加载
    # 直接读取 CSV，假设其中已包含所有特征列和时间列
    df = load_test_df()
    # df = load_train_df()
    # 【关键】检查时间列是否存在
    # if "open_time_date_utc" not in df.columns:
    #     logger.error("CRITICAL: 'open_time_date_utc' column missing.")
    #     sys.exit(1)
    # # 【关键】解析时间列
    # # 不再调用 attach_attr，避免重复计算和潜在的数据修改
    df["open_time_date_utc"] = pd.to_datetime(df["open_time_date_utc"], utc=True)

    # -----------------------------------------------------------
    # 2. 封装的模型预测 (一行代码搞定加载和推理)
    # -----------------------------------------------------------
    try:
        # 初始化处理类
        tarin_out_path = r"output\the5ers"
        tarin_out_path = os.path.join(PROJECT_DIR,tarin_out_path)
        handler = model_loader.ModelHandler(tarin_out_path=tarin_out_path) #Best_F1/Best_Loss
        # 执行预测，获取结果和指标
        df_with_pred, model_stats = handler.predict(df, kline_interval_ms = load_interval_ms(), is_live = False, diff_thresh = None,
                                                       cache_path=os.path.join(TEMPORARY_DIR,"trade_cache.pt"), use_cache = False )
        # handler.scan_thresholds(df, thresholds=[0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
        # exit()
        # 过滤掉没有预测结果的前面部分数据（用于 Backtrader）
        # 2. 【核心修改】：寻找第一个有效预测的索引
        # 这样可以跳过最开始特征还没算出来的“预热期”
        # 但会保留中间因为时间不连续产生的 NaN “空洞”
        first_valid_idx = df_with_pred['pred'].first_valid_index()

        if first_valid_idx is not None:
            # 从第一个信号开始，保留后续所有行（包含中间的 NaN）
            df_with_pred = df_with_pred.loc[first_valid_idx:].copy()
            logger.info(f"Backtest starts from first signal at {df_with_pred.index[0]}")
        else:
            logger.error("No valid predictions found in the entire dataset!")
            sys.exit(1)

        logger.info(
            f"Backtest range: {df_with_pred['open_time_date_utc'].min()} to {df_with_pred['open_time_date_utc'].max()}"
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
        holdbar=Parameters.holdbar,
        allow_short=Parameters.allow_short,
        allow_long=Parameters.allow_long,
        thresh=Parameters.thresh,
        stop_loss_long = Parameters.stop_loss_long,
        stop_loss_short = Parameters.stop_loss_short,
        atr_sl_mult_long = Parameters.atr_sl_mult_long,
        atr_sl_mult_short = Parameters.atr_sl_mult_short,
        take_profit = Parameters.take_profit,
        trade_risk = Parameters.trade_risk,
        max_daily_loss_pct = Parameters.max_daily_loss_pct,
    )

    data = PandasDataWithPred(
        dataname=df_with_pred,
        datetime="open_time_date_utc",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        atr = "atr_14",
        # slow_atr = "atr_5000",
        # vol_regime = "vol_regime_100",
        label = "label",
        openinterest=-1,
        nocase=True,
        # fromdate=datetime(2024, 10, 1),
        # todate=datetime(2025, 1, 1),
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(Parameters.cash)
    cerebro.broker.addcommissioninfo(
        cus_comminfo.CommInfo_Cryptocurrency(commission=Parameters.commission, leverage =10)
    )
    # cerebro.broker.set_coc(True)  #

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, compression=1, factor=365)
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns' , tann=365)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(cus_analyzer.CusAnalyzer, _name="customize")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="my_trades")

    logger.info("Starting Backtest...")
    results = cerebro.run()
    strat = results[0]
    # result_analyze.analyze_pnl_distribution(strat.closed_pnl)
    # result_analyze.analyze_trade_dependency(strat.closed_pnl)
    # 5. 结果统计
    # UI
    # 封装统计数据 (合并回测数据和模型指标)
    statistics = generate_backtest_report(logger, strat, model_stats, save_path=os.path.join(TEMPORARY_DIR,'full_backtest_report.json'))

    trade_logs = cerebro.trade_logs

    # ========== 转 K线 JSON ==========
    candles = df_with_pred[["open_time_date_utc", "open", "high", "low", "close", "volume", "pred", "label"]].copy()
    candles["pred"] = candles["pred"].fillna(0.0)
    candles["label"] = candles["label"].fillna(-1).astype(int)
    candles.rename(columns={"open_time_date_utc": "time"}, inplace=True)
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


def generate_backtest_report(logger,strat, model_stats, save_path):
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
    trade_analysis = strat.analyzers.my_trades.get_analysis()

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
    calmar = (cagr*100 / abs(maxdd_pct)) if maxdd_pct > 0 else 0.0
    lost_longest = trade_analysis.streak.lost.longest
    won_longest = trade_analysis.streak.won.longest
    # --- 读取日内回撤数据 ---
    max_daily_dd = perf.get('max_daily_dd', 0.0) # 例如 -0.045
    max_daily_date = perf.get('max_daily_dd_date', 'N/A')
    max_violation_days = perf.get('daily_dd_max_violation_days', 0)
    max_3_violation_days = perf.get('daily_dd_max_3_violation_days', 0)
    violation_days = perf.get('daily_dd_violation_days', 0)
    # 1. 获取全局最低净值
    global_min_equity = perf.get('global_min_equity', 0.0)
    # 2. 计算距离初始资金的跌幅 (FTMO Max Loss)
    start_cash = strat.broker.startingcash
    dist_to_start_pct = (global_min_equity - start_cash) / start_cash
    # 3. 打印日志
    logger.info(f"FTMO LINE | Dist to Start: {dist_to_start_pct*100:.2f}% (Limit: -10%)")

    if dist_to_start_pct < -0.10:
        logger.warning("❌ FAILED: 账户曾经跌破初始本金的 10%！")
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
    avg_pct_gross = avg_pnl_gross /(avg_cost/2) * Parameters.commission
    avg_pct_net = avg_pnl_net / (avg_cost/2) * Parameters.commission

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

    # 获取仓位信息 (CusAnalyzer 已经计算好了)
    avg_pos = perf.get("avg_pos_ratio", 0)
    max_pos = perf.get("max_pos_ratio", 0)
    p95_pos = perf.get("p95_pos_ratio", 0)

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

    # 【修改日志输出】增加鲁棒指标显示
    logger.info(f"RISK(Daily)| Top 5 Losses: [{top_10_str}]")
    logger.info(f"RISK(Daily)| Robust Max Loss (Avg 2nd-5th): {robust_max_loss*100:.2f}%")
    logger.info(f"RISK(Daily)| Worst Day: {max_daily_dd*100:.2f}% ({max_daily_date}) | >3% Days: {max_3_violation_days} | >4% Days: {violation_days} | >5% Days: {max_violation_days}")

    # summary 输出
    logger.info("-" * 80)
    logger.info(f"Time    | {bt.num2date(strat.datas[0].datetime.array[0])} --> {bt.num2date(strat.datas[0].datetime.array[-1])} | end value {end_value} ")
    logger.info(f"SUMMARY | GrossRet: {gross_return*100:.2f}% | CAGR: {cagr*100:.2f}% | "
                f"Sharpe: {sr:.3f} | MaxDD: {maxdd_pct:.2f}% ({maxdd_amt:.0f}) | Calmar: {calmar:.2f}")
    # 【新增】打印仓位暴露信息
    logger.info(f"EXPOSURE| Avg Pos: {avg_pos*100:.2f}% | Max Pos: {max_pos*100:.2f}% | P95 Pos: {p95_pos*100:.2f}% | trade_risk: {Parameters().trade_risk}")
    logger.info(f"TRADES  | Total: {total_trades} | Freq: {daily_trades:.2f} trades/day | WinRate: {win_rate:.2f}% | lost_longest: {lost_longest} | won_longest: {won_longest} ")
    logger.info(f"PNL($)  | Avg Gross: {avg_pnl_gross:.2f}({avg_pct_gross:.3f}%) | Avg Net: {avg_pnl_net:.2f}({avg_pct_net:.3f}%) (Cost: {avg_cost:.2f}/trade)")
    logger.info(f"DETAILS | Long: {long_pnl_total} Winrate: {long_win_rate:.1f}% | Short: {short_pnl_total} Winrate: {short_win_rate:.1f}%")
    log_parameters(Parameters,logger.info)
    log_parameters(train_2head.TrainConfig,logger.debug)
    log_parameters(CommonDefine,logger.debug)
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
    logger, _= setup_session_logger(sub_folder='simulation',console_level= logging.INFO, file_level = logging.DEBUG)
    start_time = time.time()
    main(logger)
    end_time = time.time()
    run_time = end_time - start_time
    logger.info(f": run_time: {run_time:.4f} s")