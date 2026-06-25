from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import datetime
import os, sys, time, json,torch
import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd
import logging
from dataclasses import asdict, is_dataclass,dataclass
from typing import Optional

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..",'..'))

# Import project modules
from data_process.common import *
from data_process import common 
from model import model_loader
from model import data_loader
from trade.bt import cus_analyzer, cus_comminfo, result_analyze
from model import train
from model import train_config
from trade.bt.bt_trade_ml import BtFtmoStrategy
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
        "atr_pct",
        "slow_atr",
        "vol_regime",
    )
    params = (
        ("pred", -1),
        ("pred_prob", -1),
        ("atr_pct", -1),      # 自动匹配列名
        ("slow_atr", -1),      # 自动匹配列名
        ("vol_regime", -1),      # 自动匹配列名
        ("label", -1),
    )

def log_parameters(params_obj, logger):
    """
    Inspect an arbitrary params object and log all attributes in groups of 4.
    """
    # 1. Filter out Python internal __xx__ attributes and methods (callables).
    #    This works whether parameters are defined in __init__ or on the class.
    all_keys = [k for k in dir(params_obj) 
                if not k.startswith('__') and not callable(getattr(params_obj, k))]
    
    # 2. Optionally sort by name for easier inspection
    # all_keys.sort() 

    items_per_line = 4
    
    for i in range(0, len(all_keys), items_per_line):
        chunk_keys = all_keys[i : i + items_per_line]
        
        # 3. Build \"key: value\" strings using getattr for safety
        para_parts = []
        for k in chunk_keys:
            val = getattr(params_obj, k)
            para_parts.append(f"{k}: {val}")
            
        para_str = " | ".join(para_parts)
        
        # 4. Log with a prefix aligned with SUMMARY/EXPOSURE sections
        logger(f"Para    | {para_str}")

@dataclass
class StrategyPara:
    # switches
    allow_long: bool = True
    allow_short: bool = True
    # execution
    holdbar: int = common.BaseDefine.predict_num       # 默认值，初始化时可覆盖
    commission: float = 0.05   # 0.1 = 0.1%, can't be 0
    cash: float = 10000.0
    # signal
    thresh: Optional[float] = None
    # stop / take
    stop_loss_long: float = 0.03
    stop_loss_short: float = 0.015
    atr_sl_mult_long: float = 8.0
    atr_sl_mult_short: float = 5.0
    take_profit: float = 0.99
    # risk
    trade_risk: float = 0.4
    max_daily_loss_pct: float = 0.04
    decide_version : int = 0

# period: short / forward / long
def main(logger:logging.Logger, para = StrategyPara(), pre_para = BaseDefine(),train_cfg= train.TrainConfig(),prep_output_dir =common.DATA_OUT_DIR,train_output_dir: str = common.TRAIN_OUT_DIR,
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), period = 'short'):
    logger.info(f"prep_output_dir:{prep_output_dir}, train_output_dir:{train_output_dir}")
    if period == 'short' or period == 'forward':
        df = common.load_test_df_from_dir(prep_output_dir)
        recent_month = 2
        split_ts = pd.to_datetime(df['open_time_date_utc'].iloc[-1]) - pd.DateOffset(months=recent_month)
        if period == 'forward':
            # Forward test: use last 2 months
            df = df[df['open_time_date_utc'] >= str(split_ts)]
            logger.info(f"🚀 Using forward period (recent {recent_month} months) from {str(split_ts)[:10]}")
        elif period == 'short':
            # Short test: exclude last 2 months
            df = df[df['open_time_date_utc'] < str(split_ts)]
            logger.info(f"📊 Using short period (Prior to {str(split_ts)[:10]})")
    else:
        df = common.load_train_df_from_dir(prep_output_dir)
    logger.info(f"Using period {period} for backtest.Backtest settings: Short={para.allow_short}, Long={para.allow_long}")
    _interval_ms = common.get_interval_ms(pre_para.interval)
    df["open_time_date_utc"] = pd.to_datetime(df["open_time_date_utc"], utc=True)

    # -----------------------------------------------------------
    # 2. 封装的模型预测 (一行代码搞定加载和推理)
    # -----------------------------------------------------------
    # 使用 train_output_dir 作为模型目录（batch 实验时每个 training 独立目录，便于 training 与 simulation 对应）
    tarin_out_path = train_output_dir
    if not os.path.isabs(tarin_out_path):
        tarin_out_path = os.path.join(PROJECT_DIR, tarin_out_path)
    handler = model_loader.ModelHandler(tarin_out_path=tarin_out_path, device=device)  # Best_F1/Best_Loss
    # # 执行预测，获取结果和指标
    # df_with_pred, model_stats = handler.predict(df, kline_interval_ms=_interval_ms, is_live = False, diff_thresh = None,
    #                                                cache_path=os.path.join(TEMPORARY_DIR,"trade_cache.pt"), use_cache = False )

    # 1. 准备数据：传入 is_live 标志以控制索引记录逻辑
    ds = data_loader.TimeSeriesWindowDataset(
        df=df, 
        kline_interval_ms = _interval_ms,
        feature_cols=handler.feature_cols, 
        label_col=handler.label_col, 
        window=handler.window,
        is_live=False,
    )
    df['stop_loss_atr_pct'] = common.stop_loss_atr_pct(df, para.holdbar)
    atr_colum = 'stop_loss_atr_pct'
    df_with_pred, model_stats = handler.predict_with_ds(ds,df,is_live=False,diff_thresh = None)
    # compare with random prediction (sanity check)
    # df_with_pred['pred'] = np.random.choice([0, 1, 2], size=len(df_with_pred))
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

    # 4. Backtrader 执行
    cerebro = bt.Cerebro(runonce=False,cheat_on_open=True,maxcpus=1)
    cerebro.addstrategy(
        BtFtmoStrategy,
        predict_num = pre_para.predict_num,
        holdbar=para.holdbar,
        allow_short=para.allow_short,
        allow_long=para.allow_long,
        thresh=para.thresh,
        stop_loss_long = para.stop_loss_long,
        stop_loss_short = para.stop_loss_short,
        atr_sl_mult_long = para.atr_sl_mult_long,
        atr_sl_mult_short = para.atr_sl_mult_short,
        take_profit = para.take_profit,
        trade_risk = para.trade_risk,
        max_daily_loss_pct = para.max_daily_loss_pct,
        decide_version = para.decide_version,
    )

    data = PandasDataWithPred(
        dataname=df_with_pred,
        datetime="open_time_date_utc",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        atr_pct = atr_colum,
        # slow_atr = "atr_5000",
        # vol_regime = "vol_regime_100",
        label = "label",
        openinterest=-1,
        nocase=True,
        # fromdate=datetime(2023, 10, 1),
        # todate=datetime(2024, 1, 1),
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(para.cash)
    cerebro.broker.addcommissioninfo(
        cus_comminfo.CommInfo_Cryptocurrency(commission=para.commission, leverage =10)
    )
    # cerebro.broker.set_coc(True)  #

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, compression=1, annualize=True, factor=365)
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns' , tann=365)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(cus_analyzer.CusAnalyzer, _name="customize")

    logger.info("Starting Backtest...")
    results = cerebro.run()
    strat = results[0]
    # result_analyze.analyze_pnl_distribution(strat.closed_pnl)
    # result_analyze.analyze_trade_dependency(strat.closed_pnl)
    # 5. 结果统计
    # UI
    # 封装统计数据 (合并回测数据和模型指标)
    statistics = generate_backtest_report(logger, strat, model_stats, save_path=os.path.join(TEMPORARY_DIR,'full_backtest_report.json'), para=para,pre_para=pre_para,train_cfg=train_cfg)

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
        # t["is_buy"] = t["is_buy"] == False
        markers.append(
            {
                "time": t["dt"],
                "price": t["price"],
                # "size": t["size"],
                "position": "aboveBar" if t["is_buy"] else "belowBar",
                "color": "green" if t["is_buy"] else "red",
                "shape": "arrowUp" if t["is_buy"] else "arrowDown",
                "text": ("BUY" if t["is_buy"] else "SELL") + f"@ {t['price']:.2f}",
            }
        )

    return {"candles": candles_json, "markers": markers, "statistics": statistics}

def build_daily_df(daily_stats):
    """Convert daily stats list (dict with 'date', 'dd_pct', 'equity') to DataFrame."""
    if not daily_stats:
        return pd.DataFrame()
    df = pd.DataFrame(daily_stats)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    return df

#6 个月（crypto 推荐 180 或 365）
def rolling_calmar(df: pd.DataFrame, window_days: int = 180, step_days: int = 30):
    """
    df 必须包含: date, equity
    """
    results = []

    dates = df['date']
    equity = df['equity'].values

    start_idx = 0
    n = len(df)

    while True:
        start_date = dates.iloc[start_idx]
        end_date = start_date + pd.Timedelta(days=window_days)

        # 找到窗口结束索引
        end_idx = df.index[df['date'] <= end_date].max()
        if pd.isna(end_idx) or end_idx <= start_idx:
            break

        eq_start = equity[start_idx]
        eq_end = equity[end_idx]

        # CAGR
        years = (dates.iloc[end_idx] - dates.iloc[start_idx]).days / 365
        if years <= 0 or eq_start <= 0:
            start_idx += step_days
            continue

        cagr = (eq_end / eq_start) ** (1 / years) - 1

        # Max Drawdown（窗口内）
        window_eq = equity[start_idx:end_idx + 1]
        peak = np.maximum.accumulate(window_eq)
        dd = (window_eq - peak) / peak
        max_dd = dd.min()

        calmar = cagr / abs(max_dd) if max_dd < 0 else np.inf

        results.append({
            "start": dates.iloc[start_idx],
            "end": dates.iloc[end_idx],
            "cagr": cagr,
            "max_dd": max_dd,
            "calmar": calmar,
        })

        # 向前滚动
        next_date = start_date + pd.Timedelta(days=step_days)
        next_idx = df.index[df['date'] >= next_date].min()
        if pd.isna(next_idx):
            break
        start_idx = next_idx

    return pd.DataFrame(results)

def summarize_rolling_calmar(rc_df: pd.DataFrame) -> dict:
    """
    rc_df columns: start,end,cagr,max_dd,calmar
    返回更细的 rolling 指标：分布 + 尾部风险 + 连续性 + 稳健性
    """
    if rc_df is None or len(rc_df) == 0:
        return {"rc_n": 0}

    s = rc_df["calmar"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {"rc_n": 0}

    # 基本分布
    q = s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()

    # 连续性：最长“差窗口”/“好窗口”连续段（按窗口顺序）
    # 注意：窗口是重叠的，但连续段仍然能反映“长时间不适用”
    flags_neg = (s < 0).to_numpy()
    flags_ge1 = (s >= 1).to_numpy()
    flags_ge2 = (s >= 2).to_numpy()

    def longest_run(flags: np.ndarray) -> int:
        best = cur = 0
        for f in flags:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        return int(best)

    # 尾部风险：Expected Shortfall（条件均值）
    def expected_shortfall(x: pd.Series, alpha: float) -> float:
        thr = x.quantile(alpha)
        tail = x[x <= thr]
        return float(tail.mean()) if len(tail) else float("nan")

    # 稳健性：MAD / IQR 等
    median = float(q[0.50])
    mad = float((s - median).abs().median())  # median absolute deviation
    iqr = float(q[0.75] - q[0.25])

    # “合格窗口”比例（你偏好 Calmar>=2）
    out = {
        "rc_n": int(len(s)),  # Total number of valid rolling calmar values

        # 分位数（更细）
        "rc_q01": float(q[0.01]),  # 1st percentile of rolling calmar
        "rc_q05": float(q[0.05]),  # 5th percentile of rolling calmar
        "rc_q10": float(q[0.10]),  # 10th percentile of rolling calmar
        "rc_q25": float(q[0.25]),  # 25th percentile (lower quartile) of rolling calmar
        "rc_median": float(q[0.50]),  # Median (50th percentile) of rolling calmar
        "rc_q75": float(q[0.75]),  # 75th percentile (upper quartile) of rolling calmar
        "rc_q90": float(q[0.90]),  # 90th percentile of rolling calmar
        "rc_q95": float(q[0.95]),  # 95th percentile of rolling calmar
        "rc_q99": float(q[0.99]),  # best 99th percentile of rolling calmar

        # 比例类
        "rc_pos_ratio": float((s > 0).mean()),  # Proportion of positive rolling calmar values
        "rc_neg_ratio": float((s < 0).mean()),  # Proportion of negative rolling calmar values
        "rc_ge_1_ratio": float((s >= 1).mean()),  # Proportion of rolling calmar values >= 1
        "rc_ge_2_ratio": float((s >= 2).mean()),  # Proportion of rolling calmar values >= 2
        "rc_ge_3_ratio": float((s >= 3).mean()),  # Proportion of rolling calmar values >= 3

        # 尾部（最坏窗口的平均水平，比 min 更稳健）
        "rc_es_05": expected_shortfall(s, 0.05),  # Average of the worst 5% rolling calmar values
        "rc_es_10": expected_shortfall(s, 0.10),  # Average of the worst 10% rolling calmar values

        # 连续性（衡量“穿越2022-2023这种死亡带”的能力）
        "rc_longest_neg_run": longest_run(flags_neg),  # Longest consecutive run of negative rolling calmar values
        "rc_longest_ge1_run": longest_run(flags_ge1),  # Longest consecutive run of rolling calmar values >= 1
        "rc_longest_ge2_run": longest_run(flags_ge2),  # Longest consecutive run of rolling calmar values >= 2

        # 稳健性/离散度
        "rc_mean": float(s.mean()),  # Mean of rolling calmar values
        "rc_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,  # Standard deviation of rolling calmar values
        "rc_mad": mad,  # Median absolute deviation of rolling calmar values
        "rc_iqr": iqr,  # Interquartile range (Q3 - Q1) of rolling calmar values
        "rc_cv": float(s.std(ddof=1) / s.mean()) if len(s) > 1 and s.mean() != 0 else float("nan"),  # Coefficient of variation (std/mean) of rolling calmar values
    }

    # 可选：同时把 rolling 的 cagr / max_dd 的分布也带上（更可解释）
    if "cagr" in rc_df.columns:
        c = rc_df["cagr"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(c):
            out.update({
                "rc_cagr_median": float(c.median()),
                "rc_cagr_q10": float(c.quantile(0.10)),
                "rc_cagr_q25": float(c.quantile(0.25)),
            })
    if "max_dd" in rc_df.columns:
        d = rc_df["max_dd"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d):
            out.update({
                "rc_dd_median": float(d.median()),
                "rc_dd_q90": float(d.quantile(0.90)),  # 回撤是负值，q90更接近0
                "rc_dd_min": float(d.min()),           # 最差回撤
            })

    return out


def generate_backtest_report(logger,strat, model_stats, save_path, para:StrategyPara,pre_para:BaseDefine,train_cfg:train.TrainConfig):
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
    ret_analyzer = strat.analyzers.returns.get_analysis()

    # ----- 基础数值 -----
    start_value = strat.broker.startingcash  # 初始资金
    end_value = strat.broker.getvalue()      # 最终资金
    gross_return = (end_value - start_value) / start_value
    cagr = ret_analyzer.get('rnorm', 0.0)
    sr = sharpe.get("sharperatio", 0.0) or 0.0

    # ----- 最大回撤 -----
    maxdd_pct = dd.get("max", {}).get("drawdown", 0.0)
    maxdd_amt = dd.get("max", {}).get("moneydown", 0.0)
    maxdd_len = dd.get("max", {}).get("len", 0)
    calmar = (cagr*100 / abs(maxdd_pct)) if maxdd_pct > 0 else 0.0
    #rolling calmar
    daily_returns_list = strat.analyzers.customize.get_analysis()['daily_returns_list']
    df = build_daily_df(daily_returns_list)
    rc_df = rolling_calmar(df,window_days=180, step_days=30 )
    rc_summary = summarize_rolling_calmar(rc_df)

    lost_longest = safe_get(trades, ["streak", "lost", "longest"], 0)
    won_longest = safe_get(trades, ["streak", "won", "longest"], 0)
    # --- 读取日内回撤数据 ---
    max_daily_dd = perf.get('max_daily_dd', 0.0) # 例如 -0.045
    max_daily_date = perf.get('max_daily_dd_date', 'N/A')
    max_violation_days = perf.get('daily_dd_max_violation_days', 0)
    max_3_violation_days = perf.get('daily_dd_max_3_violation_days', 0)
    violation_days = perf.get('daily_dd_violation_days', 0)
    # 1. 获取全局最低净值
    global_min_equity = perf.get('global_min_equity', 0.0)
    max_hwm_duration_days = perf.get('max_hwm_duration_days', 0)
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
    win_rate = (total_won / total_trades) if total_trades > 0 else 0.0

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
    if total_trades > 0 and abs(avg_cost) > 1e-12:
        avg_pct_gross = avg_pnl_gross / (avg_cost / 2) * para.commission
        avg_pct_net = avg_pnl_net / (avg_cost / 2) * para.commission
    else:
        avg_pct_gross = 0.0
        avg_pct_net = 0.0

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
    daily_returns_list = perf.get('daily_returns_list', []) # 需要 CusAnalyzer 配合暴露此列表 (list of dicts)
    top_10_str = "N/A"
    robust_max_loss = 0.0
    
    if daily_returns_list:
        # 筛选负收益并排序 (最惨的排在前面)
        # daily_returns_list 现在是字典列表，每个字典有 'dd_pct' 字段
        losses_values = []
        for item in daily_returns_list:
            if isinstance(item, dict):
                val = item.get('dd_pct', 0)
            else:
                val = item
            if val < 0:
                losses_values.append(val)
        
        sorted_losses = sorted(losses_values)
        top_5_losses = sorted_losses[:20]
        top_10_str = " | ".join([f"{l*100:.2f}%" for l in top_5_losses])
        
        # 计算 Robust Max Loss: 剔除第1名离群值，取 2-5 名均值
        if len(top_5_losses) > 1:
            robust_max_loss = sum(top_5_losses[1:]) / len(top_5_losses[1:])
        else:
            robust_max_loss = top_5_losses[0] if top_5_losses else 0.0

    params_hash = calc_params_hash(
        strategy=para,
        common=pre_para,
        train=train_cfg,
    )
    report = {
        f"params": {
            f"strategy": asdict(para),
            f"common": asdict(pre_para),
            f"train": asdict(train_cfg),
            f"hash": params_hash,
            f"git_commit": common.get_git_info(logger),
            "model_stats": {
                "accuracy": model_stats["accuracy"],
                "f1_macro": model_stats["f1_macro"],
                "f1_weighted": model_stats["f1_weighted"],
            },
        },
        f"time": {
            f"start": bt.num2date(strat.datas[0].datetime.array[0]),
            f"end": bt.num2date(strat.datas[0].datetime.array[-1]),
        },

        f"performance": {
            f"gross_return": gross_return,
            f"cagr": cagr,
            f"calmar": calmar,
            f"sharpe": sr,
            f"start_value": start_value,
            f"end_value": end_value,
            f"rc_summary":rc_summary,
        },
        f"drawdown": {
            f"daily_loss_list": daily_returns_list,          # list）
            f"max_dd_pct": maxdd_pct,
            f"max_dd_amt": maxdd_amt,
            f"max_daily_dd": max_daily_dd,
            f"max_daily_date": max_daily_date,
            f"robust_max_daily_loss": robust_max_loss,
            f"dd_3_pct_days": max_3_violation_days,
            f"dd_4_pct_days": violation_days,
            f"dd_5_pct_days": max_violation_days,
            f"max_hwm_duration_days":max_hwm_duration_days,
        },

        f"exposure": {
            f"avg_pos": avg_pos,
            f"max_pos": max_pos,
            f"p95_pos": p95_pos,
            f"trade_risk": para.trade_risk,
        },

        f"trades": {
            f"total": total_trades,
            f"daily_freq": daily_trades,
            f"win_rate": win_rate,
            f"lost_longest": lost_longest,
            f"won_longest": won_longest,
            f"avg_pnl_gross": avg_pnl_gross,
            f"avg_pct_gross": avg_pct_gross,
            f"avg_pnl_net": avg_pnl_net,
            f"avg_pct_net": avg_pct_net,
            f"avg_cost": avg_cost,
            f"long_pnl": long_pnl_total,
            f"long_win_rate": long_win_rate,
            f"short_pnl": short_pnl_total,
            f"short_win_rate": short_win_rate,
        },
        f"model_metrics": model_stats,
    }

    report_additional = {
        f"raw_analyzer":{
            f"customize":perf,
        },
    }

    # common.dump_params_json(train_cfg,logger)
    # common.dump_params_json(para,logger)
    # common.dump_params_json(pre_para,logger)
        
    # summary 输出
    logger.info("-" * 29 + f"PARAMS_HASH | {params_hash}"+"-" * 29)

    logger.info(f"RISK(Daily)| Top 10 Losses: [{top_10_str}]")
    if robust_max_loss:  # Only log if we calculated it
        logger.info(
            f"RISK(Daily)| Robust Max Loss (Avg 2nd-5th): "
            f"{robust_max_loss*100:.2f}%"
        )
    logger.info(
        f"RISK(Daily)| Worst Day: "
        f"{report['drawdown']['max_daily_dd']*100:.2f}% "
        f"({report['drawdown']['max_daily_date']}) | "
        f">3% Days: {report['drawdown']['dd_3_pct_days']} | "
        f">4% Days: {report['drawdown']['dd_4_pct_days']} | "
        f">5% Days: {report['drawdown']['dd_5_pct_days']}"
    )

    logger.info(
        f"Time    | {report['time']['start']} --> {report['time']['end']} "
        f"| CAGR: {report['performance']['cagr']*100:.2f}% "
        f"| Calmar: {report['performance']['calmar']:.2f}"
    )

    logger.info(
        f"SUMMARY | GrossRet: {report['performance']['gross_return']*100:.2f}% "
        f"| Sharpe: {report['performance']['sharpe']:.3f} "
        f"| MaxDD: {report['drawdown']['max_dd_pct']:.2f}% "
        f"({report['drawdown']['max_dd_amt']:.0f}) "
        f"| end value {report['performance']['end_value']}"
    )

    logger.info(
        f"EXPOSURE| Avg Pos: {report['exposure']['avg_pos']*100:.2f}% "
        f"| Max Pos: {report['exposure']['max_pos']*100:.2f}% "
        f"| P95 Pos: {report['exposure']['p95_pos']*100:.2f}% "
        f"| trade_risk: {report['exposure']['trade_risk']}"
    )

    logger.info(
        f"TRADES  | Total: {report['trades']['total']} "
        f"| Freq: {report['trades']['daily_freq']:.2f} trades/day "
        f"| WinRate: {report['trades']['win_rate']*100:.2f}% "
        f"| lost_longest: {report['trades']['lost_longest']} "
        f"| won_longest: {report['trades']['won_longest']}"
    )

    logger.info(
        f"PNL($)  | Avg Gross: {report['trades']['avg_pnl_gross']:.2f}"
        f"({report['trades']['avg_pct_gross']:.3f}%) "
        f"| Avg Net: {report['trades']['avg_pnl_net']:.2f}"
        f"({report['trades']['avg_pct_net']:.3f}%) "
        f"(Cost: {report['trades']['avg_cost']:.2f}/trade)"
    )

    logger.info(
        f"DETAILS | Long: {report['trades']['long_pnl']} "
        f"Winrate: {report['trades']['long_win_rate']:.1f}% | "
        f"Short: {report['trades']['short_pnl']} "
        f"Winrate: {report['trades']['short_win_rate']:.1f}%"
    )

    logger.info("-" * 80)

    return (report_additional,report)

if __name__ == "__main__":
    exp_dir = common.create_experiment_dir(os.path.join(common.PERSISTENCE_DIR,'batch_experiments'),common.BaseDefine.symbol, common.BaseDefine.interval)
    logger: logging.Logger
    logger, _ = common.setup_session_logger(log_file_path=os.path.join(exp_dir, 'experiment.log'), console_level = logging.INFO,file_level=logging.INFO)
    save_dir = os.path.join(common.TRAIN_OUT_DIR)
    start_time = time.time()
    report = main(logger,train_output_dir = save_dir)
    append_jsonl(
        os.path.join(exp_dir, "reports.jsonl"),
        report["statistics"][1]
    )
    end_time = time.time()
    run_time = end_time - start_time
    logger.info(f": run_time: {run_time:.4f} s")