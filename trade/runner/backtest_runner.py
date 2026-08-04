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
from trade.runner import cus_analyzer, result_analyze
from trade.venue.bt import cus_comminfo
from model import train
from model import train_config
from trade.venue.bt.bt_venue_ml import MlBtVenue
from trade.feed.prediction_feed import PredictionFeed
log_file = os.path.join(TEMPORARY_DIR, 'trade_log_ftmo')

class TradeResult:
    def __init__(self) -> None:
        self.times = 0

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

def log_metrics_block(logger, prefix: str, metrics: dict, items_per_line: int = 4):
    """
    Generic metric printing: dump any {key: scalar} mapping N items per line.
    Every strategy returns different fields, so nothing is assumed about them here.
    """
    if not metrics:
        return
    keys = list(metrics.keys())
    for i in range(0, len(keys), items_per_line):
        parts = []
        for k in keys[i:i + items_per_line]:
            v = metrics[k]
            parts.append(f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}")
        logger.info(f"{prefix:<8}| " + " | ".join(parts))


@dataclass
class StrategyPara:
    # switches
    allow_long: bool = True
    allow_short: bool = True
    # execution
    min_hold_bars: int = 48       # default value, may be overridden on init
    commission_pct: float = 0.05   # cryptocurrency: 0.05. 0.1 = 0.1%, can't be 0   XAU: comission:0.0007, spread: 0.02
    init_equity: float = 10000.0
    # signal
    prob_thresh: Optional[float] = None
    # stop / take
    atr_sl_long_mult: float = 6
    atr_sl_short_mult: float = 6
    atr_tp_mult: float = 100
    # risk
    risk_per_trade_pct: float = 0.01
    max_daily_loss_pct: float = 0.04
    decide_version : int = 0

# period: short / forward / long
def main(logger:logging.Logger, para = StrategyPara(),train_cfg= train.TrainConfig(),prep_output_dir =common.DATA_OUT_DIR,train_output_dir: str = common.TRAIN_OUT_DIR,
         device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), period = 'long'):
    pre_para:BaseDefine = common.load_pre_params_from_dir(train_output_dir)
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
    # 2. Wrapped model inference (loading and prediction in one call)
    # -----------------------------------------------------------
    # train_output_dir is the model directory (batch experiments give every training its own dir, keeping training and simulation paired)
    tarin_out_path = train_output_dir
    if not os.path.isabs(tarin_out_path):
        tarin_out_path = os.path.join(PROJECT_DIR, tarin_out_path)
    handler = model_loader.ModelHandler(tarin_out_path=tarin_out_path, device=device)  # Best_F1/Best_Loss
    # # Run the prediction and collect the results and metrics
    # df_with_pred, model_stats = handler.predict(df, kline_interval_ms=_interval_ms, is_live = False, diff_thresh = None,
    #                                                cache_path=os.path.join(TEMPORARY_DIR,"trade_cache.pt"), use_cache = False )

    # 1. Prepare the data: the is_live flag controls how indices are recorded
    ds = data_loader.TimeSeriesWindowDataset(
        df=df, 
        kline_interval_ms = _interval_ms,
        feature_cols=handler.feature_cols, 
        label_col=handler.label_col,
        seq_len=handler.seq_len,
        is_live=False,
    )
    df['stop_loss_atr_pct'] = common.stop_loss_atr_pct(df, para.min_hold_bars)
    atr_colum = 'stop_loss_atr_pct'
    df_with_pred, model_stats = handler.predict_with_ds(ds,df,is_live=False,diff_thresh = None)
    # For composite tasks (TRIGGER_DIR/LONG_SHORT_OVR) every sub-model is evaluated on its own;
    # non composite tasks return {}, leaving the single model 3-class report structure untouched.
    sub_model_stats = handler.evaluate_sub_models(df, kline_interval_ms=_interval_ms)
    # compare with random prediction (sanity check)
    # df_with_pred['pred'] = np.random.choice([0, 1, 2], size=len(df_with_pred))
    # handler.scan_thresholds(df, thresholds=[0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    # exit()
    # Drop the leading rows without a prediction (for backtrader)
    # 2. [core change]: find the index of the first valid prediction
    # This skips the warm-up period where the features are not ready yet
    # while keeping the NaN holes caused by time discontinuities in the middle
    first_valid_idx = df_with_pred['pred'].first_valid_index()

    if first_valid_idx is not None:
        # Keep every row from the first signal onwards (NaN holes included)
        df_with_pred = df_with_pred.loc[first_valid_idx:].copy()
        logger.info(f"Backtest starts from first signal at {df_with_pred.index[0]}")
    else:
        logger.error("No valid predictions found in the entire dataset!")
        sys.exit(1)

    logger.info(
        f"Backtest range: {df_with_pred['open_time_date_utc'].min()} to {df_with_pred['open_time_date_utc'].max()}"
    )

    # 4. Run backtrader
    cerebro = bt.Cerebro(runonce=False,cheat_on_open=True,maxcpus=1)
    cerebro.addstrategy(
        MlBtVenue,
        predict_num = pre_para.predict_num,
        min_hold_bars=para.min_hold_bars,
        allow_short=para.allow_short,
        allow_long=para.allow_long,
        prob_thresh=para.prob_thresh,
        atr_sl_long_mult = para.atr_sl_long_mult,
        atr_sl_short_mult = para.atr_sl_short_mult,
        atr_tp_mult = para.atr_tp_mult,
        risk_per_trade_pct = para.risk_per_trade_pct,
        max_daily_loss_pct = para.max_daily_loss_pct,
        decide_version = para.decide_version,
        init_equity = para.init_equity,
    )

    data = PredictionFeed(
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
        bars_to_close = "bars_to_close",
        openinterest=-1,
        nocase=True,
        # fromdate=datetime(2023, 10, 1),
        # todate=datetime(2024, 1, 1),
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(para.init_equity)
    cerebro.broker.addcommissioninfo(
        cus_comminfo.CommInfo_Cryptocurrency(commission=para.commission_pct, leverage =10)
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
    # 5. Result statistics
    # UI
    # Pack the statistics (backtest numbers merged with the model metrics)
    statistics = generate_backtest_report(logger, strat, model_stats, save_path=os.path.join(TEMPORARY_DIR,'full_backtest_report.json'), para=para,pre_para=pre_para,train_cfg=train_cfg,
                                           sub_model_stats=sub_model_stats)

    # ========== convert to kline JSON ==========
    candles = df_with_pred[["open_time_date_utc", "open", "high", "low", "close", "volume", "pred", "label"]].copy()
    candles["pred"] = candles["pred"].fillna(0.0)
    candles["label"] = candles["label"].fillna(-1).astype(int)
    candles.rename(columns={"open_time_date_utc": "time"}, inplace=True)
    candles["time"] = candles["time"].apply(lambda dt: int(dt.timestamp()))
    candles_json = candles.to_dict(orient="records")

    return {"candles": candles_json, "statistics": statistics}

def build_daily_df(daily_stats):
    """Convert daily stats list (dict with 'date', 'dd_pct', 'equity') to DataFrame."""
    if not daily_stats:
        return pd.DataFrame()
    df = pd.DataFrame(daily_stats)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    return df

# 6 months (180 or 365 recommended for crypto)
def rolling_calmar(df: pd.DataFrame, window_days: int = 180, step_days: int = 30):
    """
    df must contain: date, equity
    """
    results = []

    dates = df['date']
    equity = df['equity'].values

    start_idx = 0
    n = len(df)

    while True:
        start_date = dates.iloc[start_idx]
        end_date = start_date + pd.Timedelta(days=window_days)

        # Find the index where the window ends
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

        # Max drawdown (inside the window)
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

        # Roll forward
        next_date = start_date + pd.Timedelta(days=step_days)
        next_idx = df.index[df['date'] >= next_date].min()
        if pd.isna(next_idx):
            break
        start_idx = next_idx

    return pd.DataFrame(results)

def summarize_rolling_calmar(rc_df: pd.DataFrame) -> dict:
    """
    rc_df columns: start,end,cagr,max_dd,calmar
    Returns finer grained rolling metrics: distribution + tail risk + persistence + robustness
    """
    if rc_df is None or len(rc_df) == 0:
        return {"rc_n": 0}

    s = rc_df["calmar"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {"rc_n": 0}

    # Basic distribution
    q = s.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).to_dict()

    # Persistence: longest consecutive run of "bad" / "good" windows (in window order)
    # Note: the windows overlap, but a run still shows "unusable for a long stretch"
    flags_neg = (s < 0).to_numpy()
    flags_ge1 = (s >= 1).to_numpy()
    flags_ge2 = (s >= 2).to_numpy()

    def longest_run(flags: np.ndarray) -> int:
        best = cur = 0
        for f in flags:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        return int(best)

    # Tail risk: expected shortfall (conditional mean)
    def expected_shortfall(x: pd.Series, alpha: float) -> float:
        thr = x.quantile(alpha)
        tail = x[x <= thr]
        return float(tail.mean()) if len(tail) else float("nan")

    # Robustness: MAD / IQR etc.
    median = float(q[0.50])
    mad = float((s - median).abs().median())  # median absolute deviation
    iqr = float(q[0.75] - q[0.25])

    # Share of "qualifying" windows (you prefer Calmar>=2)
    out = {
        "rc_n": int(len(s)),  # Total number of valid rolling calmar values

        # Quantiles (finer grained)
        "rc_q01": float(q[0.01]),  # 1st percentile of rolling calmar
        "rc_q05": float(q[0.05]),  # 5th percentile of rolling calmar
        "rc_q10": float(q[0.10]),  # 10th percentile of rolling calmar
        "rc_q25": float(q[0.25]),  # 25th percentile (lower quartile) of rolling calmar
        "rc_median": float(q[0.50]),  # Median (50th percentile) of rolling calmar
        "rc_q75": float(q[0.75]),  # 75th percentile (upper quartile) of rolling calmar
        "rc_q90": float(q[0.90]),  # 90th percentile of rolling calmar
        "rc_q95": float(q[0.95]),  # 95th percentile of rolling calmar
        "rc_q99": float(q[0.99]),  # best 99th percentile of rolling calmar

        # Ratios
        "rc_pos_ratio": float((s > 0).mean()),  # Proportion of positive rolling calmar values
        "rc_neg_ratio": float((s < 0).mean()),  # Proportion of negative rolling calmar values
        "rc_ge_1_ratio": float((s >= 1).mean()),  # Proportion of rolling calmar values >= 1
        "rc_ge_2_ratio": float((s >= 2).mean()),  # Proportion of rolling calmar values >= 2
        "rc_ge_3_ratio": float((s >= 3).mean()),  # Proportion of rolling calmar values >= 3

        # Tail (average level of the worst windows, more robust than min)
        "rc_es_05": expected_shortfall(s, 0.05),  # Average of the worst 5% rolling calmar values
        "rc_es_10": expected_shortfall(s, 0.10),  # Average of the worst 10% rolling calmar values

        # Persistence (ability to cross a death zone such as 2022-2023)
        "rc_longest_neg_run": longest_run(flags_neg),  # Longest consecutive run of negative rolling calmar values
        "rc_longest_ge1_run": longest_run(flags_ge1),  # Longest consecutive run of rolling calmar values >= 1
        "rc_longest_ge2_run": longest_run(flags_ge2),  # Longest consecutive run of rolling calmar values >= 2

        # Robustness / dispersion
        "rc_mean": float(s.mean()),  # Mean of rolling calmar values
        "rc_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,  # Standard deviation of rolling calmar values
        "rc_mad": mad,  # Median absolute deviation of rolling calmar values
        "rc_iqr": iqr,  # Interquartile range (Q3 - Q1) of rolling calmar values
        "rc_cv": float(s.std(ddof=1) / s.mean()) if len(s) > 1 and s.mean() != 0 else float("nan"),  # Coefficient of variation (std/mean) of rolling calmar values
    }

    # Optional: carry the distribution of the rolling cagr / max_dd as well (more interpretable)
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
                "rc_dd_q90": float(d.quantile(0.90)),  # drawdowns are negative, so q90 is the closest to 0
                "rc_dd_min": float(d.min()),           # worst drawdown
            })

    return out


def generate_backtest_report(logger,strat, model_stats, save_path, para:StrategyPara,pre_para:BaseDefine,train_cfg:train.TrainConfig, sub_model_stats=None):
    """
    Fixed report generator:
    1. Corrected the profit factor formula (Gross Won / Gross Lost)
    2. Corrected the PnL% extraction (handles backtrader's nested list structure)

    sub_model_stats: per sub-model metrics for composite tasks (TRIGGER_DIR/LONG_SHORT_OVR)
    (see ModelHandler.evaluate_sub_models); an empty dict for single model tasks.
    """
    sub_model_stats = sub_model_stats or {}

    # ========== 1. Pull the analyzers ==========
    perf = strat.analyzers.customize.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    ret_analyzer = strat.analyzers.returns.get_analysis()

    # ----- basic numbers -----
    start_value = strat.broker.startingcash  # starting capital
    end_value = strat.broker.getvalue()      # final capital
    gross_return = (end_value - start_value) / start_value
    cagr = ret_analyzer.get('rnorm', 0.0)
    sr = sharpe.get("sharperatio", 0.0) or 0.0

    # ----- max drawdown -----
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
    # --- read the intraday drawdown data ---
    max_daily_dd = perf.get('max_daily_dd', 0.0) # e.g. -0.045
    max_daily_date = perf.get('max_daily_dd_date', 'N/A')
    max_violation_days = perf.get('daily_dd_max_violation_days', 0)
    max_3_violation_days = perf.get('daily_dd_max_3_violation_days', 0)
    violation_days = perf.get('daily_dd_violation_days', 0)
    # 1. Global minimum equity
    global_min_equity = perf.get('global_min_equity', 0.0)
    max_hwm_duration_days = perf.get('max_hwm_duration_days', 0)
    # 2. Distance from the starting capital (FTMO max loss)
    start_cash = strat.broker.startingcash
    dist_to_start_pct = (global_min_equity - start_cash) / start_cash
    # 3. Log it
    logger.info(f"FTMO LINE | Dist to Start: {dist_to_start_pct*100:.2f}% (Limit: -10%)")

    if dist_to_start_pct < -0.10:
        logger.warning("❌ FAILED: 账户曾经跌破初始本金的 10%！")
    # Print it into the log
    logger.info(f"RISK(Daily)| Worst Day: {max_daily_dd*100:.2f}% ({max_daily_date}) | >4% Days: {violation_days}")

    if max_daily_dd < -0.05:
        logger.warning("❌ 严重警告：单日回撤已触发 FTMO 5% 违规红线！")

    # ----- trade statistics (overall) -----
    total_trades = safe_get(trades, ["total", "closed"], 0)
    total_won = safe_get(trades, ["won", "total"], 0)
    total_lost = safe_get(trades, ["lost", "total"], 0)
    win_rate = (total_won / total_trades) if total_trades > 0 else 0.0

    # [fix 1] correct profit factor computation
    # PF = total gross profit / abs(total gross loss)
    gross_won_total = safe_get(trades, ["won", "pnl", "total"], 0.0)
    gross_lost_total = abs(safe_get(trades, ["lost", "pnl", "total"], 0.0))
    profit_factor = (gross_won_total / gross_lost_total) if gross_lost_total != 0 else 0.0

    # ----- raw absolute PnL -----
    avg_pnl_net = safe_get(trades, ["pnl", "net", "average"], 0.0)  # after costs
    avg_pnl_gross = safe_get(trades, ["pnl", "gross", "average"], 0.0) # before costs
    avg_cost = avg_pnl_gross - avg_pnl_net

    # ============================================================
    # === [fix 2] per-trade return percentage (robust iteration) ===
    # ============================================================
    
    # 1. Daily frequency
    if len(strat.datas) > 0 and len(strat.datas[0]) > 0:
        t_start = bt.num2date(strat.datas[0].datetime.array[0])
        t_end = bt.num2date(strat.datas[0].datetime.array[-1])
        duration = t_end - t_start
        total_days = max(duration.days + (duration.seconds / 86400), 1)
    else:
        total_days = 1
    daily_trades = total_trades / total_days

    # Averages (*100 to get percent)
    if total_trades > 0 and abs(avg_cost) > 1e-12:
        avg_pct_gross = avg_pnl_gross / (avg_cost / 2) * para.commission_pct
        avg_pct_net = avg_pnl_net / (avg_cost / 2) * para.commission_pct
    else:
        avg_pct_gross = 0.0
        avg_pct_net = 0.0

    # ============================================================
    # --- 1. Long statistics ---
    long_total = safe_get(trades, ["long", "total"], 0)
    long_won   = safe_get(trades, ["long", "won"], 0)   # number of wins
    # Long total pnl (amount)
    long_pnl_total = safe_get(trades, ["long", "pnl", "total"], 0.0)
    # Long win rate
    long_win_rate = (long_won / long_total * 100) if long_total > 0 else 0.0
    # --- 2. Short statistics ---
    short_total = safe_get(trades, ["short", "total"], 0)
    short_won   = safe_get(trades, ["short", "won"], 0)
    # Short total pnl (amount)
    short_pnl_total = safe_get(trades, ["short", "pnl", "total"], 0.0)
    # Short win rate
    short_win_rate = (short_won / short_total * 100) if short_total > 0 else 0.0

    # Position information (already computed by CusAnalyzer)
    avg_pos = perf.get("avg_pos_ratio", 0)
    max_pos = perf.get("max_pos_ratio", 0)
    p95_pos = perf.get("p95_pos_ratio", 0)

    # ============================================================
    # === [added: robust risk statistics] only this block changed, to support the Top-N display ===
    # ============================================================
    daily_returns_list = perf.get('daily_returns_list', []) # requires CusAnalyzer to expose this list (list of dicts)
    top_10_str = "N/A"
    robust_max_loss = 0.0
    
    if daily_returns_list:
        # Keep the negative returns and sort them (worst first)
        # daily_returns_list is now a list of dicts, each carrying a 'dd_pct' field
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
        
        # Robust max loss: drop the #1 outlier, average ranks 2-5
        if len(top_5_losses) > 1:
            robust_max_loss = sum(top_5_losses[1:]) / len(top_5_losses[1:])
        else:
            robust_max_loss = top_5_losses[0] if top_5_losses else 0.0

    # ---- strategy specific statistics: content differs per strategy (ML holding times / martingale deaths and restarts...), the channel does not ----
    # The venue already split strategy.report() into summary(scalars, jsonl-able) + detail(list/dict records)
    strategy_stats = strat.strategy_metrics() if hasattr(strat, "strategy_metrics") else {}
    strategy_summary = strategy_stats.get("summary", {})
    strategy_detail = strategy_stats.get("detail", {})

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
            f"daily_loss_list": daily_returns_list,          # list
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
            f"risk_per_trade_pct": para.risk_per_trade_pct,
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
        f"strategy": strategy_summary,
        f"trade_logs": list(getattr(strat, "trade_logs", [])),
        f"model_metrics": model_stats,
        f"sub_model_metrics": sub_model_stats,
    }

    report_additional = {
        f"raw_analyzer":{
            f"customize":perf,
        },
        f"strategy_detail": strategy_detail,
    }

    # common.dump_params_json(train_cfg,logger)
    # common.dump_params_json(para,logger)
    # common.dump_params_json(pre_para,logger)
        
    # summary output
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
        f"| risk_per_trade_pct: {report['exposure']['risk_per_trade_pct']}"
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

    log_metrics_block(logger, "STRATEGY", report['strategy'])

    logger.info(
        f"DETAILS | Long: {report['trades']['long_pnl']} "
        f"Winrate: {report['trades']['long_win_rate']:.1f}% | "
        f"Short: {report['trades']['short_pnl']} "
        f"Winrate: {report['trades']['short_win_rate']:.1f}%"
    )

    logger.info("-" * 80)

    return (report_additional,report)

if __name__ == "__main__":
    train_output_dir = os.path.join(common.TRAIN_OUT_DIR, train_config.TrainTask.DIRECT_3CLASS)
    start_time = time.time()
    pre_para:BaseDefine = common.load_pre_params_from_dir(train_output_dir)
    exp_dir = common.create_experiment_dir(os.path.join(common.PERSISTENCE_DIR,'simulation'),pre_para.symbol, pre_para.interval)
    logger, _ = common.setup_session_logger(log_file_path=os.path.join(exp_dir, 'experiment.log'), console_level = logging.INFO,file_level=logging.INFO)
    para = StrategyPara()
    para.min_hold_bars = pre_para.predict_num
    report = main(logger=logger, para=para, train_output_dir = train_output_dir)
    append_jsonl(
        os.path.join(exp_dir, "reports.jsonl"),
        report["statistics"][1]
    )
    end_time = time.time()
    run_time = end_time - start_time
    logger.info(f": run_time: {run_time:.4f} s")
