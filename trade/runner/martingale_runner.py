"""
Runner for the restartable martingale strategy (trade/strategy/strategy_martingale.py).

It assembles the three layers and prints/returns the backtest report:
  feed     : PredictionFeed  (raw klines, plus model signals when entry_mode="signal")
  venue    : MartingaleBtVenue (backtrader)
  strategy : RestartableMartingaleStrategy

Two data sources are supported:
  - "csv"   : raw klines from PROJECT_DATA_DIR/{symbol}_{interval}.csv, no model needed.
              Only signal-free entry modes (long / short / reversion) work here.
  - "model" : same pipeline as backtest_runner.py (prepared dataset + trained model),
              required by entry_mode="signal".

The martingale specific statistics (deaths / restarts / swept profit / expectancy edge)
come from RestartableMartingaleStrategy.report() through BtVenue.strategy_metrics().
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import os, sys, time, json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import backtrader as bt
import backtrader.analyzers as btanalyzers
import pandas as pd

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", ".."))

# Import project modules
from data_process.common import *
from data_process import common
from trade.runner import cus_analyzer
from trade.runner.backtest_runner import (
    build_daily_df,
    log_metrics_block,
    rolling_calmar,
    summarize_rolling_calmar,
)
from trade.venue.bt import cus_comminfo
from trade.venue.bt.bt_venue_martingale import MartingaleBtVenue
from trade.feed.prediction_feed import PredictionFeed


@dataclass
class MartingalePara:
    # ---- data ----
    data_source: str = "csv"          # csv (raw klines) | model (prepared dataset + model)
    symbol: str = "BTCUSDT"           # only used by data_source="csv"
    interval: str = "4h"              # only used by data_source="csv"
    from_date: Optional[str] = None   # "2022-01-01", None = from the first bar
    to_date: Optional[str] = None
    atr_ref_bars: int = 24            # reference holding length used to derive the ATR window
    # ---- broker ----
    init_equity: float = 10000.0
    commission_pct: float = 0.05      # 0.05 = 0.05%, can't be 0
    leverage: float = 5.0
    # ---- capital isolation ----
    reserve_pct: float = 0.7
    restart_capital_pct: float = 0.3
    min_restart_capital_pct: float = 0.05
    restart_cost_pct: float = 0.0
    pause_days: int = 7
    # ---- profit sweep ----
    sweep_trigger_pct: float = 0.10
    compound_pct: float = 0.0
    sweep_min_interval_days: int = 0
    # ---- martingale grid ----
    base_order_pct: float = 0.02
    max_safety_orders: int = 8
    price_deviation_pct: float = 0.01
    step_mult: float = 1.2
    volume_mult: float = 1.6
    tp_pct: float = 0.01
    atr_grid_mult: Optional[float] = None
    atr_tp_mult: Optional[float] = None
    max_hold_bars: Optional[int] = None
    margin_usage_cap_pct: float = 0.9
    # ---- death rule ----
    death_equity_pct: float = 0.2
    cycle_stop_pct: Optional[float] = None
    # ---- entry ----
    entry_mode: str = "reversion"     # signal / long / short / reversion
    prob_thresh: Optional[float] = None
    allow_long: bool = True
    allow_short: bool = True


# Params forwarded as-is to MartingaleBtVenue (the venue passes them down to the strategy).
VENUE_PARAM_KEYS = (
    "init_equity",
    "reserve_pct", "restart_capital_pct", "min_restart_capital_pct",
    "restart_cost_pct", "pause_days",
    "sweep_trigger_pct", "compound_pct", "sweep_min_interval_days",
    "base_order_pct", "max_safety_orders", "price_deviation_pct",
    "step_mult", "volume_mult", "tp_pct",
    "atr_grid_mult", "atr_tp_mult", "max_hold_bars", "margin_usage_cap_pct",
    "death_equity_pct", "cycle_stop_pct",
    "entry_mode", "prob_thresh", "allow_long", "allow_short",
)


# ======================================================================
# Data loading
# ======================================================================
def load_klines(logger: logging.Logger, para: MartingalePara) -> pd.DataFrame:
    """Raw OHLCV klines; the martingale only needs price (+ATR when the grid is ATR driven)."""
    data_path = os.path.join(PROJECT_DATA_DIR, f"{para.symbol}_{para.interval}.csv")
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path, encoding="utf-8")
    if "open_time_date_utc" not in df.columns:
        logger.error("CRITICAL: 'open_time_date_utc' column missing.")
        sys.exit(1)
    df["open_time_date_utc"] = pd.to_datetime(df["open_time_date_utc"], utc=True)
    df["stop_loss_atr_pct"] = common.stop_loss_atr_pct(df, para.atr_ref_bars)
    logger.info(f"Loaded {len(df)} bars from {data_path}")
    return df


def load_predictions(logger: logging.Logger, para: MartingalePara, prep_output_dir: str,
                     train_output_dir: str, device) -> pd.DataFrame:
    """Prepared dataset + trained model, same pipeline as backtest_runner.main()."""
    import torch  # noqa: F401  (kept local so csv mode does not pay for the model stack)
    from model import model_loader, data_loader

    pre_para: BaseDefine = common.load_pre_params_from_dir(train_output_dir)
    df = common.load_test_df_from_dir(prep_output_dir)
    df["open_time_date_utc"] = pd.to_datetime(df["open_time_date_utc"], utc=True)
    interval_ms = common.get_interval_ms(pre_para.interval)

    train_out_path = train_output_dir
    if not os.path.isabs(train_out_path):
        train_out_path = os.path.join(PROJECT_DIR, train_out_path)
    handler = model_loader.ModelHandler(tarin_out_path=train_out_path, device=device)

    ds = data_loader.TimeSeriesWindowDataset(
        df=df,
        kline_interval_ms=interval_ms,
        feature_cols=handler.feature_cols,
        label_col=handler.label_col,
        seq_len=handler.seq_len,
        is_live=False,
    )
    df["stop_loss_atr_pct"] = common.stop_loss_atr_pct(df, para.atr_ref_bars)
    df_with_pred, model_stats = handler.predict_with_ds(ds, df, is_live=False, diff_thresh=None)

    # Drop the warm-up head (features not ready yet), keep the NaN holes in the middle.
    first_valid_idx = df_with_pred["pred"].first_valid_index()
    if first_valid_idx is None:
        logger.error("No valid predictions found in the entire dataset!")
        sys.exit(1)
    df_with_pred = df_with_pred.loc[first_valid_idx:].copy()
    logger.info(
        f"Model signals ready | input macro-F1={model_stats.get('f1_macro', float('nan')):.4f} "
        f"| range {df_with_pred['open_time_date_utc'].min()} -> {df_with_pred['open_time_date_utc'].max()}"
    )
    return df_with_pred


# ======================================================================
# Main
# ======================================================================
def main(logger: logging.Logger, para: MartingalePara = MartingalePara(),
         prep_output_dir: str = common.DATA_OUT_DIR, train_output_dir: str = common.TRAIN_OUT_DIR,
         device=None, save_path: Optional[str] = None):
    if para.entry_mode == "signal" and para.data_source != "model":
        logger.error("entry_mode='signal' needs data_source='model' (the csv feed carries no prediction).")
        sys.exit(1)

    logger.info(
        f"Martingale backtest | source={para.data_source} | entry_mode={para.entry_mode} "
        f"| reserve={para.reserve_pct:.0%} | safety_orders={para.max_safety_orders} "
        f"| commission={para.commission_pct}% | leverage={para.leverage}"
    )

    if para.data_source == "model":
        if device is None:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df = load_predictions(logger, para, prep_output_dir, train_output_dir, device)
    else:
        df = load_klines(logger, para)

    cerebro = bt.Cerebro(runonce=False, cheat_on_open=True, maxcpus=1)
    cerebro.addstrategy(
        MartingaleBtVenue,
        **{key: getattr(para, key) for key in VENUE_PARAM_KEYS},
    )

    data = PredictionFeed(
        dataname=df,
        datetime="open_time_date_utc",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        atr_pct="stop_loss_atr_pct",
        # pred / pred_prob / label / bars_to_close are auto-detected; missing ones stay NaN
        openinterest=-1,
        nocase=True,
        fromdate=pd.Timestamp(para.from_date).to_pydatetime() if para.from_date else None,
        todate=pd.Timestamp(para.to_date).to_pydatetime() if para.to_date else None,
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(para.init_equity)
    cerebro.broker.addcommissioninfo(
        cus_comminfo.CommInfo_Cryptocurrency(commission=para.commission_pct, leverage=para.leverage)
    )

    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days,
                        compression=1, annualize=True, factor=365)
    cerebro.addanalyzer(btanalyzers.Returns, _name="returns", tann=365)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(cus_analyzer.CusAnalyzer, _name="customize")

    logger.info("Starting Backtest...")
    results = cerebro.run()
    strat = results[0]

    save_path = save_path or os.path.join(TEMPORARY_DIR, "martingale_backtest_report.json")
    report_additional, report = generate_martingale_report(logger, strat, save_path=save_path, para=para)

    candles = df[["open_time_date_utc", "open", "high", "low", "close", "volume"]].copy()
    candles.rename(columns={"open_time_date_utc": "time"}, inplace=True)
    candles["time"] = candles["time"].apply(lambda dt: int(dt.timestamp()))
    return {"candles": candles.to_dict(orient="records"), "statistics": (report_additional, report)}


# ======================================================================
# Report
# ======================================================================
def generate_martingale_report(logger: logging.Logger, strat, save_path: str, para: MartingalePara):
    """
    Same report skeleton as backtest_runner.generate_backtest_report, minus the model
    sections, plus the martingale life-cycle block (deaths / restarts / swept profit).
    """
    perf = strat.analyzers.customize.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    ret_analyzer = strat.analyzers.returns.get_analysis()

    # ---- headline numbers ----
    start_value = strat.broker.startingcash
    end_value = strat.broker.getvalue()
    gross_return = (end_value - start_value) / start_value
    cagr = ret_analyzer.get("rnorm", 0.0)
    sr = sharpe.get("sharperatio", 0.0) or 0.0

    maxdd_pct = dd.get("max", {}).get("drawdown", 0.0)
    maxdd_amt = dd.get("max", {}).get("moneydown", 0.0)
    calmar = (cagr * 100 / abs(maxdd_pct)) if maxdd_pct > 0 else 0.0

    daily_returns_list = perf.get("daily_returns_list", [])
    rc_summary = summarize_rolling_calmar(
        rolling_calmar(build_daily_df(daily_returns_list), window_days=180, step_days=30)
    ) if daily_returns_list else {"rc_n": 0}

    # ---- worst days (the martingale's tail risk lives here) ----
    losses_values = []
    for item in daily_returns_list:
        val = item.get("dd_pct", 0) if isinstance(item, dict) else item
        if val < 0:
            losses_values.append(val)
    top_losses = sorted(losses_values)[:20]
    top_losses_str = " | ".join(f"{l*100:.2f}%" for l in top_losses) if top_losses else "N/A"
    if len(top_losses) > 1:
        robust_max_loss = sum(top_losses[1:]) / len(top_losses[1:])
    else:
        robust_max_loss = top_losses[0] if top_losses else 0.0

    # ---- trade statistics ----
    total_trades = safe_get(trades, ["total", "closed"], 0)
    total_won = safe_get(trades, ["won", "total"], 0)
    win_rate = (total_won / total_trades) if total_trades > 0 else 0.0
    gross_won_total = safe_get(trades, ["won", "pnl", "total"], 0.0)
    gross_lost_total = abs(safe_get(trades, ["lost", "pnl", "total"], 0.0))
    # A pure martingale can run a whole backtest without a single losing close,
    # so keep PF undefined instead of reporting a misleading 0.
    profit_factor = (gross_won_total / gross_lost_total) if gross_lost_total != 0 else None
    avg_pnl_net = safe_get(trades, ["pnl", "net", "average"], 0.0)
    avg_pnl_gross = safe_get(trades, ["pnl", "gross", "average"], 0.0)
    avg_cost = avg_pnl_gross - avg_pnl_net
    long_pnl_total = safe_get(trades, ["long", "pnl", "total"], 0.0)
    short_pnl_total = safe_get(trades, ["short", "pnl", "total"], 0.0)

    if len(strat.datas) > 0 and len(strat.datas[0]) > 0:
        t_start = bt.num2date(strat.datas[0].datetime.array[0])
        t_end = bt.num2date(strat.datas[0].datetime.array[-1])
        duration = t_end - t_start
        total_days = max(duration.days + (duration.seconds / 86400), 1)
    else:
        t_start = t_end = None
        total_days = 1
    daily_trades = total_trades / total_days

    # ---- strategy specific block: {'summary': scalars, 'detail': per-death records} ----
    strategy_stats = strat.strategy_metrics() if hasattr(strat, "strategy_metrics") else {}
    strategy_summary = strategy_stats.get("summary", {})
    strategy_detail = strategy_stats.get("detail", {})

    report = {
        "params": {
            "strategy": asdict(para),
            "git_commit": common.get_git_info(logger),
        },
        "time": {"start": t_start, "end": t_end},
        "performance": {
            "gross_return": gross_return,
            "cagr": cagr,
            "calmar": calmar,
            "sharpe": sr,
            "profit_factor": profit_factor,
            "start_value": start_value,
            "end_value": end_value,
            "rc_summary": rc_summary,
        },
        "drawdown": {
            "daily_loss_list": daily_returns_list,
            "max_dd_pct": maxdd_pct,
            "max_dd_amt": maxdd_amt,
            "max_daily_dd": perf.get("max_daily_dd", 0.0),
            "max_daily_date": perf.get("max_daily_dd_date", "N/A"),
            "robust_max_daily_loss": robust_max_loss,
            "dd_3_pct_days": perf.get("daily_dd_max_3_violation_days", 0),
            "dd_4_pct_days": perf.get("daily_dd_violation_days", 0),
            "dd_5_pct_days": perf.get("daily_dd_max_violation_days", 0),
            "max_hwm_duration_days": perf.get("max_hwm_duration_days", 0),
        },
        "exposure": {
            "avg_pos": perf.get("avg_pos_ratio", 0),
            "max_pos": perf.get("max_pos_ratio", 0),
            "p95_pos": perf.get("p95_pos_ratio", 0),
            "base_order_pct": para.base_order_pct,
        },
        "trades": {
            "total": total_trades,
            "daily_freq": daily_trades,
            "win_rate": win_rate,
            "avg_pnl_gross": avg_pnl_gross,
            "avg_pnl_net": avg_pnl_net,
            "avg_cost": avg_cost,
            "long_pnl": long_pnl_total,
            "short_pnl": short_pnl_total,
        },
        "strategy": strategy_summary,
        "trade_logs": list(getattr(strat, "trade_logs", [])),
    }
    report_additional = {
        "raw_analyzer": {"customize": perf},
        "strategy_detail": strategy_detail,
    }

    logger.info("-" * 80)
    logger.info(
        f"Time    | {report['time']['start']} --> {report['time']['end']} "
        f"| CAGR: {cagr*100:.2f}% | Calmar: {calmar:.2f}"
    )
    logger.info(
        f"SUMMARY | GrossRet: {gross_return*100:.2f}% | Sharpe: {sr:.3f} "
        f"| MaxDD: {maxdd_pct:.2f}% ({maxdd_amt:.0f}) "
        f"| PF: {'n/a' if profit_factor is None else format(profit_factor, '.2f')} "
        f"| end value {end_value:.2f}"
    )
    logger.info(f"RISK    | Top losses: [{top_losses_str}]")
    logger.info(
        f"RISK    | Robust max daily loss: {robust_max_loss*100:.2f}% "
        f"| Worst day: {report['drawdown']['max_daily_dd']*100:.2f}% "
        f"({report['drawdown']['max_daily_date']})"
    )
    logger.info(
        f"EXPOSURE| Avg Pos: {report['exposure']['avg_pos']*100:.2f}% "
        f"| Max Pos: {report['exposure']['max_pos']*100:.2f}% "
        f"| P95 Pos: {report['exposure']['p95_pos']*100:.2f}%"
    )
    logger.info(
        f"TRADES  | Total: {total_trades} | Freq: {daily_trades:.2f} trades/day "
        f"| WinRate: {win_rate*100:.2f}%"
    )
    logger.info(
        f"PNL($)  | Avg Gross: {avg_pnl_gross:.2f} | Avg Net: {avg_pnl_net:.2f} "
        f"(Cost: {avg_cost:.2f}/trade) | Long: {long_pnl_total:.2f} | Short: {short_pnl_total:.2f}"
    )
    log_metrics_block(logger, "STRATEGY", strategy_summary)
    logger.info("-" * 80)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({"report": report, "additional": report_additional}, f, indent=4, default=str)
    logger.info(f"Report saved to {save_path}")

    return report_additional, report


if __name__ == "__main__":
    para = MartingalePara()
    exp_dir = common.create_experiment_dir(
        os.path.join(common.PERSISTENCE_DIR, "martingale"), para.symbol, para.interval
    )
    logger, _ = common.setup_session_logger(
        log_file_path=os.path.join(exp_dir, "experiment.log"),
        console_level=logging.INFO, file_level=logging.DEBUG,
    )
    start_time = time.time()
    result = main(logger=logger, para=para, save_path=os.path.join(exp_dir, "report.json"))
    append_jsonl(os.path.join(exp_dir, "reports.jsonl"), result["statistics"][1])
    logger.info(f": run_time: {time.time() - start_time:.4f} s")
