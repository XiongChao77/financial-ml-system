"""Standalone backtest and Monte Carlo driver for the martingale simulator.

Reads raw OHLC straight from the canonical market-data CSV, replays it through
:mod:`trade.sim.martingale_engine` and aggregates the run statistics.  Nothing
here touches backtrader, the venue layer or the strategy layer.

Monte Carlo starts a fresh account at ``runs`` random bars and trades each one
until it is liquidated (or the data runs out), which is what makes the survival
distribution of a martingale visible instead of the single lucky path a plain
start-to-end backtest shows.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", ".."))

from data_process import common
from trade.sim.martingale_engine import (
    SIDE_NAME,
    AccountParams,
    MartingaleParams,
    MartingaleSimulator,
)


# ============================================================
# Configuration
# ============================================================
@dataclass(kw_only=True)
class DataParams(common.MarketDataSourceConfig):
    """Which raw OHLC file to read, and which slice of it to use."""

    from_date: Optional[str] = None
    to_date: Optional[str] = None
    root_dir: Optional[str] = None


@dataclass(frozen=True)
class MonteCarloParams:
    runs: int = 200
    seed: int = 42
    min_bars: int = 2_000          # bars that must remain ahead of a random start
    max_bars: Optional[int] = None  # None: run until ruin or end of data
    start_fraction: float = 0.30    # random starts are limited to the first 30%
    # Runs are fully independent, so they fan out over processes.  0 picks
    # cpu_count - 1, 1 stays in this process.  Start offsets are drawn in the
    # parent, so the result never depends on how many workers ran it.
    workers: int = 0

    def __post_init__(self):
        if not 0.0 < self.start_fraction <= 1.0:
            raise ValueError("start_fraction must be in (0, 1]")


@dataclass(frozen=True)
class CapitalParams:
    """How realized profits are handled during one Monte Carlo run.

    ``withdraw`` moves every positive closed-trade PnL to an isolated reserve,
    so the next grid continues with the same trading capital. ``compound``
    leaves profits in the trading account. ``both`` is accepted by the
    top-level driver and replays identical random starts once per mode.
    """

    profit_handling: str = "compound"  # "withdraw", "compound", or "both"
    stop_at_first_grid_breach: bool = False
    double_target_multiple: float = 2.0

    def __post_init__(self):
        if self.profit_handling not in ("withdraw", "compound", "both"):
            raise ValueError(
                "profit_handling must be 'withdraw', 'compound', or 'both'"
            )
        if self.double_target_multiple <= 1.0:
            raise ValueError("double_target_multiple must be > 1")


@dataclass(frozen=True)
class BacktestConfig:
    strategy: MartingaleParams = field(default_factory=MartingaleParams)
    account: AccountParams = field(default_factory=AccountParams)
    data: DataParams = field(default_factory=DataParams)
    monte_carlo: MonteCarloParams = field(default_factory=MonteCarloParams)
    capital: CapitalParams = field(default_factory=CapitalParams)
    gap_analysis: "GapAnalysisParams" = field(default_factory=lambda: GapAnalysisParams())
    save_path: Optional[str] = None
    trades_path: Optional[str] = None   # one row per closed trade
    fills_path: Optional[str] = None    # one row per filled layer
    plot_path: Optional[str] = None      # price chart with every failure marked
    pressure_path: Optional[str] = None  # time x window-scale pressure heatmap
    cluster_path: Optional[str] = None   # gap allowance x cluster-size heatmap
    pressure_min_scale: int = 8
    pressure_max_scale: Optional[int] = None
    pressure_scale_count: int = 40
    cluster_min_scale: int = 1
    cluster_max_scale: int = 5_000
    cluster_scale_count: int = 40


# ============================================================
# Data
# ============================================================
@dataclass
class BarSeries:
    """Column-wise OHLC: cheap to slice and to index inside the hot loop."""

    time: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    # Bar frequency is a property of the series, not of a single run: computed
    # once in load_bars instead of re-derived from the whole time column on
    # every Monte Carlo run.
    periods_per_year: float = 0.0

    def __len__(self) -> int:
        return len(self.open)


def _periods_per_year(times: np.ndarray) -> float:
    """Annualization factor from the median positive bar spacing."""
    if len(times) < 2:
        return 0.0
    deltas = np.diff(pd.to_datetime(times).astype("int64")) / 1e9
    positive = deltas[deltas > 0]
    if not len(positive):
        return 0.0
    median_seconds = float(np.median(positive))
    if median_seconds <= 0.0:
        return 0.0
    return 365.25 * 24.0 * 3600.0 / median_seconds


def load_bars(logger: logging.Logger, data: DataParams) -> BarSeries:
    """Read the canonical raw CSV and drop rows that are not valid candles."""
    path = common.market_data_path(data, root_dir=data.root_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    frame = pd.read_csv(path, encoding="utf-8")
    if "open_time_date_utc" not in frame.columns:
        raise ValueError("'open_time_date_utc' column missing from market data")
    frame["open_time_date_utc"] = pd.to_datetime(frame["open_time_date_utc"], utc=True)
    frame = frame.sort_values("open_time_date_utc")

    if data.from_date:
        frame = frame[frame["open_time_date_utc"] >= pd.Timestamp(data.from_date, tz="UTC")]
    if data.to_date:
        frame = frame[frame["open_time_date_utc"] <= pd.Timestamp(data.to_date, tz="UTC")]

    raw_rows = len(frame)
    ohlc = frame[["open", "high", "low", "close"]].astype(float)
    valid = (
        ohlc.notna().all(axis=1)
        & (ohlc > 0.0).all(axis=1)
        & (ohlc["high"] >= ohlc[["open", "low", "close"]].max(axis=1))
        & (ohlc["low"] <= ohlc[["open", "high", "close"]].min(axis=1))
    )
    frame = frame[valid]
    if not len(frame):
        raise ValueError(f"No usable bars left after filtering {path}")
    times = frame["open_time_date_utc"].to_numpy()
    periods_per_year = _periods_per_year(times)
    logger.info(
        f"Loaded {len(frame)} bars from {path} "
        f"({raw_rows - len(frame)} malformed rows dropped) | "
        f"{frame['open_time_date_utc'].iloc[0]} -> {frame['open_time_date_utc'].iloc[-1]} "
        f"| {periods_per_year:.0f} bars/year"
    )
    return BarSeries(
        time=times,
        open=frame["open"].to_numpy(dtype=float),
        high=frame["high"].to_numpy(dtype=float),
        low=frame["low"].to_numpy(dtype=float),
        close=frame["close"].to_numpy(dtype=float),
        periods_per_year=periods_per_year,
    )


# ============================================================
# One run
# ============================================================
def _elapsed_years(start_time, end_time) -> float:
    seconds = (pd.Timestamp(end_time) - pd.Timestamp(start_time)).total_seconds()
    return max(seconds / (365.25 * 24.0 * 3600.0), 0.0)


# Annualizing a window shorter than a day is meaningless and overflows: on 1m
# bars a 2.6x peak reached in one bar compounds to ~1e17 or straight to inf.
MIN_CAGR_YEARS = 1.0 / 365.25


def _cagr(start_value: float, end_value: float, years: float) -> Optional[float]:
    if start_value <= 0.0 or end_value < 0.0 or years < MIN_CAGR_YEARS:
        return None
    if end_value == 0.0:
        return -1.0
    rate = (end_value / start_value) ** (1.0 / years) - 1.0
    return float(rate) if np.isfinite(rate) else None


def _annualized_sharpe(equity_curve: Sequence[float], periods_per_year: float) -> Optional[float]:
    values = np.asarray(equity_curve, dtype=float)
    if len(values) < 3 or periods_per_year <= 0.0:
        return None
    previous = values[:-1]
    current = values[1:]
    valid = (previous > 0.0) & np.isfinite(previous) & np.isfinite(current)
    if valid.sum() < 2:
        return None
    returns = current[valid] / previous[valid] - 1.0
    std = float(np.std(returns, ddof=1))
    if std <= 0.0 or not np.isfinite(std):
        return None
    return float(np.mean(returns) / std * np.sqrt(periods_per_year))


def run_once(
    bars: BarSeries,
    config: BacktestConfig,
    *,
    start_index: int = 0,
    max_bars: Optional[int] = None,
) -> dict:
    """Trade until failure, limit, data end, or the configured first breach.

    A breach is the first full-layer stop, liquidation, ruin threshold, or grid
    that cannot be opened with the available margin.
    """
    total_bars = len(bars)
    if not 0 <= start_index < total_bars:
        raise ValueError(f"start_index {start_index} outside 0..{total_bars - 1}")

    account = config.account
    capital = config.capital
    if capital.profit_handling == "both":
        raise ValueError(
            "run_once needs one profit_handling mode; use main() to compare both"
        )
    simulator = MartingaleSimulator(config.strategy, account)
    state = simulator.new_state()
    initial_equity = account.initial_equity
    end_index = total_bars if max_bars is None else min(total_bars, start_index + max_bars)
    periods_per_year = bars.periods_per_year

    peak_equity = initial_equity
    peak_index = start_index
    peak_time = bars.time[start_index]
    equity_curve = [initial_equity]
    exit_reason = "end_of_data"
    index = start_index
    processed_trades = 0
    reserve_balance = 0.0
    first_grid_breach_reason: Optional[str] = None
    first_grid_breach_index: Optional[int] = None
    first_grid_breach_time = None
    withdrawn_profit_before_breach: Optional[float] = None
    doubled_before_grid_breach = False
    double_index: Optional[int] = None
    double_time = None
    double_balance: Optional[float] = None

    for index in range(start_index, end_index):
        previous_peak = state.max_equity_seen
        state = simulator.process_bar(
            state,
            index=index,
            time=bars.time[index],
            open_=bars.open[index],
            high=bars.high[index],
            low=bars.low[index],
            close=bars.close[index],
        )

        new_trades = state.trades[processed_trades:]
        processed_trades = len(state.trades)
        if capital.profit_handling == "withdraw":
            for trade in new_trades:
                withdrawn = max(float(trade["net_pnl"]), 0.0)
                if withdrawn:
                    # The reserve is outside the trading account: it cannot
                    # size the next ladder or be lost by the active grid.
                    state.equity = max(0.0, state.equity - withdrawn)
                    reserve_balance += withdrawn
                trade["withdrawn_profit"] = withdrawn
                trade["reserve_after"] = reserve_balance
                trade["trading_balance_after_withdrawal"] = state.equity
        else:
            for trade in new_trades:
                trade["withdrawn_profit"] = 0.0
                trade["reserve_after"] = 0.0
                trade["trading_balance_after_withdrawal"] = state.equity

        if (
            capital.profit_handling == "compound"
            and first_grid_breach_reason is None
            and not doubled_before_grid_breach
            and state.max_balance_seen
            >= initial_equity * capital.double_target_multiple - 1e-12
        ):
            doubled_before_grid_breach = True
            double_index = index
            double_time = bars.time[index]
            double_balance = float(state.max_balance_seen)

        breach_reason = next(
            (trade["reason"] for trade in new_trades if trade["reason"] == "stop_loss"),
            None,
        )
        if breach_reason is None:
            if state.bankrupt:
                breach_reason = "liquidation"
            elif state.ruin_threshold_hit:
                breach_reason = "ruin_threshold"
            elif state.grid_infeasible:
                breach_reason = "grid_infeasible"
        if breach_reason is not None and first_grid_breach_reason is None:
            first_grid_breach_reason = breach_reason
            first_grid_breach_index = index
            first_grid_breach_time = bars.time[index]
            withdrawn_profit_before_breach = reserve_balance

        if state.max_equity_seen > peak_equity + 1e-12:
            peak_equity = state.max_equity_seen
            peak_index = index
            peak_time = bars.time[index]
        elif state.max_equity_seen > previous_peak + 1e-12:
            peak_equity = max(peak_equity, state.max_equity_seen)

        if state.bankrupt:
            current_equity = float(state.failure_equity if state.failure_equity is not None else state.equity)
            exit_reason = "liquidation"
        elif state.ruin_threshold_hit:
            current_equity = float(
                state.failure_equity
                if state.failure_equity is not None
                else initial_equity * account.ruin_equity_pct
            )
            exit_reason = "ruin_threshold"
        elif state.grid_infeasible:
            current_equity = float(
                state.failure_equity
                if state.failure_equity is not None
                else simulator.equity_at(state, bars.close[index])
            )
            exit_reason = "grid_infeasible"
        else:
            current_equity = simulator.equity_at(state, bars.close[index])

        equity_curve.append(max(current_equity, 0.0))
        if state.failed:
            break
        if breach_reason is not None and capital.stop_at_first_grid_breach:
            exit_reason = "grid_breach"
            break

    final_equity = equity_curve[-1]
    end_time = bars.time[index]
    elapsed_years = _elapsed_years(bars.time[start_index], end_time)
    peak_years = _elapsed_years(bars.time[start_index], peak_time)
    max_drawdown = state.max_drawdown_seen
    trades = state.trades
    exit_counts: dict[str, int] = {}
    for trade in trades:
        exit_counts[trade["reason"]] = exit_counts.get(trade["reason"], 0) + 1

    profitable = sum(trade["net_pnl"] > 0.0 for trade in trades)
    stop_losses = [trade for trade in trades if trade["reason"] == "stop_loss"]
    if withdrawn_profit_before_breach is None:
        withdrawn_profit_before_breach = reserve_balance
    summary = {
        "start_index": start_index,
        "start_time": bars.time[start_index],
        "end_index": index,
        "end_time": end_time,
        "bars_survived": index - start_index + 1,
        "elapsed_years": elapsed_years,
        "initial_equity": initial_equity,
        "final_equity": final_equity,
        "final_total_wealth": final_equity + reserve_balance,
        "profit_handling": capital.profit_handling,
        "reserve_balance": reserve_balance,
        "withdrawn_profit_before_grid_breach": withdrawn_profit_before_breach,
        "grid_breached": first_grid_breach_reason is not None,
        "grid_breach_reason": first_grid_breach_reason,
        "grid_breach_index": first_grid_breach_index,
        "grid_breach_time": first_grid_breach_time,
        "trades_before_grid_breach": (
            len(trades)
            if first_grid_breach_index is None
            else sum(
                trade["exit_bar"] <= first_grid_breach_index for trade in trades
            )
        ),
        "doubled_before_grid_breach": doubled_before_grid_breach,
        "double_target_multiple": capital.double_target_multiple,
        "double_index": double_index,
        "double_time": double_time,
        "double_balance": double_balance,
        "bars_to_double": (
            None if double_index is None else double_index - start_index + 1
        ),
        "min_equity": float(state.min_equity_seen),
        "peak_equity": float(peak_equity),
        "peak_realized_balance": float(state.max_balance_seen),
        "peak_equity_multiple": float(peak_equity / initial_equity),
        "peak_return_pct": float(peak_equity / initial_equity - 1.0),
        "peak_index": peak_index,
        "peak_time": peak_time,
        "bars_to_peak": peak_index - start_index + 1,
        "years_to_peak": peak_years,
        "peak_cagr": _cagr(initial_equity, peak_equity, peak_years),
        "total_return_pct": (final_equity - initial_equity) / initial_equity,
        "cagr": _cagr(initial_equity, final_equity, elapsed_years),
        "sharpe": _annualized_sharpe(equity_curve, periods_per_year),
        "periods_per_year": periods_per_year,
        "max_drawdown_pct": float(max_drawdown),
        "max_margin_usage": float(state.max_margin_usage_seen),
        "exit_reason": exit_reason,
        "ruined": state.failed,
        "failure_price": state.failure_price,
        "total_trades": len(trades),
        "profitable_trades": int(profitable),
        "losing_trades": int(len(trades) - profitable),
        "win_rate": float(profitable / len(trades)) if trades else 0.0,
        "total_net_profit_abs": sum(trade["net_pnl"] for trade in trades),
        "total_fees": sum(trade["fees"] for trade in trades),
        "stop_loss_trades": len(stop_losses),
        "stop_loss_net_pnl": sum(trade["net_pnl"] for trade in stop_losses),
        "same_bar_trades": int(sum(trade["same_bar"] for trade in trades)),
        "multiple_fills_same_bar_trades": int(sum(
            trade["multiple_fills_same_bar"] for trade in trades
        )),
        "max_layers_seen": max(
            max((trade["layers"] for trade in trades), default=0),
            state.cycle.layers,
        ),
        "open_layers_at_end": state.cycle.layers,
        "unfilled_layers": state.unfilled_layers,
        "cooldown_blocked_bars": state.cooldown_blocked_bars,
        "reversals": state.reversals,
        "final_direction": SIDE_NAME[state.next_side],
        "dual_path_bars": state.dual_path_bars,
        "up_first_worse_bars": state.up_first_worse_bars,
        "exit_counts": exit_counts,
    }
    return {"summary": summary, "trades": trades, "layer_stats": layer_statistics(trades)}


def direction_statistics(trades: Sequence[dict]) -> dict:
    """Split the trades by the side the cycle was opened on.

    With ``reverse_after_stop_loss`` on, this is what says whether following
    the break actually paid: compare the two sides' win rate and net PnL.
    """
    buckets: dict[str, dict] = {}
    for trade in trades:
        bucket = buckets.setdefault(
            trade.get("direction", "long"),
            {"trades": 0, "profitable_trades": 0, "entry_notional": 0.0,
             "net_profit_abs": 0.0, "fees": 0.0, "profit_vs_balance": 0.0,
             "stop_losses": 0},
        )
        bucket["trades"] += 1
        bucket["profitable_trades"] += int(trade["net_pnl"] > 0.0)
        bucket["entry_notional"] += trade["entry_notional"]
        bucket["net_profit_abs"] += trade["net_pnl"]
        bucket["fees"] += trade["fees"]
        bucket["profit_vs_balance"] += trade.get("profit_vs_start_balance", 0.0)
        bucket["stop_losses"] += int(trade["reason"] == "stop_loss")

    total = len(trades)
    for bucket in buckets.values():
        bucket["trade_pct"] = bucket["trades"] / total if total else 0.0
        bucket["win_rate"] = bucket["profitable_trades"] / bucket["trades"]
        bucket["stop_loss_pct"] = bucket["stop_losses"] / bucket["trades"]
        bucket["net_profit_pct"] = (
            bucket["net_profit_abs"] / bucket["entry_notional"]
            if bucket["entry_notional"] else 0.0
        )
        bucket["avg_net_profit_abs"] = bucket["net_profit_abs"] / bucket["trades"]
        bucket["avg_profit_vs_balance"] = bucket["profit_vs_balance"] / bucket["trades"]
    return {name: buckets[name] for name in sorted(buckets)}


def layer_statistics(trades: Sequence[dict]) -> dict:
    """Report terminal PnL and take-profit probability at every reached layer.

    ``take_profit_pct`` uses every cycle that reached the layer as its
    denominator. Its complement is simply "did not take profit at this layer".
    """

    def new_bucket() -> dict:
        return {
            # Cycles whose terminal exit happened at exactly this layer.
            "trades": 0,
            "profitable_trades": 0,
            "entry_notional": 0.0,
            "net_profit_abs": 0.0,
            "fees": 0.0,
            "profit_vs_balance": 0.0,
            "same_bar_trades": 0,
            "liquidations": 0,
            "stop_losses": 0,
            "stop_loss_net_pnl": 0.0,
            # Cycles that reached this layer and what happened next.
            "reached_trades": 0,
            "take_profits_at_layer": 0,
        }

    buckets: dict[str, dict] = {}
    for trade in trades:
        final_layer = int(trade["layers"])
        terminal = buckets.setdefault(str(final_layer), new_bucket())
        terminal["trades"] += 1
        terminal["profitable_trades"] += int(trade["net_pnl"] > 0.0)
        terminal["entry_notional"] += trade["entry_notional"]
        terminal["net_profit_abs"] += trade["net_pnl"]
        terminal["fees"] += trade["fees"]
        terminal["profit_vs_balance"] += trade.get("profit_vs_start_balance", 0.0)
        terminal["same_bar_trades"] += int(trade["same_bar"])
        terminal["liquidations"] += int(trade["reason"] == "liquidation")
        if trade["reason"] == "stop_loss":
            terminal["stop_losses"] += 1
            terminal["stop_loss_net_pnl"] += trade["net_pnl"]

        for reached_layer in range(1, final_layer + 1):
            bucket = buckets.setdefault(str(reached_layer), new_bucket())
            bucket["reached_trades"] += 1
            if reached_layer == final_layer and trade["reason"] == "take_profit":
                bucket["take_profits_at_layer"] += 1

    total = len(trades)
    for bucket in buckets.values():
        closed = bucket["trades"]
        reached = bucket["reached_trades"]
        bucket["trade_pct"] = closed / total if total else 0.0
        bucket["reach_pct"] = reached / total if total else 0.0
        bucket["win_rate"] = (
            bucket["profitable_trades"] / closed if closed else 0.0
        )
        bucket["take_profit_pct"] = (
            bucket["take_profits_at_layer"] / reached if reached else 0.0
        )
        bucket["same_bar_pct"] = (
            bucket["same_bar_trades"] / closed if closed else 0.0
        )
        bucket["net_profit_pct"] = (
            bucket["net_profit_abs"] / bucket["entry_notional"]
            if bucket["entry_notional"]
            else 0.0
        )
        bucket["avg_net_profit_abs"] = (
            bucket["net_profit_abs"] / closed if closed else 0.0
        )
        bucket["avg_profit_vs_balance"] = (
            bucket["profit_vs_balance"] / closed if closed else 0.0
        )
        bucket["avg_stop_loss_abs"] = (
            bucket["stop_loss_net_pnl"] / bucket["stop_losses"]
            if bucket["stop_losses"]
            else 0.0
        )
    return {layer: buckets[layer] for layer in sorted(buckets, key=int)}


# ============================================================
# Monte Carlo
# ============================================================
def _percentiles(values: Sequence[float], points=(5, 25, 50, 75, 95)) -> dict:
    if not len(values):
        return {f"p{point}": 0.0 for point in points}
    array = np.asarray(values, dtype=float)
    return {f"p{point}": float(np.percentile(array, point)) for point in points}


def _distribution(values: Sequence[Optional[float]]) -> dict:
    clean = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=float,
    )
    if not len(clean):
        return {"count": 0, "mean": None, "p5": None, "p25": None,
                "p50": None, "p75": None, "p95": None}
    result = {"count": int(len(clean)), "mean": float(np.mean(clean))}
    result.update(_percentiles(clean))
    return result


def _merge_layer_stats(runs: Sequence[dict]) -> dict:
    """Pool every run's trades into one layer distribution."""
    return layer_statistics([trade for run in runs for trade in run["trades"]])


_WORKER_BARS: Optional[BarSeries] = None
_WORKER_CONFIG: Optional[BacktestConfig] = None


def _worker_init(bars: BarSeries, config: BacktestConfig):
    """Hand each process the bars and the config once, not once per run."""
    global _WORKER_BARS, _WORKER_CONFIG
    _WORKER_BARS = bars
    _WORKER_CONFIG = config


def _tag_run(result: dict, run_id: int, start_index: int) -> dict:
    result["summary"]["run_id"] = run_id
    for trade in result["trades"]:
        trade["run_id"] = run_id
        trade["start_index"] = start_index
    return result


def _worker_run(task) -> dict:
    run_id, start_index = task
    return _tag_run(
        run_once(
            _WORKER_BARS,
            _WORKER_CONFIG,
            start_index=start_index,
            max_bars=_WORKER_CONFIG.monte_carlo.max_bars,
        ),
        run_id,
        start_index,
    )


def resolve_workers(requested: int, tasks: int) -> int:
    if requested and requested > 0:
        return max(1, min(requested, tasks))
    return max(1, min((os.cpu_count() or 1) - 1, tasks))


# ============================================================
# Do the losses cluster, or are they scattered at random?
# ============================================================
@dataclass(frozen=True)
class GapAnalysisParams:
    """How the inter-loss spacing is measured and what it is compared against."""

    window_bars: int = 500    # window width for the count dispersion (Fano) test
    short_gap_bars: int = 10  # "another loss almost immediately" threshold
    repeats: int = 20         # random baselines drawn per run
    seed: int = 17


def _point_process_stats(
    positions: np.ndarray, span: int, params: GapAnalysisParams
) -> Optional[dict]:
    """Spacing statistics of one run's loss positions, in bars.

    The reference is a homogeneous random process over the same span with the
    same number of losses, for which gap CV == 1, lag-1 autocorrelation == 0
    and the Fano factor == 1.  Values above those mean the losses arrive in
    bursts; below means they are more evenly spaced than chance.
    """
    positions = np.sort(np.asarray(positions, dtype=float))
    if len(positions) < 3 or span <= 1:
        return None
    gaps = np.diff(positions)
    mean_gap = float(gaps.mean())
    if mean_gap <= 0.0:
        return None

    stats = {
        "count": int(len(positions)),
        "span": int(span),
        "gap_mean": mean_gap,
        "gap_cv": float(gaps.std(ddof=1) / mean_gap) if len(gaps) > 1 else 0.0,
        "short_gap_pct": float((gaps <= params.short_gap_bars).mean()),
        "normalized_gaps": gaps / mean_gap,
    }

    if len(gaps) >= 4:
        first, second = gaps[:-1], gaps[1:]
        if first.std() > 0.0 and second.std() > 0.0:
            stats["gap_lag1"] = float(np.corrcoef(first, second)[0, 1])

    # Count dispersion needs several windows to mean anything.
    if span >= params.window_bars * 3:
        counts, _ = np.histogram(
            positions, bins=np.arange(0, span + params.window_bars, params.window_bars)
        )
        mean_count = counts.mean()
        if mean_count > 0.0:
            stats["fano"] = float(counts.var(ddof=1) / mean_count)
    return stats


def _pool_point_process(entries: Sequence[dict]) -> dict:
    """Average the per-run measures and pool the rate-normalized gaps."""
    if not entries:
        return {}
    normalized = np.concatenate([entry["normalized_gaps"] for entry in entries])
    pooled = {
        "runs": len(entries),
        "losses": int(sum(entry["count"] for entry in entries)),
        # Gaps are normalized by each run's own mean before pooling: runs have
        # different loss rates, and mixing those would inflate the CV on its own
        # and fake a clustering signal.
        "gap_cv_pooled": float(normalized.std(ddof=1) / normalized.mean()),
        "gap_mean_bars": float(np.mean([entry["gap_mean"] for entry in entries])),
        "gap_cv": float(np.mean([entry["gap_cv"] for entry in entries])),
        "short_gap_pct": float(np.mean([entry["short_gap_pct"] for entry in entries])),
    }
    for key in ("gap_lag1", "fano"):
        values = [entry[key] for entry in entries if key in entry]
        pooled[key] = float(np.mean(values)) if values else None
    pooled["gap_percentiles"] = _percentiles(
        np.concatenate([entry["normalized_gaps"] * entry["gap_mean"] for entry in entries])
    )
    return pooled


def _ratio(observed: Optional[float], reference: Optional[float]) -> Optional[float]:
    if observed is None or reference is None or reference == 0.0:
        return None
    return float(observed / reference)


def loss_gap_statistics(
    summaries: Sequence[dict],
    trades: Sequence[dict],
    params: GapAnalysisParams = GapAnalysisParams(),
) -> dict:
    """Compare the spacing of losing trades against a random null of the same rate."""
    spans = {summary["run_id"]: summary["bars_survived"] for summary in summaries}
    positions_by_run: dict[int, list] = {}
    for trade in trades:
        if trade["net_pnl"] < 0.0:
            positions_by_run.setdefault(trade["run_id"], []).append(
                trade["exit_bar"] - trade["start_index"]
            )

    rng = np.random.default_rng(params.seed)
    observed_entries, random_entries = [], []
    skipped = 0
    for run_id, positions in positions_by_run.items():
        span = spans.get(run_id)
        if span is None:
            continue
        observed = _point_process_stats(np.asarray(positions), span, params)
        if observed is None:
            skipped += 1
            continue
        observed_entries.append(observed)
        # Same span, same number of losses, thrown down uniformly at random.
        for _ in range(params.repeats):
            draw = _point_process_stats(rng.random(len(positions)) * span, span, params)
            if draw is not None:
                random_entries.append(draw)

    observed = _pool_point_process(observed_entries)
    reference = _pool_point_process(random_entries)
    ratios = {
        key: _ratio(observed.get(key), reference.get(key))
        for key in ("gap_cv_pooled", "gap_cv", "fano", "short_gap_pct")
    }
    lag1 = observed.get("gap_lag1")

    dispersion = [value for value in (ratios["gap_cv_pooled"], ratios["fano"]) if value]
    strongest = max(dispersion) if dispersion else 1.0
    if strongest >= 1.2 or (lag1 is not None and lag1 >= 0.15):
        verdict = "clustered"
    elif strongest <= 0.85:
        verdict = "more regular than random"
    else:
        verdict = "indistinguishable from random"

    return {
        "params": asdict(params),
        "runs_analyzed": len(observed_entries),
        "runs_skipped": skipped,
        "observed": observed,
        "random": reference,
        "ratios": ratios,
        "gap_lag1_excess": (
            None if lag1 is None or reference.get("gap_lag1") is None
            else float(lag1 - reference["gap_lag1"])
        ),
        "verdict": verdict,
    }


def run_monte_carlo(
    logger: logging.Logger,
    bars: BarSeries,
    config: BacktestConfig,
) -> dict:
    if config.capital.profit_handling == "both":
        raise ValueError(
            "run_monte_carlo needs one profit_handling mode; use main() to compare both"
        )
    monte_carlo = config.monte_carlo
    available_last_start = len(bars) - monte_carlo.min_bars
    if available_last_start < 0:
        raise ValueError(
            f"Only {len(bars)} bars available, need at least "
            f"min_bars={monte_carlo.min_bars}"
        )

    fraction_last_start = max(
        0, int(len(bars) * monte_carlo.start_fraction) - 1
    )
    last_start = min(available_last_start, fraction_last_start)
    logger.info(
        f"Monte Carlo start range | 0..{last_start} "
        f"(first {monte_carlo.start_fraction:.0%} of {len(bars)} bars)"
    )

    # Drawn here, so a 1-worker run and a 16-worker run replay the same starts.
    rng = random.Random(monte_carlo.seed)
    tasks = [
        (run_id, rng.randint(0, last_start)) for run_id in range(monte_carlo.runs)
    ]
    workers = resolve_workers(monte_carlo.workers, len(tasks))
    report_every = max(1, monte_carlo.runs // 10)
    runs: List[dict] = []

    def progress():
        if len(runs) % report_every == 0 or len(runs) == len(tasks):
            ruined = sum(run["summary"]["ruined"] for run in runs)
            logger.info(
                f"Monte Carlo | {len(runs)}/{len(tasks)} runs "
                f"| ruined so far: {ruined}"
            )

    if workers == 1:
        _worker_init(bars, config)
        for task in tasks:
            runs.append(_worker_run(task))
            progress()
    else:
        logger.info(f"Monte Carlo | {len(tasks)} runs over {workers} processes")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(bars, config),
        ) as pool:
            futures = [pool.submit(_worker_run, task) for task in tasks]
            for future in as_completed(futures):
                runs.append(future.result())
                progress()
        runs.sort(key=lambda run: run["summary"]["run_id"])

    summaries = [run["summary"] for run in runs]
    ruined = [s for s in summaries if s["ruined"]]
    breached = [s for s in summaries if s["grid_breached"]]
    doubled = [s for s in summaries if s["doubled_before_grid_breach"]]
    all_trades = [trade for run in runs for trade in run["trades"]]

    total_trades = len(all_trades)
    profitable = sum(trade["net_pnl"] > 0.0 for trade in all_trades)
    same_bar = sum(trade["same_bar"] for trade in all_trades)
    multi_fill = sum(trade["multiple_fills_same_bar"] for trade in all_trades)
    net_profit = sum(trade["net_pnl"] for trade in all_trades)
    fees = sum(trade["fees"] for trade in all_trades)
    entry_notional = sum(trade["entry_notional"] for trade in all_trades)

    exit_counts: dict[str, int] = {}
    for trade in all_trades:
        exit_counts[trade["reason"]] = exit_counts.get(trade["reason"], 0) + 1

    # A full-layer stop trades ruin for a large, recurring loss.  The point of
    # these numbers is the exchange rate between the two: how many take-profit
    # cycles one stop gives back, and how often it fires.
    stop_trades = [trade for trade in all_trades if trade["reason"] == "stop_loss"]
    winners = [trade for trade in all_trades if trade["net_pnl"] > 0.0]
    stop_net = sum(trade["net_pnl"] for trade in stop_trades)
    avg_stop = stop_net / len(stop_trades) if stop_trades else 0.0
    avg_win = (
        sum(trade["net_pnl"] for trade in winners) / len(winners) if winners else 0.0
    )

    aggregate = {
        "runs": len(summaries),
        "profit_handling": config.capital.profit_handling,
        "stop_at_first_grid_breach": config.capital.stop_at_first_grid_breach,
        "grid_breached_runs": len(breached),
        "grid_breach_rate": len(breached) / len(summaries) if summaries else 0.0,
        "grid_breach_reasons": {
            reason: sum(s["grid_breach_reason"] == reason for s in breached)
            for reason in sorted({s["grid_breach_reason"] for s in breached})
        },
        "withdrawn_profit_before_grid_breach_mean": float(np.mean([
            s["withdrawn_profit_before_grid_breach"] for s in summaries
        ])),
        "withdrawn_profit_before_grid_breach": _percentiles([
            s["withdrawn_profit_before_grid_breach"] for s in summaries
        ]),
        "withdrawn_profit_on_breached_runs": _distribution([
            s["withdrawn_profit_before_grid_breach"] for s in breached
        ]),
        "doubled_before_grid_breach_runs": len(doubled),
        "double_before_grid_breach_rate": (
            len(doubled) / len(summaries) if summaries else 0.0
        ),
        "double_before_grid_breach_rate_on_breached_runs": (
            sum(s["doubled_before_grid_breach"] for s in breached) / len(breached)
            if breached else 0.0
        ),
        "bars_to_double": _distribution([s["bars_to_double"] for s in doubled]),
        "ruined_runs": len(ruined),
        "ruin_rate": len(ruined) / len(summaries) if summaries else 0.0,
        "liquidation_runs": sum(s["exit_reason"] == "liquidation" for s in summaries),
        "ruin_threshold_runs": sum(s["exit_reason"] == "ruin_threshold" for s in summaries),
        "grid_infeasible_runs": sum(s["exit_reason"] == "grid_infeasible" for s in summaries),
        "survived_runs": len(summaries) - len(ruined),
        "bars_survived_mean": float(np.mean([s["bars_survived"] for s in summaries])),
        "bars_survived": _percentiles([s["bars_survived"] for s in summaries]),
        "bars_to_ruin": _percentiles([s["bars_survived"] for s in ruined]),
        "elapsed_years_mean": float(np.mean([s["elapsed_years"] for s in summaries])),
        "elapsed_years": _percentiles([s["elapsed_years"] for s in summaries]),
        "total_return_pct_mean": float(np.mean([s["total_return_pct"] for s in summaries])),
        "total_return_pct": _percentiles([s["total_return_pct"] for s in summaries]),
        "max_drawdown_pct_mean": float(np.mean([s["max_drawdown_pct"] for s in summaries])),
        "max_margin_usage_mean": float(np.mean([s["max_margin_usage"] for s in summaries])),
        "total_trades": total_trades,
        "trades_per_run": total_trades / len(summaries) if summaries else 0.0,
        "trades_per_run_percentiles": _percentiles([s["total_trades"] for s in summaries]),
        "profitable_trades": profitable,
        "losing_trades": total_trades - profitable,
        "win_rate": profitable / total_trades if total_trades else 0.0,
        "total_net_profit_abs": net_profit,
        "total_fees_abs": fees,
        "total_entry_notional": entry_notional,
        "net_profit_pct_of_notional": net_profit / entry_notional if entry_notional else 0.0,
        "fees_pct_of_notional": fees / entry_notional if entry_notional else 0.0,
        "avg_net_profit_abs_per_trade": net_profit / total_trades if total_trades else 0.0,
        # --- stop loss ---
        "stop_loss_trades": len(stop_trades),
        "stop_loss_pct_of_trades": len(stop_trades) / total_trades if total_trades else 0.0,
        "stop_loss_runs": sum(s["stop_loss_trades"] > 0 for s in summaries),
        "stop_losses_per_run": len(stop_trades) / len(summaries) if summaries else 0.0,
        "stop_loss_net_pnl_abs": stop_net,
        "avg_stop_loss_abs": avg_stop,
        "avg_stop_loss_pct_of_balance": (
            float(np.mean([trade["return_on_balance"] for trade in stop_trades]))
            if stop_trades else 0.0
        ),
        "avg_win_abs": avg_win,
        "wins_per_stop_loss": abs(avg_stop) / avg_win if avg_win > 0.0 and stop_trades else 0.0,
        "same_bar_trades": same_bar,
        "same_bar_trade_pct": same_bar / total_trades if total_trades else 0.0,
        "multiple_fills_same_bar_trades": multi_fill,
        "multiple_fills_same_bar_pct": multi_fill / total_trades if total_trades else 0.0,
        "max_layers_seen": max((s["max_layers_seen"] for s in summaries), default=0),
        "unfilled_layers": sum(s["unfilled_layers"] for s in summaries),
        "cooldown_blocked_bars": sum(s["cooldown_blocked_bars"] for s in summaries),
        "reversals": sum(s["reversals"] for s in summaries),
        "dual_path_bars": sum(s["dual_path_bars"] for s in summaries),
        "up_first_worse_bars": sum(s["up_first_worse_bars"] for s in summaries),
        "exit_counts": exit_counts,
    }
    aggregate["up_first_worse_pct"] = (
        aggregate["up_first_worse_bars"] / aggregate["dual_path_bars"]
        if aggregate["dual_path_bars"]
        else 0.0
    )
    aggregate["distributions"] = {
        "final_equity": _distribution([s["final_equity"] for s in summaries]),
        "final_total_wealth": _distribution([
            s["final_total_wealth"] for s in summaries
        ]),
        "reserve_balance": _distribution([
            s["reserve_balance"] for s in summaries
        ]),
        "peak_equity": _distribution([s["peak_equity"] for s in summaries]),
        "peak_realized_balance": _distribution([s["peak_realized_balance"] for s in summaries]),
        "peak_equity_multiple": _distribution([s["peak_equity_multiple"] for s in summaries]),
        "peak_return_pct": _distribution([s["peak_return_pct"] for s in summaries]),
        "bars_to_peak": _distribution([s["bars_to_peak"] for s in summaries]),
        "years_to_peak": _distribution([s["years_to_peak"] for s in summaries]),
        "total_return_pct": _distribution([s["total_return_pct"] for s in summaries]),
        "cagr": _distribution([s["cagr"] for s in summaries]),
        "peak_cagr": _distribution([s["peak_cagr"] for s in summaries]),
        "sharpe": _distribution([s["sharpe"] for s in summaries]),
        "max_drawdown_pct": _distribution([s["max_drawdown_pct"] for s in summaries]),
        "max_margin_usage": _distribution([s["max_margin_usage"] for s in summaries]),
        "bars_survived": _distribution([s["bars_survived"] for s in summaries]),
        "elapsed_years": _distribution([s["elapsed_years"] for s in summaries]),
        "total_trades": _distribution([s["total_trades"] for s in summaries]),
        "stop_loss_trades": _distribution([s["stop_loss_trades"] for s in summaries]),
    }
    return {
        "aggregate": aggregate,
        "direction_stats": direction_statistics(all_trades),
        "layer_stats": _merge_layer_stats(runs),
        "loss_gaps": loss_gap_statistics(summaries, all_trades, config.gap_analysis),
        "runs": summaries,
        "trades": all_trades,
        "workers": workers,
    }


# ============================================================
# Trade level export (for downstream statistical analysis)
# ============================================================
TRADE_COLUMNS = [
    "run_id", "start_index", "index", "reason",
    "opened_at", "closed_at", "entry_bar", "exit_bar", "bars_held",
    "duration_seconds", "same_bar", "max_fills_in_one_bar",
    "layers", "full_layers", "qty",
    "entry_notional", "exit_notional",
    "first_entry_price", "last_entry_price", "avg_entry_price", "exit_price",
    "worst_price", "best_price", "mae_pct", "mfe_pct", "exit_pct_from_avg",
    "gross_pnl", "entry_fees", "exit_fee", "fees", "net_pnl", "profit_pct",
    "balance_before", "balance_after", "return_on_balance", "shortfall",
]


def trades_dataframe(trades: Sequence[dict]) -> pd.DataFrame:
    """One row per closed trade: time, price, size and pnl of every cycle."""
    if not trades:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    frame = pd.DataFrame(
        [{key: value for key, value in trade.items() if key != "fills"} for trade in trades]
    )
    if frame["opened_at"].dtype.kind == "M" and frame["closed_at"].dtype.kind == "M":
        frame["duration_seconds"] = (
            frame["closed_at"] - frame["opened_at"]
        ).dt.total_seconds()
    else:
        frame["duration_seconds"] = np.nan
    ordered = [column for column in TRADE_COLUMNS if column in frame.columns]
    return frame[ordered + [c for c in frame.columns if c not in ordered]]


def fills_dataframe(trades: Sequence[dict]) -> pd.DataFrame:
    """One row per filled layer, joinable back on (run_id, trade_index)."""
    rows = [
        {
            "run_id": trade.get("run_id"),
            "capital_mode": trade.get("capital_mode"),
            "trade_index": trade["index"],
            "trade_reason": trade["reason"],
            **fill,
        }
        for trade in trades
        for fill in trade.get("fills", ())
    ]
    return pd.DataFrame(rows)


def write_table(logger: logging.Logger, frame: pd.DataFrame, path: str, label: str):
    """Write parquet when the suffix asks for it, CSV otherwise."""
    if frame.empty:
        logger.warning(f"No {label} to export, skipping {path}")
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if path.endswith(".parquet"):
        try:
            frame.to_parquet(path, index=False)
        except (ImportError, ValueError) as error:
            path = f"{os.path.splitext(path)[0]}.csv"
            logger.warning(f"Parquet unavailable ({error}), writing {path} instead")
            frame.to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)
    logger.info(f"Exported {len(frame)} {label} rows to {path}")


# ============================================================
# Reporting
# ============================================================
def log_direction_table(logger: logging.Logger, direction_stats: dict, aggregate: dict):
    """Only interesting once both sides were traded, i.e. reversal is on."""
    if len(direction_stats) < 2 and not aggregate.get("reversals"):
        return
    logger.info(
        f"{'SIDE':<7}{'TRADES':>9}{'SHARE':>8}{'WINRATE':>9}{'STOPPED':>9}"
        f"{'NET_PNL':>18}{'PNL/NOTIONAL':>15}   reversals "
        f"{aggregate.get('reversals', 0)}"
    )
    for name, bucket in direction_stats.items():
        logger.info(
            f"{name:<7}{bucket['trades']:>9}"
            f"{bucket['trade_pct'] * 100:>7.2f}%"
            f"{bucket['win_rate'] * 100:>8.2f}%"
            f"{bucket['stop_loss_pct'] * 100:>8.2f}%"
            f"{bucket['net_profit_abs']:>18.2f}"
            f"{bucket['net_profit_pct'] * 100:>14.3f}%"
            f"{bucket['avg_profit_vs_balance'] * 100:>10.3f}%"
        )


def log_loss_gaps(logger: logging.Logger, gaps: dict):
    """Observed spacing of losing trades next to a same-rate random null."""
    observed, reference = gaps.get("observed"), gaps.get("random")
    if not observed or not reference:
        return
    percentiles = observed["gap_percentiles"]
    logger.info(
        f"LOSSGAP | runs {gaps['runs_analyzed']} | losses {observed['losses']} "
        f"| mean gap {observed['gap_mean_bars']:.0f} bars "
        f"| p5 {percentiles['p5']:.0f} | p50 {percentiles['p50']:.0f} "
        f"| p95 {percentiles['p95']:.0f}"
    )
    ratios = gaps["ratios"]
    logger.info(
        f"CLUSTER | gap CV {observed['gap_cv_pooled']:.3f} vs random "
        f"{reference['gap_cv_pooled']:.3f} (x{_fmt(ratios['gap_cv_pooled'], 2)}) "
        f"| Fano {_fmt(observed.get('fano'), 2)} vs {_fmt(reference.get('fano'), 2)} "
        f"(x{_fmt(ratios['fano'], 2)}) "
        f"| lag-1 {_fmt(observed.get('gap_lag1'), 3)} "
        f"| short gaps {observed['short_gap_pct'] * 100:.1f}% vs "
        f"{reference['short_gap_pct'] * 100:.1f}% "
        f"(x{_fmt(ratios['short_gap_pct'], 2)}) -> {gaps['verdict']}"
    )


def log_layer_table(logger: logging.Logger, layer_stats: dict):
    logger.info(
        f"{'LAYERS':<7}{'REACHED':>9}{'REACH%':>8}{'TP%':>8}{'CLOSED':>9}"
        f"{'SAMEBAR':>9}"
        f"{'LIQ':>7}{'STOP':>7}{'AVG_STOP':>12}{'NET_PNL':>18}"
        f"{'PNL/NOTIONAL':>15}{'AVG %BAL':>11}"
    )
    for layer, bucket in layer_stats.items():
        logger.info(
            f"{layer:<7}{bucket['reached_trades']:>9}"
            f"{bucket['reach_pct'] * 100:>7.2f}%"
            f"{bucket['take_profit_pct'] * 100:>7.2f}%"
            f"{bucket['trades']:>9}"
            f"{bucket['same_bar_pct'] * 100:>8.2f}%"
            f"{bucket['liquidations']:>7}"
            f"{bucket['stop_losses']:>7}"
            f"{bucket['avg_stop_loss_abs']:>12.2f}"
            f"{bucket['net_profit_abs']:>18.2f}"
            f"{bucket['net_profit_pct'] * 100:>14.3f}%"
            f"{bucket['avg_profit_vs_balance'] * 100:>10.3f}%"
        )


def _fmt(value: Optional[float], digits: int = 2, scale: float = 1.0) -> str:
    return "n/a" if value is None else f"{value * scale:.{digits}f}"


def log_report(logger: logging.Logger, result: dict, config: BacktestConfig):
    aggregate = result["aggregate"]
    strategy = config.strategy
    account = config.account
    logger.info("-" * 96)
    logger.info(
        f"STRATEGY| target {strategy.take_profit_pct:.3%} of cycle balance "
        f"| symmetric boundary {strategy.price_deviation_pct:.3%} "
        f"| step x{strategy.deviation_step_mult} "
        f"| max_layers {strategy.max_layers} "
        f"| preflight full grid"
    )
    logger.info(
        f"STOP    | full-layer stop {strategy.stop_at_full_layers}"
    )
    logger.info(
        f"ACCOUNT | equity {account.initial_equity:.0f} "
        f"| leverage {account.leverage}x "
        f"| fee taker {account.taker_fee_pct}% / maker {account.maker_fee_pct}% "
        f"| maint {account.maintenance_margin_pct}% "
        f"| margin cap {account.margin_usage_cap_pct:.0%}"
    )
    breach_profit = aggregate["withdrawn_profit_before_grid_breach"]
    logger.info(
        f"CAPITAL | mode {config.capital.profit_handling} "
        f"| stop at first breach {config.capital.stop_at_first_grid_breach} "
        f"| breached {aggregate['grid_breached_runs']}/{aggregate['runs']} "
        f"({aggregate['grid_breach_rate'] * 100:.2f}%) "
        f"| reasons {aggregate['grid_breach_reasons']}"
    )
    if config.capital.profit_handling == "withdraw":
        logger.info(
            f"WITHDRAW| accumulated before breach mean "
            f"{aggregate['withdrawn_profit_before_grid_breach_mean']:.2f} "
            f"| p5 {breach_profit['p5']:.2f} | p50 {breach_profit['p50']:.2f} "
            f"| p95 {breach_profit['p95']:.2f}"
        )
    else:
        logger.info(
            f"COMPOUND| reached {config.capital.double_target_multiple:g}x before breach "
            f"{aggregate['doubled_before_grid_breach_runs']}/{aggregate['runs']} "
            f"({aggregate['double_before_grid_breach_rate'] * 100:.2f}%)"
        )
    logger.info(
        f"SURVIVE | runs {aggregate['runs']} "
        f"| ruined {aggregate['ruined_runs']} ({aggregate['ruin_rate'] * 100:.2f}%) "
        f"| liquidation {aggregate['liquidation_runs']} "
        f"| threshold {aggregate['ruin_threshold_runs']} "
        f"| grid infeasible {aggregate['grid_infeasible_runs']} "
        f"| survived {aggregate['survived_runs']}"
    )
    bars = aggregate["bars_survived"]
    logger.info(
        f"BARS    | mean {aggregate['bars_survived_mean']:.0f} "
        f"| p5 {bars['p5']:.0f} | p25 {bars['p25']:.0f} | p50 {bars['p50']:.0f} "
        f"| p75 {bars['p75']:.0f} | p95 {bars['p95']:.0f}"
    )
    years = aggregate["elapsed_years"]
    logger.info(
        f"TIME    | survived mean {aggregate['elapsed_years_mean']:.3f}y "
        f"| p5 {years['p5']:.3f}y | p25 {years['p25']:.3f}y "
        f"| p50 {years['p50']:.3f}y | p75 {years['p75']:.3f}y "
        f"| p95 {years['p95']:.3f}y"
    )
    returns = aggregate["total_return_pct"]
    logger.info(
        f"RETURN  | mean {aggregate['total_return_pct_mean'] * 100:.2f}% "
        f"| p5 {returns['p5'] * 100:.2f}% | p50 {returns['p50'] * 100:.2f}% "
        f"| p95 {returns['p95'] * 100:.2f}% "
        f"| avg MaxDD {aggregate['max_drawdown_pct_mean'] * 100:.2f}% "
        f"| avg peak IM/equity {aggregate['max_margin_usage_mean'] * 100:.1f}%"
    )
    distributions = aggregate["distributions"]
    peak = distributions["peak_equity"]
    peak_multiple = distributions["peak_equity_multiple"]
    realized_peak = distributions["peak_realized_balance"]
    logger.info(
        f"PEAK    | equity mean {_fmt(peak['mean'])} | p50 {_fmt(peak['p50'])} "
        f"| p95 {_fmt(peak['p95'])} | realized p50 {_fmt(realized_peak['p50'])} "
        f"| multiple mean {_fmt(peak_multiple['mean'])}x "
        f"| p50 {_fmt(peak_multiple['p50'])}x | p95 {_fmt(peak_multiple['p95'])}x"
    )
    cagr = distributions["cagr"]
    peak_cagr = distributions["peak_cagr"]
    sharpe = distributions["sharpe"]
    logger.info(
        f"RISKADJ | CAGR mean {_fmt(cagr['mean'], scale=100)}% "
        f"| p50 {_fmt(cagr['p50'], scale=100)}% | p95 {_fmt(cagr['p95'], scale=100)}% "
        f"| peak CAGR mean {_fmt(peak_cagr['mean'], scale=100)}% "
        f"| Sharpe mean {_fmt(sharpe['mean'], 3)} "
        f"| p50 {_fmt(sharpe['p50'], 3)} | p95 {_fmt(sharpe['p95'], 3)}"
    )
    trade_counts = aggregate["trades_per_run_percentiles"]
    logger.info(
        f"TRADES  | total {aggregate['total_trades']} "
        f"| per run mean {aggregate['trades_per_run']:.1f} "
        f"| p5 {trade_counts['p5']:.0f} | p25 {trade_counts['p25']:.0f} "
        f"| p50 {trade_counts['p50']:.0f} | p75 {trade_counts['p75']:.0f} "
        f"| p95 {trade_counts['p95']:.0f} "
        f"| win {aggregate['win_rate'] * 100:.2f}%"
    )
    logger.info(
        f"PNL     | net {aggregate['total_net_profit_abs']:.2f} "
        f"| fees {aggregate['total_fees_abs']:.2f} "
        f"({aggregate['fees_pct_of_notional'] * 100:.4f}% of notional) "
        f"| avg {aggregate['avg_net_profit_abs_per_trade']:.4f}/trade"
    )
    logger.info(
        f"STOPLOSS| trades {aggregate['stop_loss_trades']} "
        f"({aggregate['stop_loss_pct_of_trades'] * 100:.2f}% of trades) "
        f"| runs hit {aggregate['stop_loss_runs']}/{aggregate['runs']} "
        f"| per run {aggregate['stop_losses_per_run']:.2f} "
        f"| avg {aggregate['avg_stop_loss_abs']:.2f} "
        f"({aggregate['avg_stop_loss_pct_of_balance'] * 100:.2f}% of balance) "
        f"| avg win {aggregate['avg_win_abs']:.4f} "
        f"| wins to recover one stop {aggregate['wins_per_stop_loss']:.1f}"
    )
    logger.info(
        f"SAMEBAR | open+close in one bar {aggregate['same_bar_trades']} "
        f"({aggregate['same_bar_trade_pct'] * 100:.2f}%) "
        f"| multi-layer fills in one bar {aggregate['multiple_fills_same_bar_trades']} "
        f"({aggregate['multiple_fills_same_bar_pct'] * 100:.2f}%)"
    )
    logger.info(
        f"PATH    | dual-path bars {aggregate['dual_path_bars']} "
        f"| up-first was worse {aggregate['up_first_worse_bars']} "
        f"({aggregate['up_first_worse_pct'] * 100:.2f}%) "
        f"| exits {aggregate['exit_counts']} "
        f"| layers refused by margin {aggregate['unfilled_layers']}"
    )
    log_direction_table(logger, result.get("direction_stats", {}), aggregate)
    log_loss_gaps(logger, result.get("loss_gaps", {}))
    log_layer_table(logger, result["layer_stats"])
    logger.info("-" * 96)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def json_safe(value):
    """Replace inf/NaN with null so a finished run always writes its report.

    ``allow_nan=False`` is what makes the report readable by strict JSON
    parsers, but it raises at the very end of a run that already cost minutes.
    Sanitizing up front keeps that guarantee without the crash.
    """
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


GRID_FAILURE_HINTS = {
    "target_already_met_without_new_order": (
        "the existing position already exceeds the cash target at this layer's "
        "take-profit price; reduce max_layers to the last OK layer or lower "
        "deviation_step_mult"
    ),
    "non_positive_required_notional": (
        "the sizing equation produced no positive order; reduce max_layers or "
        "deviation_step_mult and inspect existing_net_at_take_profit"
    ),
    "non_positive_incremental_return": (
        "the layer's price edge does not cover its entry and exit fees; increase "
        "price_deviation_pct or reduce fees"
    ),
    "margin_cap_exceeded": (
        "initial margin exceeds margin_usage_cap_pct; reduce take_profit_pct or "
        "max_layers, or increase leverage/margin_usage_cap_pct"
    ),
    "non_positive_equity_after_order": (
        "mark-to-market equity after the order fee is non-positive; reduce grid "
        "size, take_profit_pct, or max_layers"
    ),
    "adverse_step_ge_100pct": (
        "an adverse grid step reached 100%; reduce deviation_step_mult, "
        "price_deviation_pct, or max_layers"
    ),
    "take_profit_deviation_ge_100pct": (
        "the take-profit deviation reached 100%; reduce deviation_step_mult, "
        "price_deviation_pct, or max_layers"
    ),
    "non_positive_layer_price": (
        "the planned layer price is non-positive; reduce grid spacing or layers"
    ),
}


def _diagnostic_number(value: Optional[float], digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def log_grid_preflight(
    logger: logging.Logger,
    diagnostics: dict[str, dict],
    config: BacktestConfig,
    start_price: float,
):
    """Print enough sizing state to explain an invalid grid configuration."""
    logger.info(
        f"GRID PREFLIGHT | start {start_price:.12g} "
        f"| target {config.strategy.take_profit_pct:.4%} of equity "
        f"| deviation {config.strategy.price_deviation_pct:.4%} "
        f"| step x{config.strategy.deviation_step_mult:g} "
        f"| layers {config.strategy.max_layers} "
        f"| leverage {config.account.leverage:g}x "
        f"| margin cap {config.account.margin_usage_cap_pct:.2%}"
    )
    for side, result in diagnostics.items():
        outcome = "OK" if result["executable"] else (
            f"FAIL L{result['failure_layer']:02d} {result['failure_reason']}"
        )
        logger.info(
            f"GRID SIDE | {side.upper():<5} | {outcome} "
            f"| cash target {result['target_cash']:.6f}"
        )
        for row in result["layers"]:
            if "take_profit_deviation_pct" not in row:
                entry = _diagnostic_number(row.get("entry_price"), 12)
                adverse = row.get("adverse_step_pct")
                logger.info(
                    f"GRID LAYER | {side:<5} L{row['layer']:02d} FAILED "
                    f"| reason {row['failure_reason']} "
                    f"| entry {entry} "
                    f"| adverse step {_fmt(adverse, 4, 100.0)}%"
                )
                continue
            average_before = (
                row["cumulative_notional_before"] / row["position_qty_before"]
                if row["position_qty_before"] > 0.0 else None
            )
            margin_relation = (
                "<=" if row.get("initial_margin_after", 0.0)
                <= row.get("margin_cap_amount", 0.0) + 1e-12 else ">"
            )
            logger.info(
                f"GRID LAYER | {side:<5} L{row['layer']:02d} {row['status'].upper():<6} "
                f"| entry {row['entry_price']:.12g} "
                f"| avg_before {_diagnostic_number(average_before, 10)} "
                f"| tp {row.get('take_profit_price', float('nan')):.12g} "
                f"| tp_dev {row['take_profit_deviation_pct']:.4%} "
                f"| existing@tp {row.get('existing_net_at_take_profit', 0.0):.6f} "
                f"/ target {row['target_cash']:.6f} "
                f"| add raw {_diagnostic_number(row.get('raw_required_notional'))} "
                f"used {row.get('required_notional', 0.0):.6f} "
                f"| cumulative {row.get('cumulative_notional_after', 0.0):.6f} "
                f"| fee {row.get('order_fee', 0.0):.6f} "
                f"| equity_mtm {row.get('equity_mtm_before_order', 0.0):.6f} "
                f"| IM {row.get('initial_margin_after', 0.0):.6f} "
                f"{margin_relation} cap {row.get('margin_cap_amount', 0.0):.6f}"
            )
        if not result["executable"]:
            reason = result["failure_reason"]
            logger.error(
                f"GRID FAILURE | {side.upper()} layer {result['failure_layer']} "
                f"| {reason} | {GRID_FAILURE_HINTS.get(reason, 'inspect the failed layer')}"
            )


def main(logger: logging.Logger, config: BacktestConfig) -> dict:
    bars = load_bars(logger, config.data)
    start_price = float(bars.open[0])
    simulator = MartingaleSimulator(config.strategy, config.account)
    diagnostics = simulator.grid_diagnostics(start_price)
    log_grid_preflight(logger, diagnostics, config, start_price)
    simulator.validate_initial_grid(start_price, diagnostics)
    modes = (
        ("withdraw", "compound")
        if config.capital.profit_handling == "both"
        else (config.capital.profit_handling,)
    )
    scenario_results = {}
    export_trades = []
    for mode in modes:
        scenario_config = replace(
            config, capital=replace(config.capital, profit_handling=mode)
        )
        logger.info(f"CAPITAL SCENARIO | {mode}")
        scenario_result = run_monte_carlo(logger, bars, scenario_config)
        for trade in scenario_result["trades"]:
            trade["capital_mode"] = mode
        export_trades.extend(scenario_result["trades"])
        scenario_results[mode] = scenario_result
        log_report(logger, scenario_result, scenario_config)

    # Keep the existing result shape for plots and callers. In comparison mode
    # the primary path is compound, while both complete summary sets are nested.
    primary_mode = "compound" if "compound" in scenario_results else modes[0]
    result = scenario_results[primary_mode]
    if len(scenario_results) > 1:
        result["capital_scenarios"] = {
            mode: {
                key: value
                for key, value in scenario_result.items()
                if key != "trades"
            }
            for mode, scenario_result in scenario_results.items()
        }
        result["capital_comparison"] = {
            "withdraw": {
                "accumulated_before_breach": scenario_results["withdraw"]["aggregate"][
                    "withdrawn_profit_before_grid_breach"
                ],
                "mean": scenario_results["withdraw"]["aggregate"][
                    "withdrawn_profit_before_grid_breach_mean"
                ],
                "on_breached_runs": scenario_results["withdraw"]["aggregate"][
                    "withdrawn_profit_on_breached_runs"
                ],
            },
            "compound": {
                "double_target_multiple": config.capital.double_target_multiple,
                "doubled_runs": scenario_results["compound"]["aggregate"][
                    "doubled_before_grid_breach_runs"
                ],
                "double_rate": scenario_results["compound"]["aggregate"][
                    "double_before_grid_breach_rate"
                ],
                "double_rate_on_breached_runs": scenario_results["compound"][
                    "aggregate"
                ]["double_before_grid_breach_rate_on_breached_runs"],
            },
        }

    if config.trades_path:
        write_table(logger, trades_dataframe(export_trades), config.trades_path, "trade")
    if config.fills_path:
        write_table(logger, fills_dataframe(export_trades), config.fills_path, "fill")
    if config.pressure_path or config.cluster_path:
        from trade.sim.martingale_plot import plot_cluster_scales, plot_pressure_heatmap

        label = (
            f"{config.data.symbol} {config.data.interval} | "
            f"layers {config.strategy.max_layers} "
            f"dev {config.strategy.price_deviation_pct:.2%} "
            f"cooldown {config.strategy.loss_cooldown_bars}"
        )
        if config.pressure_path:
            plot_pressure_heatmap(
                logger, bars, result, config.pressure_path,
                title=f"Stop-loss pressure Z(t, m) — {label}",
                min_scale=config.pressure_min_scale,
                max_scale=config.pressure_max_scale,
                scale_count=config.pressure_scale_count,
            )
        if config.cluster_path:
            plot_cluster_scales(
                logger, result, config.cluster_path,
                title=f"Stop-loss clusters vs gap allowance — {label}",
                min_scale=config.cluster_min_scale,
                max_scale=config.cluster_max_scale,
                scale_count=config.cluster_scale_count,
            )

    if config.plot_path:
        # Imported here so the headless matplotlib backend is only pulled in
        # when a chart was actually asked for.
        from trade.sim.martingale_plot import plot_failures

        plot_failures(
            logger, bars, result, config.plot_path,
            title=(
                f"{config.data.symbol} {config.data.interval} | "
                f"dev {config.strategy.price_deviation_pct:.2%} "
                f"tp {config.strategy.take_profit_pct:.2%} "
                f"layers {config.strategy.max_layers} "
                f"preflight grid "
                f"dir {config.strategy.initial_direction}"
                f"{'+reverse' if config.strategy.reverse_after_stop_loss else ''} "
                f"full-layer stop {config.strategy.stop_at_full_layers} | "
                f"lev {config.account.leverage:g}x "
                f"fee {config.account.taker_fee_pct}/{config.account.maker_fee_pct}%"
            ),
        )

    result["params"] = {
        "strategy": asdict(config.strategy),
        "account": asdict(config.account),
        "data": asdict(config.data),
        "monte_carlo": asdict(config.monte_carlo),
        "capital": asdict(config.capital),
    }
    if config.save_path:
        os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
        # The per-trade records go to their own table; keeping them here would
        # bloat the summary JSON by orders of magnitude.
        report = json_safe(
            {key: value for key, value in result.items() if key != "trades"}
        )
        with open(config.save_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=4, default=_json_default, allow_nan=False)
        logger.info(f"Report saved to {config.save_path}")
    return result


# ============================================================
# Run settings - edit these values directly
# ============================================================
@dataclass
class Args:
    """Every knob of one experiment, in one place.

    Edit the defaults here (or build ``Args(...)`` in a loop for a sweep) and
    run the module; nothing is read from the command line.
    """

    # --- data ---
    symbol: str = common.DOGE_1m.symbol
    interval: str = common.DOGE_1m.interval
    trading_type: str = common.DOGE_1m.trading_type
    market_category: str = common.DOGE_1m.market_category
    data_source: str = common.DOGE_1m.data_source
    from_date: Optional[str] = None
    to_date: Optional[str] = None

    # --- strategy ---
    price_deviation_pct: float = 0.01   # first-layer adverse/favourable distance
    take_profit_pct: float = 0.0008       # fixed net cash target / cycle-start balance
    deviation_step_mult: float = 1.5
    max_layers: int = 10
    initial_direction: str = "long"       # "long" or "short"
    reverse_after_stop_loss: bool = False  # flip the side after every stop
    loss_cooldown_bars: int = 0           # bars to sit out after a losing cycle
    stop_at_full_layers: bool = True      # stop at the rung the ladder cannot reach

    # --- account ---
    initial_equity: float = 10_000.0
    leverage: float = 50.0
    taker_fee_pct: float = 0.05           # percent, base order / stop / liquidation
    maker_fee_pct: float = 0.02           # percent, ladder and take profit
    maintenance_margin_pct: float = 0.5   # percent of position notional
    margin_usage_cap_pct: float = 0.90    # refuse orders past this share of equity
    liquidation_penalty_pct: float = 0.0
    ruin_equity_pct: float = 0.05         # a run is dead below this share of start

    # --- profit handling experiment ---
    profit_handling: str = "both"         # "withdraw", "compound", or "both"
    stop_at_first_grid_breach: bool = True
    double_target_multiple: float = 2.0

    # --- monte carlo ---
    runs: int = 120
    seed: int = 42
    min_bars: int = 2_000                 # bars that must remain ahead of a start
    max_bars: Optional[int] = None        # None: run until ruin or end of data
    monte_carlo_start_fraction: float = 0.30
    workers: int = 30                     # 0 = cpu_count - 1, 1 = serial

    # --- loss clustering ---
    gap_window_bars: int = 500            # window for the count dispersion test
    gap_short_bars: int = 10              # "another loss almost immediately"
    gap_repeats: int = 20                 # random baselines drawn per run
    gap_seed: int = 17

    # --- output ---  (None = a default name inside the experiment directory)
    save_path: Optional[str] = None       # summary JSON
    trades_path: Optional[str] = None     # .csv or .parquet, one row per trade
    fills_path: Optional[str] = None      # .csv or .parquet, one row per layer
    plot_path: Optional[str] = None       # .png loss/failure chart
    pressure_path: Optional[str] = None   # .png time x window-scale pressure map
    cluster_path: Optional[str] = None    # .png cluster size vs gap allowance
    # Window-scale ladders for those two charts, in bars.
    pressure_min_scale: int = 8
    pressure_max_scale: Optional[int] = None   # None: total_bars // 8
    pressure_scale_count: int = 40
    cluster_min_scale: int = 1
    cluster_max_scale: int = 5_000
    cluster_scale_count: int = 40
    no_plot: bool = False
    no_export: bool = False


def config_from_args(args: Args) -> BacktestConfig:
    return BacktestConfig(
        strategy=MartingaleParams(
            price_deviation_pct=args.price_deviation_pct,
            take_profit_pct=args.take_profit_pct,
            deviation_step_mult=args.deviation_step_mult,
            max_layers=args.max_layers,
            loss_cooldown_bars=args.loss_cooldown_bars,
            initial_direction=args.initial_direction,
            reverse_after_stop_loss=args.reverse_after_stop_loss,
            stop_at_full_layers=args.stop_at_full_layers,
        ),
        account=AccountParams(
            initial_equity=args.initial_equity,
            leverage=args.leverage,
            taker_fee_pct=args.taker_fee_pct,
            maker_fee_pct=args.maker_fee_pct,
            maintenance_margin_pct=args.maintenance_margin_pct,
            margin_usage_cap_pct=args.margin_usage_cap_pct,
            liquidation_penalty_pct=args.liquidation_penalty_pct,
            ruin_equity_pct=args.ruin_equity_pct,
        ),
        capital=CapitalParams(
            profit_handling=args.profit_handling,
            stop_at_first_grid_breach=args.stop_at_first_grid_breach,
            double_target_multiple=args.double_target_multiple,
        ),
        data=DataParams(
            market_category=args.market_category,
            data_source=args.data_source,
            symbol=args.symbol,
            interval=args.interval,
            trading_type=args.trading_type,
            from_date=args.from_date,
            to_date=args.to_date,
        ),
        monte_carlo=MonteCarloParams(
            runs=args.runs,
            seed=args.seed,
            min_bars=args.min_bars,
            max_bars=args.max_bars,
            start_fraction=args.monte_carlo_start_fraction,
            workers=args.workers,
        ),
        gap_analysis=GapAnalysisParams(
            window_bars=args.gap_window_bars,
            short_gap_bars=args.gap_short_bars,
            repeats=args.gap_repeats,
            seed=args.gap_seed,
        ),
        save_path=args.save_path,
        trades_path=None if args.no_export else args.trades_path,
        fills_path=None if args.no_export else args.fills_path,
        plot_path=None if args.no_plot else args.plot_path,
        pressure_path=None if args.no_plot else args.pressure_path,
        cluster_path=None if args.no_plot else args.cluster_path,
        pressure_min_scale=args.pressure_min_scale,
        pressure_max_scale=args.pressure_max_scale,
        pressure_scale_count=args.pressure_scale_count,
        cluster_min_scale=args.cluster_min_scale,
        cluster_max_scale=args.cluster_max_scale,
        cluster_scale_count=args.cluster_scale_count,
    )


def run(args: Optional[Args] = None) -> dict:
    """Set up the experiment directory and logger, then run one experiment."""
    # Not a default argument: Args is mutable, and one shared instance would
    # carry edits from a previous call into the next one.
    args = args if args is not None else Args()
    exp_dir = common.create_experiment_dir(
        os.path.join(common.PERSISTENCE_DIR, "martingale_sim"),
        args.symbol,
        args.interval,
    )
    logger, _ = common.setup_session_logger(
        log_file_path=os.path.join(exp_dir, "simulation.log"),
        console_level=logging.INFO,
        file_level=logging.INFO,
    )
    config = config_from_args(args)
    defaults = {}
    if config.save_path is None:
        defaults["save_path"] = os.path.join(exp_dir, "monte_carlo_report.json")
    if not args.no_export:
        if config.trades_path is None:
            defaults["trades_path"] = os.path.join(exp_dir, "trades.parquet")
        if config.fills_path is None:
            defaults["fills_path"] = os.path.join(exp_dir, "fills.parquet")
    if not args.no_plot:
        if config.plot_path is None:
            defaults["plot_path"] = os.path.join(exp_dir, "failures.png")
        if config.pressure_path is None:
            defaults["pressure_path"] = os.path.join(exp_dir, "pressure.png")
        if config.cluster_path is None:
            defaults["cluster_path"] = os.path.join(exp_dir, "clusters.png")
    config = replace(config, **defaults)
    started = time.time()
    result = main(logger, config)
    logger.info(f"run_time: {time.time() - started:.2f} s")
    return result


if __name__ == "__main__":
    run(Args())