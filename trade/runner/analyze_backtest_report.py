"""Analyze in-memory backtest diagnostics and render report equity curves."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Callable, Iterable


def safe_get(data: dict, path: Iterable[str], default=None):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def to_float(value, default=0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def date_from_order(order: dict) -> str | None:
    if order.get("date_utc"):
        return str(order["date_utc"])
    dt = order.get("dt")
    if dt is None:
        return None
    try:
        return datetime.fromtimestamp(int(dt), tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return None


def daily_loss_rows(report: dict) -> list[dict]:
    rows = safe_get(report, ["drawdown", "daily_loss_list"], []) or []
    return rows if isinstance(rows, list) else []


_DURATION_PERCENTILES = (
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)


def _duration_summary(values) -> dict:
    """Return JSON-safe duration statistics without inventing empty values."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "max": None,
            **{name: None for name, _ in _DURATION_PERCENTILES},
        }

    quantiles = np.quantile(
        values,
        [quantile for _, quantile in _DURATION_PERCENTILES],
    )
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        **{
            name: float(value)
            for (name, _), value in zip(_DURATION_PERCENTILES, quantiles)
        },
    }


def _daily_equity_frame(report: dict):
    """Load the report's end-of-day equity samples as a clean time series."""
    import pandas as pd

    frame = pd.DataFrame(daily_loss_rows(report))
    if frame.empty or "date" not in frame or "equity" not in frame:
        return pd.DataFrame(columns=["equity"])

    frame = frame[["date", "equity"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
    frame = frame.dropna(subset=["date", "equity"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame["date"] = frame["date"].dt.tz_convert(None)
    return frame.set_index("date")


def _simulate_ftmo_equity_segment(
    equity_frame,
    *,
    profit_target_pct: float,
    loss_limit_pct: float,
) -> dict:
    """Evaluate every daily equity sample as an independent challenge start."""
    import numpy as np

    equity = equity_frame["equity"].to_numpy(dtype=float)
    dates = equity_frame.index
    outcome_counts = {
        "profit_target": 0,
        "loss_limit": 0,
        "unresolved": 0,
    }
    duration_days = {
        "profit_target": [],
        "loss_limit": [],
        "resolved": [],
        "unresolved_observed": [],
    }
    duration_observations = {
        "profit_target": [],
        "loss_limit": [],
        "resolved": [],
        "unresolved_observed": [],
    }
    valid_start_count = 0

    for start_index, start_equity in enumerate(equity):
        if not np.isfinite(start_equity) or start_equity <= 0:
            continue
        valid_start_count += 1
        future = equity[start_index + 1 :]
        future_returns = future / start_equity - 1.0
        comparison_tolerance = np.finfo(float).eps * 8
        upper_hits = np.flatnonzero(
            future_returns >= profit_target_pct - comparison_tolerance
        )
        lower_hits = np.flatnonzero(
            future_returns <= -loss_limit_pct + comparison_tolerance
        )
        upper_offset = int(upper_hits[0]) + 1 if len(upper_hits) else None
        lower_offset = int(lower_hits[0]) + 1 if len(lower_hits) else None

        if upper_offset is None and lower_offset is None:
            outcome = "unresolved"
            end_index = len(equity) - 1
            elapsed_observations = end_index - start_index
            elapsed_days = (
                dates[end_index] - dates[start_index]
            ).total_seconds() / 86400.0
            duration_days["unresolved_observed"].append(elapsed_days)
            duration_observations["unresolved_observed"].append(
                elapsed_observations
            )
        else:
            if lower_offset is None or (
                upper_offset is not None and upper_offset < lower_offset
            ):
                outcome = "profit_target"
                hit_offset = upper_offset
            else:
                outcome = "loss_limit"
                hit_offset = lower_offset

            hit_index = start_index + hit_offset
            elapsed_days = (
                dates[hit_index] - dates[start_index]
            ).total_seconds() / 86400.0
            duration_days[outcome].append(elapsed_days)
            duration_days["resolved"].append(elapsed_days)
            duration_observations[outcome].append(hit_offset)
            duration_observations["resolved"].append(hit_offset)

        outcome_counts[outcome] += 1

    resolved_count = (
        outcome_counts["profit_target"] + outcome_counts["loss_limit"]
    )

    def rate(count: int, denominator: int) -> float:
        return float(count / denominator) if denominator else 0.0

    return {
        "start_count": valid_start_count,
        "resolved_count": resolved_count,
        "profit_target_count": outcome_counts["profit_target"],
        "loss_limit_count": outcome_counts["loss_limit"],
        "unresolved_count": outcome_counts["unresolved"],
        "profit_target_rate": rate(
            outcome_counts["profit_target"], valid_start_count
        ),
        "loss_limit_rate": rate(outcome_counts["loss_limit"], valid_start_count),
        "unresolved_rate": rate(outcome_counts["unresolved"], valid_start_count),
        "profit_target_rate_of_resolved": rate(
            outcome_counts["profit_target"], resolved_count
        ),
        "duration_days": {
            name: _duration_summary(values)
            for name, values in duration_days.items()
        },
        "duration_observations": {
            name: _duration_summary(values)
            for name, values in duration_observations.items()
        },
    }


def simulate_ftmo_challenges(
    report: dict,
    *,
    profit_target_pct: float = 0.10,
    loss_limit_pct: float = 0.10,
) -> dict:
    """Run an exhaustive historical-start FTMO barrier simulation.

    Every end-of-day equity sample is treated as a possible challenge start.
    The first later sample reaching ``+profit_target_pct`` or
    ``-loss_limit_pct`` determines the outcome; samples with no hit before the
    period ends are right-censored as ``unresolved``.

    An ``all`` backtest is split at the OOD boundary and simulated separately,
    so a long-period start can never use forward-period equity to finish.
    """
    import pandas as pd

    if not 0 < profit_target_pct < 1:
        raise ValueError("profit_target_pct must be between 0 and 1")
    if not 0 < loss_limit_pct < 1:
        raise ValueError("loss_limit_pct must be between 0 and 1")

    equity_frame = _daily_equity_frame(report)
    period = str(safe_get(report, ["params", "data", "period"], "backtest")).lower()
    segments = {}
    split_at = None

    if period == "all":
        raw_boundary = safe_get(report, ["time", "regions", "ood", "start"])
        split_at = pd.to_datetime(raw_boundary, utc=True, errors="coerce")
        if pd.isna(split_at):
            raise ValueError(
                "period='all' FTMO simulation requires time.regions.ood.start"
            )
        split_at = split_at.tz_convert(None).normalize()
        segments = {
            "long": equity_frame[equity_frame.index < split_at],
            "forward": equity_frame[equity_frame.index >= split_at],
        }
    else:
        segments[period] = equity_frame

    return {
        "method": "exhaustive_historical_start_dates",
        "equity_resolution": "daily_close",
        "profit_target_pct": float(profit_target_pct),
        "loss_limit_pct": float(loss_limit_pct),
        "split_at": split_at.isoformat() if split_at is not None else None,
        "periods": {
            name: _simulate_ftmo_equity_segment(
                frame,
                profit_target_pct=profit_target_pct,
                loss_limit_pct=loss_limit_pct,
            )
            for name, frame in segments.items()
        },
    }


def log_ftmo_challenge_summary(
    result: dict,
    logger=None,
    *,
    emit: Callable[[str], None] | None = None,
) -> None:
    """Print the compact counts and duration percentiles of a simulation."""
    if emit is None:
        emit = logger.info if logger is not None else print

    def fmt_duration(summary: dict) -> str:
        if not summary or not summary.get("count"):
            return "N/A"
        return (
            f"P10 {summary['p10']:.1f} | P25 {summary['p25']:.1f} | "
            f"P50 {summary['p50']:.1f} | P75 {summary['p75']:.1f} | "
            f"P90 {summary['p90']:.1f} | P95 {summary['p95']:.1f} | "
            f"P99 {summary['p99']:.1f}"
        )

    profit_target_pct = to_float(result.get("profit_target_pct"), 0.10) * 100
    loss_limit_pct = to_float(result.get("loss_limit_pct"), 0.10) * 100
    for period, stats in (result.get("periods") or {}).items():
        emit(
            f"FTMO MC | {period.upper():7} | Starts: {stats['start_count']} "
            f"| +{profit_target_pct:g}%: {stats['profit_target_count']} "
            f"({stats['profit_target_rate'] * 100:.2f}%) "
            f"| -{loss_limit_pct:g}%: {stats['loss_limit_count']} "
            f"({stats['loss_limit_rate'] * 100:.2f}%) "
            f"| Unresolved: {stats['unresolved_count']} "
            f"({stats['unresolved_rate'] * 100:.2f}%)"
        )
        emit(
            f"FTMO +  | {period.upper():7} | Days: "
            f"{fmt_duration(stats['duration_days']['profit_target'])}"
        )
        emit(
            f"FTMO -  | {period.upper():7} | Days: "
            f"{fmt_duration(stats['duration_days']['loss_limit'])}"
        )


def build_continuous_equity_path(
    reports_by_period,
    *,
    normalize_equity: bool = False,
):
    """Build a continuous equity path from one or more backtest periods.

    Each report contains absolute daily broker equity and its own starting
    capital. Subsequent periods are chained by applying their equity return to
    the previous period's ending equity. The first point is always the report's
    starting capital (or ``1.0`` when normalization is explicitly requested).
    """
    import pandas as pd

    segments = []
    split_dates = {}
    current_equity = None
    path_start_equity = None
    path_start_time = None

    for period, report in reports_by_period:
        daily = daily_loss_rows(report)
        if not daily:
            continue
        frame = pd.DataFrame(daily)
        if "date" not in frame or "equity" not in frame:
            continue
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None)
        frame["equity"] = pd.to_numeric(frame["equity"], errors="coerce")
        frame = frame.dropna(subset=["date", "equity"]).sort_values("date")
        if frame.empty:
            continue

        segment_start_equity = to_float(
            safe_get(report, ["performance", "start_value"]),
            None,
        )
        if segment_start_equity is None or segment_start_equity <= 0:
            continue

        frame.set_index("date", inplace=True)
        split_dates.setdefault(str(period), frame.index[0])
        if current_equity is None:
            path_start_equity = 1.0 if normalize_equity else segment_start_equity
            current_equity = path_start_equity
            path_start_time = safe_get(report, ["time", "start"])

        frame["continuous_equity"] = (
            frame["equity"] / segment_start_equity * current_equity
        )
        segments.append(frame[["continuous_equity"]])
        current_equity = frame["continuous_equity"].iloc[-1]

    if not segments:
        return pd.DataFrame(columns=["continuous_equity"]), split_dates

    full_path = pd.concat(segments).sort_index()
    full_path = full_path[~full_path.index.duplicated(keep="first")]

    # Daily rows represent end-of-day equity but only carry a calendar date.
    # Put the exact initial-capital point immediately before the first row so
    # the plotted curve cannot appear to start at zero (or at first-day PnL).
    first_daily_time = full_path.index[0]
    baseline_time = pd.to_datetime(path_start_time, utc=True, errors="coerce")
    if pd.isna(baseline_time):
        baseline_time = first_daily_time - pd.Timedelta(nanoseconds=1)
    else:
        baseline_time = baseline_time.tz_convert(None)
        baseline_time = min(
            baseline_time,
            first_daily_time - pd.Timedelta(nanoseconds=1),
        )
    baseline = pd.DataFrame(
        {"continuous_equity": [path_start_equity]},
        index=pd.DatetimeIndex([baseline_time], name=full_path.index.name),
    )
    return pd.concat([baseline, full_path]).sort_index(), split_dates


def time_region_boundaries(time_regions: dict) -> list[tuple[str, object]]:
    """Parse ordered model-region starts into timezone-naive timestamps."""
    import pandas as pd

    boundaries = []
    for name in ("train", "valid", "test", "ood"):
        start = safe_get(time_regions, [name, "start"])
        timestamp = pd.to_datetime(start, utc=True, errors="coerce")
        if not pd.isna(timestamp):
            boundaries.append((name, timestamp.tz_convert(None)))
    return boundaries


def plot_equity_curves(
    all_results,
    output_dir: str,
    file_name: str = "equity_full_combined.png",
    start_index: int = 0,
    *,
    price_file: str | None = None,
    normalize_equity: bool = False,
    equity_scale: str = "both",
    logger=None,
) -> str | None:
    """Plot one or more report equity curves over the underlying market price.

    ``all_results`` accepts either one report returned by ``backtest_runner`` or
    selected-config records containing ``long``/``short``/``forward`` reports.
    Batch records are chained in chronological period order. By default the
    curve starts at the report's absolute initial equity; callers may request a
    starting value of 1 with ``normalize_equity=True``. By default both the
    absolute equity (linear axis) and logarithmic equity are drawn with separate
    right-side axes. Pass ``equity_scale="linear"`` or ``"log"`` to draw only
    one representation. The saved image path is returned.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from data_process import common

    if equity_scale not in {"both", "log", "linear"}:
        raise ValueError("equity_scale must be one of: 'both', 'log', 'linear'")

    results = [all_results] if isinstance(all_results, dict) else list(all_results)
    if not results:
        return None

    period_order = ("long", "short", "forward")

    def period_reports(result: dict):
        if not isinstance(result, dict):
            return []
        if isinstance(result.get("params"), dict):
            period = safe_get(result, ["params", "data", "period"], "backtest")
            return [(str(period), result)]
        return [
            (period, result.get(period))
            for period in period_order
            if isinstance(result.get(period), dict) and result.get(period)
        ]

    first_report = next(
        (
            report
            for result in results
            for _, report in period_reports(result)
            if isinstance(safe_get(report, ["params", "common"]), dict)
        ),
        None,
    )
    if first_report is None:
        return None
    time_regions = next(
        (
            regions
            for result in results
            for _, report in period_reports(result)
            if isinstance(
                (regions := safe_get(report, ["time", "regions"])),
                dict,
            )
            and regions
        ),
        {},
    )

    if price_file is None:
        market_params = safe_get(first_report, ["params", "common"])
        price_source = common.MarketDataSourceConfig(
            market_category=market_params["market_category"],
            data_source=market_params["data_source"],
            trading_type=market_params["trading_type"],
            symbol=market_params["symbol"],
            interval=market_params["interval"],
        )
        price_file = common.market_data_path(price_source)
    if not os.path.isfile(price_file):
        raise FileNotFoundError(f"Price data file not found: {price_file}")

    price_df = pd.read_csv(price_file, usecols=["open_time_date_utc", "close"])
    price_df["open_time_date_utc"] = pd.to_datetime(
        price_df["open_time_date_utc"], utc=True
    ).dt.tz_convert(None)
    price_df.set_index("open_time_date_utc", inplace=True)
    price_series = pd.to_numeric(price_df["close"], errors="coerce").sort_index()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, file_name)
    fig, price_axis = plt.subplots(figsize=(16, 8))
    price_axis.plot(
        price_series.index,
        price_series,
        color="black",
        linewidth=0.8,
        alpha=0.15,
        label="Market Price",
    )
    price_axis.set_ylabel("Market Price")

    split_dates = {}
    equity_paths = []
    has_multi_period_record = False
    minimum_equity = None
    for index, result in enumerate(results):
        reports_by_period = period_reports(result)
        has_multi_period_record |= len(reports_by_period) > 1
        full_path, result_split_dates = build_continuous_equity_path(
            reports_by_period,
            normalize_equity=normalize_equity,
        )
        for period, split_date in result_split_dates.items():
            split_dates.setdefault(period, split_date)
        if full_path.empty:
            continue
        path_minimum = full_path["continuous_equity"].dropna().min()
        if pd.notna(path_minimum):
            minimum_equity = (
                float(path_minimum)
                if minimum_equity is None
                else min(minimum_equity, float(path_minimum))
            )
        report_for_label = reports_by_period[0][1]
        params_hash = safe_get(report_for_label, ["params", "hash"])
        label = str(params_hash) if params_hash else f"S{start_index + index}"
        equity_paths.append((label, full_path))

    if not equity_paths:
        plt.close(fig)
        return None

    show_linear = equity_scale in {"both", "linear"}
    show_log = equity_scale in {"both", "log"}
    if equity_scale == "log" and (minimum_equity is None or minimum_equity <= 0):
        show_log = False
        show_linear = True
        if logger is not None:
            logger.warning(
                "Equity contains zero or negative values; falling back to a linear axis"
            )

    linear_equity_axis = price_axis.twinx() if show_linear else None
    log_equity_axis = price_axis.twinx() if show_log else None
    if log_equity_axis is not None:
        log_equity_axis.set_yscale("log")
        if linear_equity_axis is not None:
            log_equity_axis.spines["right"].set_position(("axes", 1.09))
            log_equity_axis.patch.set_visible(False)

    color_map = plt.get_cmap("tab10")
    for path_index, (label, full_path) in enumerate(equity_paths):
        values = full_path["continuous_equity"]
        if linear_equity_axis is not None:
            linear_equity_axis.plot(
                full_path.index,
                values,
                color=color_map((path_index * 2) % 10),
                linewidth=1.5,
                alpha=0.8,
                label=f"{label} Absolute",
            )
        if log_equity_axis is not None:
            log_equity_axis.plot(
                full_path.index,
                values.where(values > 0),
                color=color_map((path_index * 2 + 1) % 10),
                linewidth=1.5,
                alpha=0.8,
                label=f"{label} Log",
            )

    normalized_suffix = " (Normalized)" if normalize_equity else ""
    if linear_equity_axis is not None:
        linear_equity_axis.set_ylabel(
            f"Absolute Strategy Equity{normalized_suffix}",
            color=color_map(0),
        )
        linear_equity_axis.tick_params(axis="y", labelcolor=color_map(0))
    if log_equity_axis is not None:
        log_equity_axis.set_ylabel(
            f"Strategy Equity{normalized_suffix} (Log Scale)",
            color=color_map(1),
        )
        log_equity_axis.tick_params(axis="y", labelcolor=color_map(1))

    region_starts = time_region_boundaries(time_regions)
    if region_starts:
        region_colors = {
            "train": "#2e8b57",
            "valid": "#d98c00",
            "test": "#2878b5",
            "ood": "#c43c39",
        }
        for name, boundary in region_starts[1:]:
            price_axis.axvline(
                boundary,
                color=region_colors[name],
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
            )

        for position, (name, start) in enumerate(region_starts):
            if position + 1 < len(region_starts):
                end = region_starts[position + 1][1]
            else:
                end_value = safe_get(time_regions, [name, "end"])
                end = pd.to_datetime(end_value, utc=True, errors="coerce")
                end = (
                    price_series.index.max()
                    if pd.isna(end)
                    else end.tz_convert(None)
                )
            if end <= start:
                continue
            midpoint = start + (end - start) / 2
            price_axis.text(
                midpoint,
                0.985,
                name.upper(),
                transform=price_axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                color=region_colors[name],
            )
    elif has_multi_period_record:
        if "short" in split_dates:
            price_axis.axvline(
                split_dates["short"], color="blue", linestyle="--", linewidth=1, alpha=0.3
            )
        if "forward" in split_dates:
            price_axis.axvline(
                split_dates["forward"], color="red", linestyle="--", linewidth=1, alpha=0.3
            )

    legend_axes = [price_axis]
    if linear_equity_axis is not None:
        legend_axes.append(linear_equity_axis)
    if log_equity_axis is not None:
        legend_axes.append(log_equity_axis)
    legend_entries = [
        entry
        for axis in legend_axes
        for entry in zip(*axis.get_legend_handles_labels())
    ]
    legend_handles = [handle for handle, _ in legend_entries]
    legend_labels = [label for _, label in legend_entries]
    price_axis.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        ncol=4,
        fontsize=8,
    )
    if region_starts:
        title = "Strategy Equity Curve: Train -> Valid -> Test -> OOD"
    elif has_multi_period_record:
        title = "Strategy Performance: Long -> Short -> Forward"
    else:
        title = "Strategy Equity Curve"
    plt.title(title)
    if linear_equity_axis is not None and log_equity_axis is not None:
        fig.subplots_adjust(right=0.82)
    else:
        fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    if logger is not None:
        logger.info("Equity plot saved to %s", save_path)
    else:
        print(f"[SAVE] {save_path}")
    return save_path


def daily_equity_context(report: dict, target_date: str | None) -> dict:
    """Build the equity bridge used by the reported daily drawdown metric."""
    rows = daily_loss_rows(report)
    index = next(
        (i for i, row in enumerate(rows) if str(row.get("date")) == target_date),
        None,
    )
    if index is None:
        return {
            "date": target_date,
            "start_equity": None,
            "min_equity": None,
            "end_equity": None,
            "equity_change": None,
            "equity_change_pct": None,
            "dd_pct": None,
        }

    row = rows[index]
    start_equity = (
        to_float(rows[index - 1].get("equity"), None)
        if index > 0
        else safe_get(report, ["performance", "start_value"])
    )
    end_equity = to_float(row.get("equity"), None)
    dd_pct = to_float(row.get("dd_pct"), 0.0)
    min_equity = start_equity * (1.0 + dd_pct) if start_equity is not None else None
    equity_change = (
        end_equity - start_equity
        if start_equity is not None and end_equity is not None
        else None
    )
    equity_change_pct = (
        equity_change / start_equity
        if equity_change is not None and start_equity
        else None
    )
    return {
        "date": target_date,
        "start_equity": start_equity,
        "min_equity": min_equity,
        "end_equity": end_equity,
        "equity_change": equity_change,
        "equity_change_pct": equity_change_pct,
        "dd_pct": dd_pct,
    }


def worst_daily_loss(report: dict) -> dict:
    rows = daily_loss_rows(report)
    if not rows:
        return {
            "date": safe_get(report, ["drawdown", "max_daily_date"]),
            "dd_pct": safe_get(report, ["drawdown", "max_daily_dd"], 0.0),
            "equity": None,
        }
    return min(rows, key=lambda item: to_float(item.get("dd_pct"), 0.0))


def trade_logs(report: dict) -> list[dict]:
    rows = report.get("trade_logs") or []
    return rows if isinstance(rows, list) else []


def datetime_from_order(order: dict) -> str | None:
    dt = order.get("dt")
    if dt is None:
        return None
    try:
        return datetime.fromtimestamp(int(dt), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def pair_round_trips(report: dict) -> list[dict]:
    """Pair flat-to-position fills with their next full exit and estimate PnL."""
    commission_pct = to_float(
        safe_get(report, ["params", "broker", "commission_pct"]),
        0.0,
    )
    commission_rate = commission_pct / 100.0
    pending_open = None
    rows = []

    for order in trade_logs(report):
        role = order.get("role")
        if role == "open":
            pending_open = order
            continue
        if role not in {"tp", "sl", "close"} or pending_open is None:
            continue

        entry = pending_open
        pending_open = None
        entry_price = to_float(entry.get("price"))
        exit_price = to_float(order.get("price"))
        signed_size = to_float(entry.get("size"))
        qty = abs(signed_size)
        is_long = bool(entry.get("is_long"))
        gross_pnl = signed_size * (exit_price - entry_price)
        estimated_commission = qty * (entry_price + exit_price) * commission_rate
        estimated_net_pnl = gross_pnl - estimated_commission
        price_return_pct = 0.0
        if entry_price > 0:
            price_return_pct = (
                (exit_price - entry_price) / entry_price
                if is_long
                else (entry_price - exit_price) / entry_price
            )

        rows.append({
            "entry_date": date_from_order(entry),
            "exit_date": date_from_order(order),
            "entry_time_utc": datetime_from_order(entry),
            "exit_time_utc": datetime_from_order(order),
            "side": "long" if is_long else "short",
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_role": role,
            "gross_pnl": gross_pnl,
            "estimated_commission": estimated_commission,
            "estimated_net_pnl": estimated_net_pnl,
            "price_return_pct": price_return_pct,
        })

    return rows


def focus_trade_diagnostics(report: dict, target_date: str | None, top_n: int) -> dict:
    round_trips = [
        item for item in pair_round_trips(report)
        if item["exit_date"] == target_date
    ]
    round_trips.sort(key=lambda item: item["estimated_net_pnl"])
    return {
        "closed_trade_count": len(round_trips),
        "carry_in_trade_count": sum(item["entry_date"] != target_date for item in round_trips),
        "gross_pnl": sum(item["gross_pnl"] for item in round_trips),
        "estimated_commission": sum(item["estimated_commission"] for item in round_trips),
        "estimated_net_pnl": sum(item["estimated_net_pnl"] for item in round_trips),
        "worst_trades": round_trips[:top_n],
    }


def fmt_pct(value) -> str:
    return f"{to_float(value) * 100:.4f}%"


def fmt_number(value, decimals: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{to_float(value):,.{decimals}f}"


def short_trade_line(trade: dict) -> str:
    return (
        f"{trade.get('side')} entry={trade.get('entry_time_utc')}@{to_float(trade.get('entry_price')):.6g} "
        f"exit={trade.get('exit_time_utc')}@{to_float(trade.get('exit_price')):.6g} "
        f"role={trade.get('exit_role')} qty={to_float(trade.get('qty')):.6g} "
        f"move={fmt_pct(trade.get('price_return_pct'))} "
        f"gross={fmt_number(trade.get('gross_pnl'))} "
        f"est_fee={fmt_number(trade.get('estimated_commission'))} "
        f"est_net={fmt_number(trade.get('estimated_net_pnl'))}"
    )


def analyze_record(record: dict, top_n: int = 10, focus_date: str | None = None) -> dict:
    report = dict(record["report"])
    additional = record["additional"]
    report["trade_logs"] = additional.get("trade_logs") or report.get("trade_logs") or []
    worst_day = worst_daily_loss(report)
    target_date = focus_date or worst_day.get("date")
    ftmo_challenge = report.get("ftmo_challenge")
    if not isinstance(ftmo_challenge, dict):
        ftmo_challenge = simulate_ftmo_challenges(report)

    return {
        "params_hash": safe_get(report, ["params", "hash"]),
        "worst_day": worst_day,
        "focus_date": target_date,
        "focus_equity": daily_equity_context(report, target_date),
        "focus_trades": focus_trade_diagnostics(report, target_date, top_n),
        "ftmo_challenge": ftmo_challenge,
    }


def print_analysis(
    item: dict,
    index: int = 1,
    total: int = 1,
    *,
    emit: Callable[[str], None] = print,
) -> None:
    prefix = f"[{index}/{total}] " if total > 1 else ""
    emit(f"{prefix}params_hash: {item.get('params_hash')}")
    worst = item["worst_day"]
    emit(
        "Worst daily loss: "
        f"date={worst.get('date')} dd={fmt_pct(worst.get('dd_pct'))} equity={worst.get('equity')}"
    )
    emit(f"Focus date: {item['focus_date']}")

    equity = item["focus_equity"]
    emit(
        "Focus equity: "
        f"start={fmt_number(equity.get('start_equity'))} "
        f"min={fmt_number(equity.get('min_equity'))} "
        f"end={fmt_number(equity.get('end_equity'))} "
        f"change={fmt_number(equity.get('equity_change'))} "
        f"change_pct={fmt_pct(equity.get('equity_change_pct')) if equity.get('equity_change_pct') is not None else 'NA'}"
    )

    focus_trades = item["focus_trades"]
    emit(
        "Focus closed trades: "
        f"count={focus_trades['closed_trade_count']} "
        f"carry_in={focus_trades['carry_in_trade_count']} "
        f"gross={fmt_number(focus_trades['gross_pnl'])} "
        f"est_fee={fmt_number(focus_trades['estimated_commission'])} "
        f"est_net={fmt_number(focus_trades['estimated_net_pnl'])}"
    )
    if focus_trades["carry_in_trade_count"]:
        emit(
            "  note: PnL totals include full round trips opened before the focus day, "
            "so they are diagnostic and may not exactly reconcile to the daily equity change."
        )
    for trade in focus_trades["worst_trades"]:
        emit(f"  focus trade {short_trade_line(trade)}")
    log_ftmo_challenge_summary(item["ftmo_challenge"], emit=emit)
    emit("")


def analyze_backtest_report(
    report: dict,
    additional: dict | None = None,
    *,
    top_n: int = 10,
    focus_date: str | None = None,
    logger=None,
) -> dict:
    """Analyze one completed in-memory report and emit a readable diagnostic block.

    ``report`` and ``additional`` are the two dictionaries returned by
    ``backtest_runner.generate_backtest_report``. No command-line arguments or
    report files are involved. The returned dictionary is suitable for callers
    that need to consume the diagnostics programmatically.
    """
    if not isinstance(report, dict):
        raise TypeError(f"report must be a dict, got {type(report).__name__}")
    if additional is not None and not isinstance(additional, dict):
        raise TypeError(f"additional must be a dict or None, got {type(additional).__name__}")

    analysis = analyze_record(
        {"report": report, "additional": additional or {}},
        top_n=max(int(top_n), 0),
        focus_date=focus_date,
    )
    emit = logger.info if logger is not None else print
    print_analysis(analysis, emit=emit)
    return analysis
