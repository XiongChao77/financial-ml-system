"""Analyze in-memory backtest diagnostics and render report equity curves."""

from __future__ import annotations

from collections import defaultdict
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


def plot_equity_curves(
    all_results,
    output_dir: str,
    file_name: str = "equity_full_combined.png",
    start_index: int = 0,
    *,
    price_file: str | None = None,
    logger=None,
) -> str | None:
    """Plot one or more report equity curves over the underlying market price.

    ``all_results`` accepts either one report returned by ``backtest_runner`` or
    selected-config records containing ``long``/``short``/``forward`` reports.
    Batch records are chained in chronological period order and normalized to a
    starting equity of 1. The saved image path is returned.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from data_process import common

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
    equity_axis = price_axis.twinx()
    equity_axis.set_ylabel("Continuous Strategy Equity (Normalized)")

    split_dates = {}
    plotted_count = 0
    has_multi_period_record = False
    for index, result in enumerate(results):
        reports_by_period = period_reports(result)
        has_multi_period_record |= len(reports_by_period) > 1
        segments = []
        current_multiplier = 1.0
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
            frame.set_index("date", inplace=True)
            initial_equity = frame["equity"].iloc[0]
            if initial_equity == 0:
                continue
            split_dates.setdefault(period, frame.index[0])
            frame["continuous_equity"] = (
                frame["equity"] / initial_equity * current_multiplier
            )
            segments.append(frame[["continuous_equity"]])
            current_multiplier = frame["continuous_equity"].iloc[-1]

        if not segments:
            continue
        full_path = pd.concat(segments)
        full_path = full_path[~full_path.index.duplicated(keep="first")]
        params_hash = safe_get(first_report if len(results) == 1 else result, ["params", "hash"])
        label = str(params_hash) if params_hash else f"S{start_index + index}"
        equity_axis.plot(
            full_path.index,
            full_path["continuous_equity"],
            linewidth=1.5,
            alpha=0.8,
            label=label,
        )
        plotted_count += 1

    if plotted_count == 0:
        plt.close(fig)
        return None

    if has_multi_period_record:
        if "short" in split_dates:
            price_axis.axvline(
                split_dates["short"], color="blue", linestyle="--", linewidth=1, alpha=0.3
            )
        if "forward" in split_dates:
            price_axis.axvline(
                split_dates["forward"], color="red", linestyle="--", linewidth=1, alpha=0.3
            )

    price_handles, price_labels = price_axis.get_legend_handles_labels()
    equity_handles, equity_labels = equity_axis.get_legend_handles_labels()
    price_axis.legend(
        price_handles + equity_handles,
        price_labels + equity_labels,
        loc="upper left",
        ncol=4,
        fontsize=8,
    )
    title = (
        "Strategy Performance: Long (Train) -> Short (Val) -> Forward (Test)"
        if has_multi_period_record
        else "Strategy Equity Curve"
    )
    plt.title(title)
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

        sl_price = entry.get("sl_price")
        tp_price = entry.get("tp_price")
        bar_high = to_float(entry.get("bar_high"))
        bar_low = to_float(entry.get("bar_low"))
        entry_bar_stop_breached = False
        entry_bar_tp_touched = False
        if sl_price is not None:
            sl_price = to_float(sl_price)
            entry_bar_stop_breached = bar_low <= sl_price if is_long else bar_high >= sl_price
        if tp_price is not None:
            tp_price = to_float(tp_price)
            entry_bar_tp_touched = bar_high >= tp_price if is_long else bar_low <= tp_price

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
            "planned_stop_loss_pct": entry.get("planned_stop_loss_pct") or entry.get("sl_pct"),
            "sl_price": sl_price,
            "tp_price": tp_price,
            "entry_bar_open": entry.get("bar_open"),
            "entry_bar_high": entry.get("bar_high"),
            "entry_bar_low": entry.get("bar_low"),
            "entry_bar_close": entry.get("bar_close"),
            "entry_bar_stop_breached": entry_bar_stop_breached,
            "entry_bar_tp_touched": entry_bar_tp_touched,
        })

    return rows


def focus_trade_diagnostics(report: dict, target_date: str | None, top_n: int) -> dict:
    round_trips = [
        item for item in pair_round_trips(report)
        if item["exit_date"] == target_date
    ]
    round_trips.sort(key=lambda item: item["estimated_net_pnl"])
    entry_bar_stop_breaches = [
        item for item in round_trips
        if item["entry_date"] == target_date and item["entry_bar_stop_breached"]
    ]
    return {
        "closed_trade_count": len(round_trips),
        "carry_in_trade_count": sum(item["entry_date"] != target_date for item in round_trips),
        "gross_pnl": sum(item["gross_pnl"] for item in round_trips),
        "estimated_commission": sum(item["estimated_commission"] for item in round_trips),
        "estimated_net_pnl": sum(item["estimated_net_pnl"] for item in round_trips),
        "worst_trades": round_trips[:top_n],
        "entry_bar_stop_breaches": entry_bar_stop_breaches[:top_n],
    }


def detail_rows(additional: dict, key: str) -> list[dict]:
    rows = safe_get(additional, ["strategy_detail", key], []) or []
    return rows if isinstance(rows, list) else []


def gap_stop_orders(report: dict, additional: dict) -> list[dict]:
    rows = detail_rows(additional, "gap_through_stop_orders")
    if rows:
        return rows
    return [
        item for item in trade_logs(report)
        if item.get("role") == "sl" and to_float(item.get("gap_through_stop_pct"), 0.0) > 0.0
    ]


def same_bar_orders(report: dict, additional: dict) -> list[dict]:
    rows = detail_rows(additional, "same_bar_tp_sl_orders")
    if rows:
        return rows
    return [item for item in trade_logs(report) if item.get("same_bar_tp_sl_hit")]


def group_gap_by_day(orders: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for order in orders:
        date_key = date_from_order(order)
        if date_key is None:
            continue
        row = grouped.setdefault(
            date_key,
            {
                "date": date_key,
                "count": 0,
                "at_open_count": 0,
                "max_gap_through_stop_pct": 0.0,
                "sum_gap_through_stop_pct": 0.0,
            },
        )
        gap_pct = to_float(order.get("gap_through_stop_pct"), 0.0)
        row["count"] += 1
        row["at_open_count"] += 1 if order.get("gap_stop_at_open") else 0
        row["sum_gap_through_stop_pct"] += gap_pct
        row["max_gap_through_stop_pct"] = max(row["max_gap_through_stop_pct"], gap_pct)
    return sorted(
        grouped.values(),
        key=lambda item: (item["max_gap_through_stop_pct"], item["count"]),
        reverse=True,
    )


def summarize_same_bar(orders: list[dict]) -> dict:
    by_role = defaultdict(int)
    for order in orders:
        by_role[order.get("role") or order.get("same_bar_outcome") or "unknown"] += 1
    return {
        "count": len(orders),
        "as_stop_loss": by_role.get("sl", 0),
        "as_take_profit": by_role.get("tp", 0),
        "unknown": sum(v for k, v in by_role.items() if k not in {"sl", "tp"}),
    }


def fmt_pct(value) -> str:
    return f"{to_float(value) * 100:.4f}%"


def fmt_number(value, decimals: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{to_float(value):,.{decimals}f}"


def short_trade_line(trade: dict) -> str:
    flags = []
    if trade.get("entry_bar_stop_breached"):
        flags.append("ENTRY_BAR_SL_BREACH")
    if trade.get("entry_bar_tp_touched"):
        flags.append("ENTRY_BAR_TP_TOUCH")
    flag_text = f" flags={','.join(flags)}" if flags else ""
    return (
        f"{trade.get('side')} entry={trade.get('entry_time_utc')}@{to_float(trade.get('entry_price')):.6g} "
        f"exit={trade.get('exit_time_utc')}@{to_float(trade.get('exit_price')):.6g} "
        f"role={trade.get('exit_role')} qty={to_float(trade.get('qty')):.6g} "
        f"move={fmt_pct(trade.get('price_return_pct'))} "
        f"gross={fmt_number(trade.get('gross_pnl'))} "
        f"est_fee={fmt_number(trade.get('estimated_commission'))} "
        f"est_net={fmt_number(trade.get('estimated_net_pnl'))} "
        f"planned_sl={fmt_pct(trade.get('planned_stop_loss_pct'))}"
        f"{flag_text}"
    )


def short_order_line(order: dict) -> str:
    return (
        f"date={date_from_order(order)} "
        f"role={order.get('role')} "
        f"price={to_float(order.get('price')):.6g} "
        f"entry={to_float(order.get('entry_ref_price')):.6g} "
        f"sl={to_float(order.get('sl_price')):.6g} "
        f"tp={to_float(order.get('tp_price')):.6g} "
        f"bar=[o={to_float(order.get('bar_open')):.6g}, "
        f"h={to_float(order.get('bar_high')):.6g}, "
        f"l={to_float(order.get('bar_low')):.6g}, "
        f"c={to_float(order.get('bar_close')):.6g}] "
        f"open_gap={fmt_pct(order.get('open_gap_pct')) if order.get('open_gap_pct') is not None else 'NA'} "
        f"gap_stop={fmt_pct(order.get('gap_through_stop_pct')) if order.get('gap_through_stop_pct') is not None else 'NA'} "
        f"actual_sl={fmt_pct(order.get('actual_stop_loss_pct')) if order.get('actual_stop_loss_pct') is not None else 'NA'}"
    )


def analyze_record(record: dict, top_n: int = 10, focus_date: str | None = None) -> dict:
    report = record["report"]
    additional = record["additional"]
    worst_day = worst_daily_loss(report)
    target_date = focus_date or worst_day.get("date")

    gaps = gap_stop_orders(report, additional)
    same_bar = same_bar_orders(report, additional)
    gap_by_day = detail_rows(additional, "gap_through_stop_by_day") or group_gap_by_day(gaps)
    target_gap_orders = [item for item in gaps if date_from_order(item) == target_date]
    target_same_bar_orders = [item for item in same_bar if date_from_order(item) == target_date]

    strategy_summary = report.get("strategy") or {}
    return {
        "params_hash": safe_get(report, ["params", "hash"]),
        "worst_day": worst_day,
        "focus_date": target_date,
        "focus_equity": daily_equity_context(report, target_date),
        "focus_trades": focus_trade_diagnostics(report, target_date, top_n),
        "strategy_summary": {
            "gap_through_stop_count": strategy_summary.get("gap_through_stop_count", len(gaps)),
            "gap_through_stop_at_open_count": strategy_summary.get("gap_through_stop_at_open_count"),
            "max_gap_through_stop_pct": strategy_summary.get(
                "max_gap_through_stop_pct",
                max((to_float(item.get("gap_through_stop_pct"), 0.0) for item in gaps), default=0.0),
            ),
            "max_gap_through_stop_date": strategy_summary.get("max_gap_through_stop_date"),
            "same_bar_tp_sl_hit_count": strategy_summary.get("same_bar_tp_sl_hit_count", len(same_bar)),
            "same_bar_tp_sl_as_stop_count": strategy_summary.get("same_bar_tp_sl_as_stop_count"),
            "same_bar_tp_sl_as_tp_count": strategy_summary.get("same_bar_tp_sl_as_tp_count"),
        },
        "gap_by_day": gap_by_day[:top_n],
        "focus_gap_orders": sorted(
            target_gap_orders,
            key=lambda item: to_float(item.get("gap_through_stop_pct"), 0.0),
            reverse=True,
        )[:top_n],
        "same_bar_summary": summarize_same_bar(same_bar),
        "focus_same_bar_summary": summarize_same_bar(target_same_bar_orders),
        "focus_same_bar_orders": target_same_bar_orders[:top_n],
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
    for trade in focus_trades["entry_bar_stop_breaches"]:
        emit(
            "  ENTRY-BAR STOP BREACH: "
            f"planned_sl={to_float(trade.get('sl_price')):.6g} "
            f"entry_bar=[o={to_float(trade.get('entry_bar_open')):.6g}, "
            f"h={to_float(trade.get('entry_bar_high')):.6g}, "
            f"l={to_float(trade.get('entry_bar_low')):.6g}, "
            f"c={to_float(trade.get('entry_bar_close')):.6g}] "
            f"eventual_exit_role={trade.get('exit_role')} "
            f"eventual_exit={to_float(trade.get('exit_price')):.6g}"
        )

    summary = item["strategy_summary"]
    emit(
        "Gap-through stops: "
        f"count={summary['gap_through_stop_count']} "
        f"at_open={summary['gap_through_stop_at_open_count']} "
        f"max_gap={fmt_pct(summary['max_gap_through_stop_pct'])} "
        f"max_gap_date={summary['max_gap_through_stop_date']}"
    )
    for row in item["gap_by_day"]:
        emit(
            "  gap_day "
            f"date={row.get('date')} count={row.get('count')} "
            f"at_open={row.get('at_open_count')} "
            f"max_gap={fmt_pct(row.get('max_gap_through_stop_pct'))} "
            f"sum_gap={fmt_pct(row.get('sum_gap_through_stop_pct'))}"
        )

    if item["focus_gap_orders"]:
        emit("  focus gap stop orders:")
        for order in item["focus_gap_orders"]:
            emit(f"    {short_order_line(order)}")
    else:
        emit("  focus gap stop orders: none")

    same_bar = item["same_bar_summary"]
    focus_same = item["focus_same_bar_summary"]
    emit(
        "Same-bar TP/SL: "
        f"count={same_bar['count']} "
        f"as_sl={same_bar['as_stop_loss']} "
        f"as_tp={same_bar['as_take_profit']} "
        f"unknown={same_bar['unknown']}"
    )
    emit(
        "Focus same-bar TP/SL: "
        f"count={focus_same['count']} "
        f"as_sl={focus_same['as_stop_loss']} "
        f"as_tp={focus_same['as_take_profit']} "
        f"unknown={focus_same['unknown']}"
    )
    if item["focus_same_bar_orders"]:
        emit("  focus same-bar orders:")
        for order in item["focus_same_bar_orders"]:
            emit(f"    {short_order_line(order)}")
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
