"""Where the martingale lost money, drawn on the price series.

Two different things go wrong in a martingale run, and the chart marks both:

* a **losing trade** - the ladder was driven through every layer and the cycle
  was stopped out.  Frequent, survivable, and the thing that actually grinds an
  account down.
* an **account failure** - liquidation, or equity through the ruin threshold.
  Rare, terminal, and always preceded by some of the above.

One figure, four stacked panels sharing the time axis:

1. the market, with every losing trade and every account failure on it;
2. one horizontal segment per run, from its random entry to its exit;
3. losing trades per time bucket;
4. account failures per time bucket, split by cause.

Panels 3 and 4 are small multiples, not a dual axis: losses outnumber failures
by two orders of magnitude, so sharing a y-scale would flatten one of them.

Colour never carries meaning alone here - each class also has its own marker
shape and a legend entry.  The failure causes use the reserved status colours
(serious, critical); a losing trade is not a status, so it takes a categorical
slot instead of a third, lighter red that no one could tell apart.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")   # headless: this module only ever writes files

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# Validated against the light chart surface (#fcfcfb) with --pairs all:
# worst CVD ΔE 13.0, worst normal-vision ΔE 15.7, both clear of the floors.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
PRICE_LINE = "#52514e"
PRICE_BAND = "#c9c8c4"

LOSS_STYLE = {"color": "#4a3aa7", "marker": ".", "label": "Losing trade"}
FAILURE_STYLE = {
    "ruin_threshold": {"color": "#ec835a", "marker": "v", "label": "Ruin threshold"},
    "liquidation": {"color": "#d03b3b", "marker": "X", "label": "Liquidation"},
}
SURVIVED = {"color": "#2a78d6", "marker": "o", "label": "Survived"}

# Above this many losing trades the price panel becomes an ink blot and the
# render crawls, so it draws an evenly spaced subsample - and says so.
MAX_LOSS_MARKERS = 20_000


def _decimate(times: np.ndarray, low: np.ndarray, high: np.ndarray,
              close: np.ndarray, max_points: int = 3_000):
    """Bucket a long series down to a drawable width, keeping the extremes.

    A stride would drop the spikes that killed the accounts, which are the
    whole point of the chart; per-bucket min/max keeps them.
    """
    count = len(close)
    if count <= max_points:
        return times, low, high, close

    edges = np.unique(np.linspace(0, count, max_points + 1).astype(int))
    starts, stops = edges[:-1], edges[1:]
    keep = stops > starts
    starts, stops = starts[keep], stops[keep]

    return (
        times[starts],
        np.minimum.reduceat(low, starts),
        np.maximum.reduceat(high, starts),
        close[stops - 1],
    )


def _style_axis(axis, ylabel: str):
    axis.set_facecolor(SURFACE)
    axis.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    axis.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK_MUTED, labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def _failure_points(runs: Sequence[dict], bars) -> dict:
    """Group the runs that died by cause into (time, price) arrays."""
    grouped: dict[str, dict] = {}
    for run in runs:
        if not run.get("ruined"):
            continue
        reason = run.get("exit_reason", "liquidation")
        price = run.get("failure_price")
        if price is None or not np.isfinite(price):
            price = float(bars.close[run["end_index"]])
        bucket = grouped.setdefault(reason, {"time": [], "price": []})
        bucket["time"].append(run["end_time"])
        bucket["price"].append(float(price))
    return grouped


def _loss_points(trades: Sequence[dict]) -> dict:
    """Every cycle that closed at a loss, at the time and price it closed."""
    losses = [trade for trade in trades if trade["net_pnl"] < 0.0]
    return {
        "time": [trade["closed_at"] for trade in losses],
        "price": [float(trade["exit_price"]) for trade in losses],
        "net_pnl": [float(trade["net_pnl"]) for trade in losses],
        "full_ladder": sum(bool(trade.get("full_layers")) for trade in losses),
        "count": len(losses),
    }


def _subsample(values: Sequence, limit: int):
    """Evenly spaced subsample, deterministic and order preserving."""
    if len(values) <= limit:
        return list(values), False
    keep = np.linspace(0, len(values) - 1, limit).astype(int)
    return [values[index] for index in keep], True


def _bucket_counts(times, edges) -> np.ndarray:
    if not len(times):
        return np.zeros(len(edges) - 1)
    counts, _ = np.histogram(
        mdates.date2num(pd.to_datetime(list(times))), bins=mdates.date2num(edges)
    )
    return counts


def plot_failures(
    logger: logging.Logger,
    bars,
    result: dict,
    path: str,
    *,
    title: Optional[str] = None,
    max_points: int = 3_000,
    max_loss_markers: int = MAX_LOSS_MARKERS,
    buckets: int = 60,
) -> Optional[str]:
    """Draw the market with every losing trade and account failure marked."""
    runs = result.get("runs", [])
    if not runs:
        logger.warning(f"No Monte Carlo runs to plot, skipping {path}")
        return None

    times = pd.to_datetime(bars.time)
    plot_time, plot_low, plot_high, plot_close = _decimate(
        times.to_numpy(), bars.low, bars.high, bars.close, max_points
    )
    plot_time = pd.to_datetime(plot_time)

    losses = _loss_points(result.get("trades", []))
    failures = _failure_points(runs, bars)
    failed_total = sum(len(group["time"]) for group in failures.values())

    figure, (price_axis, span_axis, loss_axis, fail_axis) = plt.subplots(
        4, 1,
        figsize=(15, 12.5),
        height_ratios=(3.0, 1.9, 0.95, 0.95),
        sharex=True,
        constrained_layout=True,
    )
    figure.patch.set_facecolor(SURFACE)

    # ---------- 1. the market, with both kinds of loss on it ----------
    _style_axis(price_axis, "Price (log)")
    price_axis.fill_between(
        plot_time, plot_low, plot_high,
        color=PRICE_BAND, linewidth=0, alpha=0.9, zorder=1,
    )
    price_axis.plot(plot_time, plot_close, color=PRICE_LINE, linewidth=0.9, zorder=2)
    price_axis.set_yscale("log")

    handles = []
    if losses["count"]:
        shown_time, trimmed = _subsample(losses["time"], max_loss_markers)
        shown_price, _ = _subsample(losses["price"], max_loss_markers)
        price_axis.scatter(
            pd.to_datetime(shown_time), shown_price,
            s=9, marker=LOSS_STYLE["marker"], color=LOSS_STYLE["color"],
            alpha=0.30, linewidths=0, zorder=3,
        )
        label = (
            f"{LOSS_STYLE['label']} ({losses['count']}, "
            f"{losses['full_ladder']} at full ladder)"
        )
        if trimmed:
            logger.info(
                f"Price panel shows {len(shown_time)} of {losses['count']} "
                f"losing trades (evenly spaced subsample)"
            )
            label += f" — showing {len(shown_time)}"
        handles.append(
            Line2D([], [], color=LOSS_STYLE["color"], marker="o", linestyle="none",
                   markersize=5, label=label)
        )

    # Failures go on top: two orders of magnitude rarer, and terminal.
    for reason in ("ruin_threshold", "liquidation"):
        group = failures.get(reason)
        if not group:
            continue
        style = FAILURE_STYLE[reason]
        price_axis.scatter(
            pd.to_datetime(group["time"]), group["price"],
            s=70, marker=style["marker"], color=style["color"],
            edgecolors=SURFACE, linewidths=0.8, zorder=4,
        )
        handles.append(
            Line2D([], [], color=style["color"], marker=style["marker"],
                   linestyle="none", markersize=8,
                   label=f"{style['label']} ({len(group['time'])})")
        )
    if handles:
        price_axis.legend(
            handles=handles, loc="upper left", frameon=False,
            fontsize=9, labelcolor=INK_MUTED,
        )
    price_axis.set_title(
        title or "Martingale losses and account failures on the price series",
        color=INK, fontsize=12, loc="left", pad=8,
    )
    aggregate = result.get("aggregate", {})
    price_axis.text(
        0.995, 0.02,
        f"{aggregate.get('runs', len(runs))} runs · "
        f"{failed_total} failed ({aggregate.get('ruin_rate', 0.0) * 100:.1f}%) · "
        f"{aggregate.get('total_trades', 0)} trades",
        transform=price_axis.transAxes, ha="right", va="bottom",
        color=INK_MUTED, fontsize=9,
    )

    # ---------- 2. one segment per run ----------
    _style_axis(span_axis, "Run (by entry)")
    ordered = sorted(runs, key=lambda run: pd.Timestamp(run["start_time"]))
    observed = {}
    for row, run in enumerate(ordered):
        reason = run["exit_reason"] if run.get("ruined") else None
        style = FAILURE_STYLE.get(reason, SURVIVED) if reason else SURVIVED
        observed[style["label"]] = style
        start, end = pd.Timestamp(run["start_time"]), pd.Timestamp(run["end_time"])
        span_axis.plot(
            [start, end], [row, row],
            color=style["color"], linewidth=1.1, alpha=0.65, solid_capstyle="butt",
        )
        span_axis.plot(
            [end], [row],
            marker=style["marker"], color=style["color"],
            markersize=4.5, markeredgewidth=0.0,
        )
    span_axis.set_ylim(-1, len(ordered))
    # Only outcomes that actually happened: an unused legend entry reads as a
    # category with zero observations rather than one that never applied.
    span_axis.legend(
        handles=[
            Line2D([], [], color=style["color"], marker=style["marker"],
                   linewidth=1.6, markersize=6, label=label)
            for label, style in observed.items()
        ],
        loc="upper left", frameon=False, fontsize=9,
        ncol=len(observed), labelcolor=INK_MUTED,
    )

    # ---------- 3 & 4. counts per bucket, as small multiples ----------
    edges = pd.date_range(times.min(), times.max(), periods=buckets + 1)
    centres = mdates.date2num(edges[:-1])
    width = (edges[1] - edges[0]).total_seconds() / 86400.0 * 0.9

    _style_axis(loss_axis, "Losing trades")
    loss_axis.bar(
        centres + width / 2, _bucket_counts(losses["time"], edges),
        width=width, color=LOSS_STYLE["color"], linewidth=0.8, edgecolor=SURFACE,
    )

    _style_axis(fail_axis, "Account failures")
    bottom = np.zeros(len(edges) - 1)
    for reason in ("ruin_threshold", "liquidation"):
        group = failures.get(reason)
        if not group:
            continue
        style = FAILURE_STYLE[reason]
        counts = _bucket_counts(group["time"], edges)
        fail_axis.bar(
            centres + width / 2, counts, width=width, bottom=bottom,
            color=style["color"], linewidth=0.8, edgecolor=SURFACE,
        )
        bottom = bottom + counts
    fail_axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator())
    )

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    logger.info(
        f"Loss/failure chart saved to {path} "
        f"({losses['count']} losing trades, {failed_total} account failures)"
    )
    return path


# ============================================================
# Where and at what scale the stop losses pile up
# ============================================================
# Diverging pair for the pressure map: blue (calmer than random) through a
# neutral gray midpoint to red (more stop losses than random).  Both arms are
# monotone in lightness and step-matched, so neither pole dominates the other.
DIVERGING_LOW = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = ["#fbd7d5", "#f4b0ac", "#ec8a85", "#e34948", "#bf3232", "#952727", "#6b1c1c"]
# Single-hue ramp for a magnitude (probability) heatmap.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Ordinal steps for the quantile curves: they are an ordered family, so they
# take one hue by lightness rather than four unrelated colours.
ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#0d366b"]


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "pressure", DIVERGING_LOW + [DIVERGING_MID] + DIVERGING_HIGH
    )


def _sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("density", SEQUENTIAL)


def scale_grid(min_bars: int, max_bars: int, count: int) -> np.ndarray:
    """Geometric ladder of window sizes, in bars, with no repeats."""
    if max_bars <= min_bars:
        return np.array([max(1, min_bars)], dtype=int)
    grid = np.geomspace(max(1, min_bars), max_bars, count)
    return np.unique(np.round(grid).astype(int))


def _loss_and_exposure(runs: Sequence[dict], trades: Sequence[dict], total_bars: int):
    """Per-bar count of stop losses, and how many runs were trading that bar.

    Exposure matters: runs start at random bars and die, so a period that simply
    had more accounts alive would otherwise look like a period of more danger.
    """
    losses = np.zeros(total_bars, dtype=float)
    for trade in trades:
        if trade["net_pnl"] < 0.0:
            index = int(trade["exit_bar"])
            if 0 <= index < total_bars:
                losses[index] += 1.0

    # Difference array: +1 at each run's first bar, -1 just past its last.
    edges = np.zeros(total_bars + 1, dtype=float)
    for run in runs:
        start = max(0, int(run["start_index"]))
        stop = min(total_bars, int(run["end_index"]) + 1)
        if stop > start:
            edges[start] += 1.0
            edges[stop] -= 1.0
    return losses, np.cumsum(edges[:-1])


def _rolling_sums(values: np.ndarray, centres: np.ndarray, window: int):
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    half = window // 2
    starts = np.clip(centres - half, 0, len(values) - 1)
    stops = np.clip(starts + window, 0, len(values))
    starts = np.clip(stops - window, 0, len(values))
    return cumulative[stops] - cumulative[starts]


def pressure_field(
    runs: Sequence[dict],
    trades: Sequence[dict],
    total_bars: int,
    scales: np.ndarray,
    columns: int = 900,
):
    """Z(t, m): stop losses in a window of m bars, in sigmas above random.

    The null is a Poisson process whose rate is the pooled losses per
    active-run-bar, so Z compares each window against what that much trading
    would have produced if losses were scattered uniformly.
    """
    losses, exposure = _loss_and_exposure(runs, trades, total_bars)
    total_exposure = exposure.sum()
    if total_exposure <= 0.0 or losses.sum() <= 0.0:
        return None

    rate = losses.sum() / total_exposure
    centres = np.unique(np.linspace(0, total_bars - 1, min(columns, total_bars)).astype(int))
    field = np.full((len(scales), len(centres)), np.nan)
    for row, window in enumerate(scales):
        observed = _rolling_sums(losses, centres, int(window))
        expected = rate * _rolling_sums(exposure, centres, int(window))
        usable = expected > 1e-9
        field[row, usable] = (observed[usable] - expected[usable]) / np.sqrt(expected[usable])
    return {
        "z": field,
        "centres": centres,
        "scales": scales,
        "exposure": exposure,
        "rate": rate,
    }


def plot_pressure_heatmap(
    logger: logging.Logger,
    bars,
    result: dict,
    path: str,
    *,
    title: Optional[str] = None,
    min_scale: int = 8,
    max_scale: Optional[int] = None,
    scale_count: int = 40,
    columns: int = 900,
) -> Optional[str]:
    """Time x window-scale map of standardized stop-loss pressure."""
    runs, trades = result.get("runs", []), result.get("trades", [])
    total_bars = len(bars)
    if not runs or not trades:
        logger.warning(f"Nothing to map, skipping {path}")
        return None

    scales = scale_grid(min_scale, max_scale or max(min_scale * 2, total_bars // 8), scale_count)
    field = pressure_field(runs, trades, total_bars, scales, columns)
    if field is None:
        logger.warning(f"No stop losses to map, skipping {path}")
        return None

    times = pd.to_datetime(bars.time)
    x = times[field["centres"]]
    finite = field["z"][np.isfinite(field["z"])]
    # Symmetric limits so the neutral midpoint really sits at Z = 0, clipped at
    # a high percentile so one extreme window cannot flatten the whole map.
    limit = float(np.percentile(np.abs(finite), 99)) if len(finite) else 1.0
    limit = max(limit, 1.0)

    figure, (exposure_axis, heat_axis) = plt.subplots(
        2, 1, figsize=(15, 7.5), height_ratios=(1.0, 4.0),
        sharex=True, constrained_layout=True,
    )
    figure.patch.set_facecolor(SURFACE)

    _style_axis(exposure_axis, "Runs active")
    exposure_axis.fill_between(
        times[field["centres"]], field["exposure"][field["centres"]],
        color=PRICE_BAND, linewidth=0,
    )
    exposure_axis.set_title(
        title or "Stop-loss pressure by time and window scale",
        color=INK, fontsize=12, loc="left", pad=8,
    )

    _style_axis(heat_axis, "Window m (bars, log)")
    heat_axis.grid(False)
    mesh = heat_axis.pcolormesh(
        x, field["scales"], np.ma.masked_invalid(field["z"]),
        cmap=_diverging_cmap(), vmin=-limit, vmax=limit, shading="nearest",
    )
    heat_axis.set_yscale("log")
    heat_axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(mdates.AutoDateLocator())
    )
    bar = figure.colorbar(mesh, ax=heat_axis, pad=0.01)
    bar.set_label(
        "Z: sigmas above a random process of the same rate",
        color=INK_MUTED, fontsize=9,
    )
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_edgecolor(GRID)
    heat_axis.text(
        0.995, 0.03,
        f"{int(sum(t['net_pnl'] < 0 for t in trades))} stop losses · "
        f"rate {field['rate'] * 1000:.2f} per 1000 run-bars · "
        f"|Z| clipped at {limit:.1f}",
        transform=heat_axis.transAxes, ha="right", va="bottom",
        color=INK_MUTED, fontsize=9,
    )

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    logger.info(f"Pressure heatmap saved to {path}")
    return path


# ============================================================
# How small clusters merge into big ones as the gap allowance grows
# ============================================================
def cluster_sizes(positions: Sequence[float], max_gap: float) -> np.ndarray:
    """Single-linkage cluster sizes: split wherever the gap exceeds max_gap."""
    ordered = np.sort(np.asarray(positions, dtype=float))
    if not len(ordered):
        return np.array([], dtype=int)
    breaks = np.flatnonzero(np.diff(ordered) > max_gap) + 1
    return np.diff(np.concatenate(([0], breaks, [len(ordered)]))).astype(int)


def cluster_size_by_scale(trades: Sequence[dict], scales: np.ndarray) -> dict:
    """Pool cluster sizes over every run, once per scale.

    Clustering is done inside a run: two accounts being stopped out at the same
    moment are not one cluster, they are the same market event hitting twice.
    """
    positions_by_run: dict[int, list] = {}
    for trade in trades:
        if trade["net_pnl"] < 0.0:
            positions_by_run.setdefault(trade.get("run_id", 0), []).append(
                trade["exit_bar"] - trade.get("start_index", 0)
            )
    runs = [np.sort(np.asarray(values, dtype=float)) for values in positions_by_run.values()]
    if not runs:
        return {}

    sizes_by_scale, summary = {}, []
    for scale in scales:
        sizes = np.concatenate([cluster_sizes(run, float(scale)) for run in runs])
        sizes_by_scale[int(scale)] = sizes
        summary.append({
            "m": int(scale),
            "clusters": int(len(sizes)),
            "median": float(np.median(sizes)),
            "p75": float(np.percentile(sizes, 75)),
            "p95": float(np.percentile(sizes, 95)),
            "max": int(sizes.max()),
            "mean": float(sizes.mean()),
        })
    return {"sizes": sizes_by_scale, "curves": summary}


def plot_cluster_scales(
    logger: logging.Logger,
    result: dict,
    path: str,
    *,
    title: Optional[str] = None,
    min_scale: int = 1,
    max_scale: int = 5_000,
    scale_count: int = 40,
) -> Optional[str]:
    """Cluster-size distribution and quantile curves against the gap allowance."""
    trades = result.get("trades", [])
    scales = scale_grid(min_scale, max_scale, scale_count)
    analysis = cluster_size_by_scale(trades, scales)
    if not analysis:
        logger.warning(f"No stop losses to cluster, skipping {path}")
        return None

    curves = pd.DataFrame(analysis["curves"])
    largest = max(int(sizes.max()) for sizes in analysis["sizes"].values())
    edges = np.unique(np.round(np.geomspace(1, max(largest, 2) + 1, 45)).astype(int))

    # Column-normalized: each m is its own distribution, so the panel answers
    # "what share of clusters have this size", not "which m had more losses".
    density = np.full((len(edges) - 1, len(scales)), np.nan)
    for column, scale in enumerate(scales):
        counts, _ = np.histogram(analysis["sizes"][int(scale)], bins=edges)
        total = counts.sum()
        if total:
            density[:, column] = counts / total

    figure, (heat_axis, curve_axis, count_axis) = plt.subplots(
        3, 1, figsize=(13, 11), height_ratios=(2.4, 1.8, 1.0),
        sharex=True, constrained_layout=True,
    )
    figure.patch.set_facecolor(SURFACE)

    _style_axis(heat_axis, "Cluster size (stop losses)")
    heat_axis.grid(False)
    mesh = heat_axis.pcolormesh(
        scales, edges[:-1], np.ma.masked_invalid(density),
        cmap=_sequential_cmap(), shading="nearest",
    )
    heat_axis.set_yscale("log")
    heat_axis.set_title(
        title or "How stop losses merge into clusters as the gap allowance grows",
        color=INK, fontsize=12, loc="left", pad=8,
    )
    bar = figure.colorbar(mesh, ax=heat_axis, pad=0.01)
    bar.set_label("Share of clusters at this m", color=INK_MUTED, fontsize=9)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    bar.outline.set_edgecolor(GRID)

    _style_axis(curve_axis, "Cluster size")
    series = (("median", "Median"), ("p75", "P75"), ("p95", "P95"), ("max", "Max"))
    endings = []
    for (column, label), colour in zip(series, ORDINAL):
        curve_axis.plot(
            curves["m"], curves[column], color=colour, linewidth=2.0, label=label,
        )
        endings.append([float(curves[column].iloc[-1]), label, colour])

    # Direct labels at the right end, decluttered: P95 and Max converge once m
    # swallows every gap, and stacked text at the same y is unreadable.
    endings.sort(key=lambda item: item[0])
    minimum_ratio = 1.18   # the axis is log, so separation is multiplicative
    for lower, upper in zip(endings, endings[1:]):
        if upper[0] < lower[0] * minimum_ratio:
            upper[0] = lower[0] * minimum_ratio
    for y, label, colour in endings:
        curve_axis.annotate(
            label, (curves["m"].iloc[-1], y),
            xytext=(8, 0), textcoords="offset points",
            color=colour, fontsize=9, va="center", annotation_clip=False,
        )
    curve_axis.set_xscale("log")
    curve_axis.set_yscale("log")
    curve_axis.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_MUTED)

    _style_axis(count_axis, "Clusters")
    count_axis.plot(curves["m"], curves["clusters"], color=ORDINAL[1], linewidth=2.0)
    count_axis.set_xscale("log")
    count_axis.set_yscale("log")
    count_axis.set_xlabel("Gap allowance m (bars, log)", color=INK_MUTED, fontsize=9)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    logger.info(
        f"Cluster-scale chart saved to {path} "
        f"(m from {scales[0]} to {scales[-1]} bars, largest cluster {largest})"
    )
    return path
