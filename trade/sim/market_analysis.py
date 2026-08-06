"""Compare Martingale stop counts on real prices and random walks.

For every ``max_layers`` value from 1 through M and every configured
``price_deviation_pct``, the scanner replays independent Long and Short
ladders. Each crossed adverse boundary enters the next layer, and both
boundaries reset symmetrically around the latest layer entry. A stop is counted
when price reaches the first forbidden rung after the ladder is full. At layer ``i``, both boundary distances are
``price_deviation_pct * deviation_step_mult ** (i - 1)``.

By default, the baseline uses Gaussian IID zero-drift arithmetic-return increments calibrated
to the observed volatility. This is a thin-tail null for exposing excess
tail risk. The optional ``bootstrap`` model samples observed returns with
replacement; it preserves heavy tails and is intended for testing serial
dependence instead. All panels share the same seeded random walks.

Both real and synthetic scans use closes.  Applying real intrabar extremes
without a corresponding synthetic OHLC model would bias the comparison.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from numba import njit


@dataclass(frozen=True)
class MarketAnalysisParams:
    """Grid of the sweep and the size of the random baseline."""

    max_layers: int = 10            # one panel per layer count 1..max_layers
    deviation_min: float = 0.002
    deviation_max: float = 0.05
    deviation_count: int = 20
    deviation_step_mult: float = 1.0
    baseline_model: str = "gaussian"  # thin-tail null; "bootstrap" preserves tails
    baseline_repeats: int = 20      # repeated null paths
    seed: int = 11
    workers: int = 0                # 0 = cpu_count - 1, 1 = serial

    def __post_init__(self):
        if self.max_layers < 1:
            raise ValueError("max_layers must be >= 1")
        if not 0.0 < self.deviation_min <= self.deviation_max < 1.0:
            raise ValueError("need 0 < deviation_min <= deviation_max < 1")
        if self.deviation_count < 2:
            raise ValueError("deviation_count must be >= 2")
        if self.deviation_step_mult <= 0.0:
            raise ValueError("deviation_step_mult must be positive")
        if self.baseline_model not in {"gaussian", "bootstrap"}:
            raise ValueError("baseline_model must be gaussian or bootstrap")
        if self.baseline_repeats < 0:
            raise ValueError("baseline_repeats must be >= 0")

    @property
    def deviations(self) -> np.ndarray:
        return np.geomspace(
            self.deviation_min, self.deviation_max, self.deviation_count
        )


# ============================================================
# Market-only martingale cycle simulation
# ============================================================
@njit(cache=True)
def _scan_ladder_kernel(
    prices: np.ndarray,
    side: int,
    deviation: float,
    max_layers: int,
    deviation_step_mult: float,
) -> tuple[int, int]:
    stops = 0
    profits = 0
    active = False
    previous_price = float(prices[0])
    last_entry = 0.0
    filled_layers = 0

    for price in prices:
        if not active:
            active = True
            last_entry = price
            filled_layers = 1
            previous_price = price
            continue

        if side * (price - previous_price) < 0.0:
            while active:
                step = deviation * (
                    deviation_step_mult ** (filled_layers - 1)
                )
                if step >= 1.0:
                    break
                if filled_layers < max_layers:
                    rung = last_entry * (1.0 - side * step)
                    tolerance = max(abs(price), abs(rung)) * 1e-12
                    if side * (price - rung) > tolerance:
                        break
                    last_entry = rung
                    filled_layers += 1
                    continue

                stop_price = last_entry * (1.0 - side * step)
                tolerance = max(abs(price), abs(stop_price)) * 1e-12
                if side * (price - stop_price) <= tolerance:
                    stops += 1
                    active = False
                break
        else:
            step = deviation * (
                deviation_step_mult ** (filled_layers - 1)
            )
            if step >= 1.0:
                previous_price = price
                continue
            take_profit = last_entry * (1.0 + side * step)
            tolerance = max(abs(price), abs(take_profit)) * 1e-12
            if side * (price - take_profit) >= -tolerance:
                profits += 1
                active = False

        previous_price = price

    return stops, profits


def scan_ladder(
    prices: np.ndarray,
    side: int,
    deviation: float,
    max_layers: int,
    params: MarketAnalysisParams,
) -> tuple[int, int]:
    """Replay a ladder with symmetric, step-scaled boundaries per layer."""
    return _scan_ladder_kernel(
        np.asarray(prices, dtype=np.float64),
        side,
        deviation,
        max_layers,
        params.deviation_step_mult,
    )


def stop_curve(prices: np.ndarray, layers: int, params: MarketAnalysisParams) -> dict:
    """Exact market-only ladder outcomes over the configured deviation grid."""
    long_stops, short_stops = [], []
    long_exits, short_exits = [], []
    for deviation in params.deviations:
        stops, profits = scan_ladder(prices, 1, deviation, layers, params)
        long_stops.append(stops)
        long_exits.append(profits)
        stops, profits = scan_ladder(prices, -1, deviation, layers, params)
        short_stops.append(stops)
        short_exits.append(profits)
    return {
        "long_stops": np.asarray(long_stops, dtype=float),
        "short_stops": np.asarray(short_stops, dtype=float),
        "long_exits": np.asarray(long_exits, dtype=float),
        "short_exits": np.asarray(short_exits, dtype=float),
    }


def random_walk_prices(
    start_price: float,
    returns: np.ndarray,
    rng: np.random.Generator,
    model: str = "gaussian",
) -> np.ndarray:
    """Random walk under either a thin-tail or empirical-return null."""
    if model == "gaussian":
        volatility = float(np.std(returns, ddof=1))
        increments = rng.normal(0.0, volatility, size=len(returns))
    elif model == "bootstrap":
        centered = returns - np.mean(returns)
        increments = rng.choice(centered, size=len(returns), replace=True)
    else:
        raise ValueError(f"Unsupported random-walk model: {model}")
    factors = np.maximum(1.0 + increments, np.finfo(float).tiny)
    return start_price * np.concatenate(([1.0], np.cumprod(factors)))


# ============================================================
# The sweep
# ============================================================
_WORKER_PRICES: Optional[np.ndarray] = None
_WORKER_RETURNS: Optional[np.ndarray] = None
_WORKER_PARAMS: Optional[MarketAnalysisParams] = None


def _worker_init(prices, returns, params):
    global _WORKER_PRICES, _WORKER_RETURNS, _WORKER_PARAMS
    _WORKER_PRICES, _WORKER_RETURNS, _WORKER_PARAMS = prices, returns, params


def _layer_task(layers: int) -> dict:
    params = _WORKER_PARAMS
    real = stop_curve(_WORKER_PRICES, layers, params)
    # Every layer count reuses the same seed, so the panels are driven by one
    # set of random walks and differ only in the ladder depth.
    rng = np.random.default_rng(params.seed)
    draws = {"long_stops": [], "short_stops": []}
    for _ in range(params.baseline_repeats):
        curve = stop_curve(
            random_walk_prices(
                _WORKER_PRICES[0], _WORKER_RETURNS, rng, params.baseline_model
            ),
            layers,
            params,
        )
        draws["long_stops"].append(curve["long_stops"])
        draws["short_stops"].append(curve["short_stops"])

    baseline = {}
    for key, values in draws.items():
        if not values:
            continue
        stacked = np.vstack(values)
        baseline[key] = {
            "mean": stacked.mean(axis=0),
            "p25": np.percentile(stacked, 25, axis=0),
            "p75": np.percentile(stacked, 75, axis=0),
        }
    return {"layers": layers, "real": real, "baseline": baseline}


def analyze_market(
    logger: logging.Logger,
    closes: Sequence[float],
    params: MarketAnalysisParams = MarketAnalysisParams(),
) -> dict:
    """Sweep every layer count, comparing real prices with IID random walks."""
    prices = np.asarray(closes, dtype=float)
    if (
        len(prices) < 100
        or not np.all(np.isfinite(prices))
        or np.any(prices <= 0.0)
    ):
        raise ValueError("need at least 100 positive closes to analyze")

    returns = np.diff(prices) / prices[:-1]
    layer_counts = list(range(1, params.max_layers + 1))
    workers = params.workers if params.workers > 0 else max(1, (os.cpu_count() or 1) - 1)
    workers = max(1, min(workers, len(layer_counts)))
    logger.info(
        f"Market analysis | {len(layer_counts)} layer counts x "
        f"{params.deviation_count} deviations x "
        f"{params.baseline_repeats} {params.baseline_model} IID random walks over {workers} processes"
    )

    if workers == 1:
        _worker_init(prices, returns, params)
        panels = [_layer_task(layers) for layers in layer_counts]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(prices, returns, params),
        ) as pool:
            panels = list(pool.map(_layer_task, layer_counts))

    panels.sort(key=lambda panel: panel["layers"])
    return {
        "params": params,
        "deviations": params.deviations,
        "bars": len(prices),
        "panels": panels,
        "summary": _summarize(panels, params.deviations),
    }


def _summarize(panels: Sequence[dict], deviations: np.ndarray) -> list:
    """Where each ladder depth departs most from the random baseline."""
    rows = []
    for panel in panels:
        row = {"layers": panel["layers"]}
        for side in ("long", "short"):
            key = f"{side}_stops"
            real = panel["real"][key]
            reference = panel["baseline"].get(key, {}).get("mean")
            if reference is None:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(reference > 0, real / reference, np.nan)
            if np.all(np.isnan(ratio)):
                continue
            worst = int(np.nanargmax(ratio))
            row[f"{side}_max_ratio"] = float(ratio[worst])
            row[f"{side}_max_ratio_deviation"] = float(deviations[worst])
            row[f"{side}_mean_ratio"] = float(np.nanmean(ratio))
        rows.append(row)
    return rows


def log_summary(logger: logging.Logger, analysis: dict):
    logger.info(
        f"{'LAYERS':<8}{'LONG x RANDOM':>16}{'at dev':>9}"
        f"{'SHORT x RANDOM':>17}{'at dev':>9}{'LONG mean':>12}{'SHORT mean':>12}"
    )
    for row in analysis["summary"]:
        logger.info(
            f"{row['layers']:<8}"
            f"{row.get('long_max_ratio', float('nan')):>15.2f}x"
            f"{row.get('long_max_ratio_deviation', float('nan')) * 100:>8.2f}%"
            f"{row.get('short_max_ratio', float('nan')):>16.2f}x"
            f"{row.get('short_max_ratio_deviation', float('nan')) * 100:>8.2f}%"
            f"{row.get('long_mean_ratio', float('nan')):>11.2f}x"
            f"{row.get('short_mean_ratio', float('nan')):>11.2f}x"
        )
