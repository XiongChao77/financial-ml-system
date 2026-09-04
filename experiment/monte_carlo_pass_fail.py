#!/usr/bin/env python3
"""Estimate account pass/fail probabilities with Monte Carlo simulation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Rules and trade outcome assumptions for one account simulation."""

    initial_equity: float = 10_000.0
    profit_target_pct: float = 10.0
    trailing_drawdown_pct: float = 10.0
    win_probability: float = 0.5
    trade_win_pct: float = 1.0
    trade_loss_pct: float = 1.0
    max_trades: int = 10_000

    @property
    def pass_equity(self) -> float:
        return self.initial_equity * (1.0 + self .profit_target_pct / 100.0)

    @property
    def trailing_drawdown_amount(self) -> float:
        return self.initial_equity * self.trailing_drawdown_pct / 100.0

    @property
    def trade_win_amount(self) -> float:
        return self.initial_equity * self.trade_win_pct / 100.0

    @property
    def trade_loss_amount(self) -> float:
        return self.initial_equity * self.trade_loss_pct / 100.0


@dataclass(frozen=True)
class ChunkResult:
    simulations: int
    pass_count: int
    fail_count: int
    unresolved_count: int
    pass_trade_sum: int
    fail_trade_sum: int


def validate_config(config: SimulationConfig) -> None:
    if not math.isfinite(config.initial_equity) or config.initial_equity <= 0:
        raise ValueError("Initial equity must be a positive finite number")
    for name in (
        "profit_target_pct",
        "trailing_drawdown_pct",
        "trade_win_pct",
        "trade_loss_pct",
    ):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name.replace('_', ' ').title()} must be positive")
    if (
        not math.isfinite(config.win_probability)
        or not 0.0 <= config.win_probability <= 1.0
    ):
        raise ValueError("Win probability must be between 0 and 1")
    if config.max_trades <= 0:
        raise ValueError("Maximum trades must be positive")


def classify_trade_path(
    pnl_values: Iterable[float],
    config: SimulationConfig,
) -> dict[str, float | int | str]:
    """Classify a deterministic closed-trade PnL path for rule verification."""
    validate_config(config)
    equity = config.initial_equity
    peak_equity = equity

    for trade_number, pnl in enumerate(pnl_values, start=1):
        if not math.isfinite(pnl):
            raise ValueError("Trade PnL values must be finite")
        equity += pnl
        peak_equity = max(peak_equity, equity)
        tolerance = max(1.0, abs(equity), abs(peak_equity)) * 1e-12

        if equity + tolerance >= config.pass_equity:
            status = "pass"
        elif equity <= peak_equity - config.trailing_drawdown_amount + tolerance:
            status = "fail"
        else:
            continue

        return {
            "status": status,
            "trades": trade_number,
            "final_equity": equity,
            "peak_equity": peak_equity,
            "fail_equity": peak_equity - config.trailing_drawdown_amount,
        }

    return {
        "status": "unresolved",
        "trades": trade_number if "trade_number" in locals() else 0,
        "final_equity": equity,
        "peak_equity": peak_equity,
        "fail_equity": peak_equity - config.trailing_drawdown_amount,
    }


def simulate_chunk(
    config: SimulationConfig,
    simulations: int,
    seed: int,
) -> ChunkResult:
    """Simulate a chunk of independent account paths using vectorized arrays."""
    validate_config(config)
    if simulations <= 0:
        raise ValueError("Simulation count must be positive")

    rng = np.random.default_rng(seed)
    equity = np.full(simulations, config.initial_equity, dtype=np.float64)
    peak_equity = equity.copy()
    status = np.zeros(simulations, dtype=np.int8)
    resolution_trade = np.zeros(simulations, dtype=np.int32)

    for trade_number in range(1, config.max_trades + 1):
        active_indices = np.flatnonzero(status == 0)
        if active_indices.size == 0:
            break

        wins = rng.random(active_indices.size) < config.win_probability
        pnl = np.where(wins, config.trade_win_amount, -config.trade_loss_amount)
        equity[active_indices] += pnl
        peak_equity[active_indices] = np.maximum(
            peak_equity[active_indices],
            equity[active_indices],
        )

        active_equity = equity[active_indices]
        active_peaks = peak_equity[active_indices]
        tolerance = (
            np.maximum.reduce(
                (
                    np.ones(active_indices.size),
                    np.abs(active_equity),
                    np.abs(active_peaks),
                )
            )
            * 1e-12
        )
        passed = active_equity + tolerance >= config.pass_equity
        failed = (~passed) & (
            active_equity <= active_peaks - config.trailing_drawdown_amount + tolerance
        )

        resolved_indices = active_indices[passed | failed]
        status[active_indices[passed]] = 1
        status[active_indices[failed]] = -1
        resolution_trade[resolved_indices] = trade_number

    passed_mask = status == 1
    failed_mask = status == -1
    unresolved_mask = status == 0
    return ChunkResult(
        simulations=simulations,
        pass_count=int(passed_mask.sum()),
        fail_count=int(failed_mask.sum()),
        unresolved_count=int(unresolved_mask.sum()),
        pass_trade_sum=int(resolution_trade[passed_mask].sum()),
        fail_trade_sum=int(resolution_trade[failed_mask].sum()),
    )


def split_simulations(simulations: int, workers: int) -> list[int]:
    chunk_count = min(simulations, workers)
    base, remainder = divmod(simulations, chunk_count)
    return [base + (index < remainder) for index in range(chunk_count)]


def wilson_interval(successes: int, total: int) -> list[float | None]:
    """Return a 95% Wilson score interval for a binomial proportion."""
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def probability_result(count: int, total: int) -> dict[str, Any]:
    return {
        "count": count,
        "probability": count / total if total else None,
        "confidence_interval_95": wilson_interval(count, total),
    }


def run_simulation(
    config: SimulationConfig,
    simulations: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    validate_config(config)
    if simulations <= 0:
        raise ValueError("Simulation count must be positive")
    if workers <= 0:
        raise ValueError("Worker count must be positive")

    chunks = split_simulations(simulations, workers)
    child_sequences = np.random.SeedSequence(seed).spawn(len(chunks))
    child_seeds = [
        int(sequence.generate_state(1, dtype=np.uint64)[0])
        for sequence in child_sequences
    ]

    if len(chunks) == 1:
        chunk_results = [simulate_chunk(config, chunks[0], child_seeds[0])]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(chunks)
        ) as executor:
            futures = [
                executor.submit(simulate_chunk, config, chunk, child_seed)
                for chunk, child_seed in zip(chunks, child_seeds, strict=True)
            ]
            chunk_results = [future.result() for future in futures]

    pass_count = sum(result.pass_count for result in chunk_results)
    fail_count = sum(result.fail_count for result in chunk_results)
    unresolved_count = sum(result.unresolved_count for result in chunk_results)
    resolved_count = pass_count + fail_count
    pass_trade_sum = sum(result.pass_trade_sum for result in chunk_results)
    fail_trade_sum = sum(result.fail_trade_sum for result in chunk_results)

    result = {
        "config": asdict(config),
        "thresholds": {
            "pass_equity": config.pass_equity,
            "initial_fail_equity": (
                config.initial_equity - config.trailing_drawdown_amount
            ),
            "trailing_drawdown_amount": config.trailing_drawdown_amount,
            "trade_win_amount": config.trade_win_amount,
            "trade_loss_amount": config.trade_loss_amount,
        },
        "simulation": {
            "simulations": simulations,
            "workers": len(chunks),
            "seed": seed,
        },
        "results": {
            "pass": probability_result(pass_count, simulations),
            "fail": probability_result(fail_count, simulations),
            "unresolved": probability_result(unresolved_count, simulations),
            "resolved": resolved_count,
            "conditional_on_resolution": {
                "pass_probability": (
                    pass_count / resolved_count if resolved_count else None
                ),
                "fail_probability": (
                    fail_count / resolved_count if resolved_count else None
                ),
            },
            "mean_trades_to_pass": (
                pass_trade_sum / pass_count if pass_count else None
            ),
            "mean_trades_to_fail": (
                fail_trade_sum / fail_count if fail_count else None
            ),
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate pass/fail probabilities for an account with a fixed profit "
            "target and a trailing drawdown measured from peak closed equity."
        )
    )
    parser.add_argument("--simulations", type=int, default=100_000)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--profit-target-pct", type=float, default=10.0)
    parser.add_argument("--trailing-drawdown-pct", type=float, default=10.0)
    parser.add_argument("--win-probability", type=float, default=0.5)
    parser.add_argument(
        "--trade-win-pct",
        type=float,
        default=1.0,
        help="Winning trade PnL as a percentage of initial equity",
    )
    parser.add_argument(
        "--trade-loss-pct",
        type=float,
        default=1.0,
        help="Losing trade PnL as a percentage of initial equity",
    )
    parser.add_argument("--max-trades", type=int, default=10_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        initial_equity=args.initial_equity,
        profit_target_pct=args.profit_target_pct,
        trailing_drawdown_pct=args.trailing_drawdown_pct,
        win_probability=args.win_probability,
        trade_win_pct=args.trade_win_pct,
        trade_loss_pct=args.trade_loss_pct,
        max_trades=args.max_trades,
    )
    result = run_simulation(
        config=config,
        simulations=args.simulations,
        workers=args.workers,
        seed=args.seed,
    )
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    print(serialized)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
