#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
import pandas as pd

REPORTS_FILE = "reports.jsonl"
# ==========================
# Utils
# ==========================

def load_jsonl_iter(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            row["_line_no"] = line_no
            yield row


def get_nested(d: Dict[str, Any], path: str, default=None):
    cur = d

    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]

    return cur


def to_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def to_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def weighted_avg(x: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & w.notna() & (w > 0)

    if mask.sum() == 0:
        return np.nan

    return float(np.average(x[mask], weights=w[mask]))


# ==========================
# Model Info
# ==========================

def model_name(model: Dict[str, Any]) -> str:
    return f"{model.get('model_type')}v{model.get('model_version')}"


def extract_model_fields(prefix: str, model: Dict[str, Any]) -> Dict[str, Any]:
    best = get_nested(model, "metrics.Best_F1", {}) or {}
    train_params = model.get("train_params", {}) or {}
    model_cfg = train_params.get("model_cfg", {}) or {}
    cls1 = get_nested(best, "per_class.1", {}) or {}

    return {
        f"{prefix}_task_hash": model.get("task_hash"),
        f"{prefix}_task_type": model.get("task_type"),
        f"{prefix}_model": model_name(model),
        f"{prefix}_model_type": model.get("model_type"),
        f"{prefix}_model_version": model.get("model_version"),

        f"{prefix}_score": to_float(model.get("score")),
        f"{prefix}_macro_f1": to_float(best.get("macro_f1")),
        f"{prefix}_mcc": to_float(best.get("mcc")),
        f"{prefix}_accuracy": to_float(best.get("accuracy")),
        f"{prefix}_balanced_accuracy": to_float(best.get("balanced_accuracy")),
        f"{prefix}_test_loss": to_float(best.get("test_loss")),

        f"{prefix}_class1_f1": to_float(cls1.get("f1")),
        f"{prefix}_class1_recall": to_float(cls1.get("recall")),
        f"{prefix}_class1_precision": to_float(cls1.get("precision")),
        f"{prefix}_class1_precision_lift": to_float(cls1.get("precision_lift")),

        f"{prefix}_seq_len": model_cfg.get("seq_len"),
        f"{prefix}_stride": train_params.get("stride"),
        f"{prefix}_miss_penalty": train_params.get("miss_penalty"),
        f"{prefix}_flip_penalty": train_params.get("flip_penalty"),
        f"{prefix}_loss_fun_version": train_params.get("loss_fun_version"),
        f"{prefix}_save_dir": model.get("save_dir"),
    }


# ==========================
# Flatten
# ==========================

def flatten_one(
    row: Dict[str, Any],
    period: str,
    fee_key: str,
) -> Dict[str, Any] | None:

    if "signal_return" not in row:
        return None

    if fee_key not in row["signal_return"]:
        return None

    if "simulation" not in row:
        return None

    if period not in row["simulation"]:
        return None

    sig = row["signal_return"][fee_key]
    bt = row["simulation"][period]

    trigger = row["trigger"]
    direction = row["direction"]

    signal_count = to_int(sig.get("signal_count"))
    long_count = to_int(sig.get("long_count"))
    short_count = to_int(sig.get("short_count"))
    trade_signal_count = long_count + short_count

    out = {
        "line_no": row.get("_line_no"),
        "fusion_hash": row.get("fusion_hash"),
        "period": period,
        "pre_key": row.get("pre_key"),
        "train_compatibility": row.get("train_compatibility"),
        "fusion_dir": row.get("fusion_dir"),
        "prep_output_dir": row.get("prep_output_dir"),
        "elapsed_sec": to_float(row.get("elapsed_sec")),

        # quick signal return
        "fee_key": fee_key,
        "signal_avg_return": to_float(sig.get("signal_avg_return")),
        "signal_median_return": to_float(sig.get("signal_median_return")),
        "signal_win_rate": to_float(sig.get("signal_win_rate")),
        "signal_count": signal_count,
        "long_count": long_count,
        "short_count": short_count,
        "trade_signal_count": trade_signal_count,
        "signal_count_mismatch": signal_count != trade_signal_count,

        # backtest performance
        "bt_gross_return": to_float(get_nested(bt, "performance.gross_return")),
        "bt_cagr": to_float(get_nested(bt, "performance.cagr")),
        "bt_calmar": to_float(get_nested(bt, "performance.calmar")),
        "bt_sharpe": to_float(get_nested(bt, "performance.sharpe")),
        "bt_start_value": to_float(get_nested(bt, "performance.start_value")),
        "bt_end_value": to_float(get_nested(bt, "performance.end_value")),

        # drawdown
        "bt_max_dd_pct": to_float(get_nested(bt, "drawdown.max_dd_pct")),
        "bt_max_dd_amt": to_float(get_nested(bt, "drawdown.max_dd_amt")),
        "bt_max_daily_dd": to_float(get_nested(bt, "drawdown.max_daily_dd")),
        "bt_robust_max_daily_loss": to_float(get_nested(bt, "drawdown.robust_max_daily_loss")),
        "bt_dd_3_pct_days": to_int(get_nested(bt, "drawdown.dd_3_pct_days")),
        "bt_dd_4_pct_days": to_int(get_nested(bt, "drawdown.dd_4_pct_days")),
        "bt_dd_5_pct_days": to_int(get_nested(bt, "drawdown.dd_5_pct_days")),
        "bt_max_hwm_duration_days": to_int(get_nested(bt, "drawdown.max_hwm_duration_days")),

        # exposure
        "bt_avg_pos": to_float(get_nested(bt, "exposure.avg_pos")),
        "bt_max_pos": to_float(get_nested(bt, "exposure.max_pos")),
        "bt_p95_pos": to_float(get_nested(bt, "exposure.p95_pos")),
        "bt_trade_risk": to_float(get_nested(bt, "exposure.trade_risk")),

        # trades
        "bt_total_trades": to_int(get_nested(bt, "trades.total")),
        "bt_daily_freq": to_float(get_nested(bt, "trades.daily_freq")),
        "bt_win_rate": to_float(get_nested(bt, "trades.win_rate")),
        "bt_lost_longest": to_int(get_nested(bt, "trades.lost_longest")),
        "bt_won_longest": to_int(get_nested(bt, "trades.won_longest")),

        "bt_avg_pnl_gross": to_float(get_nested(bt, "trades.avg_pnl_gross")),
        "bt_avg_pct_gross": to_float(get_nested(bt, "trades.avg_pct_gross")),
        "bt_avg_pnl_net": to_float(get_nested(bt, "trades.avg_pnl_net")),
        "bt_avg_pct_net": to_float(get_nested(bt, "trades.avg_pct_net")),
        "bt_avg_cost": to_float(get_nested(bt, "trades.avg_cost")),

        "bt_long_pnl": to_float(get_nested(bt, "trades.long_pnl")),
        "bt_long_win_rate": to_float(get_nested(bt, "trades.long_win_rate")),
        "bt_short_pnl": to_float(get_nested(bt, "trades.short_pnl")),
        "bt_short_win_rate": to_float(get_nested(bt, "trades.short_win_rate")),

        # backtest model metrics
        "bt_model_accuracy": to_float(get_nested(bt, "model_metrics.accuracy")),
        "bt_model_f1_macro": to_float(get_nested(bt, "model_metrics.f1_macro")),
        "bt_model_f1_weighted": to_float(get_nested(bt, "model_metrics.f1_weighted")),
        "bt_model_signal_count": to_int(get_nested(bt, "model_metrics.signal.signal_count")),
        "bt_model_signal_coverage": to_float(get_nested(bt, "model_metrics.signal.coverage")),
        "bt_model_directional_accuracy": to_float(
            get_nested(bt, "model_metrics.signal.directional_accuracy")
        ),
    }

    out.update(extract_model_fields("trigger", trigger))
    out.update(extract_model_fields("direction", direction))

    return out


def build_rows(
    reports_path: str,
    fee_key: str,
    periods: List[str],
) -> pd.DataFrame:

    rows = []

    for row in load_jsonl_iter(reports_path):
        for period in periods:
            flat = flatten_one(row, period=period, fee_key=fee_key)

            if flat is not None:
                rows.append(flat)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No valid rows parsed.")

    return df


# ==========================
# Summary
# ==========================

def summarize_by_role(df: pd.DataFrame, role: str, sort_metric: str) -> pd.DataFrame:
    group_cols = [
        f"{role}_task_hash",
        f"{role}_model",
        f"{role}_model_type",
        f"{role}_model_version",
        f"{role}_seq_len",
        f"{role}_stride",
        f"{role}_miss_penalty",
        f"{role}_flip_penalty",
        f"{role}_loss_fun_version",
    ]

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        item = dict(zip(group_cols, keys))

        item.update({
            "n_rows": len(g),
            "n_fusions": g["fusion_hash"].nunique(),

            # signal return
            "mean_signal_avg_return": float(g["signal_avg_return"].mean()),
            "median_signal_avg_return": float(g["signal_avg_return"].median()),
            "max_signal_avg_return": float(g["signal_avg_return"].max()),
            "positive_signal_ratio": float((g["signal_avg_return"] > 0).mean()),
            "total_signal_count": int(g["signal_count"].sum()),
            "total_trade_signal_count": int(g["trade_signal_count"].sum()),
            "weighted_signal_return_by_signal_count": weighted_avg(
                g["signal_avg_return"],
                g["signal_count"],
            ),
            "weighted_signal_return_by_trade_count": weighted_avg(
                g["signal_avg_return"],
                g["trade_signal_count"],
            ),

            # backtest return
            "mean_bt_gross_return": float(g["bt_gross_return"].mean()),
            "median_bt_gross_return": float(g["bt_gross_return"].median()),
            "max_bt_gross_return": float(g["bt_gross_return"].max()),
            "positive_bt_return_ratio": float((g["bt_gross_return"] > 0).mean()),

            # avg gross
            "mean_bt_avg_pnl_gross": float(g["bt_avg_pnl_gross"].mean()),
            "median_bt_avg_pnl_gross": float(g["bt_avg_pnl_gross"].median()),
            "max_bt_avg_pnl_gross": float(g["bt_avg_pnl_gross"].max()),

            "mean_bt_avg_pct_gross": float(g["bt_avg_pct_gross"].mean()),
            "median_bt_avg_pct_gross": float(g["bt_avg_pct_gross"].median()),
            "max_bt_avg_pct_gross": float(g["bt_avg_pct_gross"].max()),

            # net
            "mean_bt_avg_pnl_net": float(g["bt_avg_pnl_net"].mean()),
            "mean_bt_avg_pct_net": float(g["bt_avg_pct_net"].mean()),

            # risk / quality
            "mean_bt_total_trades": float(g["bt_total_trades"].mean()),
            "mean_bt_daily_freq": float(g["bt_daily_freq"].mean()),
            "mean_bt_win_rate": float(g["bt_win_rate"].mean()),
            "mean_bt_sharpe": float(g["bt_sharpe"].mean()),
            "mean_bt_calmar": float(g["bt_calmar"].mean()),
            "mean_bt_max_dd_pct": float(g["bt_max_dd_pct"].mean()),

            # ML metric
            f"mean_{role}_macro_f1": float(g[f"{role}_macro_f1"].mean()),
            f"mean_{role}_mcc": float(g[f"{role}_mcc"].mean()),
            f"mean_{role}_score": float(g[f"{role}_score"].mean()),
            f"mean_{role}_class1_f1": float(g[f"{role}_class1_f1"].mean()),
            f"mean_{role}_class1_recall": float(g[f"{role}_class1_recall"].mean()),
            f"mean_{role}_class1_precision": float(g[f"{role}_class1_precision"].mean()),
        })

        rows.append(item)

    out = pd.DataFrame(rows)

    if len(out) and sort_metric in out.columns:
        out = out.sort_values(sort_metric, ascending=False)

    return out


def summarize_by_arch(df: pd.DataFrame, role: str, sort_metric: str) -> pd.DataFrame:
    group_cols = [
        f"{role}_model",
        f"{role}_model_type",
        f"{role}_model_version",
    ]

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        item = dict(zip(group_cols, keys))

        item.update({
            "n_rows": len(g),
            "n_fusions": g["fusion_hash"].nunique(),

            "mean_signal_avg_return": float(g["signal_avg_return"].mean()),
            "positive_signal_ratio": float((g["signal_avg_return"] > 0).mean()),

            "mean_bt_gross_return": float(g["bt_gross_return"].mean()),
            "positive_bt_return_ratio": float((g["bt_gross_return"] > 0).mean()),

            "mean_bt_avg_pnl_gross": float(g["bt_avg_pnl_gross"].mean()),
            "mean_bt_avg_pct_gross": float(g["bt_avg_pct_gross"].mean()),

            f"mean_{role}_macro_f1": float(g[f"{role}_macro_f1"].mean()),
            f"mean_{role}_mcc": float(g[f"{role}_mcc"].mean()),
            f"mean_{role}_score": float(g[f"{role}_score"].mean()),
        })

        rows.append(item)

    out = pd.DataFrame(rows)

    if len(out) and sort_metric in out.columns:
        out = out.sort_values(sort_metric, ascending=False)

    return out


# ==========================
# Correlation
# ==========================

def corr_pair(df: pd.DataFrame, x: str, y: str) -> Dict[str, Any]:
    tmp = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) < 3:
        return {
            "x": x,
            "y": y,
            "n": len(tmp),
            "pearson_corr": np.nan,
            "spearman_corr": np.nan,
        }

    return {
        "x": x,
        "y": y,
        "n": len(tmp),
        "pearson_corr": float(tmp[x].corr(tmp[y], method="pearson")),
        "spearman_corr": float(tmp[x].corr(tmp[y], method="spearman")),
    }


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        # signal_return vs backtest return / avg gross
        ("signal_avg_return", "bt_gross_return"),
        ("signal_avg_return", "bt_avg_pnl_gross"),
        ("signal_avg_return", "bt_avg_pct_gross"),
        ("signal_avg_return", "bt_avg_pnl_net"),
        ("signal_avg_return", "bt_avg_pct_net"),

        # signal win rate
        ("signal_win_rate", "bt_gross_return"),
        ("signal_win_rate", "bt_avg_pnl_gross"),
        ("signal_win_rate", "bt_avg_pct_gross"),

        # count
        ("signal_count", "bt_gross_return"),
        ("signal_count", "bt_avg_pnl_gross"),
        ("signal_count", "bt_avg_pct_gross"),
        ("trade_signal_count", "bt_gross_return"),
        ("trade_signal_count", "bt_avg_pnl_gross"),
        ("trade_signal_count", "bt_avg_pct_gross"),

        # ML metrics vs backtest
        ("trigger_macro_f1", "bt_gross_return"),
        ("trigger_macro_f1", "bt_avg_pnl_gross"),
        ("direction_macro_f1", "bt_gross_return"),
        ("direction_macro_f1", "bt_avg_pnl_gross"),
        ("direction_mcc", "bt_gross_return"),
        ("direction_mcc", "bt_avg_pnl_gross"),
    ]

    rows = [corr_pair(df, x, y) for x, y in pairs]

    out = pd.DataFrame(rows)

    if len(out):
        out["abs_pearson_corr"] = out["pearson_corr"].abs()
        out = out.sort_values("abs_pearson_corr", ascending=False)

    return out


# ==========================
# Save Outputs
# ==========================

def save_outputs(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, "fusion_all_rows.csv"), index=False)

    # 1. signal_return 分析
    fusion_by_signal = df.sort_values(
        ["signal_avg_return", "signal_count"],
        ascending=[False, False],
    )
    fusion_by_signal.to_csv(
        os.path.join(out_dir, "fusion_rank_by_signal_return.csv"),
        index=False,
    )

    summarize_by_role(df, "trigger", "mean_signal_avg_return").to_csv(
        os.path.join(out_dir, "trigger_summary_by_signal_return.csv"),
        index=False,
    )
    summarize_by_role(df, "direction", "mean_signal_avg_return").to_csv(
        os.path.join(out_dir, "direction_summary_by_signal_return.csv"),
        index=False,
    )
    summarize_by_arch(df, "trigger", "mean_signal_avg_return").to_csv(
        os.path.join(out_dir, "trigger_arch_by_signal_return.csv"),
        index=False,
    )
    summarize_by_arch(df, "direction", "mean_signal_avg_return").to_csv(
        os.path.join(out_dir, "direction_arch_by_signal_return.csv"),
        index=False,
    )

    # 2. 回测收益分析
    fusion_by_bt = df.sort_values(
        ["bt_gross_return", "bt_sharpe", "bt_calmar"],
        ascending=[False, False, False],
    )
    fusion_by_bt.to_csv(
        os.path.join(out_dir, "fusion_rank_by_backtest_return.csv"),
        index=False,
    )

    summarize_by_role(df, "trigger", "mean_bt_gross_return").to_csv(
        os.path.join(out_dir, "trigger_summary_by_backtest_return.csv"),
        index=False,
    )
    summarize_by_role(df, "direction", "mean_bt_gross_return").to_csv(
        os.path.join(out_dir, "direction_summary_by_backtest_return.csv"),
        index=False,
    )
    summarize_by_arch(df, "trigger", "mean_bt_gross_return").to_csv(
        os.path.join(out_dir, "trigger_arch_by_backtest_return.csv"),
        index=False,
    )
    summarize_by_arch(df, "direction", "mean_bt_gross_return").to_csv(
        os.path.join(out_dir, "direction_arch_by_backtest_return.csv"),
        index=False,
    )

    # 3. Avg Gross 分析
    fusion_by_avg_gross = df.sort_values(
        ["bt_avg_pnl_gross", "bt_avg_pct_gross", "bt_gross_return"],
        ascending=[False, False, False],
    )
    fusion_by_avg_gross.to_csv(
        os.path.join(out_dir, "fusion_rank_by_avg_gross.csv"),
        index=False,
    )

    summarize_by_role(df, "trigger", "mean_bt_avg_pnl_gross").to_csv(
        os.path.join(out_dir, "trigger_summary_by_avg_gross.csv"),
        index=False,
    )
    summarize_by_role(df, "direction", "mean_bt_avg_pnl_gross").to_csv(
        os.path.join(out_dir, "direction_summary_by_avg_gross.csv"),
        index=False,
    )
    summarize_by_arch(df, "trigger", "mean_bt_avg_pnl_gross").to_csv(
        os.path.join(out_dir, "trigger_arch_by_avg_gross.csv"),
        index=False,
    )
    summarize_by_arch(df, "direction", "mean_bt_avg_pnl_gross").to_csv(
        os.path.join(out_dir, "direction_arch_by_avg_gross.csv"),
        index=False,
    )

    # 4. 相关性
    corr = correlation_table(df)
    corr.to_csv(
        os.path.join(out_dir, "correlation_signal_return_vs_backtest.csv"),
        index=False,
    )

    # 5. sanity
    sanity = df[df["signal_count_mismatch"]].copy()
    sanity.to_csv(
        os.path.join(out_dir, "sanity_signal_count_mismatch.csv"),
        index=False,
    )

    return {
        "fusion_by_signal": fusion_by_signal,
        "fusion_by_bt": fusion_by_bt,
        "fusion_by_avg_gross": fusion_by_avg_gross,
        "corr": corr,
        "sanity": sanity,
    }


# ==========================
# Main
# ==========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--reports",
        default="/home/chao/work/quant_output/batch_train/DOGEUSDT_15m/2026-06-24/17_56_42/batch_simulation",
    )

    parser.add_argument(
        "--fee-key",
        default="0",
        help='Signal return fee key, e.g. "0" or "0.005"',
    )

    parser.add_argument(
        "--period",
        default="all",
        help='forward / short / long / all',
    )

    parser.add_argument(
        "-o",
        "--out-dir",
        default=None,
    )

    args = parser.parse_args()

    reports_path = os.path.join(args.reports,REPORTS_FILE)

    if args.period == "all":
        periods = ["forward", "short", "long"]
    else:
        periods = [args.period]

    if args.out_dir is None:
        out_dir = os.path.join(
            str(Path(reports_path).parent),
            "analysis_backtest_signal",
        )
    else:
        out_dir = args.out_dir

    df = build_rows(
        reports_path=reports_path,
        fee_key=args.fee_key,
        periods=periods,
    )

    df = df.dropna(
        subset=[
            "signal_avg_return",
            "bt_gross_return",
            "bt_avg_pnl_gross",
            "bt_avg_pct_gross",
        ]
    ).copy()

    if len(df) == 0:
        raise RuntimeError("No valid rows after dropping NaN metrics.")

    outputs = save_outputs(df, out_dir)

    print(f"Rows parsed: {len(df)}")
    print(f"Output dir: {out_dir}")

    print("\nTop by signal_return:")
    print(
        outputs["fusion_by_signal"][
            [
                "fusion_hash",
                "period",
                "signal_avg_return",
                "signal_count",
                "trade_signal_count",
                "bt_gross_return",
                "bt_avg_pnl_gross",
                "bt_avg_pct_gross",
                "trigger_model",
                "direction_model",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nTop by backtest gross_return:")
    print(
        outputs["fusion_by_bt"][
            [
                "fusion_hash",
                "period",
                "bt_gross_return",
                "bt_avg_pnl_gross",
                "bt_avg_pct_gross",
                "bt_sharpe",
                "bt_calmar",
                "signal_avg_return",
                "trigger_model",
                "direction_model",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nTop by Avg Gross:")
    print(
        outputs["fusion_by_avg_gross"][
            [
                "fusion_hash",
                "period",
                "bt_avg_pnl_gross",
                "bt_avg_pct_gross",
                "bt_gross_return",
                "signal_avg_return",
                "bt_total_trades",
                "trigger_model",
                "direction_model",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nCorrelation:")
    print(outputs["corr"].to_string(index=False))

    if len(outputs["sanity"]) > 0:
        print(f"\nWARNING: signal_count mismatch rows: {len(outputs['sanity'])}")
        print("Check sanity_signal_count_mismatch.csv")


if __name__ == "__main__":
    main()