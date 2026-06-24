#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPORTS_FILE = "reports.jsonl"

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                row["_line_no"] = line_no
                rows.append(row)
            except json.JSONDecodeError as e:
                print(f"Skip bad json line={line_no}: {e}")

    return rows


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


def get_signal_block(row: Dict[str, Any], fee_key: str) -> Optional[Dict[str, Any]]:
    """
    兼容两种字段名：
    - signal_return
    - signal_avg_return
    """

    signal_all = row.get("signal_return")
    if signal_all is None:
        signal_all = row.get("signal_avg_return")

    if not isinstance(signal_all, dict):
        return None

    if fee_key in signal_all:
        return signal_all[fee_key]

    # 兼容 "0.0" / "0"
    alt_key = str(float(fee_key)) if fee_key != "0" else "0.0"
    if alt_key in signal_all:
        return signal_all[alt_key]

    return None


def model_name(model: Dict[str, Any]) -> str:
    return f"{model.get('model_type')}v{model.get('model_version')}"


def extract_model_fields(prefix: str, model: Dict[str, Any]) -> Dict[str, Any]:
    best = get_nested(model, "metrics.Best_F1", {})

    train_params = model.get("train_params", {})
    model_cfg = train_params.get("model_cfg", {})

    out = {
        f"{prefix}_task_hash": model.get("task_hash"),
        f"{prefix}_task_type": model.get("task_type"),
        f"{prefix}_model_type": model.get("model_type"),
        f"{prefix}_model_version": model.get("model_version"),
        f"{prefix}_model": model_name(model),
        f"{prefix}_score": to_float(model.get("score")),
        f"{prefix}_score_source": model.get("score_source"),

        f"{prefix}_macro_f1": to_float(best.get("macro_f1")),
        f"{prefix}_macro_precision": to_float(best.get("macro_precision")),
        f"{prefix}_macro_recall": to_float(best.get("macro_recall")),
        f"{prefix}_mcc": to_float(best.get("mcc")),
        f"{prefix}_accuracy": to_float(best.get("accuracy")),
        f"{prefix}_balanced_accuracy": to_float(best.get("balanced_accuracy")),
        f"{prefix}_test_loss": to_float(best.get("test_loss")),
        f"{prefix}_sample_count": to_int(best.get("sample_count")),

        f"{prefix}_seq_len": model_cfg.get("seq_len"),
        f"{prefix}_stride": train_params.get("stride"),
        f"{prefix}_miss_penalty": train_params.get("miss_penalty"),
        f"{prefix}_flip_penalty": train_params.get("flip_penalty"),
        f"{prefix}_loss_fun_version": train_params.get("loss_fun_version"),
        f"{prefix}_save_dir": model.get("save_dir"),
    }

    # per_class.1 常用于 trigger 正类
    cls1 = get_nested(best, "per_class.1", {})
    out.update({
        f"{prefix}_class1_f1": to_float(cls1.get("f1")),
        f"{prefix}_class1_recall": to_float(cls1.get("recall")),
        f"{prefix}_class1_precision": to_float(cls1.get("precision")),
        f"{prefix}_class1_precision_lift": to_float(cls1.get("precision_lift")),
        f"{prefix}_class1_pred_ratio": to_float(cls1.get("pred_ratio")),
        f"{prefix}_class1_base_rate": to_float(cls1.get("base_rate")),
    })

    return out


def flatten_report(row: Dict[str, Any], fee_key: str) -> Optional[Dict[str, Any]]:
    sig = get_signal_block(row, fee_key)
    if sig is None:
        return None

    trigger = row.get("trigger", {})
    direction = row.get("direction", {})

    signal_count = to_int(sig.get("signal_count"))
    long_count = to_int(sig.get("long_count"))
    short_count = to_int(sig.get("short_count"))
    trade_signal_count = long_count + short_count

    out = {
        "line_no": row.get("_line_no"),
        "fusion_hash": row.get("fusion_hash"),
        "pre_key": row.get("pre_key"),
        "train_compatibility": row.get("train_compatibility"),
        "fusion_dir": row.get("fusion_dir"),
        "prep_output_dir": row.get("prep_output_dir"),
        "device": row.get("device"),

        "fee_key": fee_key,
        "fee_per_trade": to_float(sig.get("fee_per_trade")),
        "horizon": to_int(sig.get("horizon")),

        "signal_avg_return": to_float(sig.get("signal_avg_return")),
        "signal_median_return": to_float(sig.get("signal_median_return")),
        "signal_win_rate": to_float(sig.get("signal_win_rate")),
        "signal_count": signal_count,
        "long_count": long_count,
        "short_count": short_count,
        "trade_signal_count": trade_signal_count,
        "count_mismatch": signal_count != trade_signal_count,

        "quick_eval_passed": row.get("quick_eval_passed"),
        "has_simulation": bool(row.get("simulation")),
    }

    out.update(extract_model_fields("trigger", trigger))
    out.update(extract_model_fields("direction", direction))

    return out


def weighted_avg(x: pd.Series, w: pd.Series) -> float:
    mask = x.notna() & w.notna() & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(x[mask], weights=w[mask]))


def summarize_by_model(df: pd.DataFrame, role: str) -> pd.DataFrame:
    """
    role = trigger / direction
    """

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
        row = dict(zip(group_cols, keys))

        row.update({
            "n_fusions": len(g),
            "mean_signal_avg_return": float(g["signal_avg_return"].mean()),
            "median_signal_avg_return": float(g["signal_avg_return"].median()),
            "max_signal_avg_return": float(g["signal_avg_return"].max()),
            "min_signal_avg_return": float(g["signal_avg_return"].min()),

            "weighted_avg_return_by_signal_count": weighted_avg(
                g["signal_avg_return"],
                g["signal_count"],
            ),
            "weighted_avg_return_by_trade_signal_count": weighted_avg(
                g["signal_avg_return"],
                g["trade_signal_count"],
            ),

            "positive_return_ratio": float((g["signal_avg_return"] > 0).mean()),
            "total_signal_count": int(g["signal_count"].sum()),
            "total_trade_signal_count": int(g["trade_signal_count"].sum()),
            "mean_signal_count": float(g["signal_count"].mean()),
            "mean_trade_signal_count": float(g["trade_signal_count"].mean()),

            f"mean_{role}_macro_f1": float(g[f"{role}_macro_f1"].mean()),
            f"mean_{role}_mcc": float(g[f"{role}_mcc"].mean()),
            f"mean_{role}_score": float(g[f"{role}_score"].mean()),
            f"mean_{role}_class1_f1": float(g[f"{role}_class1_f1"].mean()),
            f"mean_{role}_class1_recall": float(g[f"{role}_class1_recall"].mean()),
            f"mean_{role}_class1_precision": float(g[f"{role}_class1_precision"].mean()),
        })

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            ["weighted_avg_return_by_signal_count", "mean_signal_avg_return", "n_fusions"],
            ascending=[False, False, False],
        )

    return out


def summarize_by_arch(df: pd.DataFrame, role: str) -> pd.DataFrame:
    group_cols = [
        f"{role}_model",
        f"{role}_model_type",
        f"{role}_model_version",
    ]

    rows = []

    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))

        row.update({
            "n_fusions": len(g),
            "mean_signal_avg_return": float(g["signal_avg_return"].mean()),
            "median_signal_avg_return": float(g["signal_avg_return"].median()),
            "weighted_avg_return_by_signal_count": weighted_avg(
                g["signal_avg_return"],
                g["signal_count"],
            ),
            "positive_return_ratio": float((g["signal_avg_return"] > 0).mean()),
            "total_signal_count": int(g["signal_count"].sum()),
            f"mean_{role}_macro_f1": float(g[f"{role}_macro_f1"].mean()),
            f"mean_{role}_mcc": float(g[f"{role}_mcc"].mean()),
            f"mean_{role}_score": float(g[f"{role}_score"].mean()),
        })

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out):
        out = out.sort_values(
            ["weighted_avg_return_by_signal_count", "mean_signal_avg_return"],
            ascending=[False, False],
        )

    return out


def calc_correlations(df: pd.DataFrame, min_signal_count: int = 0) -> pd.DataFrame:
    if min_signal_count > 0:
        d = df[df["signal_count"] >= min_signal_count].copy()
    else:
        d = df.copy()

    target = "signal_avg_return"

    candidates = [
        "trigger_macro_f1",
        "trigger_mcc",
        "trigger_score",
        "trigger_class1_f1",
        "trigger_class1_recall",
        "trigger_class1_precision",

        "direction_macro_f1",
        "direction_mcc",
        "direction_score",
        "direction_class1_f1",
        "direction_class1_recall",
        "direction_class1_precision",

        "signal_count",
        "trade_signal_count",
    ]

    rows = []

    for col in candidates:
        if col not in d.columns:
            continue

        tmp = d[[target, col]].dropna()

        if len(tmp) < 3:
            rows.append({
                "metric": col,
                "n": len(tmp),
                "pearson_corr": np.nan,
                "spearman_corr": np.nan,
            })
            continue

        rows.append({
            "metric": col,
            "n": len(tmp),
            "pearson_corr": float(tmp[target].corr(tmp[col], method="pearson")),
            "spearman_corr": float(tmp[target].corr(tmp[col], method="spearman")),
        })

    out = pd.DataFrame(rows)

    if len(out):
        out["abs_pearson_corr"] = out["pearson_corr"].abs()
        out = out.sort_values("abs_pearson_corr", ascending=False)

    return out


def save_outputs(
    out_dir: str,
    df: pd.DataFrame,
    trigger_summary: pd.DataFrame,
    direction_summary: pd.DataFrame,
    trigger_arch_summary: pd.DataFrame,
    direction_arch_summary: pd.DataFrame,
    corr_df: pd.DataFrame,
):
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, "fusion_rows.csv"), index=False)
    trigger_summary.to_csv(os.path.join(out_dir, "trigger_model_summary.csv"), index=False)
    direction_summary.to_csv(os.path.join(out_dir, "direction_model_summary.csv"), index=False)
    trigger_arch_summary.to_csv(os.path.join(out_dir, "trigger_arch_summary.csv"), index=False)
    direction_arch_summary.to_csv(os.path.join(out_dir, "direction_arch_summary.csv"), index=False)
    corr_df.to_csv(os.path.join(out_dir, "correlation_summary.csv"), index=False)

    sanity = df[df["count_mismatch"]].copy()
    sanity.to_csv(os.path.join(out_dir, "sanity_issues.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--reports",
        default="/home/chao/work/quant_output/batch_train/DOGEUSDT_15m/2026-06-24/11_54_09/batch_simulation",
    )
    parser.add_argument(
        "--fee-key",
        default="0",
        help='Which fee result to analyze, e.g. "0" or "0.005"',
    )
    parser.add_argument(
        "--min-signal-count",
        type=int,
        default=0,
        help="Filter rows for correlation analysis only.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        default=None,
    )

    args = parser.parse_args()

    reports_path = os.path.join(args.reports,REPORTS_FILE)
    out_dir = os.path.join(args.reports,'analysis')

    raw_rows = load_jsonl(reports_path)

    rows = []
    for r in raw_rows:
        flat = flatten_report(r, fee_key=args.fee_key)
        if flat is not None:
            rows.append(flat)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No valid rows parsed from reports.jsonl")

    trigger_summary = summarize_by_model(df, "trigger")
    direction_summary = summarize_by_model(df, "direction")

    trigger_arch_summary = summarize_by_arch(df, "trigger")
    direction_arch_summary = summarize_by_arch(df, "direction")

    corr_df = calc_correlations(
        df,
        min_signal_count=args.min_signal_count,
    )

    save_outputs(
        out_dir=out_dir,
        df=df,
        trigger_summary=trigger_summary,
        direction_summary=direction_summary,
        trigger_arch_summary=trigger_arch_summary,
        direction_arch_summary=direction_arch_summary,
        corr_df=corr_df,
    )

    print(f"Rows parsed: {len(df)}")
    print(f"Output dir: {out_dir}")

    print("\nTop trigger models:")
    print(trigger_summary.head(10).to_string(index=False))

    print("\nTop direction models:")
    print(direction_summary.head(10).to_string(index=False))

    print("\nCorrelation with signal_avg_return:")
    print(corr_df.to_string(index=False))

    mismatch_count = int(df["count_mismatch"].sum())
    if mismatch_count > 0:
        print(f"\nWARNING: count_mismatch rows found: {mismatch_count}")
        print("Check analysis/sanity_issues.csv")


if __name__ == "__main__":
    main()