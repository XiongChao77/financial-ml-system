from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, time, json, math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))
import copy
# Import project modules
from data_process.common import *
from data_process import common 

REPORTS_FILE = "reports.jsonl"

def iter_reports_jsonl(root_list):
    """
    Recursively scan for all reports.jsonl files under given roots.
    """
    for root in root_list:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname == REPORTS_FILE:
                    yield os.path.join(dirpath, fname)

def extract_row(report, src_path):
    """
    Extract key fields from a single report.
    """
    fusion_hash = report.get("fusion_hash", report)
    fusion_dir = report.get("fusion_dir", report)
    trigger = report.get("trigger", report)
    direction = report.get("direction", report)
    signal_return = report.get("signal_return", report)
    signal_avg_return = signal_return['0'].get("signal_avg_return", report)
    signal_count = signal_return['0'].get("signal_count", report)
    simulation = report.get("simulation", report)
    short = simulation.get("short", report)  # Support both separated short/long storage and merged formats
    long = simulation.get("long", report)
    forward = simulation.get("forward", report)
    perf = short.get("performance", {})
    params = short.get("params", {})
    common = params.get("common", {})
    long_perf = long.get("performance", {})
    long_params = long.get("params", {})
    long_common = long_params.get("common", {})
    forward_perf = forward.get("performance", {})
    triger_metrics = report["trigger"]["metrics"]
    direction_metrics = report["direction"]["metrics"]
    triger_macro_f1 = triger_metrics["Best_F1"]["macro_f1"]
    triger_macro_precision = triger_metrics.get("macro_precision", 0)
    triger_accuracy = triger_metrics.get("accuracy", 0)
    triger_macro_recall = triger_metrics.get("macro_recall", 0)
    direction_macro_f1 = direction_metrics["Best_F1"]["macro_f1"]
    directionmacro_precision = direction_metrics.get("macro_precision", 0)
    directionaccuracy = direction_metrics.get("accuracy", 0)
    directionmacro_recall = direction_metrics.get("macro_recall", 0)
    trigger_pos_lift = triger_metrics["Best_F1"]["per_class"]["1"]["precision_lift"]
    lift_long = (report["direction"]["metrics"]["Best_F1"]["per_class"]['1']["precision_lift"])
    lift_short = (report["direction"]["metrics"]["Best_F1"]["per_class"]["0"]["precision_lift"])

    return {
        "signal_avg_return":signal_avg_return,
        "signal_count":signal_count,
        "cagr": perf.get("cagr"),
        "calmar": perf.get("calmar"),
        "daily_freq" : short.get("trades", {}).get("daily_freq"),
        "l_cagr": long_perf.get("cagr"),
        "l_calmar": long_perf.get("calmar"),
        "l_daily_freq" : long.get("trades", {}).get("daily_freq"),
        "l_win_rate" : long.get("trades", {}).get("win_rate"),
        "l_avg_pct_gross" : long.get("trades", {}).get("avg_pct_gross"),
        "l_sharpe" : long_perf.get("sharpe"),
        "f_cagr": forward_perf.get("cagr"),
        "f_calmar": forward_perf.get("calmar"),
        "f_daily_freq" : forward.get("trades", {}).get("daily_freq"),
        "hash": params.get('hash',0),
        "path": src_path,
        "short" : short,
        "long": long,
        "forward": report.get("forward", report),
        "fusion_hash":fusion_hash,
        "fusion_dir":fusion_dir,
        "trigger":trigger,
        "direction":direction,

        "triger_macro_f1":triger_macro_f1,
        "triger_macro_precision":triger_macro_precision,
        "triger_accuracy":triger_accuracy,
        "triger_macro_recall":triger_macro_recall,
        "direction_macro_f1":direction_macro_f1,
        "directionmacro_precision":directionmacro_precision,
        "directionaccuracy":directionaccuracy,
        "directionmacro_recall":directionmacro_recall,

        "trigger_pos_lift":trigger_pos_lift,
        "lift_long":lift_long ,
        "lift_short": lift_short,
        "raw" : report,
    }

def plot_model_param_heatmaps(
    records,
    output_dir,
    metric="l_cagr",
    side="trigger",      # "trigger" or "direction"
    agg="best",          # "best", "mean", "top10_mean"
    top_pct=0.10,
):
    """
    Plot predict_num x miss_penalty heatmaps for each model class,
    then combine all model heatmaps into one large figure.

    side:
        "trigger"   -> use trigger model class and trigger miss_penalty
        "direction" -> use direction model class and direction miss_penalty
    """

    os.makedirs(output_dir, exist_ok=True)

    rows = []

    for r in records:
        model_info = r.get(side, {})
        if not isinstance(model_info, dict):
            continue

        model_type = model_info.get("model_type")
        model_version = model_info.get("model_version")

        train_params = model_info.get("train_params", {})
        model_cfg = train_params.get("model_cfg", {})

        model_class = f"{model_type}_v{model_version}"
        miss_penalty =  common.recursive_get(model_info,"miss_penalty")

        # predict_num 来自 common params
        predict_num = common.recursive_get(model_info, "predict_num")

        value = _safe_float(r.get(metric))

        rows.append({
            "model_class": model_class,
            "predict_num": predict_num,
            "miss_penalty": miss_penalty,
            metric: value,
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["model_class", "predict_num", "miss_penalty", metric])

    if df.empty:
        print(f"[WARN] no valid rows for {side} {metric}")
        return None

    df["predict_num"] = df["predict_num"]
    df["miss_penalty"] = df["miss_penalty"]

    raw_csv = os.path.join(
        output_dir,
        f"model_param_heatmap_raw_{side}_{metric}_{agg}.csv"
    )
    df.to_csv(raw_csv, index=False)
    print(f"[SAVE] {raw_csv}")

    model_classes = sorted(df["model_class"].unique())

    pivots = {}

    for model_class in model_classes:
        sub = df[df["model_class"] == model_class]

        grouped = sub.groupby(["predict_num", "miss_penalty"])[metric]

        if agg == "best":
            summary = grouped.max().reset_index(name="value")
        elif agg == "mean":
            summary = grouped.mean().reset_index(name="value")
        elif agg == "top10_mean":
            summary = grouped.apply(
                lambda x: _top_pct_mean(
                    x,
                    pct=top_pct,
                    higher_is_better=True,
                )
            ).reset_index(name="value")
        else:
            raise ValueError(f"unsupported agg: {agg}")

        pivot = summary.pivot(
            index="predict_num",
            columns="miss_penalty",
            values="value",
        )

        pivot = pivot.sort_index().sort_index(axis=1)
        pivots[model_class] = pivot

        # # 单独保存每个模型一张图
        # _plot_heatmap(
        #     pivot,
        #     output_dir,
        #     f"heatmap_{side}_{model_class}_{metric}_by_predict_seq_{agg}.png",
        #     f"{side} {model_class} | {metric} {agg} by predict_num x miss_penalty",
        #     fmt=".2f",
        #     cmap="RdYlGn",
        # )

    # ===== n 合一大图 =====

    all_values = pd.concat(
        [p.stack() for p in pivots.values()],
        axis=0
    ).dropna()

    if all_values.empty:
        print("[WARN] all pivot values are NaN")
        return None

    vmin = float(all_values.min())
    vmax = float(all_values.max())

    n = len(model_classes)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))

    fig_width = 5.0 * ncols + 0.8
    fig_height = 4.2 * nrows

    fig = plt.figure(figsize=(fig_width, fig_height))

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[1] * ncols + [0.05],
        wspace=0.25,
        hspace=0.35,
    )

    axes = []

    for i, model_class in enumerate(model_classes):
        row = i // ncols
        col = i % ncols

        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

        pivot = pivots[model_class]

        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            linewidths=0.4,
            linecolor="white",
            cbar=False,
        )

        ax.set_title(model_class)
        ax.set_xlabel("miss_penalty")
        ax.set_ylabel("predict_num")

    # colorbar 只放在最右边，跨所有行
    cbar_ax = fig.add_subplot(gs[:, -1])

    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn",
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(metric)

    fig.suptitle(
        f"{side} models | {metric} ({agg}) by predict_num x miss_penalty",
        fontsize=14,
    )

    combined_path = os.path.join(
        output_dir,
        f"heatmap_all_{side}_models_{metric}_by_predict_seq_{agg}.png"
    )

    fig.savefig(combined_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] {combined_path}")

    return {
        "raw_csv": raw_csv,
        "combined_path": combined_path,
        "pivots": pivots,
    }

def show_performance(all_results,output_dir, batch_size=5):
    print("-"*20 + 'Key strategy indicators' +"-"*20)
    print(f"{'Num':>5}"
          f"{'Hash':>10}"
          f"{'CAGR':>10}"
          f"{'Sharpe':>10}"
          f"{'Calmar':>10}"
          f"{'Max_DD':>12}"
          f"{'DailyFreq':>12}"
          f"{'WinRate':>10}"
          f"{'RC_Median':>12}"
          f"{'RC_PosRatio':>12}"
          f"{'MAX_DD_DAYS':>12}")

    print("-" * 98)

    for i, r in enumerate(all_results):
        g = lambda k: common.recursive_get(r['long'], k)
        rc_median = g('rc_median')
        rc_pos_ratio = g('rc_pos_ratio')
        print(f"{i:>4}"
              f"{str(g('hash')):>12}"
              f"{g('cagr'):10.2f}"
              f"{g('sharpe'):10.2f}"
              f"{g('calmar'):10.2f}"
              f"{g('max_dd_pct'):12.2f}"
              f"{g('daily_freq'):12.2f}"
              f"{g('win_rate'):10.2f}"
              f"{(rc_median if rc_median is not None else 0):12.2f}"
              f"{(rc_pos_ratio if rc_pos_ratio is not None else 0):12.2f}"
              f"{g('max_hwm_duration_days'):12.2f}"
              )
    plot_in_batches(all_results,output_dir,batch_size)

def plot_equity_curves(all_results, output_dir, file_name="equity_full_combined.png", start_index=0):
    save_path = os.path.join(output_dir, file_name)

    # ===== 1. Get price background data =====
    pere_para = all_results[0]['long']['params']['common']
    price_file = os.path.join(
        common.PROJECT_DATA_DIR, pere_para['market_category'], pere_para['data_source'],
        pere_para['trading_type'],
        f"{pere_para['symbol']}_{pere_para['interval']}.csv"
    )
    
    price_df = pd.read_csv(price_file)
    price_df['open_time_date_utc'] = pd.to_datetime(price_df['open_time_date_utc'])
    price_df.set_index('open_time_date_utc', inplace=True)
    price_series = price_df['close'].sort_index()

    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax1.plot(price_series.index, price_series, color='black', linewidth=0.8, alpha=0.15, label='Market Price')
    ax1.set_ylabel('Market Price (USD)')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Continuous Strategy Equity (Normalized)')

    # ===== 2. Concatenate and plot curves =====
    # Record split timestamps for vertical lines
    split_dates = {}

    for i, r in enumerate(all_results):
        segments = []
        current_multiplier = 1.0
        
        for period in ['long', 'short', 'forward']:
            period_data = r.get(period)
            daily_list = common.recursive_get(period_data,'daily_loss_list')

            df = pd.DataFrame(daily_list)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').set_index('date')
            
            # Record the start time of this phase (used to draw vertical lines)
            if period not in split_dates:
                split_dates[period] = df.index[0]
            
            returns_sequence = df['equity'] / df['equity'].iloc[0]
            df['continuous_equity'] = returns_sequence * current_multiplier
            segments.append(df[['continuous_equity']])
            current_multiplier = df['continuous_equity'].iloc[-1]

        full_path_df = pd.concat(segments)
        full_path_df = full_path_df[~full_path_df.index.duplicated(keep='first')]
        
        ax2.plot(full_path_df.index, full_path_df['continuous_equity'], 
                 linewidth=1.5, alpha=0.8, label=f"S{start_index + i}")

    # ===== 3. Draw phase-split vertical lines =====
    # Only draw lines, no labels, with lighter color (alpha=0.3)
    if 'short' in split_dates:
        s_start = split_dates['short']
        # Use thin blue dashed line to mark transition from Long to Short
        ax1.axvline(x=s_start, color='blue', linestyle='--', linewidth=1, alpha=0.3)

    if 'forward' in split_dates:
        f_start = split_dates['forward']
        # Use red dashed line to mark entering the critical Forward phase
        ax1.axvline(x=f_start, color='red', linestyle='--', linewidth=1, alpha=0.3)

    # ===== 4. Legend and styling =====
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', ncol=4, fontsize=8)

    plt.title(f"Strategy Performance: Long (Train) -> Short (Val) -> Forward (Test)")
    print(f"[SAVE] {save_path}")
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_in_batches(all_results, output_dir, batch_size=5):
    total = len(all_results)
    
    sns.set_theme(style="white")
    for i in range(0, total, batch_size):
        batch = all_results[i:i + batch_size]
        filename = f"batch_{i//batch_size + 1}.png"
        # ✨ Pass current loop index i as the starting number
        plot_equity_curves(batch, output_dir, filename, start_index=i)

def _safe_float(v):
    """Convert values to float; return NaN for None/non-numeric values."""
    try:
        if v is None:
            return np.nan
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _model_axis_label(model_info, prefix=""):
    """
    Axis category for heatmap.

    IMPORTANT:
    Only model_type + model_version define the model category.
    task_hash / score / miss_penalty / stride / penalties / feature params are
    different training instances under the same category, so they must NOT
    appear in the axis label.
    """
    if not isinstance(model_info, dict):
        return "unknown"

    model_type = model_info.get("model_type")
    model_version = model_info.get("model_version")

    if model_type is None:
        model_type = "unknown"

    if model_version is None:
        label = str(model_type)
    else:
        label = f"{model_type}_v{model_version}"

    if prefix:
        label = f"{prefix}:{label}"
    return label


def _top_pct_mean(values, pct=0.10, higher_is_better=True):
    """
    Mean of the best top pct values.
    For metrics such as max_dd_pct where smaller is better, set higher_is_better=False.
    """
    s = pd.Series(values, dtype="float64").dropna()
    if s.empty:
        return np.nan
    n = max(1, int(math.ceil(len(s) * pct)))
    s = s.sort_values(ascending=not higher_is_better)
    return float(s.head(n).mean())


def _best_value(values, higher_is_better=True):
    s = pd.Series(values, dtype="float64").dropna()
    if s.empty:
        return np.nan
    return float(s.max() if higher_is_better else s.min())


def _plot_heatmap(matrix, output_dir, file_name, title, fmt=".2f", cmap="RdYlGn"):
    """Save one heatmap from a pivot table."""
    if matrix.empty:
        print(f"[SKIP] empty heatmap: {file_name}")
        return None

    # Sort labels for stable output; keep rows/cols with at least one value.
    matrix = matrix.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if matrix.empty:
        print(f"[SKIP] all-NaN heatmap: {file_name}")
        return None

    width = max(10, min(36, 1.4 * len(matrix.columns) + 5))
    height = max(8, min(36, 1.0 * len(matrix.index) + 4))

    plt.figure(figsize=(width, height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.4,
        linecolor="white",
        cbar=True,
    )
    plt.title(title)
    plt.xlabel("Direction model")
    plt.ylabel("Trigger model")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SAVE] {save_path}")
    return save_path


def performance_compare_trigger(
    records,
    output_dir,
    metrics=None,
    top_pct=0.10,
    lower_is_better_metrics=None,
):
    """
    Compare trigger/direction model combinations using heatmaps and summary tables.

    X axis: direction model
    Y axis: trigger model

    For each metric, the function saves:
      1. heatmap_<metric>_mean.png       - mean value per trigger/direction pair
      2. heatmap_<metric>_best.png       - best value per pair
      3. heatmap_<metric>_top10_mean.png - mean of best top 10% per pair
      4. heatmap_<metric>_count.png      - sample count per pair
      5. CSV summaries for pair/trigger/direction level comparison

    `records` should be the extracted rows returned by `extract_row`.
    """
    if metrics is None:
        metrics = [
            "l_cagr", "l_calmar", "l_sharpe",
            "l_avg_pct_gross",
            "f_cagr", "f_calmar", "f_daily_freq",
            "signal_avg_return",
            "triger_macro_f1","triger_macro_precision","triger_accuracy","triger_macro_recall",
            "direction_macro_f1","directionmacro_precision","directionaccuracy","directionmacro_recall",
            "trigger_pos_lift","lift_long","lift_short","lift_neutral"
        ]

    if lower_is_better_metrics is None:
        lower_is_better_metrics = set()
    else:
        lower_is_better_metrics = set(lower_is_better_metrics)

    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for r in records:
        trigger_info = r.get("trigger", {})
        direction_info = r.get("direction", {})
        trigger_label = _model_axis_label(trigger_info)
        direction_label = _model_axis_label(direction_info)

        row = {
            "trigger_model": trigger_label,
            "direction_model": direction_label,
            "trigger_hash": trigger_info.get("task_hash") if isinstance(trigger_info, dict) else None,
            "direction_hash": direction_info.get("task_hash") if isinstance(direction_info, dict) else None,
            "trigger_score": _safe_float(trigger_info.get("score")) if isinstance(trigger_info, dict) else np.nan,
            "direction_score": _safe_float(direction_info.get("score")) if isinstance(direction_info, dict) else np.nan,
            "fusion_hash": r.get("fusion_hash"),
            "hash": r.get("hash"),
            "path": r.get("path"),
        }
        for m in metrics:
            row[m] = _safe_float(r.get(m))
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] no records for trigger/direction comparison")
        return None

    raw_csv = os.path.join(output_dir, "trigger_direction_raw_metrics.csv")
    df.to_csv(raw_csv, index=False)
    print(f"[SAVE] {raw_csv}")

    pair_summary_frames = []
    summary_rows = []

    for metric in metrics:
        if metric not in df.columns:
            continue

        metric_df = df.dropna(subset=[metric]).copy()
        if metric_df.empty:
            print(f"[SKIP] metric has no valid values: {metric}")
            continue

        higher_is_better = metric not in lower_is_better_metrics

        grouped = metric_df.groupby(["trigger_model", "direction_model"], dropna=False)[metric]
        pair_summary = grouped.agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
        ).reset_index()
        pair_summary["best"] = grouped.apply(
            lambda x: _best_value(x, higher_is_better=higher_is_better)
        ).values
        pair_summary[f"top{int(top_pct * 100)}_mean"] = grouped.apply(
            lambda x: _top_pct_mean(x, pct=top_pct, higher_is_better=higher_is_better)
        ).values
        pair_summary.insert(0, "metric", metric)
        pair_summary_frames.append(pair_summary)

        for agg_name in ["mean", "best", f"top{int(top_pct * 100)}_mean", "count"]:
            pivot = pair_summary.pivot(
                index="trigger_model",
                columns="direction_model",
                values=agg_name,
            )
            fmt = ".0f" if agg_name == "count" else ".2f"
            cmap = "Blues" if agg_name == "count" else "RdYlGn"
            _plot_heatmap(
                pivot,
                output_dir,
                f"heatmap_{metric}_{agg_name}.png",
                f"{metric} | {agg_name} by Trigger x Direction",
                fmt=fmt,
                cmap=cmap,
            )

        # Overall metric summary: useful for quick console comparison.
        best_idx = metric_df[metric].idxmax() if higher_is_better else metric_df[metric].idxmin()
        best_row = metric_df.loc[best_idx]
        summary_rows.append({
            "metric": metric,
            "count": int(metric_df[metric].count()),
            "mean": float(metric_df[metric].mean()),
            "median": float(metric_df[metric].median()),
            "std": float(metric_df[metric].std()) if metric_df[metric].count() > 1 else 0.0,
            "best": float(best_row[metric]),
            f"top{int(top_pct * 100)}_mean": _top_pct_mean(
                metric_df[metric], pct=top_pct, higher_is_better=higher_is_better
            ),
            "best_trigger_model": best_row["trigger_model"],
            "best_direction_model": best_row["direction_model"],
            "best_fusion_hash": best_row.get("fusion_hash"),
        })

        # Single-axis summaries answer: "which trigger/direction model is robust?"
        for axis in ["trigger_model", "direction_model"]:
            axis_summary = metric_df.groupby(axis)[metric].agg(
                count="count",
                mean="mean",
                median="median",
                std="std",
            ).reset_index()
            axis_summary["best"] = metric_df.groupby(axis)[metric].apply(
                lambda x: _best_value(x, higher_is_better=higher_is_better)
            ).values
            axis_summary[f"top{int(top_pct * 100)}_mean"] = metric_df.groupby(axis)[metric].apply(
                lambda x: _top_pct_mean(x, pct=top_pct, higher_is_better=higher_is_better)
            ).values
            sort_col = f"top{int(top_pct * 100)}_mean"
            axis_summary = axis_summary.sort_values(sort_col, ascending=not higher_is_better)
            axis_csv = os.path.join(output_dir, f"summary_{axis}_{metric}.csv")
            axis_summary.to_csv(axis_csv, index=False)
            print(f"[SAVE] {axis_csv}")

    if pair_summary_frames:
        pair_all = pd.concat(pair_summary_frames, ignore_index=True)
        pair_csv = os.path.join(output_dir, "trigger_direction_pair_summary.csv")
        pair_all.to_csv(pair_csv, index=False)
        print(f"[SAVE] {pair_csv}")
    else:
        pair_csv = None

    metric_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "trigger_direction_metric_summary.csv")
    metric_summary.to_csv(summary_csv, index=False)
    print(f"[SAVE] {summary_csv}")

    print("-" * 20 + " Trigger/Direction Metric Summary " + "-" * 20)
    if not metric_summary.empty:
        print(metric_summary[[
            "metric", "count", "mean", "median", "best",
            f"top{int(top_pct * 100)}_mean", "best_fusion_hash"
        ]].to_string(index=False))

    return {
        "raw_csv": raw_csv,
        "pair_summary_csv": pair_csv,
        "metric_summary_csv": summary_csv,
        "df": df,
        "metric_summary": metric_summary,
    }

def main():
    sim_dir1 = os.path.join(common.PERSISTENCE_DIR,'batch_train/DOGEUSDT_30m/2026-06-28/19_15_16/batch_simulation')
    output_dir = os.path.join(sim_dir1, 'report_view')
    os.makedirs(output_dir, exist_ok=True)
    exp_dir_list = [sim_dir1]
    report_files = []
    rows = []
    for jsonl_path in iter_reports_jsonl(exp_dir_list):
        report_files.append(jsonl_path)
    error_count = 0
    for report_file in report_files:
        records = common.load_reports(report_file)
        for r in records:
            if r['status'] != 'ok':
                error_count += 1
                continue
            row = extract_row(r, report_file)
            rows.append(row)
    symbol = rows[0]['short']['params']['common']['symbol']
    interval = rows[0]['short']['params']['common']['interval']
    print(f"Total reports loaded: {len(rows)}, {error_count} errors ")
    # performance_compare_trigger(rows, output_dir)

    plot_model_param_heatmaps(rows, output_dir, metric="l_cagr", side="trigger", agg="top10_mean")
    plot_model_param_heatmaps(rows, output_dir, metric="l_cagr", side="direction", agg="top10_mean")

    plot_model_param_heatmaps(rows, output_dir, metric="l_cagr", side="trigger", agg="best")
    plot_model_param_heatmaps(rows, output_dir, metric="l_cagr", side="direction", agg="best")
    sorted_records = sorted(rows, key=itemgetter("l_cagr"), reverse=True)
    select_records = sorted_records[:10]
    show_performance(select_records,output_dir,3)
    out_path = os.path.join(output_dir,"selected_configs.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in select_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[SAVE] {out_path} | total={len(select_records)}")
    print("done")

if __name__ == "__main__":
    main()
    # filter_by_short_longs()
