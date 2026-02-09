#!/usr/bin/env python3
"""
读取 reports.jsonl，以 predict_num / candlestick_num 为坐标轴，
cagr / calmar 为目标值，生成两幅热力图。
"""
from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# 默认输入路径（相对项目根或可传参）
DEFAULT_REPORTS_PATH = r"/home/chao/work/quant_output/batch_experiments/2026-02-08/ETHUSDT_30m/18_26_11/reports.jsonl"


def load_reports(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_penalty_metrics(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        try:
            r = r['short']
            fp = r["params"]["common"]["predict_num"]
            mp = r["params"]["common"]["candlestick_num"]
            cagr = r["performance"]["cagr"]
            calmar = r["performance"]["calmar"]
            rows.append({"predict_num": fp, "candlestick_num": mp, "cagr": cagr, "calmar": calmar})
        except (KeyError, TypeError):
            continue
    return pd.DataFrame(rows)


def build_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """同一 (predict_num, candlestick_num) 取均值."""
    agg = df.groupby(["predict_num", "candlestick_num"], as_index=False)[value_col].mean()
    pivot = agg.pivot(index="predict_num", columns="candlestick_num", values=value_col)
    return pivot


def plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    out_path: Path,
    cmap: str = "RdYlGn",
    xlabel: str = "candlestick_num",
    ylabel: str = "predict_num",
) -> None:
    nrows, ncols = pivot.shape
    # 按格子数量放大画布，保证每个格子足够大
    cell_size = 0.9
    fig_w = max(8, ncols * cell_size)
    fig_h = max(6, nrows * cell_size)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # 格子多时略缩小字号，避免数字重叠；基准字号约放大一倍
    annot_font = max(10, 22 - max(nrows, ncols))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        annot_kws={"size": annot_font, "rotation": 0},
        cbar_kws={"label": title.split(" ")[0]},
        linewidths=0.5,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot CAGR/Calmar heatmaps by predict_num & candlestick_num")
    parser.add_argument(
        "reports",
        nargs="?",
        default=str(DEFAULT_REPORTS_PATH),
        help="Path to reports.jsonl",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=None,
        help="Output directory for heatmap images (default: same dir as reports)",
    )
    args = parser.parse_args()

    path = Path(args.reports)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[1] / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reports file not found: {path}")

    outdir = Path(args.outdir) if args.outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_reports(path)
    df = extract_penalty_metrics(records)
    if df.empty:
        raise ValueError("No valid records with predict_num, candlestick_num, cagr, calmar")

    pivot_cagr = build_pivot(df, "cagr")
    pivot_calmar = build_pivot(df, "calmar")

    # 热力图：CAGR 用 RdYlGn（绿好红差），Calmar 同理
    plot_heatmap(
        pivot_cagr,
        "CAGR (predict_num × candlestick_num)",
        outdir / "heatmap_cagr.png",
        cmap="RdYlGn",
    )
    plot_heatmap(
        pivot_calmar,
        "Calmar (predict_num × candlestick_num)",
        outdir / "heatmap_calmar.png",
        cmap="RdYlGn",
    )
    print(f"CAGR heatmap saved: {outdir / 'heatmap_cagr.png'}")
    print(f"Calmar heatmap saved: {outdir / 'heatmap_calmar.png'}")


if __name__ == "__main__":
    main()
