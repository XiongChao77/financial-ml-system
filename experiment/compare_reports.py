#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取指定目录下所有子目录中的 reports.jsonl，统计 cagr/calmar 并选出最好的一个。
用法: python compare_reports.py [目录路径]
"""
from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import numpy as np
import os

DEFAULT_DIR = "/home/chao/work/quant_output/batch_experiments/2026-02-03/DOGEUSDT_15m/"


def load_reports(path):
    """逐行读取 jsonl，跳过损坏行。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    reports = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return reports


def find_reports_jsonl(root_dir):
    """在 root_dir 下查找所有 reports.jsonl 路径（子目录内）。"""
    pattern = os.path.join(root_dir, "*", "reports.jsonl")
    paths = sorted(glob.glob(pattern))
    return [p for p in paths if os.path.isfile(p)]


def extract_cagr_calmar(reports):
    """从 report 列表中提取 (cagr, calmar)，过滤掉缺失的。"""
    pairs = []
    for r in reports:
        perf = r.get("performance") or {}
        cagr = perf.get("cagr")
        calmar = perf.get("calmar")
        if cagr is not None and calmar is not None:
            pairs.append((float(cagr), float(calmar)))
    return pairs


def stats(arr):
    """返回 min, max, mean, median, std（空数组则全为 nan）。"""
    a = np.array(arr, dtype=float)
    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    return (
        np.min(a),
        np.max(a),
        np.mean(a),
        np.median(a),
        np.std(a) if len(a) > 1 else 0.0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="读取目录下所有 reports.jsonl，统计 cagr/calmar 并选出最好的一个。"
    )
    parser.add_argument(
        "dir",
        nargs="?",
        default=DEFAULT_DIR,
        help="包含子目录（每目录下有 reports.jsonl）的根目录",
    )
    parser.add_argument(
        "--prefer",
        choices=["cagr", "calmar", "both"],
        default="both",
        help="优先依据哪个指标判定更好: cagr / calmar / both（默认 both 看综合）",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        print(f"❌ 目录不存在: {root}")
        return

    paths = find_reports_jsonl(root)
    if not paths:
        print(f"❌ 在 {root} 下未找到任何 reports.jsonl。")
        return

    # 每个文件对应一条汇总： (rel_path, n, mean_cagr, mean_calmar, ...)
    rows = []
    for path in paths:
        reports = load_reports(path)
        pairs = extract_cagr_calmar(reports)
        if not pairs:
            continue
        cagrs = [p[0] for p in pairs]
        calmars = [p[1] for p in pairs]
        min_c, max_c, mean_c, med_c, std_c = stats(cagrs)
        min_m, max_m, mean_m, med_m, std_m = stats(calmars)
        rel = os.path.relpath(path, root)
        rows.append({
            "path": path,
            "rel": rel,
            "n": len(pairs),
            "mean_cagr": mean_c,
            "mean_calmar": mean_m,
            "max_cagr": max_c,
            "max_calmar": max_m,
            "min_cagr": min_c,
            "min_calmar": min_m,
            "std_cagr": std_c,
            "std_calmar": std_m,
        })

    if not rows:
        print(f"❌ 所有 reports.jsonl 中都没有有效的 cagr/calmar 记录。")
        return

    # 表头
    header = f"{'实验(相对路径)':<28} | {'N':>4} | {'mean_cagr%':>10} | {'mean_calmar':>10} | {'max_cagr%':>10} | {'max_calmar':>10}"
    sep = "-" * 95

    print("\n" + "=" * 95)
    print("目录:", root)
    print("共", len(rows), "个 reports.jsonl")
    print(sep)
    print(header)
    print(sep)

    for r in rows:
        print(
            f"{r['rel']:<28} | {r['n']:>4} | {r['mean_cagr']*100:>9.2f}% | {r['mean_calmar']:>10.3f} | {r['max_cagr']*100:>9.2f}% | {r['max_calmar']:>10.3f}"
        )

    # 选最好的
    if args.prefer == "cagr":
        best_idx = max(range(len(rows)), key=lambda i: rows[i]["mean_cagr"])
        reason = "mean CAGR 最高"
    elif args.prefer == "calmar":
        best_idx = max(range(len(rows)), key=lambda i: rows[i]["mean_calmar"])
        reason = "mean Calmar 最高"
    else:
        scale = 0.01
        scores = [
            r["mean_cagr"] + scale * r["mean_calmar"]
            for r in rows
        ]
        best_idx = int(np.argmax(scores))
        reason = f"综合得分 (mean_cagr + 0.01*mean_calmar) 最高 = {scores[best_idx]:.4f}"

    best = rows[best_idx]
    print(sep)
    print(">>> 最好的一组:")
    print(f"    路径: {best['path']}")
    print(f"    相对: {best['rel']}")
    print(f"    记录数: {best['n']}")
    print(f"    mean CAGR: {best['mean_cagr']*100:.2f}%")
    print(f"    mean Calmar: {best['mean_calmar']:.3f}")
    print(f"    依据: {reason}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
