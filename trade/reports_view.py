from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, time, json

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

# 引入自定义模块
from data_process.common import *
from data_process import common 

exp_dir = (os.path.join(common.PERSISTENCE_DIR,'batch_experiments'))

TOP_K = 20

def merge_selected(selected, report, rule_name, src_path):
    """
    selected: dict[params_hash] -> full_report
    report: 原始 report（json 读出来的 dict）
    row: extract_row 的结果（用于取 cagr / calmar）
    """
    h = report['params']['hash']
    if h is None:
        return

    if h not in selected:
        r = dict(report)  # 浅拷贝，避免改原对象
        r["rule"] = [rule_name]
        r["path"] = src_path
        selected[h] = r
    else:
        if rule_name not in selected[h]["rule"]:
            selected[h]["rule"].append(rule_name)


def iter_reports_jsonl(root):
    """
    递归扫描所有 reports.jsonl
    """
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname == "reports.jsonl":
                yield os.path.join(dirpath, fname)


def load_reports(path):
    """
    逐行读取 jsonl，跳过损坏行
    """
    reports = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                reports.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return reports


def extract_row(report, src_path):
    """
    从单条 report 中抽取关键信息
    """
    perf = report.get("performance", {})
    params = report.get("params", {})
    common = params.get("common", {})

    return {
        "cagr": perf.get("cagr"),
        "calmar": perf.get("calmar"),
        "symbol": common.get("symbol"),
        "interval": common.get("interval"),
        "candlestick_num": common.get("candlestick_num"),
        "params_hash": report.get("params",{}).get('hash',0),
        "path": src_path,
        "report" : report,
    }


def main():
    rows = []

    for jsonl_path in iter_reports_jsonl(exp_dir):
        reports = load_reports(jsonl_path)
        for r in reports:
            row = extract_row(r, jsonl_path)
            if row["cagr"] is not None and row["calmar"] is not None:
                rows.append(row)

    print(f"Total reports loaded: {len(rows)}")

    # ===== 按 CAGR 排序 =====
    top_cagr = sorted(rows, key=lambda x: x["cagr"], reverse=True)[:TOP_K]

    print("\n" + "=" * 80)
    print(f"Top {TOP_K} by CAGR")
    print("=" * 80)

    for i, r in enumerate(top_cagr, 1):
        print(
            f"[{i:02d}] CAGR={r['cagr']:.2%} | Calmar={r['calmar']:.2f} | "
            f"{r['symbol']} {r['interval']} | win={r['candlestick_num']} | "
            f"hash={r['params_hash']} | {r['path']}"
        )

    # ===== 按 Calmar 排序 =====
    top_calmar = sorted(rows, key=lambda x: x["calmar"], reverse=True)[:TOP_K]

    print("\n" + "=" * 80)
    print(f"Top {TOP_K} by Calmar")
    print("=" * 80)

    for i, r in enumerate(top_calmar, 1):
        print(
            f"[{i:02d}] Calmar={r['calmar']:.2f} | CAGR={r['cagr']:.2%} | "
            f"{r['symbol']} {r['interval']} | win={r['candlestick_num']} | "
            f"hash={r['params_hash']} | {r['path']}"
        )
    selected = {}
    for row in top_cagr:
        merge_selected(selected, row["report"], "top_cagr", row["path"])
    for row in top_calmar:
        merge_selected(selected, row["report"], "top_calmar", row["path"])
    out_path = os.path.join(exp_dir, "selected_configs.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in selected.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[SAVE] {out_path} | total={len(selected)}")

    

if __name__ == "__main__":
    main()
