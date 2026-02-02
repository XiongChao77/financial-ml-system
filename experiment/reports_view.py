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
    # para_evaluation(rows)
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

def para_evaluation(rows, label1="Vol 1.9", label2="Vol 1.7"):
    """
    通用参数评估函数
    label1/label2: 用于记录这两组分别代表什么，但在表格中显示为 Group 1/2
    """
    group_1_data = []
    group_2_data = []

    # 1. 灵活分类逻辑
    for row in rows:
        # 你可以根据需要修改这里的判断条件，函数整体是通用的
        # vol = row["report"]["params"]["common"]["vol_multiplier_long"]
        # if vol == 1.9:
        #     group_1_data.append(row)
        # elif vol == 1.7:
        #     group_2_data.append(row)
        predict_num = row["report"]["params"]["common"]["predict_num"]
        holdbar = row["report"]["params"]["strategy"]["holdbar"]
        if predict_num == 20 and holdbar ==20:
            group_1_data.append(row)
        elif predict_num == 20 and holdbar ==16:
            group_2_data.append(row)

    # 2. 内部指标提取器
    def extract_metrics(group_list):
        if not group_list: return None
        return {
            "cagr": [r["report"]["performance"]["cagr"] for r in group_list],
            "calmar": [r["report"]["performance"]["calmar"] for r in group_list],
            "sharpe": [r["report"]["performance"]["sharpe"] for r in group_list],
            "max_dd": [r["report"]["drawdown"]["max_dd_pct"] for r in group_list],
            "count": len(group_list)
        }

    g1_metrics = extract_metrics(group_1_data)
    g2_metrics = extract_metrics(group_2_data)

    # 3. 构造对比表格
    summary_rows = []
    for i, (m, label_desc) in enumerate([(g1_metrics, label1), (g2_metrics, label2)]):
        if m:
            summary_rows.append({
                "Group": f"Group {i+1}",
                "Desc": label_desc,  # 备注说明，方便你知道 Group1 是哪个
                "Count": m["count"],
                "Avg CAGR": f"{np.mean(m['cagr']):.2%}",
                "Max CAGR": f"{np.max(m['cagr']):.2%}",
                "Min CAGR": f"{np.min(m['cagr']):.2%}",
                "Std CAGR": f"{np.std(m['cagr']):.4f}",
                "Avg Calmar": f"{np.mean(m['calmar']):.2f}",
                "Avg Sharpe": f"{np.mean(m['sharpe']):.2f}",
                "Avg MaxDD": f"{np.mean(m['max_dd']):.2f}%" # 修正回撤显示
            })

    # 4. 完美对齐输出
    if summary_rows:
        df_final = pd.DataFrame(summary_rows).set_index("Group")
        
        print("\n" + "="*110)
        print(f"📊 参数组对比评估 (Group 1: {label1} | Group 2: {label2})")
        print("="*110)
        # 使用 pandas 的 to_string 保证列对齐
        print(df_final.to_string(justify='center', col_space=10))
        print("="*110)
        
        # 5. 极简结论
        cagr1 = np.mean(g1_metrics['cagr'])
        cagr2 = np.mean(g2_metrics['cagr'])
        winner = "Group 1" if cagr1 > cagr2 else "Group 2"
        print(f"💡 结论预览: {winner} 在收益期望上表现更优 ({max(cagr1, cagr2):.2%})")
    else:
        print("❌ 错误: 未能分类出有效数据，请检查输入 rows 的参数。")
    exit()

if __name__ == "__main__":
    main()
