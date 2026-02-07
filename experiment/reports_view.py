from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, time, json, math

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

# 引入自定义模块
from data_process.common import *
from data_process import common 

exp_dir = (os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'selected_configs'))
short_reports_file = (os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'selected_configs', 'reports_short.jsonl'))
long_reports_file = (os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'selected_configs', 'reports_long.jsonl'))

TOP_K = 50
SKIP_PERCENT = 0  # 跳过前百分之多少，0表示不跳过，从最前面开始选择

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
    sorted_cagr = sorted(rows, key=lambda x: x["cagr"], reverse=True)
    skip_count = int(len(sorted_cagr) * SKIP_PERCENT / 100)
    top_cagr = sorted_cagr[skip_count:skip_count + TOP_K]

    print("\n" + "=" * 80)
    if SKIP_PERCENT > 0:
        print(f"Top {TOP_K} by CAGR (skipped top {SKIP_PERCENT}%, starting from rank {skip_count + 1})")
    else:
        print(f"Top {TOP_K} by CAGR")
    print("=" * 80)

    for i, r in enumerate(top_cagr, 1):
        actual_rank = skip_count + i
        print(
            f"[{actual_rank:02d}] CAGR={r['cagr']:.2%} | Calmar={r['calmar']:.2f} | "
            f"{r['symbol']} {r['interval']} | win={r['candlestick_num']} | "
            f"hash={r['params_hash']} | {r['path']}"
        )

    # ===== 按 Calmar 排序 =====
    sorted_calmar = sorted(rows, key=lambda x: x["calmar"], reverse=True)
    skip_count = int(len(sorted_calmar) * SKIP_PERCENT / 100)
    top_calmar = sorted_calmar[skip_count:skip_count + TOP_K]

    print("\n" + "=" * 80)
    if SKIP_PERCENT > 0:
        print(f"Top {TOP_K} by Calmar (skipped top {SKIP_PERCENT}%, starting from rank {skip_count + 1})")
    else:
        print(f"Top {TOP_K} by Calmar")
    print("=" * 80)

    for i, r in enumerate(top_calmar, 1):
        actual_rank = skip_count + i
        print(
            f"[{actual_rank:02d}] Calmar={r['calmar']:.2f} | CAGR={r['cagr']:.2%} | "
            f"{r['symbol']} {r['interval']} | win={r['candlestick_num']} | "
            f"hash={r['params_hash']} | {r['path']}"
        )
    selected = {}
    for row in top_cagr:
        merge_selected(selected, row["report"], "top_cagr", row["path"])
    for row in top_calmar:
        merge_selected(selected, row["report"], "top_calmar", row["path"])
    print(f"Total reports: {len(selected)}")
    selected = filter_by_performance(selected.values(), min_cagr=0.2, min_calmar=1.2)
    print(f"After performance filter: {len(selected)} reports")
    selected = filter_by_rc_summary(selected)
    print(f"After rc_summary filter: {len(selected)} reports")
    selected = filter_by_trades(selected, min_total_trades=100, min_win_rate=35)
    print(f"After trades filter: {len(selected)} reports")
    out_path = os.path.join(exp_dir,"selected_configs" ,"selected_configs.jsonl")
    os.makedirs(os.path.join(exp_dir,"selected_configs"), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in selected:
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

def filter_by_performance(reports, min_cagr=None, min_calmar=None, min_sharpe=None):
    """
    Filter reports based on performance metrics.
    """
    def meets_criteria(report):
        perf = report.get("performance", {})
        if min_cagr is not None and perf.get("cagr", 0) < min_cagr:
            return False
        if min_calmar is not None and perf.get("calmar", 0) < min_calmar:
            return False
        if min_sharpe is not None and perf.get("sharpe", 0) < min_sharpe:
            return False
        return True

    return [r for r in reports if meets_criteria(r)]

def filter_by_rc_summary(
    reports,
    # —— 生存性 / 尾部 ——
    min_rc_es_05= None,          # 例如 > -0.8
    min_rc_q05= None,            # 例如 > -0.5

    # —— 可持有性 / 连续性 ——
    max_rc_longest_neg_run=None,  # 例如 < 300（天/窗口）
    max_rc_neg_ratio= None,        # 例如 < 0.5

    # —— 典型收益水平 ——
    min_rc_median=None,         # 例如 > 0
    min_rc_q25=None,            # 例如 > 0

    # —— 稳定性 / 离散度 ——
    max_rc_cv=None,             # 例如 < 3
    max_rc_mad=None,            # 可选
):
    """
    Filter reports based on rc_summary metrics.
    目标：剔除结构性不稳定 / 不可持有的策略
    """

    def ok(report):
        rc = report.get("performance", {}).get("rc_summary", {})
        if not rc:
            return False

        # ---------- 生存性（尾部） ----------
        if min_rc_es_05 is not None:
            if rc.get("rc_es_05", -math.inf) < min_rc_es_05:
                return False

        if min_rc_q05 is not None:
            if rc.get("rc_q05", -math.inf) < min_rc_q05:
                return False

        # ---------- 可持有性（长期折磨） ----------
        if max_rc_longest_neg_run is not None:
            if rc.get("rc_longest_neg_run", math.inf) > max_rc_longest_neg_run:
                return False

        if max_rc_neg_ratio is not None:
            if rc.get("rc_neg_ratio", 1.0) > max_rc_neg_ratio:
                return False

        # ---------- 典型收益水平 ----------
        if min_rc_median is not None:
            if rc.get("rc_median", -math.inf) < min_rc_median:
                return False

        if min_rc_q25 is not None:
            if rc.get("rc_q25", -math.inf) < min_rc_q25:
                return False

        # ---------- 稳定性 ----------
        if max_rc_cv is not None:
            rc_cv = rc.get("rc_cv", math.inf)
            if not math.isnan(rc_cv) and rc_cv > max_rc_cv:
                return False

        if max_rc_mad is not None:
            if rc.get("rc_mad", math.inf) > max_rc_mad:
                return False

        return True

    return [r for r in reports if ok(r)]

def filter_by_trades(reports, min_total_trades=100, min_win_rate=35, min_daily_freq = None):
    """
    Filter reports based on trade statistics.
    """
    def meets_criteria(report):
        trades = report.get("trades", {})
        if min_total_trades is not None and trades.get("total", 0) < min_total_trades:
            return False
        if min_win_rate is not None and trades.get("win_rate", 0) < min_win_rate:
            return False
        if min_daily_freq is not None and trades.get("daily_freq", 0) < min_daily_freq:
            return False
        return True

    return [r for r in reports if meets_criteria(r)]

def filter_by_short_long_reports():
    """
    Filter reports based on short/long trade ratio.
    """
    short_reports = load_reports(short_reports_file)
    long_reports = load_reports(long_reports_file)

    print(f"Short total reports: {len(short_reports)}")
    short_selected = filter_by_performance(short_reports, min_cagr=0.2, min_calmar=1.2)
    print(f"Short after performance filter: {len(short_selected)} reports")
    # selected = filter_by_rc_summary(selected, min_rc_es_05 = None, min_rc_q05 = None, max_rc_longest_neg_run = None, max_rc_neg_ratio = None, min_rc_median = None, min_rc_q25 = None)
    print(f"Short after rc_summary filter: {len(short_selected)} reports")
    short_selected = filter_by_trades(short_selected, min_total_trades=100)#, min_win_rate=38, min_daily_freq = 0.3)
    print(f"Short after trades filter: {len(short_selected)} reports")

    long_reports = sorted(long_reports, key=lambda x: x['performance']["cagr"], reverse=True)
    print(f"Long total reports: {len(long_reports)}")
    long_selected = filter_by_performance(long_reports, min_cagr=0.2, min_calmar=1)
    print(f"Long after performance filter: {len(long_selected)} reports")
    # long_selected = filter_by_rc_summary(long_selected, min_rc_es_05 = -2, min_rc_q05 = None, max_rc_longest_neg_run = None, max_rc_neg_ratio = None, min_rc_median = None, min_rc_q25 = None)
    print(f"Long after rc_summary filter: {len(long_selected)} reports")
    long_selected = filter_by_trades(long_selected, min_total_trades=100)#, min_win_rate=38)
    print(f"Long after trades filter: {len(long_selected)} reports")

    short_reports_dict = {r["params"]["hash"]: r for r in short_selected}
    long_reports_dict = {r["params"]["hash"]: r for r in long_selected}
    common_keys = set(short_reports_dict.keys()) & set(long_reports_dict.keys())
    merged = {k: {"short": short_reports_dict[k], "long": long_reports_dict[k]} for k in common_keys}
    
    out_path = os.path.join(exp_dir,"selected_configs" ,"candidate.jsonl")
    os.makedirs(os.path.join(exp_dir,"selected_configs"), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in merged.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # main()
    filter_by_short_long_reports()
