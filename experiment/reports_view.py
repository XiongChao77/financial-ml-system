from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, time, json, math

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, ".."))

# 引入自定义模块
from data_process.common import *
from data_process import common 

# exp_dir = (os.path.join(common.PERSISTENCE_DIR,'batch_experiments', '2026-02-08','ETHUSDT_30m'))
exp_dir = (os.path.join(common.PERSISTENCE_DIR,'batch_experiments', '2026-02-12','DOGEUSDT_15m'))
output_dir = os.path.join(common.PERSISTENCE_DIR,'batch_experiments')
short_reports_file = (os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'selected_configs', 'reports_short.jsonl'))
long_reports_file = (os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'selected_configs', 'reports_long.jsonl'))

TOP_K = 50
SKIP_PERCENT = 0  # 跳过前百分之多少，0表示不跳过，从最前面开始选择

def merge_selected(selected, report, rule_name, src_path):
    """
    selected: dict[hash] -> full_report
    report: 原始 report（json 读出来的 dict）
    row: extract_row 的结果（用于取 cagr / calmar）
    """
    h = report['hash']
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
    short_report = report.get("short", report)  # 兼容 short/long 分开存储和合并存储两种格式
    long_report = report.get("long", report)
    perf = short_report.get("performance", {})
    params = short_report.get("params", {})
    common = params.get("common", {})
    long_perf = long_report.get("performance", {})
    long_params = long_report.get("params", {})
    long_common = long_params.get("common", {})
    return {
        "cagr": perf.get("cagr"),
        "calmar": perf.get("calmar"),
        "daily_freq" : short_report.get("trades", {}).get("daily_freq"),
        "long_cagr": long_perf.get("cagr"),
        "long_calmar": long_perf.get("calmar"),
        "long_daily_freq" : long_report.get("trades", {}).get("daily_freq"),
        "symbol": common.get("symbol"),
        "interval": common.get("interval"),
        "hash": params.get('hash',0),
        "path": src_path,
        "short_report" : short_report,
        "long_report": long_report
    }


def main():
    rows = []

    for jsonl_path in iter_reports_jsonl(exp_dir):
        reports = load_reports(jsonl_path)
        for r in reports:
            if 'long' not in r and 'short' not in r:
                continue
            row = extract_row(r, jsonl_path)
            if row["cagr"] is not None and row["calmar"] is not None:
                rows.append(row)

    print(f"Total reports loaded: {len(rows)}")
    # para_evaluation(rows)
    # ===== 按 CAGR 排序 =====
    sorted_cagr = sorted(rows, key=lambda x: x["cagr"], reverse=True)

    selected = {}
    for row in sorted_cagr:
        merge_selected(selected, row, "top_cagr", row["path"])
    print(f"Total reports: {len(selected)}")
    selected = filter_by_performance(selected.values(), period ='short_report'  ,min_cagr=0.2, min_calmar=1)
    
    print(f"After short_report performance filter: {len(selected)} reports")
    selected = filter_by_performance(selected, period ='long_report', min_cagr=0.1, min_calmar=0.5)
    print(f"After long_report performance filter: {len(selected)} reports")
    analyze_candlestick_num(selected)
    selected = filter_by_trades(selected, period ='short_report', min_daily_freq = 0.2)
    print(f"After short_report trades filter: {len(selected)} reports")
    selected = filter_by_trades(selected, period ='long_report', min_daily_freq = 0.2)
    print(f"After long_report trades filter: {len(selected)} reports")
    selected = filter_by_rc_summary(selected,'short_report')
    print(f"After short_report rc_summary filter: {len(selected)} reports")
    selected = filter_by_rc_summary(selected,'long_report')
    print(f"After long_report rc_summary filter: {len(selected)} reports")
    for config in selected:
        print(f"candlestick_num: {common.recursive_get(config,'candlestick_num')} | predict_num: {common.recursive_get(config,'predict_num')} | holdbar: {common.recursive_get(config,'holdbar')}")
    out_path = os.path.join(output_dir,"selected_configs" ,"selected_configs.jsonl")
    os.makedirs(os.path.join(output_dir,"selected_configs"), exist_ok=True)
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

def filter_by_performance(reports, period= 'short_report', min_cagr=None, min_calmar=None, min_sharpe=None):
    """
    Filter reports based on performance metrics.
    """
    def meets_criteria(report):
        perf = report.get(period).get("performance", {})
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
    period= 'short_report',
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
        rc = report.get(period).get("performance", {}).get("rc_summary", {})
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

def filter_by_trades(reports, period= 'short_report', min_win_rate=35, min_daily_freq = None):
    """
    Filter reports based on trade statistics.
    """
    def meets_criteria(report):
        trades = report.get(period).get("trades", {})
        if min_win_rate is not None and trades.get("win_rate", 0) < min_win_rate:
            return False
        if min_daily_freq is not None and trades.get("daily_freq", 0) < min_daily_freq:
            return False
        return True

    return [r for r in reports if meets_criteria(r)]

def find_key_path(obj, target_key, path=None):
    """
    递归查找 target_key 在嵌套对象中的路径。
    返回一个路径列表，可以用来直接索引该值。
    
    例如: find_key_path(report, "candlestick_num") 返回 ["params", "common", "candlestick_num"]
    """
    if path is None:
        path = []
    
    if isinstance(obj, dict):
        if target_key in obj:
            return path + [target_key]
        for key, value in obj.items():
            result = find_key_path(value, target_key, path + [key])
            if result is not None:
                return result
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            result = find_key_path(item, target_key, path + [i])
            if result is not None:
                return result
    
    return None


def get_value_by_path(obj, path):
    """
    使用路径列表直接获取对象中的值。
    
    例如: get_value_by_path(report, ["params", "common", "candlestick_num"])
    """
    current = obj
    try:
        for key in path:
            current = current[key]
        return current
    except (KeyError, IndexError, TypeError):
        return None


def analyze_candlestick_num(selected, target_key="candlestick_num", metric_key="cagr"):
    """
    从 selected 中递归查找 target_key，统计数量并分析性能指标，比较最优值。
    第一次遍历会自动定位 target_key 的位置，之后直接用路径索引，提高效率。
    
    Args:
        selected: 选中的报告列表
        target_key: 要查找的键名（默认 "candlestick_num"）
        metric_key: 性能指标的键名（默认 "cagr"）
    """
    from collections import defaultdict
    import numpy as np
    
    if not selected:
        print("❌ 报告列表为空")
        return
    
    # 第一次遍历：找到 target_key 的路径
    key_path = find_key_path(selected[0], target_key)
    
    if key_path is None:
        print(f"❌ 未找到任何 {target_key}")
        return
    
    print(f"✓ 定位 {target_key} 的路径: {' -> '.join(map(str, key_path))}")
    
    # 按 target_key 分组
    groups = defaultdict(list)
    
    for report in selected:
        value = get_value_by_path(report, key_path)
        if value is not None:
            groups[value].append(report)
    
    if not groups:
        print(f"❌ 未找到任何有效的 {target_key}")
        return
    
    # 统计每个值的性能
    analysis_results = []
    total_count = sum(len(v) for v in groups.values())
    
    for value in sorted(groups.keys()):
        reports = groups[value]
        count = len(reports)
        
        # 提取性能指标 (short_report)
        metric_list = []
        calmar_list = []
        
        for report in reports:
            short_report = report.get("short_report", report)
            perf = short_report.get("performance", {})
            metric = perf.get(metric_key)
            calmar = perf.get("calmar")
            
            if metric is not None:
                metric_list.append(metric)
            if calmar is not None:
                calmar_list.append(calmar)
        
        analysis_results.append({
            target_key: value,
            "count": count,
            "percentage": (count / total_count) * 100,
            f"avg_{metric_key}": np.mean(metric_list) if metric_list else None,
            "avg_calmar": np.mean(calmar_list) if calmar_list else None,
            f"max_{metric_key}": np.max(metric_list) if metric_list else None,
            f"std_{metric_key}": np.std(metric_list) if len(metric_list) > 1 else 0,
        })
    
    # 打印结果
    print("\n" + "="*100)
    print(f"📊 {target_key} 分析结果 (总共 {total_count} 个报告)")
    print("="*100)
    print(f"{target_key:<15} {'Count':<8} {'%':<8} {f'平均{metric_key.upper()}':<12} {f'Max {metric_key.upper()}':<12} {f'Std {metric_key.upper()}':<12} {'平均Calmar':<12}")
    print("-"*100)
    
    for result in analysis_results:
        value = result[target_key]
        count = result["count"]
        pct = result["percentage"]
        avg_metric = result[f"avg_{metric_key}"]
        max_metric = result[f"max_{metric_key}"]
        std_metric = result[f"std_{metric_key}"]
        avg_calmar = result["avg_calmar"]
        
        metric_str = f"{avg_metric:.2%}" if avg_metric is not None else "N/A"
        max_metric_str = f"{max_metric:.2%}" if max_metric is not None else "N/A"
        std_metric_str = f"{std_metric:.4f}" if std_metric is not None else "N/A"
        calmar_str = f"{avg_calmar:.2f}" if avg_calmar is not None else "N/A"
        
        print(f"{value:<15} {count:<8} {pct:<7.1f}% {metric_str:<12} {max_metric_str:<12} {std_metric_str:<12} {calmar_str:<12}")
    
    print("="*100)
    
    # 找出最优值
    valid_results = [r for r in analysis_results if r[f"avg_{metric_key}"] is not None]
    if valid_results:
        best_metric = max(valid_results, key=lambda x: x[f"avg_{metric_key}"])
        best_calmar = max(valid_results, key=lambda x: x["avg_calmar"] if x["avg_calmar"] is not None else -float('inf'))
        
        print(f"\n🏆 最优值对比:")
        print(f"  平均{metric_key.upper()}最优: {target_key}={best_metric[target_key]} ({best_metric[f'avg_{metric_key}']:.2%})")
        print(f"  平均Calmar最优: {target_key}={best_calmar[target_key]} ({best_calmar['avg_calmar']:.2f})")
        print("="*100)

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
    main()
    # filter_by_short_long_reports()
