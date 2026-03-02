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
# 引入自定义模块
from data_process.common import *
from data_process import common 

output_dir = os.path.join(common.PERSISTENCE_DIR,'batch_experiments')
TOP_K = 50
SKIP_PERCENT = 0  # 跳过前百分之多少，0表示不跳过，从最前面开始选择

def analyze_short_long_correlation(selected):
    """
    分析 short 与 long 的线性相关性
    """

    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    short_cagr = []
    l_cagr = []

    short_calmar = []
    l_calmar = []

    for r in selected:
        sc = r.get("cagr")
        lc = r.get("l_cagr")
        s_cal = r.get("calmar")
        l_cal = r.get("l_calmar")

        if sc is not None and lc is not None:
            short_cagr.append(sc)
            l_cagr.append(lc)

        if s_cal is not None and l_cal is not None:
            short_calmar.append(s_cal)
            l_calmar.append(l_cal)

    if len(short_cagr) < 5:
        print("❌ 样本太少，无法计算相关性")
        return

    print("\n" + "="*100)
    print("📈 Short vs Long 相关性分析")
    print("="*100)

    # CAGR
    pearson_cagr = pearsonr(short_cagr, l_cagr)
    spearman_cagr = spearmanr(short_cagr, l_cagr)

    print(f"CAGR Pearson:  r = {pearson_cagr.statistic:.4f} | p = {pearson_cagr.pvalue:.4e}")
    print(f"CAGR Spearman: r = {spearman_cagr.statistic:.4f} | p = {spearman_cagr.pvalue:.4e}")

    # Calmar
    if len(short_calmar) > 5:
        pearson_calmar = pearsonr(short_calmar, l_calmar)
        spearman_calmar = spearmanr(short_calmar, l_calmar)

        print(f"\nCalmar Pearson:  r = {pearson_calmar.statistic:.4f} | p = {pearson_calmar.pvalue:.4e}")
        print(f"Calmar Spearman: r = {spearman_calmar.statistic:.4f} | p = {spearman_calmar.pvalue:.4e}")

    print("="*100)

    # 分位数单调性测试
    print("\n🔎 分位数单调性检验（按 short CAGR 分桶）")

    pairs = list(zip(short_cagr, l_cagr))
    pairs.sort(key=lambda x: x[0])

    buckets = np.array_split(pairs, 5)

    for i, b in enumerate(buckets):
        long_vals = [x[1] for x in b]
        print(f"Bucket {i+1}: avg long CAGR = {np.mean(long_vals):.4f}")

    print("="*100)


def merge_selected(records):
    """
    selected: dict[hash] -> full_report
    report: 原始 report（json 读出来的 dict）
    row: extract_row 的结果（用于取 cagr / calmar）
    """
    reslut_set = set()
    uni_results = []
    duplicate_r = []

    for r in records:
        h = r['hash']
        if h not in reslut_set:
            reslut_set.add(h)
            # print(f" Duplicate records {h}")
            uni_results.append(r)
        else:
            duplicate_r.append(r)
    print(f" Total:{len(records)} Duplicate records {len(duplicate_r)}, uni_results {len(uni_results)}")
    return uni_results

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

def save_raw_reports(selected_rows, exp_dir ='' ,output_filename="reports_raw.jsonl"):
    """
    将 rows 中存储的 'raw' 原始报告内容提取出来，并保存到与 exp_dir 同级的指定文件中。
    保持 jsonl 格式。
    """
    if not selected_rows:
        print("⚠️ 没有数据可供保存")
        return

    out_path = os.path.join(exp_dir, output_filename)

    print(f"📦 正在提取并保存 {len(selected_rows)} 条原始报告至: {out_path}")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for row in selected_rows:
                # 提取 raw 字段（它是原始加载时的完整字典）
                raw_data = row.get("raw")
                if raw_data:
                    f.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
        
        print(f"✅ 原始数据保存完成！")
    except Exception as e:
        print(f"❌ 保存失败: {str(e)}")


def extract_row(report, src_path):
    """
    从单条 report 中抽取关键信息
    """
    short = report.get("short", report)  # 兼容 short/long 分开存储和合并存储两种格式
    long = report.get("long", report)
    forward = report.get("forward", report)
    perf = short.get("performance", {})
    params = short.get("params", {})
    common = params.get("common", {})
    long_perf = long.get("performance", {})
    long_params = long.get("params", {})
    long_common = long_params.get("common", {})
    forward_perf = forward.get("performance", {})
    return {
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
        "raw" : report,
    }

def basic_filter(all_results):
    analyze_holdbar(all_results,target_key="holdbar", period ='short',metric_key="cagr")
    ps_results_0,unselected = filter_by_criteria(all_results, period ='short', cagr=0)
    print(f"After 0-screening short: {len(ps_results_0)}, {len(ps_results_0)/len(all_results)*100:.2f}%")
    ps_results,unselected = filter_by_criteria(ps_results_0, period ='short', cagr=0.2)
    print(f"After Pre-screening short: {len(ps_results)}, {len(ps_results)/len(ps_results_0)*100:.2f}%")
    pf_results,unselected = filter_by_criteria(ps_results, period ='forward', cagr=0.2)
    print(f"After Pre-screening forward: {len(pf_results)}, {len(pf_results)/len(ps_results)*100:.2f}%")
    l_results,unselected = filter_by_criteria(pf_results, period ='long', cagr=0.1)
    print(f"After Pre-screening long: {len(l_results)}, {len(l_results)/len(pf_results)*100:.2f}%")
    return l_results

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
          f"{'RC_PosRatio':>12}")

    print("-" * 98)

    for i, r in enumerate(all_results):
        g = lambda k: common.recursive_get(r['long'], k)
        g('daily_loss_list')
        print(f"{i:>4}"
              f"{str(g('hash')):>12}"
              f"{g('cagr'):10.2f}"
              f"{g('sharpe'):10.2f}"
              f"{g('calmar'):10.2f}"
              f"{g('max_dd_pct'):12.2f}"
              f"{g('daily_freq'):12.2f}"
              f"{g('win_rate'):10.2f}"
              f"{g('rc_median'):12.2f}"
              f"{g('rc_pos_ratio'):12.2f}")
    compute_correlation(all_results,output_dir)
    plot_in_batches(all_results,output_dir,batch_size)
    exit()

def sort_by_correlation_diversity(all_results):
    """
    计算每个策略与其他策略的平均相关性，并按独立性排序
    """
    import pandas as pd
    
    # 1. 构建收益率 DataFrame (复用你原有的 build_return_series)
    returns_dict = {}
    for i, r in enumerate(all_results):
        # 假设 build_return_series 已定义
        ret = build_return_series(r) 
        returns_dict[f"S{i}"] = ret
    
    df = pd.DataFrame(returns_dict).dropna()
    
    # 2. 计算相关性矩阵
    corr_matrix = df.corr()
    
    # 3. 计算每个策略与其他策略的平均相关性 (排除对角线的 1.0)
    # 公式: (每列总和 - 1) / (策略总数 - 1)
    n = len(corr_matrix)
    mean_corr = (corr_matrix.sum() - 1) / (n - 1)
    
    # 4. 将结果转为 DataFrame 并排序
    diversity_df = mean_corr.to_frame(name="mean_correlation").sort_values(by="mean_correlation")
    
    print("\n" + "-"*20 + " Strategy Diversity Ranking " + "-"*20)
    print(diversity_df)
    
    # 5. 根据排序结果重新组织 all_results
    sorted_indices = [int(idx.replace('S', '')) for idx in diversity_df.index]
    sorted_results = [all_results[idx] for idx in sorted_indices]
    
    return sorted_results

def build_return_series(report):
    g = lambda k: common.recursive_get(report['long'], k)
    daily = g('daily_loss_list')

    df = pd.DataFrame(daily)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    # 用 equity 计算日收益率
    df['ret'] = df['equity'].pct_change()

    return df['ret'].dropna()


def compute_correlation(all_results, output_dir):
    """
    动态计算尺寸生成相关性热力图，确保格子大小固定且文字清晰
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "correlation_heatmap_fixed_cell.png")

    # ===== 1. 构建收益率序列 =====
    returns_dict = {}
    for i, r in enumerate(all_results):
        # 使用 build_return_series
        ret = build_return_series(r)
        returns_dict[f"S{i}"] = ret
    
    df = pd.DataFrame(returns_dict).dropna()
    if df.empty:
        print("⚠️ 数据为空，跳过热力图生成")
        return

    corr_matrix = df.corr()
    num_strategies = len(corr_matrix)

    # ===== 2. 动态计算图片尺寸 =====
    # 设定每个格子的目标尺寸 (inches)
    cell_size = 0.5  
    # 边缘留白（用于显示坐标轴刻度和标题）
    margin = 3.0     
    
    # 动态总宽和总高
    fig_width = num_strategies * cell_size + margin
    fig_height = num_strategies * cell_size + margin
    
    # 根据策略数量调整字体大小，防止文字重叠
    font_scale = 1.0 if num_strategies < 20 else 0.8 if num_strategies < 50 else 0.5

    plt.figure(figsize=(fig_width, fig_height))
    sns.set_theme(font_scale=font_scale)

    # ===== 3. 绘制热力图 =====
    # cbar_pos 用于微调颜色条，防止在超大图下显得太窄
    ax = sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='RdYlBu_r', 
        vmin=-1, vmax=1, 
        center=0,
        square=True, 
        linewidths=.5,
        annot_kws={"size": 10 if num_strategies < 30 else 7}, # 动态调整格内数字大小
        cbar_kws={"shrink": 0.8}
    )

    plt.title(f"Strategy Correlation Matrix (N={num_strategies})", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # ===== 4. 保存图片 =====
    plt.savefig(save_path, dpi=150) # 由于尺寸已经很大，150 DPI 足够清晰
    plt.close()

    print(f"📊 动态尺寸热力图已保存 (Size: {fig_width:.1f}x{fig_height:.1f} in): {save_path}")

def plot_equity_curves(all_results, output_dir, file_name="equity_full_combined.png", start_index=0):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, file_name)

    # ===== 1. 获取价格背景数据 =====
    pere_para = all_results[0]['long']['params']['common']
    price_file = os.path.join(
        common.PROJECT_DATA_DIR, 
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

    # ===== 2. 拼接并绘制曲线 =====
    # 用于记录分界线的时间点
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
            
            # 记录该阶段的起始时间（用于画竖线）
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

    # ===== 3. 绘制阶段切分竖线 =====
    # 仅绘制竖线，不添加文字，且颜色调至更淡 (alpha=0.3)
    if 'short' in split_dates:
        s_start = split_dates['short']
        # 使用较细的虚线，蓝色代表从 Long 进入 Short
        ax1.axvline(x=s_start, color='blue', linestyle='--', linewidth=1, alpha=0.3)

    if 'forward' in split_dates:
        f_start = split_dates['forward']
        # 使用红色虚线代表进入最关键的 Forward 阶段
        ax1.axvline(x=f_start, color='red', linestyle='--', linewidth=1, alpha=0.3)

    # ===== 4. 图例与美化 =====
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', ncol=4, fontsize=8)

    plt.title(f"Strategy Performance: Long (Train) -> Short (Val) -> Forward (Test)")
    fig.tight_layout()
    print(f"[SAVE] {save_path}")
    plt.savefig(save_path, dpi=200)
    plt.close()

def plot_in_batches(all_results, output_dir, batch_size=5):
    total = len(all_results)
    
    sns.set_theme(style="white")
    for i in range(0, total, batch_size):
        batch = all_results[i:i + batch_size]
        filename = f"batch_{i//batch_size + 1}.png"
        # ✨ 传入当前的循环索引 i 作为起始编号
        plot_equity_curves(batch, output_dir, filename, start_index=i)

def main():
    # exp_dir = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'DOGEUSDT_15m','2026-02-21','22_58_48')
    exp_dir = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'ETHUSDT_15m','2026-02-27','14_41_59')
    filter_report = None
    filter_report =  os.path.join(exp_dir,'filtered_raw_reports.jsonl')
    report_files = []
    rows = []
    records = []
    if filter_report:
        report_files.append(filter_report)
    else:
        for jsonl_path in iter_reports_jsonl(exp_dir):
            report_files.append(jsonl_path)
    for report_file in report_files:
        records = load_reports(report_file)
        for r in records:
            row = extract_row(r, report_file)
            rows.append(row)
    symbol = rows[0]['short']['params']['common']['symbol']
    interval = rows[0]['short']['params']['common']['interval']
    print(f"Total reports loaded: {len(rows)}")
    uin_records = merge_selected(rows)
    print(f"Total uint reports: {len(uin_records)}")
    if not filter_report:
        uin_records = basic_filter(uin_records)
        save_raw_reports(uin_records,exp_dir, "filtered_raw_reports.jsonl")
        exit()
    
    sorted_selected1 = sorted(uin_records, key=itemgetter("l_cagr"), reverse=True)
    # plot_heatmap(sorted_selected1,var1_key='predict_num',var2_key='predict_num',metric_key="l_cagr",save_path=os.path.join(output_dir,f"l_cagr_heatmap_combined.png"))
    # plot_heatmap(sorted_selected1,var1_key='predict_num',var2_key='predict_num',metric_key="l_sharpe",save_path=os.path.join(output_dir,f"l_sharpe_heatmap_combined.png"))
    # plot_heatmap(sorted_selected1,var1_key='predict_num',var2_key='predict_num',metric_key="l_calmar",save_path=os.path.join(output_dir,f"l_calmar_heatmap_combined.png"))
    # exit()
    analyze_holdbar(sorted_selected1,target_key="holdbar",period ='long', metric_key="cagr")
    if symbol == 'DOGEUSDT' and interval=='15m':
        l_results,unselected = filter_by_criteria(sorted_selected1, period ='long', cagr=0.6,rc_median = 0,rc_pos_ratio = 0.6,calmar = 1.9,daily_freq = 0.3,sharpe = 1)
    if symbol == 'ETHUSDT' and interval=='15m':
        l_results,unselected = filter_by_criteria(sorted_selected1, period ='long', cagr=0.2,rc_median = 0,rc_pos_ratio = 0.6,calmar = 1,daily_freq = 0.15,sharpe = 0.5)

    # sort_by_correlation_result = sort_by_correlation_diversity(l_results)
    analyze_holdbar(l_results,target_key="holdbar",period ='long', metric_key="cagr")
    # exit()
    # sorted_by_pos_ratio = sorted(
    #     l_results, 
    #     key=lambda r: common.recursive_get(r.get('long', {}), 'rc_pos_ratio') or 0, 
    #     reverse=True
    # )
    sorted_calmar = sorted(l_results, key=itemgetter("l_calmar"), reverse=True)
    selected = l_results
    filter_hash_doge_15 = ['c48fc76e','ad40b408','dc6c1390','584eb8de','83b16fb9','b3b9b8c5','9b0facb8','08c2acf6','20d6cd8a','e6c308e5','0824cbdc',
                   'd72bb0d4','ae6b5897','48514f66','31a957db','89f39876','c780edee','64afb891','f9a98676' , '084c68b5','d22cf3db','2c2321b3'
                   '4d008904','7a430104','c0e6a3dd','436a8503','b4633eab','243b56c6','0d6533f3']
    filter_hash_eth_15 = ['943143f8', '21f9fce3', 'e4927150', '9a3f7676', '4afa85ac' ,'3163d070','ed96bd77','cc89356b']
    filter_hash = filter_hash_doge_15 + filter_hash_eth_15
    selected = [
        r for r in l_results 
        if str(common.recursive_get(r.get('long', {}), 'hash'))[:8] not in filter_hash
    ]

    print(f"🎯 Hash 过滤完成: 过滤前 {len(l_results)} 条 -> 过滤后 {len(selected)} 条")
    show_performance(selected,output_dir,3)
    # stable_selected1 = filter_stable(rc_median_results)
    # print(f"-------------After filter_stable: {len(stable_selected1)} reports")
    # # selected2 = filter_aggressive(rc_results)
    # # print(f"-------------After filter_aggressive: {len(selected2)} reports")
    # # merged_selected = merge_selected_sort(sorted_selected1[:5],selected2[:5],period ='long', sort_key='cagr')
    # # print(f"-------------After all filter: {len(merged_selected)} reports")
    
    # rc_pos_ratio_results,unselected = filter_by_criteria(stable_selected1, period ='long', rc_pos_ratio = 0.7)
    # print(f"-------------After rc_pos_ratio: {len(rc_pos_ratio_results)} reports")



    # sorted_l_sharpe = sorted(rc_pos_ratio_results, key=itemgetter("l_sharpe"), reverse=True)
    # sorted_calmar = sorted(rc_pos_ratio_results, key=itemgetter("l_calmar"), reverse=True)
    # # sorted_l_win_rate = sorted(rc_results, key=itemgetter("l_win_rate"), reverse=True)
    # # sorted_l_daily_freq = sorted(rc_results, key=itemgetter("l_daily_freq"), reverse=True)
    # top_k = 40
    # merged_selected = merge_selected_sort(sorted_l_sharpe[:top_k],sorted_calmar[:top_k],rc_pos_ratio_results[:top_k],period ='long', sort_key='cagr')
    out_path = os.path.join(output_dir,"selected_configs" ,"selected_configs.jsonl")
    os.makedirs(os.path.join(output_dir,"selected_configs"), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[SAVE] {out_path} | total={len(selected)}")

def filter_stable(selected):
    results,unselected = filter_by_criteria(selected, period ='short', cagr=0.7, calmar=0, win_rate = 30  )
    print(f"After short performance filter: {len(results)} reports")
    results,long_unresults = filter_by_performance(results, period ='long', min_cagr=0.7, min_calmar=0.5)#,min_rc_cagr_median = -0.2)#,min_rc_cagr_q25 = -0.2)
    print(f"After long cagr filter: {len(results)} reports")
    results,long_unresults = filter_by_performance(results, period ='long', min_rc_cagr_median = 0)
    print(f"After long rc_cagr_median filter: {len(results)} reports")
    results,forward_unresults = filter_by_performance(results, period ='forward', min_cagr=0.7, min_calmar=0.5)
    print(f"After forward performance filter: {len(results)} reports")
    return results

def filter_aggressive(selected):
    # 1️⃣ short 要非常强（抓当前 regime）
    results, _ = filter_by_criteria( selected, period='short', cagr=1, calmar=0)
    print(f"After short performance filter: {len(results)} reports")
    # 2️⃣ long 只保证不是垃圾，不追求极稳
    results, _ = filter_by_performance( results, period='long', min_cagr=0.6, min_calmar=0.3 )
    print(f"After long performance filter: {len(results)} reports")
    results,long_unresults = filter_by_performance(results, period ='long', min_rc_cagr_median = 0)
    print(f"After long rc_cagr_median filter: {len(results)} reports")
    results, _ = filter_by_performance( results, period='forward', min_cagr=1, min_calmar=0.4)
    print(f"After forward performance filter: {len(results)} reports")
    results, _ = filter_by_criteria( results, period='short', daily_freq = 0.7)
    print(f"After long daily_freq filter: {len(results)} reports")

    return results

def merge_selected_sort(*selected_lists, period ='short', sort_key=None, reverse=True):
    """
    合并多个 selected 列表，按 hash 去重，并按 sort_key 排序。

    Parameters
    ----------
    *selected_lists : 任意多个 selected 列表
    sort_key : str
        用于排序的字段名，例如 "l_cagr"
    reverse : bool
        True 表示降序排序（默认从大到小）

    Returns
    -------
    list
        去重+排序后的新列表
    """

    merged_dict = {}

    for selected in selected_lists:
        for row in selected:
            h = row.get("hash")
            if h not in merged_dict:
                merged_dict[h] = row

    result = list(merged_dict.values())

    # 排序
    if sort_key is not None:
        result.sort(
            key=lambda x: common.recursive_get(x.get(period), sort_key),
            reverse=reverse
        )

    return result

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
        holdbar = row["report"]["params"]["common"]["holdbar"]
        holdbar = row["report"]["params"]["strategy"]["holdbar"]
        if holdbar == 20 and holdbar ==20:
            group_1_data.append(row)
        elif holdbar == 20 and holdbar ==16:
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
                "Max Calmar": f"{np.max(m['calmar']):.2%}",
                "Avg Sharpe": f"{np.mean(m['sharpe']):.2f}",
                "Avg MaxDD": f"{np.mean(m['max_dd']):.2f}%" # 修正回撤显示
            })

    # 4. 完美对齐输出
    if summary_rows:
        pd.set_option('display.max_columns', None)      # 显示所有列
        pd.set_option('display.expand_frame_repr', False) # 禁止自动换行（保持在一行内）
        pd.set_option('display.max_colwidth', None)     # 不限制列宽
        pd.set_option('display.width', 1000)            # 设置足够大的总宽度
        pd.set_option('display.expand_frame_repr', True)
        df_final = pd.DataFrame(summary_rows).set_index("Group")
    
        print("\n" + "="*120) # 稍微拉长分割线
        print(f"📊 参数组对比评估 (Group 1: {label1} | Group 2: {label2})")
        print("="*120)
        
        # 强制不换行打印
        print(df_final.to_string(justify='center', index=True, line_width=1000))
        print("="*120)
        
        # 5. 极简结论
        cagr1 = np.mean(g1_metrics['cagr'])
        cagr2 = np.mean(g2_metrics['cagr'])
        winner = "Group 1" if cagr1 > cagr2 else "Group 2"
        print(f"💡 结论预览: {winner} 在收益期望上表现更优 ({max(cagr1, cagr2):.2%})")
    else:
        print("❌ 错误: 未能分类出有效数据，请检查输入 rows 的参数。")
    exit()

def filter_by_criteria(reports, period='short', **criteria):
    """
    分步过滤函数：
    - 顺序应用每个筛选条件
    - 实时打印本步筛选后的存活数量及相对于上一步的留存比例
    """
    if not reports:
        return [], []
    initial_len = len(reports)
    passed = reports
    
    for key, min_value in criteria.items():
        # 跳过空值
        if min_value is None:
            continue
            
        # 如果池子已经空了，直接记录后续结果为 0
        if not passed:
            print(f"After screening {period} {key:<12} >= {min_value:>3}: 0, 0.00%")
            continue

        prev_len = len(passed)
        
        # 1. 定位该指标在字典中的路径 (复用你现有的 find_key_path)
        # 注意：这里基于当前 surviving 池子的第一个样本来找路径
        key_path = find_key_path(passed[0].get(period, {}), key)
        if key_path is None:
            print(f"⚠️ Warning: key '{key}' not found in {period} reports, skipping this filter.")
            continue

        # 2. 执行本轮单项过滤
        step_passed = []
        for r in passed:
            period_data = r.get(period, {})
            # 复用你现有的 get_value_by_path
            current_value = get_value_by_path(period_data, key_path)
            
            # 比较逻辑
            if current_value is not None and current_value >= min_value:
                step_passed.append(r)
        
        # 3. 更新当前池子并打印结果
        passed = step_passed
        curr_len = len(passed)
        ratio = (curr_len / prev_len * 100) if prev_len > 0 else 0
        
        # 按照你要求的格式打印
        print(f"After screening {period} {key:<12} >= {min_value:>3}: {curr_len}, {ratio:.2f}%")
    final_len = len(passed)
    filtered_count = initial_len - final_len
    summary_desc = f"TOTAL SUMMARY ({period.upper()})"
    print(f"{summary_desc:<25}: {final_len:>6} remaining, {filtered_count:>6} filtered out, {final_len/initial_len*100:.2f}%")
    passed_ids = {id(r) for r in passed}
    failed = [r for r in reports if id(r) not in passed_ids]
    return passed, failed

def filter_by_performance(reports, period= 'short', min_cagr=None, min_calmar=None, min_sharpe=None,min_rc_cagr_median = None, min_rc_cagr_q25 = None):
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
        if min_rc_cagr_median is not None and recursive_get(perf, "rc_cagr_median") < min_rc_cagr_median:
            return False
        if min_rc_cagr_q25 is not None and recursive_get(perf, "rc_cagr_q25") < min_rc_cagr_q25:
            return False
        return True
    passed = []
    failed = []
    for r in reports:
        if meets_criteria(r):
            passed.append(r)
        else:
            failed.append(r)

    return passed, failed

def filter_by_rc_summary(
    reports,
    period= 'short',
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

def filter_by_trades(reports, period= 'short', min_win_rate=35, min_daily_freq = None):
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

    passed = []
    failed = []
    for r in reports:
        if meets_criteria(r):
            passed.append(r)
        else:
            failed.append(r)

    return passed, failed

def find_key_path(obj, target_key, path=None):
    """
    递归查找 target_key 在嵌套对象中的路径。
    返回一个路径列表，可以用来直接索引该值。
    
    例如: find_key_path(report, "holdbar") 返回 ["params", "common", "holdbar"]
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
    
    例如: get_value_by_path(report, ["params", "common", "holdbar"])
    """
    current = obj
    try:
        for key in path:
            current = current[key]
        return current
    except (KeyError, IndexError, TypeError):
        return None


def analyze_holdbar(records, target_key="holdbar", period='short', metric_key="cagr"):
    """
    从 selected 中递归查找 target_key，统计数量并分析性能指标，比较最优值。
    第一次遍历会自动定位 target_key 的位置，之后直接用路径索引，提高效率。
    
    Args:
        selected: 选中的报告列表
        target_key: 要查找的键名（默认 "holdbar"）
        metric_key: 性能指标的键名（默认 "cagr"）
    """
    from collections import defaultdict
    import numpy as np
    
    if not records:
        print("❌ 报告列表为空")
        return
    
    # 第一次遍历：找到 target_key 的路径
    key_path = find_key_path(records[0], target_key)
    
    if key_path is None:
        print(f"❌ 未找到任何 {target_key}")
        return
    
    print(f"✓ 定位 {target_key} 的路径: {' -> '.join(map(str, key_path))}")

    # 第一次遍历：找到 target_key 的路径
    key_path = find_key_path(records[0], target_key)
    
    if key_path is None:
        print(f"❌ 未找到任何 {target_key}")
        return
    
    print(f"✓ 定位 {target_key} 的路径: {' -> '.join(map(str, key_path))}")

    # 按 target_key 分组
    groups = defaultdict(list)
    
    for report in records:
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
        
        # 提取性能指标 (long)
        metric_list = []
        calmar_list = []
        
        for report in reports:
            p_report = report.get(period, report)
            perf = p_report.get("performance", {})
            metric = recursive_get(p_report,metric_key)
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
            f"max_calmar": np.max(calmar_list) if calmar_list else None,
            f"med_calmar": np.median(calmar_list) if calmar_list else None,
            f"max_{metric_key}": np.max(metric_list) if metric_list else None,
            f"std_{metric_key}": np.std(metric_list) if len(metric_list) > 1 else 0,
            f"med_{metric_key}": np.median(metric_list) if metric_list else None,
        })
    
    # 打印结果
    print("\n" + "="*100)
    print(f"📊 {target_key} {period} 分析结果 (总共 {total_count} 个报告)")
    print("="*100)
    print(f"{target_key:<15} {'Count':<8} {'%':<6} {f'{metric_key.upper()}':<12} {'':<2}{'AVG':<6}{'Max':<6}{'Std':<6}{'Med':<6} {'Calmar:':<8}{'AVG':<6}{'MAX':<6}{'Med':<6}")
    print("-"*100)
    
    for result in analysis_results:
        value = result[target_key]
        count = result["count"]
        pct = result["percentage"]
        avg_metric = result[f"avg_{metric_key}"]
        max_metric = result[f"max_{metric_key}"]
        std_metric = result[f"std_{metric_key}"]
        med_metric = result[f"med_{metric_key}"]
        avg_calmar = result["avg_calmar"]
        max_calmar = result["max_calmar"]
        med_calmar = result["med_calmar"]
        
        metric_str = f"{avg_metric:.2%}" if avg_metric is not None else "N/A"
        max_metric_str = f"{max_metric:.2%}" if max_metric is not None else "N/A"
        std_metric_str = f"{std_metric:.4f}" if std_metric is not None else "N/A"
        med_metric_str = f"{med_metric:.4f}" if med_metric is not None else "N/A"
        calmar_str = f"{avg_calmar:.2f}" if avg_calmar is not None else "N/A"
        max_calmar_str = f"{max_calmar:.2f}" if max_calmar is not None else "N/A"
        med_calmar_str = f"{med_calmar:.2f}" if med_calmar is not None else "N/A"
        
        print(f"{value:<15} {count:<8} {pct:<5.1f}% {metric_key.upper():<12}  {metric_str:<6} {max_metric_str:<6} {std_metric_str:<6} {med_metric_str:<6} {'':<8}{calmar_str:<6} {max_calmar_str:<6} {med_calmar_str:<6}")
    
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

def plot_heatmap(selected, var1_key, var2_key, metric_key="l_cagr", save_path="heatmap_combined.png"):
    """
    生成 2x2 的热力图矩阵，包含：均值、中位数、标准差、最大值
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. 数据准备
    path1 = find_key_path(selected[0], var1_key)
    path2 = find_key_path(selected[0], var2_key)
    
    matrix_data = []
    for report in selected:
        v1 = get_value_by_path(report, path1)
        v2 = get_value_by_path(report, path2)
        metric = recursive_get(report, metric_key)
        if v1 is not None and v2 is not None and metric is not None:
            matrix_data.append({var1_key: v1, var2_key: v2, "val": metric})

    df = pd.DataFrame(matrix_data)
    
    # 2. 计算四种统计维度
    # 使用 groupby 一次性聚合所有指标
    agg_df = df.groupby([var1_key, var2_key])["val"].agg(['mean', 'median', 'std', 'max']).reset_index()

    # 3. 创建 2x2 画布
    # 为了防止干扰全局背景，使用上下文管理器
    with sns.axes_style("white"):
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        stats_titles = {
            'mean': 'Mean (Expectation)',
            'median': 'Median (Robustness)',
            'std': 'Std Dev (Volatility)',
            'max': 'Max (Potential)'
        }
        
        # 循环绘制四个子图
        for i, stat in enumerate(['mean', 'median', 'std', 'max']):
            # 将当前指标转为透视表
            pivot_df = agg_df.pivot(index=var1_key, columns=var2_key, values=stat)
            
            # Std 通常用数值显示，收益类指标用百分比显示
            fmt_str = ".2f" if stat == 'std' else ".1%"
            
            sns.heatmap(
                pivot_df, 
                annot=True, 
                fmt=fmt_str, 
                cmap="RdYlBu_r", 
                ax=axes[i],
                cbar_kws={'label': stat.upper()}
            )
            axes[i].set_title(f"{stats_titles[stat]} - {metric_key.upper()}", fontsize=14, fontweight='bold')
            axes[i].set_xlabel(var2_key)
            axes[i].set_ylabel(var1_key)

    plt.suptitle(f"Parameter Sensitivity Analysis: {var1_key} vs {var2_key}", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为总标题留出空间
    
    plt.savefig(save_path, dpi=200)
    print(f"✅ 四合一热力图已保存至: {save_path}")
    plt.close() # 及时释放内存

if __name__ == "__main__":
    main()
    # filter_by_short_longs()
