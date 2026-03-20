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

output_dir = os.path.join(common.PERSISTENCE_DIR,'batch_experiments',"selected_configs")
os.makedirs(output_dir, exist_ok=True)
TOP_K = 50
SKIP_PERCENT = 0  # Percentage of front part to skip; 0 means no skip, select from the very beginning

def analyze_short_long_correlation(selected):
    """
    Analyze linear correlation between short and long periods.
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
        print("❌ Sample size too small to compute correlation")
        return

    print("\n" + "="*100)
    print("📈 Short vs Long correlation analysis")
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

    # Quantile monotonicity test
    print("\n🔎 Quantile monotonicity check (bucketed by short CAGR)")

    pairs = list(zip(short_cagr, l_cagr))
    pairs.sort(key=lambda x: x[0])

    buckets = np.array_split(pairs, 5)

    for i, b in enumerate(buckets):
        long_vals = [x[1] for x in b]
        print(f"Bucket {i+1}: avg long CAGR = {np.mean(long_vals):.4f}")

    print("="*100)

def analyze_model_performance_correlation(all_results):
    """
    Analyze correlation between model metrics (Accuracy, F1, Precision, Recall) and l_cagr.
    """
    from scipy.stats import pearsonr, spearmanr
    import pandas as pd

    # 1. Metrics to analyze (must exist in model_metrics)
    metrics_to_check = [
        'accuracy', 
        'f1_macro', 
        'f1_weighted', 
        'precision_weighted', 
        'recall_weighted'
    ]
    
    # 2. Extract data
    data_list = []
    for r in all_results:
        # Fetch target return metric
        l_cagr = r.get("l_cagr")
        # Fetch model metrics dict
        model_metrics = r['long']["model_metrics"]
        
        if l_cagr is not None and model_metrics:
            row = {"l_cagr": l_cagr}
            # Only extract the 5 metrics shown in the figure
            for m in metrics_to_check:
                val = model_metrics.get(m)
                if val is not None:
                    row[m] = val
            data_list.append(row)

    if len(data_list) < 10:
        print(f"⚠️ Sample size too small ({len(data_list)}) for meaningful correlation analysis")
        return

    df = pd.DataFrame(data_list)
    
    print("\n" + "="*80)
    print(f"📊 Model evaluation metrics vs long CAGR correlation (N={len(df)})")
    print("-" * 80)
    print(f"{'Metric Name':<20} | {'Pearson r':>10} | {'p-value':>12} | {'Spearman r':>10}")
    print("-" * 80)

    # 3. Compute correlation between each metric and l_cagr
    for m in metrics_to_check:
        if m not in df.columns:
            continue
            
        # Drop NaNs
        sub_df = df[['l_cagr', m]].dropna()
        if len(sub_df) < 5: continue

        p_r, p_val = pearsonr(sub_df[m], sub_df['l_cagr'])
        s_r, _ = spearmanr(sub_df[m], sub_df['l_cagr'])

        # Mark statistical significance
        sig = "*" if p_val < 0.05 else ""
        
        print(f"{m:<20} | {p_r:10.4f}{sig} | {p_val:12.2e} | {s_r:10.4f}")

    print("="*80)
    print("💡 Note: Pearson r close to 1 implies strong positive linear correlation; p-value < 0.05 (*) is statistically significant.")

def analyze_model_metrics_by_decile(all_results):
    """
    Bucket analysis: split return metrics (CAGR, Calmar, Sharpe) into 10 quantile buckets
    and observe the average model metrics (Accuracy, F1, etc.) within each bucket.
    """
    import pandas as pd
    import numpy as np

    # 1. Config
    trading_metrics = ['l_cagr', 'l_calmar', 'l_sharpe']
    model_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    
    # 2. Extract data
    data_list = []
    for r in all_results:
        # Fetch trading performance
        row = {
            'l_cagr': r.get('l_cagr'),
            'l_calmar': r.get('l_calmar'),
            'l_sharpe': r.get('long', {}).get('performance', {}).get('sharpe')  # Some versions may use slightly different key names
        }
        
        # Fetch model metrics
        model_metrics = r['long'].get("model_metrics", {})
        for mk in model_keys:
            row[mk] = model_metrics.get(mk)
            
        if row['l_cagr'] is not None:
            data_list.append(row)

    if len(data_list) < 20:
        print("⚠️ Not enough data for decile analysis")
        return

    df = pd.DataFrame(data_list)

    # 3. For each trading metric, run bucket analysis
    for t_metric in trading_metrics:
        if t_metric not in df.columns or df[t_metric].isnull().all():
            continue

        print("\n" + "="*100)
        print(f"📈 Bucket analysis: model metrics ranked by {t_metric.upper()} (10% quantile buckets)")
        print("="*100)

        # Use qcut to divide trading metric into 10 equal-sized buckets (deciles).
        # duplicates='drop' avoids failure when too many identical values exist.
        try:
            df['bucket'] = pd.qcut(df[t_metric], 10, labels=[f"Q{i+1}" for i in range(10)], duplicates='drop')
        except ValueError:
            # If samples are too few or values too concentrated, fall back to 5 buckets
            df['bucket'] = pd.qcut(df[t_metric], 5, labels=[f"Q{i+1}" for i in range(5)], duplicates='drop')
            print(f"Note: due to data distribution, {t_metric} bucket count automatically reduced to 5.")

        # Aggregate and compute mean per bucket
        bucket_stats = df.groupby('bucket', observed=True)[model_keys].mean()
        
        # Add bucket-wise average trading metric as reference
        bucket_stats[f'avg_{t_metric}'] = df.groupby('bucket', observed=True)[t_metric].mean()
        
        # Reorder columns to put reference metric first
        cols = [f'avg_{t_metric}'] + model_keys
        bucket_stats = bucket_stats[cols]

        # Print results
        pd.options.display.max_columns = None
        pd.options.display.width = 1000
        print(bucket_stats.to_string(formatters={
            f'avg_{t_metric}': '{:,.4f}'.format,
            'accuracy': '{:,.4f}'.format,
            'f1_macro': '{:,.4f}'.format,
            'f1_weighted': '{:,.4f}'.format
        }))
        
        # Simple monotonicity hint
        first_val = bucket_stats[model_keys[0]].iloc[0]
        last_val = bucket_stats[model_keys[0]].iloc[-1]
        trend = "✅ positively monotonic" if last_val > first_val else "❌ non-monotonic or reversed"
        print(f"\n💡 Trend check ({model_keys[0]}): from lowest to highest bucket -> {trend}")
        print("-" * 100)

def merge_selected(records):
    """
    Deduplicate records by hash and return a unique list.
    """
    reslut_set = set()
    uni_results = []
    duplicate_r = []

    for r in records:
        h = r['hash']
        if h not in reslut_set:
            reslut_set.add(h)
            # print(f" Duplicate record {h}")
            uni_results.append(r)
        else:
            duplicate_r.append(r)
    print(f" Total:{len(records)} Duplicate records {len(duplicate_r)}, uni_results {len(uni_results)}")
    return uni_results

def iter_reports_jsonl(root_list):
    """
    Recursively scan for all reports.jsonl files under given roots.
    """
    for root in root_list:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname == "reports.jsonl":
                    yield os.path.join(dirpath, fname)


def load_reports(path):
    """
    Read jsonl file line by line and skip malformed lines.
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
    Extract 'raw' report field from rows and save to a jsonl file next to exp_dir.
    """
    if not selected_rows:
        print("⚠️ No data to save")
        return

    out_path = os.path.join(exp_dir, output_filename)

    print(f"📦 Extracting and saving {len(selected_rows)} raw reports to: {out_path}")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            for row in selected_rows:
                # Extract the raw field (the full original dict loaded at the beginning)
                raw_data = row.get("raw")
                if raw_data:
                    f.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
        
        print(f"✅ Raw data saved successfully!")
    except Exception as e:
        print(f"❌ Failed to save: {str(e)}")


def extract_row(report, src_path):
    """
    Extract key fields from a single report.
    """
    short = report.get("short", report)  # Support both separated short/long storage and merged formats
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
    ps_results_0,unselected = filter_by_criteria(all_results, period ='short', cagr=0)
    print(f"After 0-screening short: {len(ps_results_0)}, {len(ps_results_0)/len(all_results)*100:.2f}%")
    ps_results,unselected = filter_by_criteria(ps_results_0, period ='short', cagr=0.2)
    print(f"After pre-screening short: {len(ps_results)}, {len(ps_results)/len(ps_results_0)*100:.2f}%")
    pf_results,unselected = filter_by_criteria(ps_results, period ='forward', cagr=0.2)
    print(f"After pre-screening forward: {len(pf_results)}, {len(pf_results)/len(ps_results)*100:.2f}%")
    l_results,unselected = filter_by_criteria(pf_results, period ='long', cagr=0)
    print(f"After pre-screening long: {len(l_results)}, {len(l_results)/len(pf_results)*100:.2f}%")
    return l_results

def filter_and_rank_strategies(data, metric, k=30, final_sort_key="l_cagr"):
    """
    Select top K strategies by a given metric, then re-sort them by final_sort_key.

    :param data: original strategy list (list of dicts)
    :param metric: metric name (str) or custom lambda function
    :param k: how many top strategies to select (int)
    :param final_sort_key: final sort key for presentation, default 'l_cagr'
    :return: list of strategies after two-stage sorting
    """
    
    # 1. Define sort key (handle lambda or normal key)
    if callable(metric):
        key_func = metric
    else:
        # Use get to handle missing keys gracefully
        key_func = lambda x: x.get(metric, 0) if x.get(metric) is not None else 0

    # 2. Select top K strategies by the given metric (descending)
    top_k = sorted(data, key=key_func, reverse=True)[:k]

    # 3. Re-sort these K results by final benchmark (e.g., CAGR)
    final_sorted = sorted(top_k, key=itemgetter(final_sort_key), reverse=True)
    
    return final_sorted

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
              f"{g('rc_pos_ratio'):12.2f}"
              f"{g('max_hwm_duration_days'):12.2f}")
    compute_correlation(all_results,output_dir)
    plot_in_batches(all_results,output_dir,batch_size)

def sort_by_correlation_diversity(all_results):
    """
    Compute the average correlation of each strategy with all others and sort by independence.
    """
    import pandas as pd
    
    # 1. Build returns DataFrame (reuse existing build_return_series)
    returns_dict = {}
    for i, r in enumerate(all_results):
        # Assume build_return_series is already defined
        ret = build_return_series(r) 
        returns_dict[f"S{i}"] = ret
    
    df = pd.DataFrame(returns_dict).dropna()
    
    # 2. Compute correlation matrix
    corr_matrix = df.corr()
    
    # 3. Compute average correlation of each strategy with others (excluding diagonal 1.0)
    # Formula: (column_sum - 1) / (num_strategies - 1)
    n = len(corr_matrix)
    mean_corr = (corr_matrix.sum() - 1) / (n - 1)
    
    # 4. Convert result to DataFrame and sort
    diversity_df = mean_corr.to_frame(name="mean_correlation").sort_values(by="mean_correlation")
    
    print("\n" + "-"*20 + " Strategy Diversity Ranking " + "-"*20)
    print(diversity_df)
    
    # 5. Reorder all_results according to ranking
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

    # Use equity to compute daily returns
    df['ret'] = df['equity'].pct_change()

    return df['ret'].dropna()


def compute_correlation(all_results, output_dir):
    """
    Dynamically compute figure size for correlation heatmap to keep cell size fixed and text clear.
    """
    save_path = os.path.join(output_dir, "correlation_heatmap_fixed_cell.png")

    # ===== 1. Build return series =====
    returns_dict = {}
    for i, r in enumerate(all_results):
        # Use build_return_series
        ret = build_return_series(r)
        returns_dict[f"S{i}"] = ret
    
    df = pd.DataFrame(returns_dict).dropna()
    if df.empty:
        print("⚠️ Data is empty, skip heatmap generation")
        return

    corr_matrix = df.corr()
    num_strategies = len(corr_matrix)

    # ===== 2. Dynamically compute figure size =====
    # Target size (inches) for each cell
    cell_size = 0.5  
    # Margin for axes ticks and title
    margin = 3.0     
    
    # Dynamic total width and height
    fig_width = num_strategies * cell_size + margin
    fig_height = num_strategies * cell_size + margin
    
    # Adjust font size based on number of strategies to avoid label overlap
    font_scale = 1.0 if num_strategies < 20 else 0.8 if num_strategies < 50 else 0.5

    plt.figure(figsize=(fig_width, fig_height))
    sns.set_theme(font_scale=font_scale)

    # ===== 3. Draw heatmap =====
    # cbar_pos can be used to tweak color bar width for large figures
    ax = sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='RdYlBu_r', 
        vmin=-1, vmax=1, 
        center=0,
        square=True, 
        linewidths=.5,
        annot_kws={"size": 10 if num_strategies < 30 else 7},  # Dynamically adjust font size inside cells
        cbar_kws={"shrink": 0.8}
    )

    plt.title(f"Strategy Correlation Matrix (N={num_strategies})", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # ===== 4. Save image =====
    plt.savefig(save_path, dpi=150)  # 150 DPI is enough because the figure is already large
    plt.close()

    print(f"📊 Dynamic-size correlation heatmap saved (Size: {fig_width:.1f}x{fig_height:.1f} in): {save_path}")

def plot_equity_curves(all_results, output_dir, file_name="equity_full_combined.png", start_index=0):
    save_path = os.path.join(output_dir, file_name)

    # ===== 1. Get price background data =====
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
        # ✨ Pass current loop index i as the starting number
        plot_equity_curves(batch, output_dir, filename, start_index=i)

def main():
    exp_dir1 = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'DOGEUSDT_15m','2026-03-07','01_03_38')
    exp_dir2 = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'DOGEUSDT_15m','2026-03-09','17_33_35')
    exp_dir5 = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'DOGEUSDT_15m','2026-03-18','02_38_16')
    exp_dir7 = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'DOGEUSDT_15m','2026-03-18','13_16_48')
    exp_dir9 = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'DOGEUSDT_15m','2026-03-19','01_15_14')
    exp_dir13 = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'DOGEUSDT_15m','2026-03-20','00_30_58')
    exp_dir10 = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'DOGEUSDT_30m','2026-03-19','11_00_31')
    exp_dir11 = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'DOGEUSDT_1h','2026-03-19','11_32_26')
    exp_dir12 = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'DOGEUSDT_1h','2026-03-19','11_53_49')
    exp_dir3 = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'ETHUSDT_15m','2026-03-15','18_41_56')
    exp_dir4 = os.path.join(common.PERSISTENCE_DIR,'batch_experiments', 'ETHUSDT_15m','2026-03-15','20_17_30')
    exp_dir_list = [exp_dir13]
    filter_report = None
    filter_report =  os.path.join(output_dir,'filtered_raw_reports.jsonl')
    report_files = []
    rows = []
    records = []
    if filter_report:
        report_files.append(filter_report)
    else:
        for jsonl_path in iter_reports_jsonl(exp_dir_list):
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
        analyze_holdbar(uin_records,target_key="candlestick_num", period ='short',metric_key="daily_freq")
        uin_records = basic_filter(uin_records)
        analyze_holdbar(uin_records,target_key="candlestick_num", period ='long',metric_key="cagr")
        plot_heatmap(uin_records,var1_key='flip_penalty',var2_key='miss_penalty', metric_key="l_cagr",save_path=os.path.join(output_dir,f"l_cagr_heatmap_combined.png"))
        plot_heatmap(uin_records,var1_key='flip_penalty',var2_key='miss_penalty', metric_key="l_sharpe",save_path=os.path.join(output_dir,f"l_sharpe_heatmap_combined.png"))
        plot_heatmap(uin_records,var1_key='flip_penalty',var2_key='miss_penalty', metric_key="l_calmar",save_path=os.path.join(output_dir,f"l_calmar_heatmap_combined.png"))
        save_raw_reports(uin_records,output_dir, "filtered_raw_reports.jsonl")
        exit()
    analyze_holdbar(uin_records,target_key="stride",period ='long', metric_key="cagr")
    analyze_holdbar(uin_records,target_key="holdbar",period ='long', metric_key="cagr")
    analyze_holdbar(uin_records,target_key="candlestick_num",period ='long', metric_key="cagr")
    analyze_holdbar(uin_records,target_key="vol_ewma_span", period ='long',metric_key="cagr")
    analyze_holdbar(uin_records,target_key="predict_num", period ='long',metric_key="cagr")
    analyze_holdbar(uin_records,target_key="vol_multiplier_long", period ='long',metric_key="cagr")
    analyze_holdbar(uin_records,target_key="atr_sl_mult_long", period ='long',metric_key="cagr")
    # analyze_model_performance_correlation(uin_records)
    # analyze_model_metrics_by_decile(uin_records)
    # exit()
    sorted_selected1 = sorted(uin_records, key=itemgetter("l_cagr"), reverse=True)
    # plot_heatmap(sorted_selected1,var1_key='predict_num',var2_key='predict_num',metric_key="l_cagr",save_path=os.path.join(output_dir,f"l_cagr_heatmap_combined.png"))
    # plot_heatmap(sorted_selected1,var1_key='predict_num',var2_key='predict_num',metric_key="l_sharpe",save_path=os.path.join(output_dir,f"l_sharpe_heatmap_combined.png"))
    # plot_heatmap(sorted_selected1,var1_key='predict_num',var2_key='predict_num',metric_key="l_calmar",save_path=os.path.join(output_dir,f"l_calmar_heatmap_combined.png"))
    # exit()
    stats, f_map, groups = analyze_holdbar(sorted_selected1,target_key="feature_conf_list",period ='long', metric_key="cagr")

    if symbol == 'DOGEUSDT' and interval=='15m':
        l_results,unselected = filter_by_criteria(sorted_selected1, period ='long', cagr=0.3,rc_median = 0,rc_pos_ratio = 0.8,calmar = 1.3 ,daily_freq = 0.1,sharpe = 0.6)
        # Define all metrics of interest
        metrics_to_test = [
            ("Calmar", "l_calmar"),
            ("CAGR", "l_cagr"),
            ("Sharpe", "l_sharpe"),
            # ("WinRate", "l_win_rate"),
            ("Median Return", lambda r: common.recursive_get(r.get('long', {}), 'rc_median') or 0),
            # ("l_daily_freq", "l_daily_freq"),
            # ("Pos Ratio", lambda r: common.recursive_get(r.get('long', {}), 'rc_pos_ratio') or 0),
        ]

        for label, metric in metrics_to_test:
            refined_data = filter_and_rank_strategies(unselected, metric, k=20)
            print(f">>> Processing top strategies by {label}...")
            # show_performance(refined_data, output_dir, 3)
            l_results = l_results + refined_data
        l_results = merge_selected(l_results)
        l_results = sorted(l_results, key=itemgetter("l_cagr"), reverse=True)
        l_results,unselected = filter_by_criteria(l_results, period ='long', cagr=0.3,calmar = 0.5,sharpe = 0.5,rc_pos_ratio = 0.5,daily_freq = 0.1)
        # l_results,unselected = filter_by_criteria(unselected, period ='long', rc_pos_ratio = 0.8)
        # l_results,unselected = filter_by_criteria(unselected, period ='long', rc_pos_ratio = 0.6)
    if symbol == 'ETHUSDT' and interval=='15m':
        l_results,unselected = filter_by_criteria(sorted_selected1, period ='long', cagr=0.2,rc_median = 0,rc_pos_ratio = 0.8,calmar = 1.2,daily_freq = 0.1,sharpe = 0.5)
    if symbol == 'ETHUSDT' and interval=='30m':
        l_results,unselected = filter_by_criteria(sorted_selected1, period ='long', cagr=0.2,rc_median = 0,rc_pos_ratio = 0.6,calmar = 0.9,daily_freq = 0.15,sharpe = 0.5)
    # sort_by_correlation_result = sort_by_correlation_diversity(l_results)
    # for h,r in groups.items():
    #     h_output_dir = os.path.join(output_dir, str(h))
    #     show_performance(r,h_output_dir,3)
    # exit()
    # sorted_by_pos_ratio = sorted(
    #     l_results, 
    #     key=lambda r: common.recursive_get(r.get('long', {}), 'rc_pos_ratio') or 0, 
    #     reverse=True
    # )
    l_results = l_results#[:2]
    selected = l_results
    filter_hash_doge_15 = ['1df68cde','2c833b42','4e6d96d6','b4643441','b1d3aa34','358126fe','404b9edc','8918c3a8','652c9e37','d5eb8b05','6810b357','ef82f618','4e1bcd26','d81ee9ea','1351d037','d40a5588'
                           ,'f1fa2ba1','978a1afb','1870ccac','5290ee6d','d1062846','6d2b9fc9','c6943cca','b7a08817','2babf5c5','f7c92570','044c22f4','e7653d4c','9cc7065e',
                           '759a6e64','04c36e97','261d23d3','a1d96a57','ecf1eed1','ab31665b','c32333fc','4f9f4307','adff58e6','bbbeece4','b4b1d10f','781b9c01','7313dcfb',
                           'c8d405bf','6483c804','493904fe','8c3c266b','a6270c43','a6ab4b8a','6d5327c1',]
    keep_hash = [
        'f75e3f11', '1b5d9b3c', '63ee07fc', '0180b8fc',
        '903e836f', '2ced173e', 'b2f09163', '31a2c243', 
        'b0a375e7', '819999ca', 'fbd5e0ca', '3c9f67cc', 
        'e543670e', '1bc723b8'
    ]
    filter_hash_eth_15 = ['943143f8', '21f9fce3', 'e4927150', '9a3f7676', '4afa85ac' ,'3163d070','ed96bd77','cc89356b','31854257']
    filter_set = set(filter_hash_doge_15 + filter_hash_eth_15)
    keep_set = set(keep_hash)
    selected = [
        r for r in l_results 
        if (h := str(common.recursive_get(r.get('long', {}), 'hash'))[:8]) not in filter_set 
        # and h in keep_set
    ]
    print(f"🎯 hash fitler, {len(l_results)} -> {len(selected)}")
    stats, f_map, groups = analyze_holdbar(selected,target_key="feature_conf_list",period ='long', metric_key="cagr")
    show_performance(selected,output_dir,3)
    # exit()
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
    out_path = os.path.join(output_dir,"selected_configs.jsonl")
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
    # 1️⃣ Short must be very strong (capture current regime)
    results, _ = filter_by_criteria( selected, period='short', cagr=1, calmar=0)
    print(f"After short performance filter: {len(results)} reports")
    # 2️⃣ Long only needs to be acceptable, not extremely stable
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
    Merge multiple selected lists, deduplicate by hash, then sort by sort_key.

    Parameters
    ----------
    *selected_lists : any number of selected lists
    sort_key : str
        Field name used for sorting, e.g. "l_cagr"
    reverse : bool
        True for descending order (default from large to small)

    Returns
    -------
    list
        New list after deduplication and sorting
    """

    merged_dict = {}

    for selected in selected_lists:
        for row in selected:
            h = row.get("hash")
            if h not in merged_dict:
                merged_dict[h] = row

    result = list(merged_dict.values())

    # Sorting
    if sort_key is not None:
        result.sort(
            key=lambda x: common.recursive_get(x.get(period), sort_key),
            reverse=reverse
        )

    return result

def para_evaluation(rows, label1="Vol 1.9", label2="Vol 1.7"):
    """
    Generic parameter evaluation function.
    label1/label2: description of what these two groups represent; shown as Group 1/2 in the table.
    """
    group_1_data = []
    group_2_data = []

    # 1. Flexible grouping logic
    for row in rows:
        # You can customize the conditions here; the function is otherwise generic
        # vol = row["report"]["params"]["common"]["vol_multiplier_long_long"]
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

    # 2. Internal metric extractor
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

    # 3. Build comparison table
    summary_rows = []
    for i, (m, label_desc) in enumerate([(g1_metrics, label1), (g2_metrics, label2)]):
        if m:
            summary_rows.append({
                "Group": f"Group {i+1}",
                "Desc": label_desc,  # Description so you know what Group 1 represents
                "Count": m["count"],
                "Avg CAGR": f"{np.mean(m['cagr']):.2%}",
                "Max CAGR": f"{np.max(m['cagr']):.2%}",
                "Min CAGR": f"{np.min(m['cagr']):.2%}",
                "Std CAGR": f"{np.std(m['cagr']):.4f}",
                "Avg Calmar": f"{np.mean(m['calmar']):.2f}",
                "Max Calmar": f"{np.max(m['calmar']):.2%}",
                "Avg Sharpe": f"{np.mean(m['sharpe']):.2f}",
                "Avg MaxDD": f"{np.mean(m['max_dd']):.2f}%"  # Corrected drawdown display
            })

    # 4. Nicely aligned output
    if summary_rows:
        pd.set_option('display.max_columns', None)       # Show all columns
        pd.set_option('display.expand_frame_repr', False)  # Do not wrap lines (keep on one line)
        pd.set_option('display.max_colwidth', None)      # No column width limit
        pd.set_option('display.width', 1000)             # Set sufficiently wide display width
        pd.set_option('display.expand_frame_repr', True)
        df_final = pd.DataFrame(summary_rows).set_index("Group")
    
        print("\n" + "="*120)  # Slightly longer separator line
        print(f"📊 Parameter group comparison (Group 1: {label1} | Group 2: {label2})")
        print("="*120)
        
        # Force single-line printing
        print(df_final.to_string(justify='center', index=True, line_width=1000))
        print("="*120)
        
        # 5. Short conclusion
        cagr1 = np.mean(g1_metrics['cagr'])
        cagr2 = np.mean(g2_metrics['cagr'])
        winner = "Group 1" if cagr1 > cagr2 else "Group 2"
        print(f"💡 Preview conclusion: {winner} has better expected return ({max(cagr1, cagr2):.2%})")
    else:
        print("❌ Error: failed to classify valid data; please check parameters in rows input.")
    exit()

def filter_by_criteria(reports, period='short', **criteria):
    """
    Step-wise filtering function:
    - Apply each filter condition in sequence
    - Print surviving count and retention ratio after each step
    """
    if not reports:
        return [], []
    initial_len = len(reports)
    passed = reports
    
    for key, min_value in criteria.items():
        # Skip empty criteria
        if min_value is None:
            continue
            
        # If pool is already empty, record 0 for subsequent steps
        if not passed:
            print(f"After screening {period} {key:<12} >= {min_value:>3}: 0, 0.00%")
            continue

        prev_len = len(passed)
        
        # 1. Locate the path of this metric in the dict (reuse find_key_path)
        # Note: path is found based on the first sample in the current surviving pool
        key_path = find_key_path(passed[0].get(period, {}), key)
        if key_path is None:
            print(f"⚠️ Warning: key '{key}' not found in {period} reports, skipping this filter.")
            continue

        # 2. Apply this single filtering step
        step_passed = []
        for r in passed:
            period_data = r.get(period, {})
            # Reuse existing get_value_by_path
            current_value = get_value_by_path(period_data, key_path)
            
            # Comparison logic
            if current_value is not None and current_value >= min_value:
                step_passed.append(r)
        
        # 3. Update pool and print result
        passed = step_passed
        curr_len = len(passed)
        ratio = (curr_len / prev_len * 100) if prev_len > 0 else 0
        
        # Print in the requested format
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
    # —— Survival / tail risk ——
    min_rc_es_05= None,          # e.g. > -0.8
    min_rc_q05= None,            # e.g. > -0.5

    # —— Holdability / continuity ——
    max_rc_longest_neg_run=None,  # e.g. < 300 (days/windows)
    max_rc_neg_ratio= None,       # e.g. < 0.5

    # —— Typical return level ——
    min_rc_median=None,          # e.g. > 0
    min_rc_q25=None,             # e.g. > 0

    # —— Stability / dispersion ——
    max_rc_cv=None,              # e.g. < 3
    max_rc_mad=None,             # optional
):
    """
    Filter reports based on rc_summary metrics.
    Goal: remove structurally unstable or un-holdable strategies.
    """

    def ok(report):
        rc = report.get(period).get("performance", {}).get("rc_summary", {})
        if not rc:
            return False

        # ---------- Survival (tail) ----------
        if min_rc_es_05 is not None:
            if rc.get("rc_es_05", -math.inf) < min_rc_es_05:
                return False

        if min_rc_q05 is not None:
            if rc.get("rc_q05", -math.inf) < min_rc_q05:
                return False

        # ---------- Holdability (long-term pain) ----------
        if max_rc_longest_neg_run is not None:
            if rc.get("rc_longest_neg_run", math.inf) > max_rc_longest_neg_run:
                return False

        if max_rc_neg_ratio is not None:
            if rc.get("rc_neg_ratio", 1.0) > max_rc_neg_ratio:
                return False

        # ---------- Typical return level ----------
        if min_rc_median is not None:
            if rc.get("rc_median", -math.inf) < min_rc_median:
                return False

        if min_rc_q25 is not None:
            if rc.get("rc_q25", -math.inf) < min_rc_q25:
                return False

        # ---------- Stability ----------
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
    Recursively find the path of target_key in a nested object.
    Returns a path list which can be used to directly index the value.
    
    Example: find_key_path(report, "holdbar") returns ["params", "common", "holdbar"]
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
    Get a value from an object using a path list.
    
    Example: get_value_by_path(report, ["params", "common", "holdbar"])
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
    Final enhanced version:
    1. Supports list-type target_key (auto sort, join, and hash).
    2. Returns grouped_records to keep original records grouped.
    
    Returns:
        analysis_results (list): list of aggregated statistics.
        hash_map (dict): mapping from hash to original list.
        grouped_records (dict): {hash_or_value: [original_records...]}.
    """
    from collections import defaultdict
    import numpy as np

    if not records:
        print("❌ Report list is empty")
        return [], {}, {}

    # 1. Locate path for target_key
    key_path = find_key_path(records[0][period], target_key)
    if key_path is None:
        print(f"❌ Could not find any {target_key}")
        return [], {}, {}

    print(f"✓ Located path for {target_key}: {' -> '.join(map(str, key_path))}")

    # 2. Group records according to key
    grouped_records = defaultdict(list)  # Store original record groups
    hash_map = {}  # Map from hash to list value

    for report in records:
        value = get_value_by_path(report[period], key_path)
        if value is None:
            continue

        # Handle list: sort, join, and hash
        if isinstance(value, list):
            key_str = ",".join(map(str, sorted(value)))
            current_key = hash(key_str)
            if current_key not in hash_map:
                hash_map[current_key] = value
        else:
            current_key = value

        grouped_records[current_key].append(report)

    if not grouped_records:
        print(f"❌ No valid {target_key} found")
        return [], {}, {}

    # 3. Compute performance statistics per group
    analysis_results = []
    total_count = sum(len(v) for v in grouped_records.values())

    # Sort by key for stable output
    for key in sorted(grouped_records.keys(), key=lambda x: str(x)):
        group_items = grouped_records[key]
        count = len(group_items)
        
        metric_list = []
        calmar_list = []
        
        for report in group_items:
            # Here report is already from the grouped original records
            p_report = report.get(period, report)
            perf = p_report.get("performance", {})
            metric = recursive_get(p_report, metric_key)
            calmar = perf.get("calmar")
            
            if metric is not None:
                metric_list.append(metric)
            if calmar is not None:
                calmar_list.append(calmar)
        
        # Label for display
        display_label = f"Hash:{str(key)[:8]}" if key in hash_map else key

        analysis_results.append({
            "group_key": key,              # Key for fetching from grouped_records
            "original_value": hash_map.get(key, key),  # Original list or scalar value
            "display_key": display_label,
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

    # 4. Print table
    print("\n" + "="*110)
    print(f"📊 {target_key} {period} analysis (total {total_count} reports)")
    print("="*110)
    header = f"{'Value/Hash':<15} {'Count':<8} {'%':<6} {f'{metric_key.upper()}':<12} {'':<2}{'AVG':<6}{'Max':<6}{'Std':<6}{'Med':<6} {'Calmar:':<8}{'AVG':<6}{'MAX':<6}{'Med':<6}"
    print(header)
    print("-" * 110)
    
    for r in analysis_results:
        fmt = lambda v, p: f"{v:.2%}" if v is not None and p else (f"{v:.2f}" if v is not None else "N/A")
        print(f"{str(r['display_key']):<15} {r['count']:<8} {r['percentage']:<5.1f}% {metric_key.upper():<12}  {fmt(r[f'avg_{metric_key}'],1):<6} {fmt(r[f'max_{metric_key}'],1):<6} {fmt(r[f'std_{metric_key}'],0):<6} {fmt(r[f'med_{metric_key}'],0):<6} {'':<8}{fmt(r['avg_calmar'],0):<6} {fmt(r['max_calmar'],0):<6} {fmt(r['med_calmar'],0):<6}")
    
    print("="*110)
    
    return analysis_results, hash_map, grouped_records

def analyze_feature_regimes(records, target_key="predict_num", period='short', metric_key="cagr"):
    """
    Specifically used to analyze how different feature configuration lists affect performance.
    """
    from collections import defaultdict
    import numpy as np

    # 1. Locate path
    key_path = find_key_path(records[0], target_key)
    if key_path is None:
        print(f"❌ Key not found: {target_key}")
        return

    # 2. Group by feature combinations
    groups = defaultdict(list)
    for report in records:
        value = get_value_by_path(report, key_path)
        if value is not None:
            # ✨ Core fix: convert list to string so it can be used as a dict key
            # For example, ['open', 'high'] becomes "open, high"
            key_repr = ", ".join(sorted(value)) if isinstance(value, list) else str(value)
            groups[key_repr].append(report)

    # 3. Aggregation logic (reuse existing statistics code)
    analysis_results = []
    for key_repr, reports in groups.items():
        metrics = [recursive_get(r.get(period, r), metric_key) for r in reports]
        calmars = [recursive_get(r.get(period, r), "calmar") for r in reports]
        
        analysis_results.append({
            "feature_set": key_repr,
            "count": len(reports),
            "avg_metric": np.mean(metrics) if metrics else 0,
            "avg_calmar": np.mean(calmars) if calmars else 0
        })

    # 4. Sort and print
    print(f"\n📊 Feature configuration set ({target_key}) impact analysis - Period: {period}")
    print("-" * 100)
    for res in sorted(analysis_results, key=lambda x: x['avg_metric'], reverse=True):
        print(f"Count: {res['count']:<4} | Avg {metric_key.upper()}: {res['avg_metric']:.2%} | Calmar: {res['avg_calmar']:.2f} | Features: {res['feature_set']}")

def plot_heatmap(selected, var1_key, var2_key, metric_key="l_cagr", save_path="heatmap_combined.png"):
    """
    Generate a 2x2 heatmap grid containing: mean, median, standard deviation, and maximum.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 1. Data preparation
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
    
    # 2. Compute four statistical dimensions
    # Use groupby to aggregate all metrics at once
    agg_df = df.groupby([var1_key, var2_key])["val"].agg(['mean', 'median', 'std', 'max']).reset_index()

    # 3. Create 2x2 canvas
    # Use context manager to avoid affecting global styling
    with sns.axes_style("white"):
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        stats_titles = {
            'mean': 'Mean (Expectation)',
            'median': 'Median (Robustness)',
            'std': 'Std Dev (Volatility)',
            'max': 'Max (Potential)'
        }
        
        # Loop and plot four subplots
        for i, stat in enumerate(['mean', 'median', 'std', 'max']):
            # Convert current metric to pivot table
            pivot_df = agg_df.pivot(index=var1_key, columns=var2_key, values=stat)
            
            # Std usually shown as raw value; return metrics shown as percentages
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
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for overall title
    
    plt.savefig(save_path, dpi=200)
    print(f"✅ Four-in-one heatmap saved to: {save_path}")
    plt.close()  # Release memory promptly

if __name__ == "__main__":
    main()
    # filter_by_short_longs()
