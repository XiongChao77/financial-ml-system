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
        "raw" : report,
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
    
    sim_dir1 = os.path.join(common.PERSISTENCE_DIR,'batch_train/DOGEUSDT_30m/2026-06-25/04_09_15/batch_simulation')
    output_dir = os.path.join(sim_dir1, 'report_view')
    os.makedirs(output_dir, exist_ok=True)
    exp_dir_list = [sim_dir1]
    report_files = []
    rows = []
    for jsonl_path in iter_reports_jsonl(exp_dir_list):
        report_files.append(jsonl_path)
    for report_file in report_files:
        records = common.load_reports(report_file)
        for r in records:
            row = extract_row(r, report_file)
            rows.append(row)
    symbol = rows[0]['short']['params']['common']['symbol']
    interval = rows[0]['short']['params']['common']['interval']
    print(f"Total reports loaded: {len(rows)}")
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
