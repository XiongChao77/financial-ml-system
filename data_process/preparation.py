from pickle import FALSE
from tkinter import TRUE
import pandas as pd
import numpy as np
import datetime, os, sys, re, math, json, logging
from dataclasses import asdict
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common

def main(logger:logging.Logger, feature_group_list = common.FEATURE_GROUP_LIST,feature_conf_list=[],para = common.BaseDefine(), prep_output_dir =common.DATA_OUT_DIR ):
    file = os.path.join(common.PROJECT_DATA_DIR, para.trading_type ,f"{para.symbol}_{para.interval}.csv")
    logger.info(f"using file :{file}")
    # 1. Convert interval string to milliseconds
    interval_ms = common.get_interval_ms(para.interval)
    
    # 2. Persist metadata for labeling and downstream model usage
    df = pd.read_csv(file)
    # Rows with volume==0 typically have little impact on price, so dropping them often won't hurt training/testing.
    # In real-world feeds, volume==0 can exist; we keep them by default.
    # Feature engineering must explicitly handle volume==0 cases.
    df = common.clean_data_quality_auto(df,logger)  
    # 3. Pass interval_ms to label logic so it can adapt its volatility window to the real time span.
    label_col = 'label'
    function =0
    if function==0:
        df = common.attach_attr(df, feature_group_list , feature_conf_list, para)
        # common.attach_label(df, para=para,label_col = label_col)
        # common.print_zret_statistics(df)
        df = common.attach_label(df, para=para,label_col = label_col)
        # common.print_label_performance_stats(df, para)
    # # common.attach_macd_event_lifecycle_label(df, interval_ms=interval_ms)
    # # common.attach_boll_event_lifecycle_label(df, interval_ms=interval_ms)
    # # common.attach_sma_7_25_crossover_label(df, interval_ms=interval_ms)
    elif function==1 :
        # 4. Run analysis
        from data_process.regime_discovery import LabelRegimeAnalyzer
        analyzer = LabelRegimeAnalyzer(df, interval_ms, para)
        
        # Define a finer grid to capture gradients
        vol_range = np.arange(0.5, 3.1, 0.05).round(2)   #not include 3.1
        stop_range = [0.1,0.2, 0.3, 0.5,2,100]  #np.arange(100, 100.1, 0.1).round(1)   #not include 1.6
        
        analyzer.run_parameter_sweep(vol_range, stop_range, common.attach_label)
        # analyzer.analyze_and_plot()
        analyzer.plot_vol_vs_distribution()
        analyzer.plot_null_hypothesis_comparison()
        analyzer.plot_long_ratio_vs_vol_multiplier()
        exit()
    else:
        def generate_strict_consensus_label(df, label_prefix='label_v'):
            """
            Dissertation logic: extremely strict label consensus.
            - All columns Positive -> Signal.POSITIVE (2)
            - All columns Negative -> Signal.NEGATIVE (0)
            - All columns Neutral  -> Signal.NEUTRAL (1)
            - Otherwise (mixed directions or trend-to-range transitions) -> Signal.INVALID (-1)
            """
            label_cols = [c for c in df.columns if c.startswith(label_prefix)]
            if not label_cols:
                return df

            # Check whether all label columns agree on each row.
            # .nunique(axis=1) == 1 means all label_vXX columns point to the same outcome.
            is_unanimous = df[label_cols].nunique(axis=1) == 1
            df[label_col] = common.Signal.INVALID
            
            # Only rows with full consensus inherit the shared label value (0, 1, or 2).
            df.loc[is_unanimous, label_col] = df.loc[is_unanimous, label_cols[0]]
            
            return df
        for v_range in np.arange(0.1, 3.1, 0.1).round(1):
            para.vol_multiplier_long = v_range
            para.stop_multiplier_rate_long = None
            para.vol_multiplier_short = v_range
            para.stop_multiplier_rate_short = None
            label_suffix = f"v{int(round(v_range * 10)):02d}"
            df = common.attach_label(df, para=para, label_col=f'label_{label_suffix}')
            # Call after the loop finishes
        df = generate_strict_consensus_label(df)

    # ---------------- Summary statistics ----------------
    start_time = df['open_time_date_utc'].iloc[0]
    end_time = df['open_time_date_utc'].iloc[-1]
    duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
    logger.info(f"Time span: {start_time} -> {end_time} (total: {duration})")
    counts = df[label_col].value_counts().sort_index()
    proportions = df[label_col].value_counts(normalize=True).sort_index()
    
    logger.info("\n=== Dynamic label distribution summary ===")
    logger.info("Thresholds are saved in columns: 'threshold_long' and 'threshold_short'")
    logger.info(f"Long threshold range: Min={df['threshold_long'].min():.4f}, Max={df['threshold_long'].max():.4f}, Mean={df['threshold_long'].mean():.4f}")
    logger.info(f"Short threshold range: Min={df['threshold_short'].min():.4f}, Max={df['threshold_short'].max():.4f}, Mean={df['threshold_short'].mean():.4f}")
    
    for label_val, cnt in counts.items():
        label_name = "Down" if label_val == 0 else ("Up" if label_val == 2 else ("Range" if label_val == 1 else "INVALID"))
        pct_val = proportions[label_val]
        logger.info(f"Label {label_val} ({label_name}): {cnt} rows, ratio {pct_val:.4%}")
    logger.info("==========================\n")
    
    # ---------------------------------------------------------
    # 3. Split and persist datasets
    # ---------------------------------------------------------
    split_ts = pd.to_datetime(df['open_time_date_utc'].iloc[-1]) - pd.DateOffset(months=8)
    train_df, test_df = df[df['open_time_date_utc'] < str(split_ts)], df[df['open_time_date_utc'] >= str(split_ts)]

    # Write to prep_output_dir (default common.DATA_OUT_DIR; independent per batch worker)
    out_dir = prep_output_dir
    os.makedirs(out_dir, exist_ok=True)
    common.save_train_df_to_dir(train_df, out_dir)
    common.save_test_df_to_dir(test_df, out_dir)
    meta_path = common.get_data_config_path_in_dir(out_dir)
    para_dict = asdict(para)
    safe_para = common.json_safe(para_dict)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(safe_para, f, indent=4, ensure_ascii=False)

    logger.info("✅ Data preparation completed.")
    logger.info(f"📍 Interval: {para.interval}")
    logger.info(f"📍 Config written to: {meta_path}")


if __name__ == "__main__":
#**********column info: open_time_date_utc,open,high,low,close,volume,close_time_ms_utc,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,ignore
    logger, _ = common.setup_session_logger(sub_folder='data_process')
    main(logger,common.FEATURE_GROUP_LIST)