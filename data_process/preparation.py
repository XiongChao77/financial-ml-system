from pickle import FALSE
from tkinter import TRUE
import pandas as pd
import numpy as np
import datetime, os, sys, re, math, json, logging
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common

def main(logger:logging.Logger, feature_group_list = common.FEATURE_GROUP_LIST,feature_conf_list=[],para = common.BaseDefine, prep_output_dir =common.DATA_OUT_DIR ):
    file = os.path.join(common.PROJECT_DATA_DIR, para.trading_type ,f"{para.symbol}_{para.interval}.csv")
    # 1. 获取周期字符串并转为毫秒
    interval_ms = common.get_interval_ms(para.interval)
    
    # 2. 存入元数据，方便 attach_label_v2 和后续模型使用
    metadata = {
        "symbol_interval": para.interval,
        "interval_ms": interval_ms, # <--- 新增
        "candlestick_num": para.predict_num,
        "predict_num": para.predict_num,
        "vol_multiplier_long": para.vol_multiplier_long,
        "stop_multiplier_rate_long": para.stop_multiplier_rate_long,
        "vol_multiplier_short": para.vol_multiplier_short,
        "stop_multiplier_rate_short": para.stop_multiplier_rate_short,
    }

    df = pd.read_csv(file)
    #成交量等为0的数据对价格不会有任何影响，因此去掉不会影响训练和测试;
    #在真实场景下确实有成交量为0的数据.还是选择保留
    #特征处理特别要主要成交量为0的情况。
    df = common.clean_data_quality_auto(df,logger)  
    # 3. 将 interval_ms 传入 label 逻辑
    # 这样 v2 逻辑就能根据实际的时间跨度来调整波动率计算窗口了
    label_col = 'label'
    function = 0
    if function==0:
        df = common.attach_attr(df, feature_group_list , feature_conf_list, para)
        common.attach_label(df, para=para,label_col = label_col)
        # common.attach_triple_barrier_label(df, interval_ms=interval_ms)
    # # common.attach_macd_event_lifecycle_label(df, interval_ms=interval_ms)
    # # common.attach_boll_event_lifecycle_label(df, interval_ms=interval_ms)
    # # common.attach_sma_7_25_crossover_label(df, interval_ms=interval_ms)
    elif function==1 :
        # 4. 执行分析
        from data_process.regime_discovery import LabelRegimeAnalyzer
        analyzer = LabelRegimeAnalyzer(df, interval_ms, para)
        
        # 定义更精细的步长以捕捉梯度变化
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
            Dissertation Logic: 极致严格的标签交集。
            - 所有列均为 Positive -> Signal.POSITIVE (2)
            - 所有列均为 Negative -> Signal.NEGATIVE (0)
            - 所有列均为 Neutral  -> Signal.NEUTRAL (1)
            - 其余所有情况（方向不一或由趋势转震荡） -> Signal.INVALID (-1)
            """
            label_cols = [c for c in df.columns if c.startswith(label_prefix)]
            if not label_cols:
                return df

            # 检查每一行是否所有标签列的值都完全相同
            # .nunique(axis=1) == 1 表示这一行所有的 label_vXX 都指向同一个结果
            is_unanimous = df[label_cols].nunique(axis=1) == 1
            
            # 初始化为 INVALID (-1)
            label_col = 'label_x'
            df[label_col] = -1 
            
            # 只有达成“全体一致”的行，才继承它们共同的标签值 (0, 1, 或 2)
            df.loc[is_unanimous, label_col] = df.loc[is_unanimous, label_cols[0]]
            
            return df
        for v_range in np.arange(0.1, 3.1, 0.1).round(1):
            para.vol_multiplier_long = v_range
            para.stop_multiplier_rate_long = None
            para.vol_multiplier_short = v_range
            para.stop_multiplier_rate_short = None
            label_suffix = f"v{int(round(v_range * 10)):02d}"
            df = common.attach_label(df, para=para, label_col=f'label_{label_suffix}')
                # 在你的循环结束后调用
        df = generate_strict_consensus_label(df)

    # ---------------- 统计输出 ----------------
    start_time = df['open_time_date_utc'].iloc[0]
    end_time = df['open_time_date_utc'].iloc[-1]
    duration = pd.to_datetime(end_time) - pd.to_datetime(start_time)
    logger.info(f"时间跨度: {start_time} 至 {end_time} (共计 {duration})")
    counts = df[label_col].value_counts().sort_index()
    proportions = df[label_col].value_counts(normalize=True).sort_index()
    
    logger.info("\n=== 动态标签分布统计 ===")
    logger.info(f"阈值已保存至列: 'threshold'")
    logger.info(f"阈值范围: Min={df['threshold_long'].min():.4f}, Max={df['threshold_long'].max():.4f}, Mean={df['threshold_long'].mean():.4f}")
    logger.info(f"阈值范围: Min={df['threshold_short'].min():.4f}, Max={df['threshold_short'].max():.4f}, Mean={df['threshold_short'].mean():.4f}")
    
    for label_val, cnt in counts.items():
        label_name = "下跌" if label_val == 0 else ("上涨" if label_val == 2 else ("震荡" if label_val == 1 else "INVALID" ))
        pct_val = proportions[label_val]
        logger.info(f"Label {label_val} ({label_name}): {cnt} 个, 占比 {pct_val:.4%}")
    logger.info("==========================\n")
    
    # ---------------------------------------------------------
    # 3. 划分数据并保存
    # ---------------------------------------------------------
    split_ts = pd.to_datetime(df['open_time_date_utc'].iloc[-1]) - pd.DateOffset(months=6)
    train_df, test_df = df[df['open_time_date_utc'] < str(split_ts)], df[df['open_time_date_utc'] >= str(split_ts)]

    # 统一写入 para.prep_output_dir（默认 common.DATA_OUT_DIR，batch 多进程时为独立目录）
    out_dir = prep_output_dir
    os.makedirs(out_dir, exist_ok=True)
    common.save_train_df_to_dir(train_df, out_dir)
    common.save_test_df_to_dir(test_df, out_dir)
    meta_path = common.get_data_config_path_in_dir(out_dir)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info(f"✅ 数据处理完成！")
    logger.info(f"📍 周期识别: {para.interval}")
    logger.info(f"📍 配置已写入: {meta_path}")


if __name__ == "__main__":
#**********column info: open_time_date_utc,open,high,low,close,volume,close_time_ms_utc,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,ignore
    logger, _ = common.setup_session_logger(sub_folder='data_process')
    main(logger,common.FEATURE_GROUP_LIST)