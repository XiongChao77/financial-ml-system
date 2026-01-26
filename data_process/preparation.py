import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime,os,sys, re, math, json, logging
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common
from data_process.regime_discovery import LabelRegimeAnalyzer

def main(logger:logging.Logger, feature_config_list = common.FEATURE_CONFIG_LIST):
    file = common.origin_data_path
    # 1. 获取周期字符串并转为毫秒
    interval_str = common.get_interval_from_filename(file)
    interval_ms = common.get_interval_ms(interval_str)
    
    # 2. 存入元数据，方便 attach_label_v2 和后续模型使用
    metadata = {
        "symbol_interval": interval_str,
        "interval_ms": interval_ms, # <--- 新增
        "candlestick_num": common.CANDLESTICK_NUM,
        "predict_num": common.PREDICT_NUM,
        "vol_multiplier_long": common.VOL_MULTIPLIER_LONG,
        "stop_multiplier_rate_long": common.STOP_MULTIPLIER_RATE_LONG,
        "vol_multiplier_short": common.VOL_MULTIPLIER_SHORT,
        "stop_multiplier_rate_short": common.STOP_MULTIPLIER_RATE_SHORT,
    }

    df = pd.read_csv(file)
    #成交量等为0的数据对价格不会有任何影响，因此去掉不会影响训练和测试;
    #在真实场景下确实有成交量为0的数据.还是选择保留
    #特征处理特别要主要成交量为0的情况。
    df = common.clean_data_quality_auto(df,logger)  
    # 3. 将 interval_ms 传入 label 逻辑
    # 这样 v2 逻辑就能根据实际的时间跨度来调整波动率计算窗口了
    common.attach_attr(df, feature_config_list , interval_ms)
    if True:
        common.attach_label(df, interval_ms=interval_ms)
        # common.attach_triple_barrier_label(df, interval_ms=interval_ms)
    # # common.attach_macd_event_lifecycle_label(df, interval_ms=interval_ms)
    # # common.attach_boll_event_lifecycle_label(df, interval_ms=interval_ms)
    # # common.attach_sma_7_25_crossover_label(df, interval_ms=interval_ms)
    else:
        # 4. 执行分析
        analyzer = LabelRegimeAnalyzer(df, interval_ms, common.symbol,common.interval)
        
        # 定义更精细的步长以捕捉梯度变化
        vol_range = np.linspace(1.5, 3, 15)
        # vol_range = np.linspace(1.25, 1.45, 5) 
        stop_range = np.linspace(0.1, 0.4, 15)
        
        analyzer.run_parameter_sweep(vol_range, stop_range)
        analyzer.analyze_and_plot(output_dir= common.PERSISTENCE_DIR)
        analyzer.plot_null_hypothesis_comparison(output_dir= common.PERSISTENCE_DIR)
        exit()
    # ---------------- 统计输出 ----------------
    counts = df['label'].value_counts().sort_index()
    proportions = df['label'].value_counts(normalize=True).sort_index()
    
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
    train_ratio = 0.8
    split_idx = math.floor(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    # 创建临时目录
    os.makedirs(common.TEMPORARY_DIR, exist_ok=True)
    os.makedirs(common.TEMPORARY_DIR, exist_ok=True)

    # A. 保存 CSV 数据
    common.save_train_df(train_df)
    common.save_test_df(test_df)

    # B. 保存元数据 JSON (关键步骤)
    meta_path = common.data_config_path
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logger.info(f"✅ 数据处理完成！")
    logger.info(f"📍 周期识别: {interval_str}")
    logger.info(f"📍 配置已写入: {meta_path}")


if __name__ == "__main__":
#**********column info: open_time_date_utc,open,high,low,close,volume,close_time_ms_utc,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,ignore
    logger, _ = common.setup_session_logger(sub_folder='data_process')
    main(logger,common.FEATURE_CONFIG_LIST)