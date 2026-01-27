from enum import IntEnum
from functools import lru_cache
import logging,math,re
import pandas as pd
import numpy as np
import os, colorlog , logging, json,platform
from datetime import datetime
from data_process.feature import *

class Signal(IntEnum):
    INVALID = -1
    SHORT = 0
    NEUTRAL = 1
    LONG = 2

# 波动率系数 (0.5 ~ 1.0 之间调整)
'''
乘数 (Multiplier),阈值位置,含义
VOL_MULTIPLIER=1.0,1σ,约 31.8% 的价格变动会超出这个阈值（上下尾部）。
VOL_MULTIPLIER=0.5,0.5σ,约 61.7% 的价格变动会超出这个阈值。信号数量适中。
VOL_MULTIPLIER=1.5,1.5σ,仅约 13.4% 的价格变动会超出这个阈值。
VOL_MULTIPLIER=2.0,2σ,仅约 4.6% 的价格变动会超出这个阈值。    
''' 
# 建议在 common.py 中增加以下定义
class CommonDefine:
    #define model
    CANDLESTICK_NUM = 128   #160 best for LSTM
    PREDICT_NUM = 12
    VOL_MULTIPLIER_LONG = 1.9
    STOP_MULTIPLIER_RATE_LONG = 0.2
    VOL_MULTIPLIER_SHORT = 1.9
    STOP_MULTIPLIER_RATE_SHORT = 0.2
    # label_decrease_weak =1 
    # label_increase_weak = 3
    model_train_rate = 0.8
    symbol = 'ETHUSDT' # option: 'BTCUSDT' 'ETHUSDT' 'BNBUSDT' 'DOGEUSDT'
    interval = '30m' # option: 1s, 15s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
log_level = logging.INFO

DATA_PROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DATA_PROCESS_DIR)
# --- 核心判断逻辑 ---
TEMPORARY_DIR = os.path.join(PROJECT_DIR, 'output')
if platform.system().lower() != 'windows':
    os.makedirs('/dev/shm/quant', exist_ok=True)
    if not os.path.islink(TEMPORARY_DIR):   os.symlink('/dev/shm/quant', TEMPORARY_DIR)  # Linux/Ubuntu 环境：直接映射到共享内存
PERSISTENCE_DIR = os.path.join(os.path.dirname(PROJECT_DIR),'quant_output')
os.makedirs(PERSISTENCE_DIR, exist_ok=True)

PROJECT_DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR),'QuantData','Cryptocurrency')
origin_data_path = os.path.join(PROJECT_DATA_DIR, f"{CommonDefine.symbol}_{CommonDefine.interval}.csv")
train_data_path = os.path.join(TEMPORARY_DIR, "train_data.csv")
test_data_path  = os.path.join(TEMPORARY_DIR, "test_data.csv")
data_config_path  = os.path.join(TEMPORARY_DIR, "data_config_meta.json")
TRAIN_OUT_DIR = os.path.join(TEMPORARY_DIR, "train")
os.makedirs(TRAIN_OUT_DIR, exist_ok=True)

CONF_DF = 'to_feather'#/'to_feather'/'to_csv'

def save_train_df(df):
    if CONF_DF == 'to_csv':
        df.to_csv(train_data_path, index=False, encoding="utf-8")
    else:
        # Feather 格式需确保列名为字符串，否则会报错
        df.columns = df.columns.astype(str)
        df.to_feather(train_data_path)

def load_train_df():
    if CONF_DF == 'to_csv':
        return pd.read_csv(train_data_path, encoding="utf-8")
    else:
        return pd.read_feather(train_data_path)

def save_test_df(df):
    if CONF_DF == 'to_csv':
        df.to_csv(test_data_path, index=False, encoding="utf-8")
    else:
        df.columns = df.columns.astype(str)
        df.to_feather(test_data_path)

def load_test_df():
    if CONF_DF == 'to_csv':
        return pd.read_csv(test_data_path, encoding="utf-8")
    else:
        return pd.read_feather(test_data_path)

def attach_attr(df, feature_config_list, kline_interval_ms):
    # 1. 基础处理
    # df.drop('ignore', axis=1, inplace=True)
    # --- 2. 指标计算 (生成所有原始、未缩放的特征列) ---
    # df = add_relative_features(df)
    FeatureFactory(feature_config_list, kline_interval_ms).generate(df)

def attach_label(df, 
                interval_ms,
                candlestick_num=CommonDefine.CANDLESTICK_NUM, 
                predict_num=CommonDefine.PREDICT_NUM, 
                vol_mult_long=CommonDefine.VOL_MULTIPLIER_LONG, 
                vol_mult_short=CommonDefine.VOL_MULTIPLIER_SHORT,
                stop_rate_long=CommonDefine.STOP_MULTIPLIER_RATE_LONG,
                stop_rate_short=CommonDefine.STOP_MULTIPLIER_RATE_SHORT):
    """
    基于路径依赖的非对称打标签逻辑
    """
    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    # 1. 计算非对称动态阈值
    df = calculate_thresholds(df, candlestick_num, predict_num, 
                               vol_mult_long, vol_mult_short, 
                               stop_rate_long, stop_rate_short)

    # 2. 物理时间锚定 (保持不变)
    target_times = time_values + (predict_num * interval_ms)
    target_indices = np.searchsorted(time_values, target_times, side='left')
    in_bounds = target_indices < len(df)
    safe_idx = np.where(in_bounds, target_indices, 0)
    final_valid_mask = in_bounds & (time_values[safe_idx] == target_times)

    # 3. 计算未来收益与极致波动 (保持不变)
    future_close = np.where(final_valid_mask, df['close'].values[safe_idx], np.nan)
    pct_final = (future_close - df['close']) / df['close']

    high_mtx = np.column_stack([df['high'].shift(-i).values for i in range(1, predict_num + 1)])
    low_mtx = np.column_stack([df['low'].shift(-i).values for i in range(1, predict_num + 1)])
    
    steps = (target_indices - np.arange(len(df))).clip(1, predict_num)
    future_high_max = np.maximum.accumulate(high_mtx, axis=1)[np.arange(len(df)), steps - 1]
    future_low_min = np.minimum.accumulate(low_mtx, axis=1)[np.arange(len(df)), steps - 1]

    max_drawdown = (future_low_min - df['close']) / df['close']
    max_runup = (future_high_max - df['close']) / df['close']

    # 4.  应用非对称逻辑
    # 做多：用 Long 专属阈值
    cond_long = final_valid_mask & \
                (pct_final > df['threshold_long']) & \
                (max_drawdown > -df['stop_threshold_long'])
                
    # 做空：用 Short 专属阈值
    cond_short = final_valid_mask & \
                 (pct_final < -df['threshold_short']) & \
                 (max_runup < df['stop_threshold_short'])

    # 5. 生成结果
    conditions = [~final_valid_mask, cond_short, cond_long]
    choices = [Signal.INVALID, Signal.SHORT, Signal.LONG]
    df['label'] = np.select(conditions, choices, default=Signal.NEUTRAL).astype(int)
    
    df['return_rate'] = pct_final 
    return df

def attach_triple_barrier_label(df, 
                                 interval_ms,
                                 candlestick_num=CommonDefine.CANDLESTICK_NUM, 
                                 predict_num=CommonDefine.PREDICT_NUM, 
                                 vol_mult_long=CommonDefine.VOL_MULTIPLIER_LONG, 
                                 vol_mult_short=CommonDefine.VOL_MULTIPLIER_SHORT,
                                 stop_rate_long=CommonDefine.STOP_MULTIPLIER_RATE_LONG,
                                 stop_rate_short=CommonDefine.STOP_MULTIPLIER_RATE_SHORT):
    """
    严苛版非对称 Triple Barrier 标签
    """
    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    # 1. 计算非对称动态阈值
    df = calculate_thresholds(df, candlestick_num, predict_num, 
                               vol_mult_long, vol_mult_short, 
                               stop_rate_long, stop_rate_short)

    # 2. 物理时间锚定
    target_times = time_values + (predict_num * interval_ms)
    target_indices = np.searchsorted(time_values, target_times, side='left')
    in_bounds = target_indices < len(df)
    safe_idx = np.where(in_bounds, target_indices, 0)
    final_valid_mask = in_bounds & (time_values[safe_idx] == target_times)

    # 3. 准备数据矩阵
    future_closes = np.column_stack([df['close'].shift(-i).values for i in range(1, predict_num + 1)])
    future_highs = np.column_stack([df['high'].shift(-i).values for i in range(1, predict_num + 1)])
    future_lows = np.column_stack([df['low'].shift(-i).values for i in range(1, predict_num + 1)])
    
    closes = df['close'].values
    labels = np.full(len(df), Signal.NEUTRAL, dtype=int)

    # 4.  循环遍历 (分别使用对应的 Long/Short 阈值列)
    for i in range(len(df) - predict_num):
        if not final_valid_mask[i]:
            labels[i] = Signal.INVALID
            continue
            
        curr_price = closes[i]
        
        # --- 多头检测 (使用 _long 结尾的阈值) ---
        idx_long_tp = np.where(future_closes[i] >= curr_price * (1 + df['threshold_long'].iloc[i]))[0]
        idx_long_sl = np.where(future_lows[i] <= curr_price * (1 - df['stop_threshold_long'].iloc[i]))[0]
        first_l_tp = idx_long_tp[0] if len(idx_long_tp) > 0 else predict_num
        first_l_sl = idx_long_sl[0] if len(idx_long_sl) > 0 else predict_num

        # --- 空头检测 (使用 _short 结尾的阈值) ---
        idx_short_tp = np.where(future_closes[i] <= curr_price * (1 - df['threshold_short'].iloc[i]))[0]
        idx_short_sl = np.where(future_highs[i] >= curr_price * (1 + df['stop_threshold_short'].iloc[i]))[0]
        first_s_tp = idx_short_tp[0] if len(idx_short_tp) > 0 else predict_num
        first_s_sl = idx_short_sl[0] if len(idx_short_sl) > 0 else predict_num

        # 判定
        if first_l_tp < first_l_sl:
            labels[i] = Signal.LONG
        elif first_s_tp < first_s_sl:
            labels[i] = Signal.SHORT

    df['label'] = labels
    df.loc[~final_valid_mask, 'label'] = Signal.INVALID
    return df

def calculate_thresholds(df, 
                         candlestick_num: int = CommonDefine.CANDLESTICK_NUM, 
                         predict_num: int = CommonDefine.PREDICT_NUM, 
                         vol_mult_long = CommonDefine.VOL_MULTIPLIER_LONG,    #  拆分
                         vol_mult_short = CommonDefine.VOL_MULTIPLIER_SHORT,   #  拆分
                         stop_rate_long = CommonDefine.STOP_MULTIPLIER_RATE_LONG,  #  拆分
                         stop_rate_short = CommonDefine.STOP_MULTIPLIER_RATE_SHORT, #  拆分
                         **kwargs): 
    """
    计算非对称动态止盈和止损阈值
    """
    assert 'close' in df.columns, "缺少 Close 数据"
    
    # 1. 计算波动率基准 (Rolling Standard Deviation)
    vol_window = candlestick_num
    returns = df['close'].pct_change()
    rolling_std = returns.rolling(window=vol_window).std()
    
    # 2. 时间扩充波动率 (sigma * sqrt(T))
    expected_vol = rolling_std * np.sqrt(predict_num)
    
    # 3.  生成非对称阈值
    # 多头 (Long) 阈值
    df['threshold_long'] = (expected_vol * vol_mult_long)
    df['stop_threshold_long'] = df['threshold_long'] * stop_rate_long
    
    # 空头 (Short) 阈值
    df['threshold_short'] = (expected_vol * vol_mult_short)
    df['stop_threshold_short'] = df['threshold_short'] * stop_rate_short
    
    return df

def attach_macd_event_lifecycle_label(df, 
                                interval_ms,
                                candlestick_num=CommonDefine.CANDLESTICK_NUM, 
                                predict_num=CommonDefine.PREDICT_NUM, 
                                vol_mult_long=CommonDefine.VOL_MULTIPLIER_LONG, 
                                vol_mult_short=CommonDefine.VOL_MULTIPLIER_SHORT,
                                stop_rate_long=CommonDefine.STOP_MULTIPLIER_RATE_LONG,
                                stop_rate_short=CommonDefine.STOP_MULTIPLIER_RATE_SHORT):
    """
    严格时间对齐版 MACD 生命周期标签 (自动匹配特征列名):
    移除 min_threshold 逻辑。
    """
    # --- 1. 自动匹配 MACD 特征列名 ---
    dif_cols = [c for c in df.columns if c.startswith('MACD_') and c.endswith('_DIF')]
    dea_cols = [c for c in df.columns if c.startswith('MACD_') and c.endswith('_DEA')]
    
    if not dif_cols or not dea_cols:
        raise ValueError("❌ 未在 DataFrame 中探测到 MACD 特征列 (需以 _DIF 和 _DEA 结尾)")
    
    dif_name = dif_cols[0]
    prefix = dif_name.replace('_DIF', '')
    dea_name = f"{prefix}_DEA"
    
    if dea_name not in df.columns:
        raise ValueError(f"❌ 找不到与 {dif_name} 匹配 of {dea_name}")

    print(f"🔍 [MACD Match] 自动匹配到特征列: {dif_name} / {dea_name}")

    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    # 2. 识别所有交叉点
    dif = df[dif_name]
    dea = df[dea_name]
    cross_mask = (dif > dea) != (dif.shift(1) > dea.shift(1))
    cross_mask.iloc[0] = False
    event_indices = df.index[cross_mask].tolist()
    
    # 3. 初始化与动态阈值
    df['label'] = Signal.INVALID
    df = calculate_thresholds(df, candlestick_num, predict_num, 
                               vol_mult_long, vol_mult_short, 
                               stop_rate_long, stop_rate_short)
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    thresholds = df['threshold'].values
    sl_thresholds = df['stop_threshold'].values

    # 4. 遍历交叉事件
    for i in range(len(event_indices)):
        curr_idx = event_indices[i]
        if i + 1 >= len(event_indices):
            df.at[curr_idx, 'label'] = Signal.INVALID
            continue
            
        next_idx = event_indices[i+1]
        
        # --- 时间对齐校验 ---
        expected_gap_ms = (next_idx - curr_idx) * interval_ms
        actual_gap_ms = time_values[next_idx] - time_values[curr_idx]
        
        if actual_gap_ms != expected_gap_ms:
            df.at[curr_idx, 'label'] = Signal.INVALID
            continue

        # --- 业务逻辑判定 ---
        is_long_event = dif.iloc[curr_idx] > dea.iloc[curr_idx]
        entry_price = closes[curr_idx]
        tp_target = thresholds[curr_idx]
        sl_target = sl_thresholds[curr_idx]
        
        pnl_at_exit = (closes[next_idx] - entry_price) / entry_price if is_long_event else \
                      (entry_price - closes[next_idx]) / entry_price
        
        window_highs = highs[curr_idx + 1 : next_idx + 1]
        window_lows = lows[curr_idx + 1 : next_idx + 1]
        
        if is_long_event:
            hit_stop = np.any(window_lows <= entry_price * (1 - sl_target))
        else:
            hit_stop = np.any(window_highs >= entry_price * (1 + sl_target))

        if pnl_at_exit >= tp_target and not hit_stop:
            df.at[curr_idx, 'label'] = Signal.LONG if is_long_event else Signal.SHORT
        else:
            df.at[curr_idx, 'label'] = Signal.NEUTRAL

    return df

def attach_boll_event_lifecycle_label(df, 
                                interval_ms,
                                candlestick_num=CommonDefine.CANDLESTICK_NUM, 
                                predict_num=CommonDefine.PREDICT_NUM, 
                                vol_mult_long=CommonDefine.VOL_MULTIPLIER_LONG, 
                                vol_mult_short=CommonDefine.VOL_MULTIPLIER_SHORT,
                                stop_rate_long=CommonDefine.STOP_MULTIPLIER_RATE_LONG,
                                stop_rate_short=CommonDefine.STOP_MULTIPLIER_RATE_SHORT):
    """
    均值回归版 布林带生命周期标签：
    移除 min_threshold 逻辑。
    """
    upper_cols = [c for c in df.columns if c.startswith('BOLL_UPPER_')]
    lower_cols = [c for c in df.columns if c.startswith('BOLL_LOWER_')]
    middle_cols = [c for c in df.columns if c.startswith('BOLL_MIDDLE_')]
    
    if not (upper_cols and lower_cols and middle_cols):
        raise ValueError("❌ 未在 DataFrame 中探测到完整的 BOLL 特征列")
    
    u_name, l_name, m_name = upper_cols[0], lower_cols[0], middle_cols[0]
    print(f"🔍 [BOLL Match] 自动匹配到特征列: {u_name}, {l_name}, {m_name}")

    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    long_trigger = df['close'] < df[l_name]
    short_trigger = df['close'] > df[u_name]
    event_mask = long_trigger | short_trigger
    event_indices = df.index[event_mask].tolist()
    
    # 3. 初始化与动态阈值
    df['label'] = Signal.INVALID
    df = calculate_thresholds(df, candlestick_num, predict_num, 
                               vol_mult_long, vol_mult_short, 
                               stop_rate_long, stop_rate_short)
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    middles = df[m_name].values
    thresholds = df['threshold'].values
    sl_thresholds = df['stop_threshold'].values

    # 4. 遍历事件
    for curr_idx in event_indices:
        is_long_event = long_trigger.iloc[curr_idx]
        entry_price = closes[curr_idx]
        tp_target = thresholds[curr_idx]
        sl_limit = sl_thresholds[curr_idx]
        
        if is_long_event:
            exit_candidates = np.where(closes[curr_idx + 1:] >= middles[curr_idx + 1:])[0]
        else:
            exit_candidates = np.where(closes[curr_idx + 1:] <= middles[curr_idx + 1:])[0]
            
        if len(exit_candidates) == 0:
            df.at[curr_idx, 'label'] = Signal.INVALID
            continue
            
        next_idx = curr_idx + 1 + exit_candidates[0]
        
        if (time_values[next_idx] - time_values[curr_idx]) != (next_idx - curr_idx) * interval_ms:
            df.at[curr_idx, 'label'] = Signal.INVALID
            continue

        pnl_at_exit = (closes[next_idx] - entry_price) / entry_price if is_long_event else \
                      (entry_price - closes[next_idx]) / entry_price
        
        window_highs = highs[curr_idx + 1 : next_idx + 1]
        window_lows = lows[curr_idx + 1 : next_idx + 1]
        
        if is_long_event:
            hit_stop = np.any(window_lows <= entry_price * (1 - sl_limit))
        else:
            hit_stop = np.any(window_highs >= entry_price * (1 + sl_limit))

        if pnl_at_exit >= tp_target and not hit_stop:
            df.at[curr_idx, 'label'] = Signal.LONG if is_long_event else Signal.SHORT
        else:
            df.at[curr_idx, 'label'] = Signal.NEUTRAL

    _boll_audit(df, event_indices)
    return df

def _boll_audit(df, event_indices):
    total = len(event_indices)
    stats = df.loc[event_indices, 'label'].value_counts()
    print(f"\n📊 [BOLL Lifecycle Audit]")
    print(f"  - 触发点总数: {total}")
    print(f"  - LONG (2) 有效: {stats.get(Signal.LONG, 0)} ({(stats.get(Signal.LONG, 0)/total)*100:.2f}%)")
    print(f"  - SHORT (0) 有效: {stats.get(Signal.SHORT, 0)} ({(stats.get(Signal.SHORT, 0)/total)*100:.2f}%)")
    print(f"  - NEUTRAL (1) 噪音: {stats.get(Signal.NEUTRAL, 0)}")

def attach_sma_7_25_crossover_label(df, 
                                interval_ms,
                                candlestick_num=CommonDefine.CANDLESTICK_NUM, 
                                predict_num=CommonDefine.PREDICT_NUM, 
                                vol_mult_long=CommonDefine.VOL_MULTIPLIER_LONG, 
                                vol_mult_short=CommonDefine.VOL_MULTIPLIER_SHORT,
                                stop_rate_long=CommonDefine.STOP_MULTIPLIER_RATE_LONG,
                                stop_rate_short=CommonDefine.STOP_MULTIPLIER_RATE_SHORT):
    """
    指定 SMA 7/25 交叉生命周期标签：
    移除 min_threshold 逻辑。
    """
    fast_ma_name = "SMA_7B"
    slow_ma_name = "SMA_25B"
    
    if fast_ma_name not in df.columns or slow_ma_name not in df.columns:
        raise ValueError(f"❌ 找不到均线列 {fast_ma_name} 或 {slow_ma_name}，请检查 FeatureMA 配置")

    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    fast_ma = df[fast_ma_name]
    slow_ma = df[slow_ma_name]
    cross_mask = (fast_ma > slow_ma) != (fast_ma.shift(1) > slow_ma.shift(1))
    cross_mask.iloc[0] = False
    event_indices = df.index[cross_mask].tolist()
    
    df['label'] = Signal.INVALID 
    df = calculate_thresholds(df, candlestick_num, predict_num, 
                               vol_mult_long, vol_mult_short, 
                               stop_rate_long, stop_rate_short)
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    thresholds = df['threshold'].values
    sl_thresholds = df['stop_threshold'].values

    for i in range(len(event_indices)):
        curr_idx = event_indices[i]
        if i + 1 >= len(event_indices):
            continue
            
        next_idx = event_indices[i+1]
        
        expected_gap = (next_idx - curr_idx) * interval_ms
        actual_gap = time_values[next_idx] - time_values[curr_idx]
        
        if actual_gap != expected_gap:
            continue

        is_long_event = fast_ma.iloc[curr_idx] > slow_ma.iloc[curr_idx]
        entry_price = closes[curr_idx]
        
        pnl_rate = (closes[next_idx] - entry_price) / entry_price if is_long_event else \
                   (entry_price - closes[next_idx]) / entry_price
        
        window_highs = highs[curr_idx + 1 : next_idx + 1]
        window_lows = lows[curr_idx + 1 : next_idx + 1]
        
        sl_limit = sl_thresholds[curr_idx]
        if is_long_event:
            hit_stop = np.any(window_lows <= entry_price * (1 - sl_limit))
        else:
            hit_stop = np.any(window_highs >= entry_price * (1 + sl_limit))

        if pnl_rate >= thresholds[curr_idx] and not hit_stop:
            df.at[curr_idx, 'label'] = Signal.LONG if is_long_event else Signal.SHORT
        else:
            df.at[curr_idx, 'label'] = Signal.NEUTRAL

    return df


def clean_data_quality_auto(df: pd.DataFrame, logger) -> pd.DataFrame:
    logger.info("启动自动化数据质量扫描...")
    initial_count = len(df)
    na_rows = df.isna().any(axis=1).sum()
    if na_rows > 0:
        logger.warning(f"检测到 {na_rows} 行数据包含空值 (NaN)，准备丢弃。")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    zero_mask = (df[numeric_cols] == 0).any(axis=1)
    zero_rows = zero_mask.sum()
    
    if zero_rows > 0:
        zero_stats = (df[numeric_cols] == 0).sum()
        logger.warning(f"检测到 {zero_rows} 行数据包含零值，分布如下:\n{zero_stats[zero_stats > 0]}")

    condition = df.isna().any(axis=1) | zero_mask
    df_cleaned = df[~condition].copy()
    df_cleaned.reset_index(drop=True, inplace=True)

    final_count = len(df_cleaned)
    dropped_count = initial_count - final_count

    if dropped_count > 0:
        logger.info(f"✅ 清洗完毕: 原始 {initial_count} 行 -> 剩余 {final_count} 行 (丢弃了 {dropped_count} 行)")
    else:
        logger.info("✅ 扫描完毕: 未发现空值或零值，数据质量完美。")

    return df_cleaned

def float_range(start, end, step):
    values = []
    v = start
    eps = step / 10
    while v <= end + eps:
        values.append(round(v, 10))
        v += step
    return values

@lru_cache(maxsize=1)
def load_interval_ms(config_path = data_config_path):
    if not os.path.exists(config_path):
        raise RuntimeError(f"❌ 找不到配置文件: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        interval_ms = meta.get("interval_ms")
        if interval_ms is None:
            raise RuntimeError(f"⚠️ 配置文件中缺失 'interval_ms' 字段！")
        return interval_ms
    except Exception as e:
        raise RuntimeError(f"💥 读取 JSON 时发生意外错误: {e}")

def setup_session_logger(sub_folder: str, symbol: str = CommonDefine.symbol, console_level: int = logging.INFO, file_level: int = logging.INFO):
    log_dir = os.path.join(PERSISTENCE_DIR, sub_folder)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sym_str = f"_{symbol}" if symbol else ""
    log_filename = f"session{sym_str}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) 
    if root_logger.handlers:
        root_logger.handlers = []
    log_format_console = "%(log_color)s%(asctime)s-%(name)s-%(levelname)s- %(message)s"
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    color_formatter = colorlog.ColoredFormatter(
        log_format_console,
        datefmt="%H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'RECORD':   'blue',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red,bg_yellow',
        }
    )
    ch.setFormatter(color_formatter)
    root_logger.addHandler(ch)
    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setLevel(file_level) 
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(file_formatter)
    root_logger.addHandler(fh)
    root_logger.info(f"Session Logger Initialized. Log file: {log_file_path}")
    return root_logger, log_file_path

def get_interval_from_filename(path: str) -> str:
    """
    从路径中提取时间周期 (如 ETHUSDT_3m.csv -> 3m)
    """
    filename = os.path.basename(path)
    # 匹配 1s, 15s, 1m, 3m... 1M 等格式
    match = re.search(r'_(\d+[smhdwM])\.csv', filename)
    if match:
        return match.group(1)
    return "unknown"

def get_interval_ms(interval_str: str) -> int:
    """
    将周期字符串转换为毫秒数
    支持: 1s, 15s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    # 定义基础单位（毫秒）
    units = {
        's': 1000,
        'm': 60 * 1000,
        'h': 60 * 60 * 1000,
        'd': 24 * 60 * 60 * 1000,
        'w': 7 * 24 * 60 * 60 * 1000,
        'M': 30 * 24 * 60 * 60 * 1000  # 按照标准 30 天计算
    }
    
    # 使用正则表达式拆分数字和单位
    match = re.match(r'(\d+)([smhdwM])', interval_str)
    if not match:
        return 0
    
    value, unit = match.groups()
    return int(value) * units[unit]

