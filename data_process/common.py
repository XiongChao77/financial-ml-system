from enum import IntEnum,Enum
from functools import lru_cache
from dataclasses import dataclass
import logging,math,re,git
import pandas as pd
import numpy as np
import os, colorlog , logging, json,platform
from dataclasses import asdict, is_dataclass,fields
from typing import Optional
from datetime import datetime
from data_process.utils import *
from data_process.feature import *

class Signal(IntEnum):
    INVALID = -1
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE  = 2

eps = 1e-8
# 波动率系数 (0.5 ~ 1.0 之间调整)
'''
乘数 (Multiplier),阈值位置,含义
VOL_MULTIPLIER=1.0,1σ,约 31.8% 的价格变动会超出这个阈值（上下尾部）。
VOL_MULTIPLIER=0.5,0.5σ,约 61.7% 的价格变动会超出这个阈值。信号数量适中。
VOL_MULTIPLIER=1.5,1.5σ,仅约 13.4% 的价格变动会超出这个阈值。
VOL_MULTIPLIER=2.0,2σ,仅约 4.6% 的价格变动会超出这个阈值。
'''
DATA_PROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DATA_PROCESS_DIR)
TEMPORARY_DIR = os.path.join(PROJECT_DIR, 'output')
if platform.system().lower() != 'windows':
    os.makedirs('/dev/shm/quant', exist_ok=True)
    if not os.path.islink(TEMPORARY_DIR):   os.symlink('/dev/shm/quant', TEMPORARY_DIR)  # Linux/Ubuntu 环境：直接映射到共享内存
else:
    os.makedirs(TEMPORARY_DIR, exist_ok=True)
PERSISTENCE_DIR = os.path.join(os.path.dirname(PROJECT_DIR),'quant_output')
os.makedirs(PERSISTENCE_DIR, exist_ok=True)
DATA_OUT_DIR = os.path.join(TEMPORARY_DIR, "data")
os.makedirs(DATA_OUT_DIR, exist_ok=True)

@dataclass
class BaseDefine:
    # model / data
    vol_ewma_span: int  = 80
    candlestick_num: int = 32     # 160 best for LSTM
    predict_num: int = 8
    # risk / vol
    vol_multiplier_long: float = 1.7
    stop_multiplier_rate_long: Optional[float] = 0.2
    vol_multiplier_short: float = 1.7
    stop_multiplier_rate_short: Optional[float] = 0.2
    # market
    symbol: str = "DOGEUSDT"    #BTCUSDT ETHUSDT DOGEUSDT
    interval: str = "15m"
    trading_type:str ='um'             #spot  / um(USDT-M Futures) / cm    (Coin-M Futures)   
    version:int = 0

log_level = logging.INFO

PROJECT_DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR),'QuantData','Cryptocurrency','binance_public_data')
origin_data_path = os.path.join(PROJECT_DATA_DIR, f"{BaseDefine.symbol}_{BaseDefine.interval}.csv")
train_data_path = os.path.join(DATA_OUT_DIR, "train_data.csv")
test_data_path  = os.path.join(DATA_OUT_DIR, "test_data.csv")
data_config_path  = os.path.join(DATA_OUT_DIR, "data_config_meta.json")
TRAIN_OUT_DIR = os.path.join(TEMPORARY_DIR, "train")
os.makedirs(TRAIN_OUT_DIR, exist_ok=True)
EXPERIMENT_DIR = os.path.join(PROJECT_DATA_DIR, "experiment")
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

CONF_DF = 'to_feather'#/'to_feather'/'to_csv'

def save_train_df(df):
    if os.path.exists(train_data_path):
        os.remove(train_data_path)
    if CONF_DF == 'to_csv':
        df.to_csv(train_data_path, index=False, encoding="utf-8")
    else:
        df.columns = df.columns.astype(str)
        df.to_feather(train_data_path)


def load_train_df():
    if CONF_DF == 'to_csv':
        return pd.read_csv(train_data_path, encoding="utf-8")
    else:
        return pd.read_feather(train_data_path)

def save_test_df(df):
    if os.path.exists(test_data_path):
        os.remove(test_data_path)
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

# ---------- 按目录读写（用于 batch 多进程：每个 preparation 独立目录） ----------
def _data_path_in_dir(base_dir, name):
    return os.path.join(base_dir, name)

def save_train_df_to_dir(df, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    path = _data_path_in_dir(base_dir, "train_data.csv" if CONF_DF == 'to_csv' else "train_data.feather")
    if os.path.exists(path):
        os.remove(path)
    if CONF_DF == 'to_csv':
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.columns = df.columns.astype(str)
        df.to_feather(path)

def save_test_df_to_dir(df, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    path = _data_path_in_dir(base_dir, "test_data.csv" if CONF_DF == 'to_csv' else "test_data.feather")
    if os.path.exists(path):
        os.remove(path)
    if CONF_DF == 'to_csv':
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        df.columns = df.columns.astype(str)
        df.to_feather(path)

def load_train_df_from_dir(base_dir):
    path = _data_path_in_dir(base_dir, "train_data.csv" if CONF_DF == 'to_csv' else "train_data.feather")
    if CONF_DF == 'to_csv':
        return pd.read_csv(path, encoding="utf-8")
    return pd.read_feather(path)

def load_test_df_from_dir(base_dir):
    path = _data_path_in_dir(base_dir, "test_data.csv" if CONF_DF == 'to_csv' else "test_data.feather")
    if CONF_DF == 'to_csv':
        return pd.read_csv(path, encoding="utf-8")
    return pd.read_feather(path)

def get_data_config_path_in_dir(base_dir):
    return _data_path_in_dir(base_dir, "data_config_meta.json")

def load_interval_ms_from_dir(base_dir) -> BaseDefine:
    """从指定目录的 data_config_meta.json 读取 interval_ms，不依赖全局路径，适用于多进程。"""
    config_path = get_data_config_path_in_dir(base_dir)
    if not os.path.exists(config_path):
        raise RuntimeError(f"❌ 找不到配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        para = BaseDefine(**meta)
    return para

def attach_attr(df, feature_group_list, feature_conf_list = [], para = BaseDefine):
    # 1. 基础处理
    # df.drop('ignore', axis=1, inplace=True)
    # --- 2. 指标计算 (生成所有原始、未缩放的特征列) ---
    # df = add_relative_features(df)
    kline_interval_ms = get_interval_ms(para.interval)
    return FeatureFactory(kline_interval_ms,feature_group_list, feature_conf_list).generate(df)

def attach_label(df, para = BaseDefine, label_col = 'label'):
    """
    基于路径依赖的非对称打标签逻辑
    """
    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    # 1. 计算非对称动态阈值
    df = calculate_thresholds(df, para)

    # 2. 物理时间锚定 (保持不变)
    interval_ms = get_interval_ms(para.interval)
    target_times = time_values + (para.predict_num * interval_ms)
    target_indices = np.searchsorted(time_values, target_times, side='left')
    in_bounds = target_indices < len(df)
    safe_idx = np.where(in_bounds, target_indices, 0)
    final_valid_mask = in_bounds & (time_values[safe_idx] == target_times)

    # 3. 计算未来收益与极致波动 (保持不变)
    future_close = np.where(final_valid_mask, df['close'].values[safe_idx], np.nan)
    pct_final = np.log(future_close / df['close'])

    high_mtx = np.column_stack([df['high'].shift(-i).values for i in range(1, para.predict_num + 1)])
    low_mtx = np.column_stack([df['low'].shift(-i).values for i in range(1, para.predict_num + 1)])
    
    steps = (target_indices - np.arange(len(df))).clip(1, para.predict_num)
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
    choices = [Signal.INVALID, Signal.NEGATIVE, Signal.POSITIVE ]
    df[label_col] = np.select(conditions, choices, default=Signal.NEUTRAL).astype(int)
    
    # volatility normalized return
    df['trend_strength'] = np.where(
        pct_final >= 0,
        pct_final / (df['threshold_long'] + eps),
        np.abs(pct_final) / (df['threshold_short'] + eps)
    )

    # 处理无效数据（物理时间越界的部分）
    df.loc[~final_valid_mask, 'trend_strength'] = np.nan
    
    return df

def calculate_thresholds(df, para=BaseDefine, **kwargs):
    """
    使用 Rogers–Satchell + EWMA 计算动态波动率阈值
    """

    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        assert col in df.columns, f"缺少 {col} 数据"

    # ===== 1️⃣ Rogers–Satchell 单期方差 =====
    log_ho = np.log(df['high'] / df['open'])
    log_hc = np.log(df['high'] / df['close'])
    log_lo = np.log(df['low'] / df['open'])
    log_lc = np.log(df['low'] / df['close'])

    rs_var = log_hc * log_ho + log_lc * log_lo

    # 防止极端异常（理论上应为非负，数值误差可能导致微小负数）
    rs_var = rs_var.clip(lower=0)

    # ===== 2️⃣ EWMA 平滑方差 =====
    span = para.vol_ewma_span
    ewma_var = rs_var.ewm(span=span, adjust=False).mean()
    # 开方得到波动率
    ewma_vol = np.sqrt(ewma_var)
    # ===== 3️⃣ 时间扩展到预测区间 =====
    # 假设方差线性扩展
    expected_vol = ewma_vol * np.sqrt(para.predict_num)
    df['expected_vol'] = expected_vol

    # ===== 4️⃣ 非对称阈值 =====
    df['threshold_long'] = expected_vol * para.vol_multiplier_long
    df['threshold_short'] = expected_vol * para.vol_multiplier_short

    if para.stop_multiplier_rate_long is not None:
        df['stop_threshold_long'] = df['threshold_long'] * para.stop_multiplier_rate_long
    else:
        df['stop_threshold_long'] = np.inf

    if para.stop_multiplier_rate_short is not None:
        df['stop_threshold_short'] = df['threshold_short'] * para.stop_multiplier_rate_short
    else:
        df['stop_threshold_short'] = np.inf

    return df

def print_zret_statistics(df, label_col='label'):
    print("\n================ trend_strength Statistics ================\n")

    valid = df['trend_strength'].notna()

    overall = df.loc[valid, 'trend_strength']

    print("Overall trend_strength distribution:")
    print(overall.describe(percentiles=[0.5,0.75,0.9,0.95,0.99]))

    print("\nBy label:")

    for label in sorted(df[label_col].unique()):
        sub = df.loc[(df[label_col] == label) & valid, 'trend_strength']

        if len(sub) == 0:
            continue

        print(f"\nLabel {label}  count={len(sub)}")
        print(sub.describe(percentiles=[0.5,0.75,0.9,0.95,0.99]))

def attach_triple_barrier_label(df, 
                                 interval_ms,
                                 para = BaseDefine,):
    """
    严苛版非对称 Triple Barrier 标签
    """
    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    # 1. 计算非对称动态阈值
    df = calculate_thresholds(df, para)

    # 2. 物理时间锚定
    target_times = time_values + (para.predict_num * interval_ms)
    target_indices = np.searchsorted(time_values, target_times, side='left')
    in_bounds = target_indices < len(df)
    safe_idx = np.where(in_bounds, target_indices, 0)
    final_valid_mask = in_bounds & (time_values[safe_idx] == target_times)

    # 3. 准备数据矩阵
    future_closes = np.column_stack([df['close'].shift(-i).values for i in range(1, para.predict_num + 1)])
    future_highs = np.column_stack([df['high'].shift(-i).values for i in range(1, para.predict_num + 1)])
    future_lows = np.column_stack([df['low'].shift(-i).values for i in range(1, para.predict_num + 1)])
    
    closes = df['close'].values
    labels = np.full(len(df), Signal.NEUTRAL, dtype=int)

    # 4.  循环遍历 (分别使用对应的 Long/Short 阈值列)
    for i in range(len(df) - para.predict_num):
        if not final_valid_mask[i]:
            labels[i] = Signal.INVALID
            continue
            
        curr_price = closes[i]
        
        # --- 多头检测 (使用 _long 结尾的阈值) ---
        idx_long_tp = np.where(future_closes[i] >= curr_price * (1 + df['threshold_long'].iloc[i]))[0]
        idx_long_sl = np.where(future_lows[i] <= curr_price * (1 - df['stop_threshold_long'].iloc[i]))[0]
        first_l_tp = idx_long_tp[0] if len(idx_long_tp) > 0 else para.predict_num
        first_l_sl = idx_long_sl[0] if len(idx_long_sl) > 0 else para.predict_num

        # --- 空头检测 (使用 _short 结尾的阈值) ---
        idx_short_tp = np.where(future_closes[i] <= curr_price * (1 - df['threshold_short'].iloc[i]))[0]
        idx_short_sl = np.where(future_highs[i] >= curr_price * (1 + df['stop_threshold_short'].iloc[i]))[0]
        first_s_tp = idx_short_tp[0] if len(idx_short_tp) > 0 else para.predict_num
        first_s_sl = idx_short_sl[0] if len(idx_short_sl) > 0 else para.predict_num

        # 判定
        if first_l_tp < first_l_sl:
            labels[i] = Signal.POSITIVE 
        elif first_s_tp < first_s_sl:
            labels[i] = Signal.NEGATIVE

    df['label'] = labels
    df.loc[~final_valid_mask, 'label'] = Signal.INVALID
    return df

def attach_macd_event_lifecycle_label(df, 
                                interval_ms,
                                para = BaseDefine,):
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
    df = calculate_thresholds(df, para)
    
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
            df.at[curr_idx, 'label'] = Signal.POSITIVE  if is_long_event else Signal.NEGATIVE
        else:
            df.at[curr_idx, 'label'] = Signal.NEUTRAL

    return df

def attach_boll_event_lifecycle_label(df, 
                                interval_ms,
                                para = BaseDefine,):
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
    df = calculate_thresholds(df, para)
    
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
            df.at[curr_idx, 'label'] = Signal.POSITIVE  if is_long_event else Signal.NEGATIVE
        else:
            df.at[curr_idx, 'label'] = Signal.NEUTRAL

    _boll_audit(df, event_indices)
    return df

def _boll_audit(df, event_indices):
    total = len(event_indices)
    stats = df.loc[event_indices, 'label'].value_counts()
    print(f"\n📊 [BOLL Lifecycle Audit]")
    print(f"  - 触发点总数: {total}")
    print(f"  - POSITIVE  (2) 有效: {stats.get(Signal.POSITIVE , 0)} ({(stats.get(Signal.POSITIVE , 0)/total)*100:.2f}%)")
    print(f"  - NEGATIVE (0) 有效: {stats.get(Signal.NEGATIVE, 0)} ({(stats.get(Signal.NEGATIVE, 0)/total)*100:.2f}%)")
    print(f"  - NEUTRAL (1) 噪音: {stats.get(Signal.NEUTRAL, 0)}")

def attach_sma_7_25_crossover_label(df, 
                                interval_ms,para = BaseDefine,):
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
    df = calculate_thresholds(df, para)
    
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
            df.at[curr_idx, 'label'] = Signal.POSITIVE  if is_long_event else Signal.NEGATIVE
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

def setup_session_logger(sub_folder: str = None, log_file_path=None, symbol: str = BaseDefine.symbol, console_level: int = logging.INFO, file_level: int = logging.INFO):
    if log_file_path ==None:
        assert sub_folder!=None
        log_dir = os.path.join(PERSISTENCE_DIR,'log', sub_folder)
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

def get_git_info(logger):
    repo = git.Repo(PROJECT_DIR)
    sha = repo.head.object.hexsha
    short_sha = repo.git.rev_parse(sha, short=8)
    
    logger.info(f"Full SHA: {sha}")
    logger.info(f"Short SHA: {short_sha}")
    logger.info(f"Commit Message: {repo.head.object.message.strip()}")
    return short_sha

def save_params(path, *, strategy, common, train):
    data = {
        "strategy": asdict(strategy),
        "common": asdict(common),
        "train": asdict(train),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_dataclass(cls, data: dict):
    """
    从 dict 构造 dataclass（支持嵌套 dataclass）
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue

        val = data[f.name]

        # 嵌套 dataclass
        if is_dataclass(f.type) and isinstance(val, dict):
            kwargs[f.name] = build_dataclass(f.type, val)
        else:
            kwargs[f.name] = val

    return cls(**kwargs)

def load_parameters(path, cls):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return build_dataclass(cls, data["strategy"])

def load_common_define(path, cls):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return build_dataclass(cls, data["common"])

def load_train_config(path, cls):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return build_dataclass(cls, data["train"])

def create_experiment_dir(base_dir, symbol, interval, now=None):
    """
    创建目录：
    base_dir / YYYY-MM-DD / SYMBOL_INTERVAL / HH_MM_SS

    返回最终实验目录路径
    """
    now = now or datetime.now()

    date_dir = now.strftime("%Y-%m-%d")
    time_dir = now.strftime("%H_%M_%S")
    sym_interval_dir = f"{symbol}_{interval}"

    exp_dir = os.path.join(base_dir, sym_interval_dir,date_dir, time_dir)
    os.makedirs(exp_dir, exist_ok=True)

    return exp_dir

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())   # 可选，但推荐