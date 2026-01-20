from enum import IntEnum
from functools import lru_cache
import logging,math
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

#define model
CANDLESTICK_NUM = 200   #160 best for LSTM
PREDICT_NUM = 16
# 波动率系数 (0.5 ~ 1.0 之间调整)
'''
乘数 (Multiplier),阈值位置,含义
VOL_MULTIPLIER=1.0,1σ,约 31.8% 的价格变动会超出这个阈值（上下尾部）。
VOL_MULTIPLIER=0.5,0.5σ,约 61.7% 的价格变动会超出这个阈值。信号数量适中。
VOL_MULTIPLIER=1.5,1.5σ,仅约 13.4% 的价格变动会超出这个阈值。
VOL_MULTIPLIER=2.0,2σ,仅约 4.6% 的价格变动会超出这个阈值。    
''' 
VOL_MULTIPLIER = 1.2
# 最小硬阈值 (覆盖手续费+滑点)
MIN_THRESHOLD = 0.005  # 0.25%
STOP_MULTIPLIER_RATE = 0.5
# label_decrease_weak =1 
# label_increase_weak = 3
model_train_rate = 0.8
symbol = 'BTCUSDT' # option: 'BTCUSDT' 'ETHUSDT' 'BNBUSDT' 'DOGEUSDT'
interval = '5m' # option: 1s, 15s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
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
origin_data_path = os.path.join(PROJECT_DATA_DIR, f"{symbol}_{interval}.csv")
train_data_path = os.path.join(TEMPORARY_DIR, "train_data.csv")
test_data_path  = os.path.join(TEMPORARY_DIR, "test_data.csv")
data_config_path  = os.path.join(TEMPORARY_DIR, "data_config_meta.json")

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
                    candlestick_num=CANDLESTICK_NUM, 
                    predict_num=PREDICT_NUM, 
                    vol_multiplier=VOL_MULTIPLIER,
                    stop_multiplier_rate=STOP_MULTIPLIER_RATE,
                    min_threshold=MIN_THRESHOLD):
    """
    融合版打标签逻辑：
    1. 【时空对齐】：严格校验未来第 N 根 K 线的物理时间戳，处理数据断层。
    2. 【路径依赖】：使用 High/Low 检查期间是否触发止损（来自 copy 版本）。
    3. 【动态阈值】：基于波动率生成的 TP/SL 阈值。
    """
    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    # 1. 计算动态阈值 (生成 df['threshold'] 和 df['stop_threshold'])
    df = calculate_thresholds(df, candlestick_num, predict_num, vol_multiplier, min_threshold, stop_multiplier_rate)

    # 2. 物理时间锚定 (时空对齐核心)
    target_times = time_values + (predict_num * interval_ms)
    target_indices = np.searchsorted(time_values, target_times, side='left')
    
    # 构建有效性掩码：索引不越界且时间戳精确匹配
    in_bounds = target_indices < len(df)
    safe_idx = np.where(in_bounds, target_indices, 0)
    actual_times = time_values[safe_idx]
    final_valid_mask = in_bounds & (actual_times == target_times)
    # # 直接覆盖原来的逻辑
    # final_valid_mask = np.ones(len(df), dtype=bool)

    # 3. 计算未来收益率 (基于锚定点收盘价)
    # pct_final 是从当前 close 到未来 T+predict_num 的 close 的变动
    future_close = np.where(final_valid_mask, df['close'].values[safe_idx], np.nan)
    pct_final = (future_close - df['close']) / df['close']

    # 4. 计算期间极致波动 (用于路径依赖校验)
    # 我们需要检查在 (t, t+predict_num] 期间，价格是否先触碰了止损
    steps = (target_indices - np.arange(len(df))).clip(1, predict_num)
    
    # 提取未来窗口内的 High 和 Low (包含当前行之后的 predict_num 根)
    high_mtx = np.column_stack([df['high'].shift(-i).values for i in range(1, predict_num + 1)])
    low_mtx = np.column_stack([df['low'].shift(-i).values for i in range(1, predict_num + 1)])
    
    # 计算累计极值
    future_high_max = np.maximum.accumulate(high_mtx, axis=1)[np.arange(len(df)), steps - 1]
    future_low_min = np.minimum.accumulate(low_mtx, axis=1)[np.arange(len(df)), steps - 1]

    # 计算期间最大反向波动
    max_drawdown = (future_low_min - df['close']) / df['close'] # 多单要看回撤
    max_runup = (future_high_max - df['close']) / df['close']    # 空单要看反弹

    # 5. 核心打标签逻辑 (融合 copy 中的路径检查)
    # 做多条件：
    # A. 时间对齐有效 
    # B. 最终收益 > 止盈阈值 
    # C. 期间最低价未跌破动态止损线 (max_drawdown > -stop_threshold)
    cond_long = final_valid_mask & \
                (pct_final > df['threshold']) & \
                (max_drawdown > -df['stop_threshold'])
                
    # 做空条件：
    # A. 时间对齐有效
    # B. 最终收益 < -止盈阈值
    # C. 期间最高价未突破动态止盈线 (max_runup < df['stop_threshold'])
    cond_short = final_valid_mask & \
                 (pct_final < -df['threshold']) & \
                 (max_runup < df['stop_threshold'])

    # 6. 应用标签
    conditions = [~final_valid_mask, cond_short, cond_long]
    choices = [Signal.INVALID, Signal.SHORT, Signal.LONG]
    
    df['label'] = np.select(conditions, choices, default=Signal.NEUTRAL).astype(int)
    
    # ==============================================================================
    # 🔍 【深度审计】打印原始 ms 与 物理跨度检查
    # ==============================================================================
    invalid_rows = df[df['label'] == Signal.INVALID]
    num_invalid = len(invalid_rows)

    print(f"\n📊 [Label Audit] 发现 {num_invalid} 个无效标签 (Signal.INVALID):")
    
    if num_invalid > 0:
        sample_size = min(10, num_invalid)
        print(f"📍 前 {sample_size} 个无效行明细 (含原始 ms):")
        # 增加一列：物理差值 (Gap_ms)，用来定位到底跳了多少时间
        print(f"{'Index':>8} | {'Raw MS':>15} | {'Target MS':>15} | {'Gap_ms':>10} | {'Reason'}")
        print("-" * 85)
        
        # 预先获取整个时间序列的 numpy 数组用于快速计算
        time_values = df['open_time_ms_utc'].values
        
        for idx, row in invalid_rows.head(sample_size).iterrows():
            curr_ms = int(row['open_time_ms_utc'])
            # 理想中，这根 K 线对应的预测终点时间
            ideal_target_ms = curr_ms + (predict_num * interval_ms)
            
            # 使用 searchsorted 找出的实际索引
            target_idx = np.searchsorted(time_values, ideal_target_ms, side='left')
            
            if target_idx < len(df):
                actual_target_ms = time_values[target_idx]
                gap_ms = actual_target_ms - ideal_target_ms
                # 如果 gap_ms > 0，说明理想的时间点在数据里是空的
                reason = f"Gap: {gap_ms}ms" if gap_ms > 0 else "Continuous"
            else:
                reason = "Out-of-Bounds"
                gap_ms = "N/A"

            print(f"{idx:8d} | {curr_ms:15d} | {ideal_target_ms:15d} | {str(gap_ms):>10} | {reason}")

    df['return_rate'] = pct_final 
    return df

def attach_triple_barrier_label(df, 
                                 interval_ms,
                                 candlestick_num=CANDLESTICK_NUM, 
                                 predict_num=PREDICT_NUM, 
                                 vol_multiplier=VOL_MULTIPLIER,
                                 stop_multiplier_rate=STOP_MULTIPLIER_RATE,
                                 min_threshold=MIN_THRESHOLD):
    """
    严苛版 Triple Barrier (混合价格触发):
    1. 止盈 (TP): 使用未来 K 线的 Close 计算，要求收盘价必须达标。
    2. 止损 (SL): 使用未来 K 线的 High/Low 计算，只要触碰即视为失败。
    3. 判定顺序: 只有当【首次收盘达标止盈】的索引 早于 【首次触碰止损】的索引时，才标注信号。
    """
    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    # 1. 计算动态阈值 (保留原逻辑)
    df = calculate_thresholds(df, candlestick_num, predict_num, vol_multiplier, min_threshold, stop_multiplier_rate)

    # 2. 物理时间锚定 (保留原逻辑)
    target_times = time_values + (predict_num * interval_ms)
    target_indices = np.searchsorted(time_values, target_times, side='left')
    in_bounds = target_indices < len(df)
    safe_idx = np.where(in_bounds, target_indices, 0)
    final_valid_mask = in_bounds & (time_values[safe_idx] == target_times)

    # 3. 向量化矩阵准备
    # 止盈用 closes，止损用 highs/lows
    future_closes = np.column_stack([df['close'].shift(-i).values for i in range(1, predict_num + 1)])
    future_highs = np.column_stack([df['high'].shift(-i).values for i in range(1, predict_num + 1)])
    future_lows = np.column_stack([df['low'].shift(-i).values for i in range(1, predict_num + 1)])
    
    closes = df['close'].values
    tp_dist = df['threshold'].values
    sl_dist = df['stop_threshold'].values
    
    # 初始化：全部设为 NEUTRAL (1)
    labels = np.full(len(df), Signal.NEUTRAL, dtype=int)

    # 4. 严苛匹配逻辑 (基于 Close 止盈 vs High/Low 止损)
    for i in range(len(df) - predict_num):
        if not final_valid_mask[i]:
            labels[i] = Signal.INVALID
            continue
            
        curr_price = closes[i]
        
        # --- 多头检测 (Long) ---
        # 止盈：Close >= 目标价
        idx_long_tp = np.where(future_closes[i] >= curr_price * (1 + tp_dist[i]))[0]
        # 止损：Low <= 止损价
        idx_long_sl = np.where(future_lows[i] <= curr_price * (1 - sl_dist[i]))[0]
        
        first_l_tp = idx_long_tp[0] if len(idx_long_tp) > 0 else predict_num
        first_l_sl = idx_long_sl[0] if len(idx_long_sl) > 0 else predict_num

        # --- 空头检测 (Short) ---
        # 止盈：Close <= 目标价 (价格下跌)
        idx_short_tp = np.where(future_closes[i] <= curr_price * (1 - tp_dist[i]))[0]
        # 止损：High >= 止损价 (价格反弹)
        idx_short_sl = np.where(future_highs[i] >= curr_price * (1 + sl_dist[i]))[0]
        
        first_s_tp = idx_short_tp[0] if len(idx_short_tp) > 0 else predict_num
        first_s_sl = idx_short_sl[0] if len(idx_short_sl) > 0 else predict_num

        # 判定：TP 索引必须严格小于 SL 索引
        # 如果 TP 和 SL 在同一根 K 线发生 (first_tp == first_sl)，
        # 鉴于 SL 是 High/Low 触发而 TP 是 Close 触发，逻辑上 SL 必然早于或等于 TP，
        # 因此这种“同归于尽”的样本在 strict 模式下应归为 NEUTRAL。
        if first_l_tp < first_l_sl:
            labels[i] = Signal.LONG
        elif first_s_tp < first_s_sl:
            labels[i] = Signal.SHORT
        # 其余情况（包含 first_tp == first_sl）均为 NEUTRAL

    df['label'] = labels
    df.loc[~final_valid_mask, 'label'] = Signal.INVALID
    
    # 保留 return_rate 计算
    future_close = np.where(final_valid_mask, df['close'].values[safe_idx], np.nan)
    df['return_rate'] = (future_close - df['close']) / df['close']
    
    return df

def calculate_thresholds(df, 
                         candlestick_num: int = CANDLESTICK_NUM, 
                         predict_num: int = PREDICT_NUM, 
                         vol_multiplier = VOL_MULTIPLIER,
                         min_threshold = MIN_THRESHOLD,
                         stop_multiplier_rate = STOP_MULTIPLIER_RATE,
                         **kwargs): # 🌟 接收多余参数防止报错
    """
    【核心逻辑提取】计算动态止盈 (threshold) 和 止损 (stop_threshold)
    该函数同时用于:
    1. 训练前的数据预处理 (attach_label)
    2. 回测/实盘中的信号生成 (simulation/production)
    """
    # 基础检查
    assert 'close' in df.columns, "缺少 Close 数据"
    
    # 1. 计算波动率基准 (Rolling Standard Deviation)
    vol_window = candlestick_num
    returns = df['close'].pct_change()
    rolling_std = returns.rolling(window=vol_window).std()
    
    # 2. 时间扩充波动率 (Time Scaling: sigma * sqrt(T))
    expected_vol = rolling_std * np.sqrt(predict_num)
    
    # 3. 计算动态阈值
    # A. 目标止盈阈值 (基于波动率)
    target_threshold = (expected_vol * vol_multiplier).clip(lower=min_threshold)
    
    # B. 目标止损阈值 (基于止盈的百分比，保持盈亏比)
    stop_threshold = target_threshold * stop_multiplier_rate
    
    # 4. 写入 DataFrame (处理 NaN)
    # 使用 copy 防止 SettingWithCopyWarning (视调用情况而定，这里直接赋值通常没问题)
    df['threshold'] = target_threshold.fillna(min_threshold)
    df['stop_threshold'] = stop_threshold.fillna(min_threshold * stop_multiplier_rate)
    
    return df

def attach_macd_event_lifecycle_label(df, 
                                     interval_ms,
                                     vol_multiplier=VOL_MULTIPLIER,
                                     stop_multiplier_rate=STOP_MULTIPLIER_RATE,
                                     min_threshold=MIN_THRESHOLD):
    """
    严格时间对齐版 MACD 生命周期标签 (自动匹配特征列名):
    1. 自动探测列名：匹配如 MACD_12_16_DIF 和 MACD_12_16_DEA。
    2. 物理时间校验：确保交叉点之间无数据断层。
    3. 严苛判定：持有至下次交叉，Close 止盈达标 (X) 且过程中 High/Low 未触及止损。
    """
    # --- 1. 自动匹配 MACD 特征列名 ---
    dif_cols = [c for c in df.columns if c.startswith('MACD_') and c.endswith('_DIF')]
    dea_cols = [c for c in df.columns if c.startswith('MACD_') and c.endswith('_DEA')]
    
    if not dif_cols or not dea_cols:
        raise ValueError("❌ 未在 DataFrame 中探测到 MACD 特征列 (需以 _DIF 和 _DEA 结尾)")
    
    # 默认取第一组匹配的 MACD (如果有多个，通常取第一个生成的)
    dif_name = dif_cols[0]
    # 寻找对应的 DEA 列名 (确保 fast/slow 参数一致)
    prefix = dif_name.replace('_DIF', '')
    dea_name = f"{prefix}_DEA"
    
    if dea_name not in df.columns:
        raise ValueError(f"❌ 找不到与 {dif_name} 匹配的 {dea_name}")

    print(f"🔍 [MACD Match] 自动匹配到特征列: {dif_name} / {dea_name}")

    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    # 2. 识别所有交叉点
    dif = df[dif_name]
    dea = df[dea_name]
    
    # 交叉判断
    cross_mask = (dif > dea) != (dif.shift(1) > dea.shift(1))
    cross_mask.iloc[0] = False
    event_indices = df.index[cross_mask].tolist()
    
    # 3. 初始化与动态阈值
    df['label'] = Signal.INVALID
    # 计算动态阈值 (用于止盈 X 和止损位)
    df = calculate_thresholds(df, vol_multiplier=vol_multiplier, min_threshold=min_threshold, stop_multiplier_rate=stop_multiplier_rate)
    
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
        # 根据当前交叉点判断方向
        is_long_event = dif.iloc[curr_idx] > dea.iloc[curr_idx]
        entry_price = closes[curr_idx]
        tp_target = thresholds[curr_idx]
        sl_target = sl_thresholds[curr_idx]
        
        # A. 止盈判断：持有至下次交叉时的收盘价收益率
        pnl_at_exit = (closes[next_idx] - entry_price) / entry_price if is_long_event else \
                      (entry_price - closes[next_idx]) / entry_price
        
        # B. 止损判断：区间内 (t+1, t_next) 的 High/Low 触碰
        window_highs = highs[curr_idx + 1 : next_idx + 1]
        window_lows = lows[curr_idx + 1 : next_idx + 1]
        
        if is_long_event:
            hit_stop = np.any(window_lows <= entry_price * (1 - sl_target))
        else:
            hit_stop = np.any(window_highs >= entry_price * (1 + sl_target))

        # C. 综合打标 (Strict Mode)
        # 只有收益超过动态阈值 X 且过程中没被止损的才算 LONG/SHORT
        if pnl_at_exit >= tp_target and not hit_stop:
            df.at[curr_idx, 'label'] = Signal.LONG if is_long_event else Signal.SHORT
        else:
            df.at[curr_idx, 'label'] = Signal.NEUTRAL

    return df

def attach_boll_event_lifecycle_label(df, 
                                     interval_ms,
                                     vol_multiplier=VOL_MULTIPLIER,
                                     stop_multiplier_rate=STOP_MULTIPLIER_RATE,
                                     min_threshold=MIN_THRESHOLD):
    """
    均值回归版 布林带生命周期标签：
    1. 自动探测列名：匹配如 BOLL_UPPER_20, BOLL_LOWER_20, BOLL_MIDDLE_20。
    2. 触发：收盘价穿越上下轨。
    3. 退出：收盘价回归中轨。
    4. 严苛判定：退出收益 > threshold 且过程中未触及 High/Low 止损。
    """
    # --- 1. 自动匹配 BOLL 特征列名 ---
    upper_cols = [c for c in df.columns if c.startswith('BOLL_UPPER_')]
    lower_cols = [c for c in df.columns if c.startswith('BOLL_LOWER_')]
    middle_cols = [c for c in df.columns if c.startswith('BOLL_MIDDLE_')]
    
    if not (upper_cols and lower_cols and middle_cols):
        raise ValueError("❌ 未在 DataFrame 中探测到完整的 BOLL 特征列")
    
    # 自动提取前缀和周期
    u_name, l_name, m_name = upper_cols[0], lower_cols[0], middle_cols[0]
    print(f"🔍 [BOLL Match] 自动匹配到特征列: {u_name}, {l_name}, {m_name}")

    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    # 2. 识别触发点 (收盘价穿越轨道)
    # 做多触发：Close < Lower
    long_trigger = df['close'] < df[l_name]
    # 做空触发：Close > Upper
    short_trigger = df['close'] > df[u_name]
    
    event_mask = long_trigger | short_trigger
    event_indices = df.index[event_mask].tolist()
    
    # 3. 初始化与动态阈值
    df['label'] = Signal.INVALID
    df = calculate_thresholds(df, vol_multiplier=vol_multiplier, min_threshold=min_threshold, stop_multiplier_rate=stop_multiplier_rate)
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    middles = df[m_name].values
    thresholds = df['threshold'].values
    sl_thresholds = df['stop_threshold'].values

    # 4. 遍历事件，寻找回归中轨的退出点
    for curr_idx in event_indices:
        is_long_event = long_trigger.iloc[curr_idx]
        entry_price = closes[curr_idx]
        tp_target = thresholds[curr_idx]
        sl_limit = sl_thresholds[curr_idx]
        
        # 寻找回归中轨的第一个位置 (退出点)
        # 多头：寻找 Close >= Middle; 空头：寻找 Close <= Middle
        if is_long_event:
            exit_candidates = np.where(closes[curr_idx + 1:] >= middles[curr_idx + 1:])[0]
        else:
            exit_candidates = np.where(closes[curr_idx + 1:] <= middles[curr_idx + 1:])[0]
            
        if len(exit_candidates) == 0:
            df.at[curr_idx, 'label'] = Signal.INVALID
            continue
            
        next_idx = curr_idx + 1 + exit_candidates[0]
        
        # --- 时间对齐校验 ---
        if (time_values[next_idx] - time_values[curr_idx]) != (next_idx - curr_idx) * interval_ms:
            df.at[curr_idx, 'label'] = Signal.INVALID
            continue

        # --- 业务逻辑判定 ---
        # A. 收益判断 (退出时的收盘价)
        pnl_at_exit = (closes[next_idx] - entry_price) / entry_price if is_long_event else \
                      (entry_price - closes[next_idx]) / entry_price
        
        # B. 路径止损检查 (区间内 High/Low)
        window_highs = highs[curr_idx + 1 : next_idx + 1]
        window_lows = lows[curr_idx + 1 : next_idx + 1]
        
        if is_long_event:
            hit_stop = np.any(window_lows <= entry_price * (1 - sl_limit))
        else:
            hit_stop = np.any(window_highs >= entry_price * (1 + sl_limit))

        # C. 打标
        if pnl_at_exit >= tp_target and not hit_stop:
            df.at[curr_idx, 'label'] = Signal.LONG if is_long_event else Signal.SHORT
        else:
            df.at[curr_idx, 'label'] = Signal.NEUTRAL

    # 审计打印
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
                                   vol_multiplier=VOL_MULTIPLIER,
                                   stop_multiplier_rate=STOP_MULTIPLIER_RATE,
                                   min_threshold=MIN_THRESHOLD):
    """
    指定 SMA 7/25 交叉生命周期标签：
    1. 信号源：SMA_7B (快线) 与 SMA_25B (慢线)。
    2. 触发逻辑：金叉(7>25)看多，死叉(7<25)看空。
    3. 严苛判定：持有至下次交叉，Close 收益需 > 动态阈值，且期间 High/Low 未触及止损。
    """
    # --- 1. 明确指定列名 ---
    fast_ma_name = "SMA_7B"
    slow_ma_name = "SMA_25B"
    
    if fast_ma_name not in df.columns or slow_ma_name not in df.columns:
        raise ValueError(f"❌ 找不到均线列 {fast_ma_name} 或 {slow_ma_name}，请检查 FeatureMA 配置")

    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    # 2. 识别交叉点 (Event Detection)
    fast_ma = df[fast_ma_name]
    slow_ma = df[slow_ma_name]
    cross_mask = (fast_ma > slow_ma) != (fast_ma.shift(1) > slow_ma.shift(1))
    cross_mask.iloc[0] = False
    event_indices = df.index[cross_mask].tolist()
    
    # 3. 初始化标签与动态阈值
    df['label'] = Signal.INVALID # 其余非交叉点均为 INVALID
    df = calculate_thresholds(df, vol_multiplier=vol_multiplier, 
                              min_threshold=min_threshold, 
                              stop_multiplier_rate=stop_multiplier_rate)
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    thresholds = df['threshold'].values
    sl_thresholds = df['stop_threshold'].values

    # 4. 遍历交叉事件
    for i in range(len(event_indices)):
        curr_idx = event_indices[i]
        
        if i + 1 >= len(event_indices):
            continue
            
        next_idx = event_indices[i+1]
        
        # --- 物理时间戳对齐检查 ---
        expected_gap = (next_idx - curr_idx) * interval_ms
        actual_gap = time_values[next_idx] - time_values[curr_idx]
        
        if actual_gap != expected_gap:
            # 如果中间有 Gap，此交叉点保持 INVALID
            continue

        # --- 判定逻辑 ---
        is_long_event = fast_ma.iloc[curr_idx] > slow_ma.iloc[curr_idx]
        entry_price = closes[curr_idx]
        
        # A. 收益判断 (使用 Exit 点的 Close)
        pnl_rate = (closes[next_idx] - entry_price) / entry_price if is_long_event else \
                   (entry_price - closes[next_idx]) / entry_price
        
        # B. 路径止损检查 (区间内 High/Low)
        window_highs = highs[curr_idx + 1 : next_idx + 1]
        window_lows = lows[curr_idx + 1 : next_idx + 1]
        
        sl_limit = sl_thresholds[curr_idx]
        if is_long_event:
            hit_stop = np.any(window_lows <= entry_price * (1 - sl_limit))
        else:
            hit_stop = np.any(window_highs >= entry_price * (1 + sl_limit))

        # C. 打标
        if pnl_rate >= thresholds[curr_idx] and not hit_stop:
            df.at[curr_idx, 'label'] = Signal.LONG if is_long_event else Signal.SHORT
        else:
            df.at[curr_idx, 'label'] = Signal.NEUTRAL

    return df

def data_analyze(df, candlestick_num:int = CANDLESTICK_NUM, predict_num:int= PREDICT_NUM , vol_multiplier = VOL_MULTIPLIER,
                 min_threshold = MIN_THRESHOLD):
    # ============================================================
    # === 【新增功能】 计算未来窗口内的最大上涨和最大下跌比例 ===
    # ============================================================
    # 逻辑说明：
    # 1. rolling(window=N).max() 计算的是 [t-N+1, t] 的最大值
    # 2. shift(-N) 将数据向上平移，使得索引 t 处的数据变成原索引 t+N 处的数据
    # 3. 结合起来：在索引 t 处，我们要的是 [t+1, t+N] 的极值
    
    # 未来的最高价序列 (窗口: t+1 到 t+predict_num)
    future_rolling_max = df['high'].rolling(window=predict_num).max().shift(-predict_num)
    # 未来的最低价序列 (窗口: t+1 到 t+predict_num)
    future_rolling_min = df['low'].rolling(window=predict_num).min().shift(-predict_num)
    
    # 计算相对于当前收盘价的涨跌幅
    # max_up: 未来窗口内能卖到的最高收益率
    df['max_up'] = (future_rolling_max - df['close']) / df['close']
    # max_down: 未来窗口内可能承受的最大亏损率
    df['max_down'] = (future_rolling_min - df['close']) / df['close']

def clean_data_quality_auto(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    自动化检查所有列：
    1. 丢弃任何含有 NaN 的行。
    2. 探测数值列，并丢弃其中含有 0 的行（排除日期等非数值列）。
    """
    logger.info("启动自动化数据质量扫描...")
    initial_count = len(df)

    # 1. 统计并丢弃 NaN (全局检查)
    na_rows = df.isna().any(axis=1).sum()
    if na_rows > 0:
        logger.warning(f"检测到 {na_rows} 行数据包含空值 (NaN)，准备丢弃。")

    # 2. 识别数值型列进行零值检查
    # 我们只对 float 和 int 类型的列检查 0，避免对字符串时间列误判
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 统计数值列中的零值
    zero_mask = (df[numeric_cols] == 0).any(axis=1)
    zero_rows = zero_mask.sum()
    
    if zero_rows > 0:
        # 找出具体是哪些列含有 0，打印详细日志
        zero_stats = (df[numeric_cols] == 0).sum()
        logger.warning(f"检测到 {zero_rows} 行数据包含零值，分布如下:\n{zero_stats[zero_stats > 0]}")

    # 3. 综合清洗：剔除 NaN 或 0 的行
    # 我们保留数值不为 0 且非空的数据
    condition = df.isna().any(axis=1) | zero_mask
    df_cleaned = df[~condition].copy()

    # 4. 重置索引 (非常重要，防止后续 rolling 计算因为索引断裂出问题)
    df_cleaned.reset_index(drop=True, inplace=True)

    final_count = len(df_cleaned)
    dropped_count = initial_count - final_count

    if dropped_count > 0:
        logger.info(f"✅ 清洗完毕: 原始 {initial_count} 行 -> 剩余 {final_count} 行 (丢弃了 {dropped_count} 行)")
    else:
        logger.info("✅ 扫描完毕: 未发现空值或零值，数据质量完美。")

    return df_cleaned

def float_range(start, end, step):
    """
    Like range(), but for float.
    Inclusive of end (with tolerance).
    """
    values = []
    v = start
    eps = step / 10

    while v <= end + eps:
        values.append(round(v, 10))
        v += step

    return values

@lru_cache(maxsize=1)   #only available when tarining
def load_interval_ms(config_path = data_config_path):
    """从元数据文件中安全读取间隔毫秒数"""
    if not os.path.exists(config_path):
        raise RuntimeError(f"❌ 找不到配置文件: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        # 提取 interval_ms
        interval_ms = meta.get("interval_ms")
        if interval_ms is None:
            raise RuntimeError(f"⚠️ 配置文件中缺失 'interval_ms' 字段！")
        return interval_ms
    except Exception as e:
        raise RuntimeError(f"💥 读取 JSON 时发生意外错误: {e}")

def setup_session_logger(sub_folder: str, symbol: str = symbol, console_level: int = logging.INFO, file_level: int = logging.INFO):
    """
    配置实盘会话日志：
    1. 生成带时间戳的文件名 (session_SYMBOL_Time.log)
    2. 将 FileHandler 挂载到 Root Logger (捕获所有模块日志)
    3. 将 StreamHandler (彩色) 挂载到 Root Logger (控制台显示所有日志)
    
    :param log_root: 日志根目录 (例如 common.TEMPORARY_DIR)
    :param sub_folder: 子文件夹名 (例如 'market_ftmo_sessions')
    :param symbol: 交易品种
    :return: (root_logger, log_file_path)
    """
    # 1. 准备目录
    log_dir = os.path.join(PERSISTENCE_DIR, sub_folder)
    os.makedirs(log_dir, exist_ok=True)

    # 2. 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sym_str = f"_{symbol}" if symbol else ""
    log_filename = f"session{sym_str}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    # 3. 获取 Root Logger
    # 注意：我们配置 Root，这样 self.logger = logging.getLogger("AnyName") 都会被捕获
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # 开启所有级别，具体过滤交给 Handler

    # 清除旧的 Handlers (防止重复打印，特别是 Jupyter 或多次调用时)
    if root_logger.handlers:
        root_logger.handlers = []

    # 4. 配置控制台输出 (复用之前的彩色格式)
    log_format_console = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    
    color_formatter = colorlog.ColoredFormatter(
        log_format_console,
        datefmt="%H:%M:%S", # 控制台时间短一点，方便看
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

    # 5. 配置文件输出 (全量记录)
    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setLevel(file_level) # 文件通常记录 INFO 以上
    
    # 文件格式包含完整日期
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(file_formatter)
    root_logger.addHandler(fh)

    # 打印一条初始化消息
    root_logger.info(f"Session Logger Initialized. Log file: {log_file_path}")

    return root_logger, log_file_path