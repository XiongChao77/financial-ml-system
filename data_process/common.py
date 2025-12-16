from enum import IntEnum
import logging,math
import pandas as pd
import numpy as np
import os, colorlog , logging
from data_process.feature import *
from data_process.logger    import setup_logger

class Signal(IntEnum):
    SHORT = 0
    NEUTRAL = 1
    LONG = 2

#define model
CANDLESTICK_NUM = 136
PREDICT_NUM = 16
# 波动率系数 (0.5 ~ 1.0 之间调整)
'''
乘数 (Multiplier),阈值位置,含义
VOL_MULTIPLIER=1.0,1σ,约 31.8% 的价格变动会超出这个阈值（上下尾部）。
VOL_MULTIPLIER=0.5,0.5σ,约 61.7% 的价格变动会超出这个阈值。信号数量适中。
VOL_MULTIPLIER=1.5,1.5σ,仅约 13.4% 的价格变动会超出这个阈值。
VOL_MULTIPLIER=2.0,2σ,仅约 4.6% 的价格变动会超出这个阈值。    
''' 
VOL_MULTIPLIER = 0.8
# 最小硬阈值 (覆盖手续费+滑点)
MIN_THRESHOLD = 0.01  # 0.25%
STOP_MULTIPLIER_RATE = 0.5
# label_decrease_weak =1 
# label_increase_weak = 3
model_train_rate = 0.8
DATA_PROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DATA_PROCESS_DIR)
TEMPORARY_DIR = os.path.join(PROJECT_DIR , 'output')
PROJECT_DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR),'QuantData','Cryptocurrency')
origin_data_path = os.path.join(PROJECT_DATA_DIR, "BTCUSDT_15m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "BTCUSDT_5m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "ETHUSDT_15m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "BNBUSDT_15m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "DOGEUSDT_15m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "DOGEUSDT_5m.csv")
DATA_PROCESS_OUT_DIR = os.path.join(DATA_PROCESS_DIR, 'output')
train_data_path = os.path.join(TEMPORARY_DIR, "train_data.csv")
test_data_path  = os.path.join(TEMPORARY_DIR, "test_data.csv")
log_level = logging.INFO

def attach_attr(df):
    # 1. 基础处理
    df.rename({'ignore':'label'},axis=1, inplace=True) 

    # --- 2. 指标计算 (生成所有原始、未缩放的特征列) ---
    # df = add_relative_features(df)
    FeatureFactory(FEATURE_CONFIG).generate(df)

def attach_label(df, 
                                 candlestick_num:int = CANDLESTICK_NUM, 
                                 predict_num:int = PREDICT_NUM, 
                                 vol_multiplier = VOL_MULTIPLIER,
                                 stop_multiplier_rate = STOP_MULTIPLIER_RATE, # 这里传入比例，如 0.3 或 0.5
                                 min_threshold = MIN_THRESHOLD):
    """
    打标签逻辑 (盈亏比绑定版):
    止损阈值不再独立计算，而是止盈阈值的一个百分比。
    
    例如: vol_multiplier=2.0 (预期涨2倍波动), stop_multiplier_rate=0.5
    => 目标涨幅 = 2.0 * Vol
    => 容忍跌幅 = 1.0 * Vol (即目标的 50%)
    """
    # 0. 基础检查
    assert 'close' in df.columns and 'high' in df.columns and 'low' in df.columns, "缺少 OHLC 数据"
    assert predict_num > 0, "predict_num 必须 > 0"
    print(f"[Labeling] Target Vol: {vol_multiplier}, Stop Rate: {stop_multiplier_rate*100}%")
    
    vol_window = candlestick_num
    df = calculate_thresholds(df, candlestick_num, predict_num, vol_multiplier, stop_multiplier_rate, min_threshold)
    # 3. 获取未来数据 (含路径依赖检查)
    # -------------------------------------------------------------------------
    # 最终收益率
    future_close = df['close'].shift(-predict_num)
    pct_final = (future_close - df['close']) / df['close']
    
    # 期间最低价 (检查多单止损)
    future_low_min = df['low'].rolling(window=predict_num).min().shift(-predict_num)
    # 期间最高价 (检查空单止损)
    future_high_max = df['high'].rolling(window=predict_num).max().shift(-predict_num)

    # 计算期间最大回撤/反向波动
    max_drawdown = (future_low_min - df['close']) / df['close'] 
    max_runup = (future_high_max - df['close']) / df['close']

    # 4. 打标签 (Labeling)
    # -------------------------------------------------------------------------
    
    # === Label 2: 做多 ===
    # 1. 最终涨幅 > 目标
    # 2. 期间最大跌幅 > -止损 (即没有跌破止损线)
    cond_long = (pct_final > df['threshold']) & \
                (max_drawdown > -df['stop_threshold'])

    # === Label 0: 做空 ===
    # 1. 最终跌幅 < -目标
    # 2. 期间最大涨幅 < 止损 (即没有涨破止损线)
    cond_short = (pct_final < -df['threshold']) & \
                 (max_runup < df['stop_threshold'])

    # 应用标签
    df['label'] = np.select(
        [cond_short, cond_long],
        [Signal.SHORT, Signal.LONG],
        default=Signal.NEUTRAL
    )

    # 5. 清洗
    df.iloc[:vol_window, df.columns.get_loc('label')] = Signal.NEUTRAL
    
    if predict_num > 0:
        df = df.iloc[:-predict_num].reset_index(drop=True)

    df['label'] = df['label'].astype(int)

    df['return_rate'] = pct_final
        
    return df

def calculate_thresholds(df, 
                         candlestick_num: int = CANDLESTICK_NUM, 
                         predict_num: int = PREDICT_NUM, 
                         vol_multiplier = VOL_MULTIPLIER,
                         stop_multiplier_rate = STOP_MULTIPLIER_RATE, 
                         min_threshold = MIN_THRESHOLD):
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

FEATURE_CONFIG:list[FeatureBase] = [
    FeatureMACD(fast=12, slow=26, signal=9),
    FeatureMA(weeks=[7,25], days=[5, 10, 20], method='sma', strict=True, add_slope = False, slope_method='reg', slope_weeks= 2),
    FeatureRsi(period=14, price_col='close', strict=True, prefix='RSI'),
    FeatureKdj(n=9, m1=3, m2=3, high_col='high', low_col='low',  close_col='close', strict=True, prefix='KDJ'),
    FeatureVolMa(vol_ma_windows = (5, 10, 20)),
    FeatureQavMa(vol_ma_windows = (5, 10, 20)),
    FeatureOBV(),
    FeaturePVT(),
    FeatureWAP(vwap_windows = (20, 48, 96)),
    FeatureCFM(cmf_window = 20),
    FeatureMFI(mfi_window=14),
    FeatureATS(), #FeatureContainer(FeatureATS),
    FeatureCandle(),
]

class FeatureFactory:
    def __init__(self, config_list:list[FeatureBase] = FEATURE_CONFIG):
        self.features:list[FeatureBase] = config_list
        # for f in config_list:
        #     self.features.append(f.feature(f.parameters))
    def generate(self,df):
        for f in self.features:     f.generate(df)
    def normalize(self, X: np.ndarray, feature_cols: list[str]):
        for f in self.features:
            f.normalize(X, feature_cols)