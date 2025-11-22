import logging,math
import pandas as pd
import numpy as np
import os
from data_process.ta_calculation import *
#define model
candlestick_num = 120
predict_num = 16
change_rate = 0.006 # 0.2%
weak_change = change_rate / 5.0
# 波动率系数 (0.5 ~ 1.0 之间调整)
'''
乘数 (Multiplier),阈值位置,含义
vol_multiplier=1.0,1σ,约 31.8% 的价格变动会超出这个阈值（上下尾部）。
vol_multiplier=0.5,0.5σ,约 61.7% 的价格变动会超出这个阈值。信号数量适中。
vol_multiplier=1.5,1.5σ,仅约 13.4% 的价格变动会超出这个阈值。
vol_multiplier=2.0,2σ,仅约 4.6% 的价格变动会超出这个阈值。    
''' 
vol_multiplier = 1.1

label_decrease = 0
# label_decrease_weak =1 
label_ignore = 1
# label_increase_weak = 3
label_increase = 2
model_train_rate = 0.8
DATA_PROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DATA_PROCESS_DIR) 
origin_data_path = os.path.join(os.path.dirname(PROJECT_DIR),'QuantData','Cryptocurrency', "BTCUSDT_15m.csv")
DATA_PROCESS_OUT_DIR = os.path.join(DATA_PROCESS_DIR, 'output')
train_data_path = os.path.join(DATA_PROCESS_OUT_DIR, "train_data.csv")
test_data_path  = os.path.join(DATA_PROCESS_OUT_DIR, "test_data.csv")
log_level = logging.INFO

# ====== 你可以按需要修改的默认特征列（9维）======
#只使用无量纲特征，让模型学习形态
DEFAULT_FEATURES = [
"open","high","low","close","volume","taker_buy_base_volume","quote_asset_volume","taker_buy_quote_volume","number_of_trades"
]
# DEFAULT_FEATURES = [
#     "open","high","low","close","volume","taker_buy_base_volume","taker_buy_quote_volume", "quote_asset_volume", "number_of_trades" ,
#     "MACD_DIF","MACD_DEA","MACD", "SMA_5D","SMA_10D","SMA_10D","SMA_20D"
# ]
#"MACD_DIF","MACD_DEA","MACD"
#EMA_7W,EMA_7W_SLOPE_REG_4W,EMA_7W_SLOPE_REG_4W_N,EMA_25W,EMA_25W_SLOPE_REG_4W,EMA_25W_SLOPE_REG_4W_N 
# , "RSI_14","KDJ_K","KDJ_D","KDJ_J"
# SMA_5D,SMA_10D,SMA_20D

def attach_attr(df):
    
    # 1. 基础处理
    df.rename({'ignore':'label'},axis=1, inplace=True) 

    # --- 2. 指标计算 (生成所有原始、未缩放的特征列) ---
    # df = add_relative_features(df)
    df = add_macd(df) 
    df = add_weekly_mas(df) 
    df = add_rsi(df, period=14, price_col="close", strict=True)
    df = add_kdj(df, n=9, m1=3, m2=3, strict=True)

    # --- 3. 标签生成与清理 ---
    
    # 生成标签 (label) 和动态阈值 (threshold)
    df = attach_label(df) 
    return df

def attach_label(df):
    """
    依据未来收益率与当前波动率的动态关系分3类，并将计算出的动态阈值保存到 'threshold' 列。
    
    Label 0: 下跌 (收益率 < -动态阈值)
    Label 1: 震荡 (绝对值 <= 动态阈值)
    Label 2: 上涨 (收益率 > 动态阈值)
    """
    assert 'close' in df.columns, "缺少列 close"
    assert predict_num > 0, "predict_num 必须 > 0"

    # ---------------- 参数设置 ----------------
    # 波动率参考窗口
    vol_window = candlestick_num 
    
    # 最小硬阈值 (覆盖手续费+滑点)
    min_threshold = 0.0025  # 0.25%
    # -----------------------------------------

    df = df.copy()

    # 1. 计算未来收益率 (Target)
    future_close = df['close'].shift(-predict_num)
    pct = (future_close - df['close']) / df['close']

    # 2. 计算动态阈值 (Dynamic Threshold)
    returns = df['close'].pct_change()
    rolling_std = returns.rolling(window=vol_window).std()
    expected_vol = rolling_std * np.sqrt(predict_num)
    
    # 计算阈值序列
    dynamic_threshold = (expected_vol * vol_multiplier).clip(lower=min_threshold)
    
    # === 【新增】 将阈值写入 DataFrame ===
    # 填充前部的 NaN (预热期)，避免保存出来的 CSV 这一列前面是空的
    # 我们可以用 min_threshold 填充，或者用第一个有效值填充
    df['threshold'] = dynamic_threshold.fillna(min_threshold)
    
    # 3. 打标签 (Labeling)
    # 注意：这里直接使用列 df['threshold'] 进行比较
    cond_decrease = (pct < -df['threshold'])
    cond_increase = (pct > df['threshold'])
    
    df['label'] = np.select(
        [cond_decrease, cond_increase],
        [label_decrease, label_increase],
        default=label_ignore
    )

    # 4. 数据清洗
    # 前 vol_window 行波动率计算不准，强制忽略
    df.iloc[:vol_window, df.columns.get_loc('label')] = label_ignore

    # 删尾（未来价格不可用的部分）
    if predict_num > 0:
        df = df.iloc[:-predict_num].reset_index(drop=True)

    df['label'] = df['label'].astype(int)

    # ---------------- 统计输出 ----------------
    counts = df['label'].value_counts().sort_index()
    proportions = df['label'].value_counts(normalize=True).sort_index()
    
    print("\n=== 动态标签分布统计 ===")
    print(f"阈值已保存至列: 'threshold'")
    print(f"阈值范围: Min={df['threshold'].min():.4f}, Max={df['threshold'].max():.4f}, Mean={df['threshold'].mean():.4f}")
    
    for label_val, cnt in counts.items():
        label_name = "下跌" if label_val == 0 else ("上涨" if label_val == 2 else "震荡")
        pct_val = proportions[label_val]
        print(f"Label {label_val} ({label_name}): {cnt} 个, 占比 {pct_val:.4%}")
    print("==========================\n")

    return df
    
def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    新增以下无量纲化特征：
    - price_change_pct: 当前 open 相对前一根 open 的变化率（%）
    - high_pct, low_pct, close_pct: 当前 high/low/close 相对 open 的变化率（%）
    
    参数:
        df: 包含 ['open','high','low','close'] 列的 DataFrame
    返回:
        新的 DataFrame（复制一份，不修改原始 df）
    """
    df = df.copy()

    eps = (1e-9) # 防止异常值和极端情况
    # 1. 当前 open 相对前一根 open 的变化率
    df['price_change_pct'] = (df['open'] / (df['open'].shift(1)+eps) - 1.0)
    df['number_of_trades_pct'] = (df['number_of_trades'] / (df['number_of_trades'].shift(1)+eps) - 1.0)
    df['quote_asset_volume_pct'] = (df['quote_asset_volume'] / (df['quote_asset_volume'].shift(1)+eps) - 1.0)
    df['volume_pct'] = (df['volume'] / (df['volume'].shift(1)+eps) - 1.0)
    # 平均成交价
    df['avg_price'] = df['quote_asset_volume'] / ((df['volume'] )+eps)
    df['avg_price_pct'] = (df['avg_price_pct'] / (df['avg_price_pct'].shift(1)+eps) - 1.0)
    # 主动买单占比
    df['taker_base_share'] = df['taker_buy_base_volume'] / ((df['volume'] )+eps)
    df['taker_quote_share'] = df['taker_buy_quote_volume'] / ((df['quote_asset_volume'] )+eps)

    # 2. 基于当前 open 计算 high/low/close 的百分比变化
    df['high_pct'] = (df['high'] / df['open'])
    df['low_pct']  = (df['low']  / df['open'])
    df['close_pct']= (df['close']/ df['open'])

    return df

