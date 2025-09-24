import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime,os
from common import *

current_work_dir = os.path.dirname(__file__) 
data_path = os.path.join(current_work_dir, 'klines15m.csv')

#**********column info: Kline_open_time,Open_price,High_price,Low_price,Close_price,Volume,Kline_close_time,Quote_asset_volume,Number_of_trades,buy_base_volume,buy_quote_volume,ignore
df = pd.read_csv(data_path)
print(df.shape)
print(df.columns)
df.drop(['Kline_open_time','Kline_close_time'],axis=1, inplace=True)
df.rename({'ignore':'label'},axis=1, inplace=True) # 0: increase >0.2%, 1:ignore ,2: decrease>0.2%
df = df.iloc[::-1].reset_index(drop=True)
print(df.columns)
print(df.head())

def attach_label(df):
    """
    依据未来收益率分5类：
      - pct < -change_rate                  -> label_decrease
      - -change_rate <= pct < -weak_change  -> label_decrease_weak
      - -weak_change <= pct <  weak_change  -> label_ignore     (小波动/噪声)
      -  weak_change <= pct <  change_rate  -> label_increase_weak
      -  pct >= change_rate                 -> label_increase
    另外：前 candlestick_num 行因历史不足，统一置为 label_ignore；
          末尾 predict_num 行被丢弃（因未来价格不可用）。
    """
    assert 'Close_price' in df.columns, "缺少列 Close_price"
    assert predict_num > 0, "predict_num 必须 > 0"
    assert change_rate > 0, "change_rate 必须 > 0"

    future_close = df['Close_price'].shift(-predict_num)
    pct = (future_close - df['Close_price']) / df['Close_price']

    weak_change = change_rate / 5.0

    df = df.copy()
    # df['label'] = np.select(
    #     [
    #         (pct < -change_rate),
    #         (pct >= -change_rate) & (pct < -weak_change),
    #         (pct >=  weak_change) & (pct <  change_rate),
    #         (pct >=  change_rate)
    #     ],
    #     [
    #         label_decrease,
    #         label_decrease_weak,
    #         label_increase_weak,
    #         label_increase
    #     ],
    #     default=label_ignore   # 覆盖 [-weak_change, weak_change) 以及任何 NaN/未命中
    # )

    df['label'] = np.select(
        [
            (pct < -change_rate),
            (pct >=  change_rate)
        ],
        [
            label_decrease,
            label_increase
        ],
        default=label_ignore   # 覆盖 [-weak_change, weak_change) 以及任何 NaN/未命中
    )

    # 前 candlestick_num 行历史不足，强制忽略
    df.iloc[:candlestick_num, df.columns.get_loc('label')] = label_ignore

    # 删尾（未来价格不可用的部分）
    if predict_num > 0:
        df = df.iloc[:-predict_num].reset_index(drop=True)

    # 可选：确保类型一致
    df['label'] = df['label'].astype(int)

    return df

def add_macd(df, fast=12, slow=26, signal=9):
    """
    在数据中加入MACD指标：
      - MACD_DIF: EMA(fast) - EMA(slow)
      - MACD_DEA: DIF的EMA(signal)
      - MACD:     2 * (DIF - DEA)
    计算基于 Close_price 列。
    """
    if 'Close_price' not in df.columns:
        raise ValueError("缺少列 Close_price，无法计算MACD")
    close = df['Close_price']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = 2 * (dif - dea)
    df['MACD_DIF'] = dif
    df['MACD_DEA'] = dea
    df['MACD'] = macd
    return df

# 计算MACD指标
df = add_macd(df)
df = attach_label(df)
print(len(df))
print(df.head())
df.to_csv(os.path.join(current_work_dir, "data.csv"), index=False, encoding="utf-8")
