import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime,os
from common import *

current_work_dir = os.path.dirname(__file__) 
data_path = os.path.join(current_work_dir, origin_data)

#**********column info: open_time_dt_utc,open,high,low,close,volume,close_time_dt_utc,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,ignore
df = pd.read_csv(data_path)
print(df.shape)
print(df.columns)
# df.drop(['open_time_dt_utc','close_time_dt_utc'],axis=1, inplace=True)
df.rename({'ignore':'label'},axis=1, inplace=True) # 0: increase >0.2%, 1:ignore ,2: decrease>0.2%
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
    assert 'close' in df.columns, "缺少列 close"
    assert predict_num > 0, "predict_num 必须 > 0"
    assert change_rate > 0, "change_rate 必须 > 0"

    future_close = df['close'].shift(-predict_num)
    pct = (future_close - df['close']) / df['close']

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

df = add_relative_features(df)
# 计算MACD指标
df = add_macd(df)
# # 计算周线
df = add_weekly_mas(
    df, weeks=(7, 25),
    method='sma',
    add_slope=True,
    slope_method='reg',
    slope_weeks=4,   # 用4周窗口拟合斜率
    normalize=True
)
# df = add_rsi(df, period=14, price_col="close", strict=True)     # 生成列：RSI_14
# df = add_kdj(df, n=9, m1=3, m2=3, strict=True)                  # 生成列：KDJ_K, KDJ_D, KDJ_J
df = attach_label(df)
print(len(df))
print(df.head())

# 2. 丢弃任意列为 NaN 的行
df = df.dropna(how='any').copy()

print(f"数据行数: {len(df)}")
print(f"最早时间点: {df['open_time_dt_utc'].iloc[0]}")

# 计算切分点
split_idx = int(len(df) * 0.8)
# 切分数据
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]
# 写入文件
train_df.to_csv(os.path.join(current_work_dir,'data', "train_data.csv"), index=False, encoding="utf-8")
test_df.to_csv(os.path.join(current_work_dir,'data', "test_data.csv"), index=False, encoding="utf-8")
