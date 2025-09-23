import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime,os
from common import *

current_work_dir = os.path.dirname(__file__) 
data_path = os.path.join(current_work_dir, 'klines15m.csv')

df = pd.read_csv(data_path)
print(df.shape)
print(df.columns)
df.drop(['Kline_open_time','Kline_close_time'],axis=1, inplace=True)
df.rename({'ignore':'label'},axis=1, inplace=True) # 0: increase >0.2%, 1:ignore ,2: decrease>0.2%
df = df.iloc[::-1].reset_index(drop=True)
print(df.columns)
print(df.head())

def attach_label(df):
    # 未来价格（向后平移）
    future_close = df['Close_price'].shift(-predict_num)
    pct = (future_close - df['Close_price']) / df['Close_price']

    df['label'] = np.select(
        [pct < -change_rate, pct > change_rate],
        [0, 2],
        default=1
    )
    df.loc[:candlestick_num, 'label'] = 1
    # 删头尾窗口
    df = df.iloc[:-predict_num].reset_index(drop=True)
    return df

df = attach_label(df)
print(len(df))
print(df.head())
df.to_csv(os.path.join(current_work_dir, "data.csv"), index=False, encoding="utf-8")