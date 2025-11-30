import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime,os,sys
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process.common import *

def main():
    df = pd.read_csv(origin_data_path)
    df = attach_attr(df)
    df = attach_label(df)
    
    # 计算切分点
    split_idx = int(len(df) * 0.8)
    # 切分数据
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    # 写入文件
    if not os.path.exists(DATA_PROCESS_OUT_DIR): os.makedirs(DATA_PROCESS_OUT_DIR)
    train_df.to_csv(train_data_path, index=False, encoding="utf-8")
    test_df.to_csv(test_data_path, index=False, encoding="utf-8")
    return df

if __name__ == "__main__":
#**********column info: open_time_dt_utc,open,high,low,close,volume,close_time_dt_utc,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,ignore
    main()