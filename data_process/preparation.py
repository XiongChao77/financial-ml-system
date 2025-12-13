import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime,os,sys
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process.common import *

def main():
    df = pd.read_csv(origin_data_path)
    attach_attr(df)
    attach_label(df)
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
    
    # 1. drop unstable data in the early stage    There will be data drop in attach_attr(SMA_W)
    # drop_ratio = 0.02
    # start_idx = math.ceil(len(df) * drop_ratio)
    # df = df.iloc[start_idx:].reset_index(drop=True)

    # 2. 时间序列切分（80% train / 20% test）
    train_ratio = 0.8
    split_idx = math.floor(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]
    # 写入文件
    os.makedirs(TEMPORARY_DIR , exist_ok=True)
    train_df.to_csv(train_data_path, index=False, encoding="utf-8")
    test_df.to_csv(test_data_path, index=False, encoding="utf-8")
    return df

if __name__ == "__main__":
#**********column info: open_time_utc,open,high,low,close,volume,close_time_utc,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,ignore
    main()