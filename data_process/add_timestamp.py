import pandas as pd
from datetime import datetime, timezone
import os

def add_timestamp_first_column(in_csv, out_csv, chunksize=200000):
    """
    从 CSV 分块读取，将 open_time_dt_utc 转成 Unix timestamp，并插入到第一列。
    """
    first = True  # 是否写表头

    for chunk in pd.read_csv(in_csv, chunksize=chunksize):
        # 转换为 datetime（UTC）→ 转 timestamp（ms级）
        ts = pd.to_datetime(chunk["open_time_dt_utc"], utc=True).astype("int64") // 10**6
        chunk.insert(0, "open_time_ts", ts)  # 在第 0 列插入

        # 写入
        chunk.to_csv(out_csv, mode="a", index=False, header=first)
        first = False

    print(f"✅ 已生成：{out_csv}")

data_dir = "/home/chao/work/Quant/data_process/data"
out_csv = os.path.join(data_dir, f"BTCUSDT_1d.csv")
new_csv = os.path.join(data_dir, f"BTCUSDT_1d_new.csv")
add_timestamp_first_column(out_csv,new_csv)