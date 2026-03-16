import pandas as pd
from datetime import datetime, timezone
import os

def add_timestamp_first_column(in_csv, out_csv, chunksize=200000):
    """
    Read CSV in chunks, convert open_time_date_utc to a Unix timestamp, and insert it as the first column.
    """
    first = True  # whether to write header

    for chunk in pd.read_csv(in_csv, chunksize=chunksize):
        # datetime (UTC) -> timestamp (ms)
        ts = pd.to_datetime(chunk["open_time_date_utc"], utc=True).astype("int64") // 10**6
        chunk.insert(0, "open_time_ts", ts)  # insert at column 0

        # Write
        chunk.to_csv(out_csv, mode="a", index=False, header=first)
        first = False

    print(f"✅ Generated: {out_csv}")

data_dir = "/home/chao/work/Quant/data_process/data"
out_csv = os.path.join(data_dir, f"BTCUSDT_1d.csv")
new_csv = os.path.join(data_dir, f"BTCUSDT_1d_new.csv")
add_timestamp_first_column(out_csv,new_csv)