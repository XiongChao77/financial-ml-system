import pandas as pd
import os
import logging
import datetime,os,sys, re, math, json, logging
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_pandas_freq(interval_str: str) -> str:
    """
    Converts crypto interval strings to Pandas frequency strings.
    Example: '1m' -> '1min', '1h' -> '1H'
    """
    unit = interval_str[-1]
    value = interval_str[:-1]
    mapping = {'s': 'S', 'm': 'min', 'h': 'H', 'd': 'D', 'w': 'W', 'M': 'ME'}
    return f"{value}{mapping.get(unit, 'min')}"

def resample_klines(df: pd.DataFrame, target_interval: str, offset: str = None) -> pd.DataFrame:
    """
    Aggregates K-lines with an optional offset.
    :param target_interval: The bin size (e.g., '10m')
    :param offset: The shift (e.g., '1min' for 1, 11, 21... or '2min' for 2, 12, 22...)
    """
    # 1. Ensure time index
    df['open_time_date_utc'] = pd.to_datetime(df['open_time_date_utc'])
    df = df.set_index('open_time_date_utc')

    # 2. Define Aggregation Logic
    agg_dict = {
        'open_time_ms_utc': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'number_of_trades': 'sum',
        'close_time_ms_utc': 'last',
        'quote_asset_volume': 'sum',
        'taker_buy_base_volume': 'sum',
        'taker_buy_quote_volume': 'sum'
    }

    freq = get_pandas_freq(target_interval)
    
    # 3. Handle Offset
    # If user provides '1m', we convert it to '1min' for pandas
    pd_offset = get_pandas_freq(offset) if offset else None
    
    # Execute resampling with the offset
    resampled = df.resample(freq, offset=pd_offset).agg(agg_dict)

    # Clean up empty bins
    resampled.dropna(subset=['open'], inplace=True)
    
    return resampled.reset_index()

def run_resampling_task(tasks: list):
    """
    Main task runner.
    :param tasks: List of tuples (target_interval, offset_string, label)
    """
    source_file = common.origin_data_path
    if not os.path.exists(source_file):
        logger.error(f"❌ Source file not found: {source_file}")
        return

    logger.info(f"📂 Reading source: {source_file}")
    df_raw = pd.read_csv(source_file)

    for interval, offset, label in tasks:
        offset_info = f" with offset {offset}" if offset else ""
        logger.info(f"⏳ Generating {interval}{offset_info} ({label})...")
        
        df_resampled = resample_klines(df_raw.copy(), interval, offset)
        
        # File naming: include the label to distinguish different offsets (e.g., ETHUSDT_10m_off1.csv)
        suffix = f"_{label}" if label else ""
        output_filename = f"{common.CommonDefine.symbol}_{interval}{suffix}.csv"
        output_path = os.path.join(common.PROJECT_DATA_DIR, output_filename)
        
        df_resampled.to_csv(output_path, index=False)
        logger.info(f"✅ Saved: {output_path}")

if __name__ == "__main__":
    # Specify tasks: (Target Interval, Offset, Custom Suffix/Label)
    # Example: 10m bins starting at minute 1 (1, 11, 21...) and minute 2 (2, 12, 22...)
    resample_tasks = [
        ('10m', '1min', 'offset1'),
        # ('10m', '2min', 'offset2'),
        # ('5m', None, 'standard'),
    ]
    
    run_resampling_task(resample_tasks)