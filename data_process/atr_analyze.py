import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime,os,sys, re, math, json, logging
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common
from data_process.regime_discovery import LabelRegimeAnalyzer

# EPS definition (consistent with project style)
EPS = 1e-9

def plot_doge_vol_variance():
    # 1. Load data
    file = common.origin_data_path
    # 2. Convert interval string to milliseconds
    interval_str = get_interval_from_filename(file)
    interval_ms = get_interval_ms(interval_str)
    
    # 3. Metadata (for downstream usage if needed)
    metadata = {
        "symbol_interval": interval_str,
        "interval_ms": interval_ms,
        "seq_len": common.BaseDefine.predict_num,
        "predict_num": common.BaseDefine.predict_num,
        "vol_multiplier": common.VOL_MULTIPLIER,
        "stop_multiplier_rate": common.STOP_MULTIPLIER_RATE,
    }

    df = pd.read_csv(file)
    
    # Convert time column to datetime and set as index
    if 'open_time_date_utc' in df.columns:
        df['time'] = pd.to_datetime(df['open_time_date_utc'])
        df.set_index('time', inplace=True)
    
    # Cast to float for numeric stability
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    prev_close = close.shift(1)

    # Base volatility (NATR - Normalized ATR)
    # TR = max(H-L, |H-PC|, |L-PC|)
    tr = np.maximum(high - low, 
                    np.maximum((high - prev_close).abs(), 
                               (low - prev_close).abs()))
    
    # Percentage volatility (range / price)
    natr_raw = tr / (close + EPS)
    
    # Windows: short-term intensity vs long-term regime
    short_w = 14
    long_w = 200
    
    # Short-term average volatility (intensity)
    short_natr = natr_raw.rolling(short_w).mean()
    
    # Volatility variance proxy (CV - coefficient of variation)
    # Logic: std(long vol) / mean(long vol)
    long_natr_mean = natr_raw.rolling(long_w).mean()
    long_natr_std = natr_raw.rolling(long_w).std()
    
    # Lower CV -> more uniform long-term volatility (trend-friendly backdrop)
    # Higher CV -> volatility itself is unstable (chaotic regime)
    vol_cv = long_natr_std / (long_natr_mean + EPS)
    
    # Composite score (burst / consistency)
    # Burst ratio: short vol / long mean
    burst_ratio = short_natr / (long_natr_mean + EPS)
    # Higher burst ratio + lower CV (higher 1/CV) -> higher score
    trend_vol_score = burst_ratio * (1.0 / (vol_cv + EPS))

    # Plotting
    plt.style.use('dark_background') # trading-terminal-like theme
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1, 1, 1.5]})
    
    # Panel 0: Price
    axes[0].plot(close, label='DOGE Price', color='#F7931A', alpha=0.9)
    axes[0].set_title('DOGE/USDT Price (Trend Discovery)')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=0.2)

    # Panel 1: Short-term volatility intensity
    axes[1].plot(short_natr * 100, label=f'Short NATR {short_w} (%)', color='cyan')
    axes[1].set_title('Short-term Volatility Intensity')
    axes[1].legend(loc='upper left')
    axes[1].grid(alpha=0.2)

    # Panel 2: Volatility consistency (1/CV); higher => more uniform/predictable
    consistency = 1.0 / (vol_cv + EPS)
    axes[2].plot(consistency, label='Volatility Consistency (1/CV)', color='magenta')
    axes[2].axhline(y=np.nanmedian(consistency), color='white', linestyle='--', alpha=0.5, label='Median')
    axes[2].set_title('Volatility Variance Filter (Higher = More Uniform/Predictable)')
    axes[2].legend(loc='upper left')
    axes[2].grid(alpha=0.2)

    # Panel 3: Final execution reference score
    # Peaks mean "burst under a highly uniform backdrop"
    axes[3].fill_between(trend_vol_score.index, trend_vol_score, 0, 
                         where=(trend_vol_score > np.nanpercentile(trend_vol_score, 75)),
                         color='green', alpha=0.5, label='High Confidence Trend Area')
    axes[3].plot(trend_vol_score, color='lime', linewidth=0.8, alpha=0.3)
    axes[3].set_title('Final Execution Score: Burst / Consistency')
    axes[3].set_yscale('log') # log scale highlights bursts
    axes[3].legend(loc='upper left')
    axes[3].grid(alpha=0.2)

    plt.tight_layout()
    # Save
    output_name = "doge_trend_vol_analysis.png"
    plt.savefig(output_name)
    print(f"✅ Plot finished. Saved to: {output_name}")
    plt.show()


def get_interval_from_filename(path: str) -> str:
    """
    Extract interval string from path (e.g. ETHUSDT_3m.csv -> 3m).
    """
    filename = os.path.basename(path)
    # Match formats like 1s, 15s, 1m, 3m... 1M
    match = re.search(r'_(\d+[smhdwM])\.csv', filename)
    if match:
        return match.group(1)
    return "unknown"

def get_interval_ms(interval_str: str) -> int:
    """
    Convert interval string to milliseconds.
    Supported: 1s, 15s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    # Base units in milliseconds
    units = {
        's': 1000,
        'm': 60 * 1000,
        'h': 60 * 60 * 1000,
        'd': 24 * 60 * 60 * 1000,
        'w': 7 * 24 * 60 * 60 * 1000,
        'M': 30 * 24 * 60 * 60 * 1000  # approximate month as 30 days
    }
    
    # Split number and unit via regex
    match = re.match(r'(\d+)([smhdwM])', interval_str)
    if not match:
        return 0
    
    value, unit = match.groups()
    return int(value) * units[unit]

# Example
plot_doge_vol_variance()
