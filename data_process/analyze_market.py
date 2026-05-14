import pandas as pd
import numpy as np
import scipy.stats as stats
import os, sys
"""
Dissect market data and quantify long/short asymmetries.
"""
# Project imports / data loading
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, '..'))
from data_process import common #

def prepare_market_numeric_columns(df):
    """
    Ensure OHLCV columns are numeric.
    This is a preprocessing step, not analysis.
    """
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    return df

def get_market_anatomy(df):
    """
    Dissect market data and quantify long/short asymmetries.
    """
    # 1. Basic preprocessing
    df['ret'] = df['close'].pct_change()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Define up candles and down candles
    up_candles = df[df['ret'] > 0]
    down_candles = df[df['ret'] < 0]
    
    # ----------------------------------------------------
    # Dimension 1: Magnitude - which side hits harder?
    # ----------------------------------------------------
    avg_up_ret = up_candles['ret'].mean()
    avg_down_ret = down_candles['ret'].mean() # negative
    
    # ----------------------------------------------------
    # Dimension 2: Velocity - how large are the extremes?
    # ----------------------------------------------------
    max_up_candle = up_candles['ret'].max()
    max_down_candle = down_candles['ret'].min()
    
    # ----------------------------------------------------
    # Dimension 3: Volatility - which side is more unstable?
    # ----------------------------------------------------
    # Downside deviation vs upside deviation
    up_vol = up_candles['ret'].std()
    down_vol = down_candles['ret'].std()
    
    # ----------------------------------------------------
    # Dimension 4: Volume - where did the money flow?
    # ----------------------------------------------------
    avg_vol_up = up_candles['volume'].mean()
    avg_vol_down = down_candles['volume'].mean()
    
    # ----------------------------------------------------
    # Dimension 5: Serial correlation - how persistent are streaks?
    # ----------------------------------------------------
    # Simple run-length logic: average consecutive up/down streak length
    price_diff = df['close'].diff()
    df['direction'] = np.sign(price_diff)
    
    # When direction changes, cumsum increments and creates groups
    g = df['direction'].ne(df['direction'].shift()).cumsum()
    streaks = df.groupby(g)['direction'].agg(['mean', 'count']) # mean is direction (1/-1), count is streak length
    
    avg_up_streak = streaks[streaks['mean'] == 1]['count'].mean()
    avg_down_streak = streaks[streaks['mean'] == -1]['count'].mean()

    # ----------------------------------------------------
    # Dimension 6: Extreme-move anatomy (top 5% volatility)
    # ----------------------------------------------------
    # Take the top 5% candles by absolute return
    threshold_top5 = df['ret'].abs().quantile(0.95)
    extreme_df = df[df['ret'].abs() > threshold_top5]
    
    ex_up = extreme_df[extreme_df['ret'] > 0]
    ex_down = extreme_df[extreme_df['ret'] < 0]
    
    print("-" * 70)
    print("-" * 70)
    print("🔥 Extreme-move anatomy (top 5% absolute returns):")
    print(f"   - Threshold: > {threshold_top5*100:.2f}% (absolute return)")
    print(f"   - Up spikes: {len(ex_up)}")
    print(f"   - Down spikes: {len(ex_down)} (ratio: x{len(ex_down)/len(ex_up):.2f})")
    print(f"   - Avg volume (up): {ex_up['volume'].mean():.2f}")
    print(f"   - Avg volume (down): {ex_down['volume'].mean():.2f} (ratio: x{ex_down['volume'].mean()/ex_up['volume'].mean():.2f})")
    print("="*60)

    # ----------------------------------------------------
    # Report output
    # ----------------------------------------------------
    print("="*60)
    print("📊 Market anatomy report (long vs short)")
    print("="*60)
    print(f"{'Metric':<25} | {'📈 Up (Long)':<15} | {'📉 Down (Short)':<15} | {'Diff Ratio':<10}")
    print("-" * 70)
    
    print(f"{'Avg Return (per candle)':<25} | {avg_up_ret*100:.4f}%          | {avg_down_ret*100:.4f}%          | x{abs(avg_down_ret/avg_up_ret):.2f}")
    print(f"{'Max Candle':<25} | {max_up_candle*100:.2f}%           | {max_down_candle*100:.2f}%           | -")
    print(f"{'Volatility':<25} | {up_vol:.5f}          | {down_vol:.5f}          | x{down_vol/up_vol:.2f}")
    print(f"{'Avg Volume':<25} | {avg_vol_up:.2f}          | {avg_vol_down:.2f}          | x{avg_vol_down/avg_vol_up:.2f}")
    print(f"{'Avg Streak':<25} | {avg_up_streak:.2f} bars         | {avg_down_streak:.2f} bars         | -")
    
    # Skew/kurtosis stats
    skew = df['ret'].skew()
    kurt = df['ret'].kurtosis()
    print("-" * 70)
    print("📐 Overall distribution stats:")
    print(f"   - Skewness: {skew:.4f} ({'left-skew (more crashes)' if skew < 0 else 'right-skew (more spikes)'})")
    print(f"   - Kurtosis: {kurt:.4f} (> 3 implies heavy tails / more extreme moves)")
    print("="*60)

def analyze_label_distribution_by_params(
    df_raw,
    seq_len,
    predict_num_list,
    vol_multiplier_list,
    stop_multiplier_rate_list,
):
    """
    Scan label distribution under different labeling parameter combinations.
    Helps understand how the rule partitions the market.
    """
    records = []

    for predict_num in predict_num_list:
        for vol_mul in vol_multiplier_list:
            for stop_rate in stop_multiplier_rate_list:

                df = df_raw.copy()

                common.attach_label(
                    df,
                    seq_len=seq_len,
                    predict_num=predict_num,
                    vol_multiplier=vol_mul,
                    stop_multiplier_rate=stop_rate,
                )

                counts = df["label"].value_counts().to_dict()
                total = len(df)

                record = {
                    "predict_num": predict_num,
                    "vol_multiplier": vol_mul,
                    "stop_multiplier_rate": stop_rate,
                    "total": total,
                    "long_cnt": counts.get(common.Signal.POSITIVE , 0),
                    "short_cnt": counts.get(common.Signal.NEGATIVE, 0),
                    "neutral_cnt": counts.get(common.Signal.NEUTRAL, 0),
                }

                # Ratios
                record["long_pct"] = record["long_cnt"] / total
                record["short_pct"] = record["short_cnt"] / total
                record["neutral_pct"] = record["neutral_cnt"] / total

                # Useful composite metric: active signal ratio
                record["active_pct"] = (
                    record["long_pct"] + record["short_pct"]
                )

                records.append(record)

    return pd.DataFrame(records)

def run_label_sensitivity_analysis(df):
    """
    Analyze label distribution under different parameter ranges.
    """

    vol_multipliers = common.float_range(0.6, 1.2 , 0.05)
    stop_rates = common.float_range(0.3, 0.6, 0.1)

    records = []

    for vol_mul in vol_multipliers:
        for stop_rate in stop_rates:

            df_tmp = df.copy()

            common.attach_label(
                df_tmp,
                seq_len=common.BaseDefine.predict_num,
                predict_num=common.BaseDefine.predict_num,
                vol_multiplier=vol_mul,
                stop_multiplier_rate=stop_rate,
            )

            counts = df_tmp["label"].value_counts().to_dict()
            total = len(df_tmp)

            record = {
                "vol_multiplier": vol_mul,
                "stop_multiplier_rate": stop_rate,

                "total": total,
                "long_cnt": counts.get(common.Signal.POSITIVE , 0),
                "short_cnt": counts.get(common.Signal.NEGATIVE, 0),
                "neutral_cnt": counts.get(common.Signal.NEUTRAL, 0),
            }

            record["long_pct"] = record["long_cnt"] / total
            record["short_pct"] = record["short_cnt"] / total
            record["neutral_pct"] = record["neutral_cnt"] / total
            record["active_pct"] = (
                record["long_pct"] + record["short_pct"]
            )

            records.append(record)

    stats_df = pd.DataFrame(records)

    out_path = os.path.join(
        os.path.dirname(__file__),
        "label_distribution_param_surface.csv",
    )
    stats_df.to_csv(out_path, index=False)

    print("\n📊 Label distribution sensitivity (head):")
    print(stats_df.head())

    return stats_df

if __name__ == "__main__":
    df = pd.read_csv(common.origin_data_path)
    df = df.dropna()

    # df = prepare_market_numeric_columns(df)
    # get_market_anatomy(df)

    run_label_sensitivity_analysis(df)