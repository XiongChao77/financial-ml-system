import pandas as pd
import numpy as np
from pathlib import Path


import pandas as pd
import numpy as np
from pathlib import Path


def _calc_window_noise(
    sub: pd.DataFrame,
    price_cols=("open", "high", "low", "close"),
    eps: float = 1e-12,
) -> dict:
    """
    Overall noise of one fixed window.
    Note: this is window level, not per bar and not over the whole sample.
    """

    o, h, l, c = price_cols

    open_arr = sub[o].to_numpy(dtype=float)
    high_arr = sub[h].to_numpy(dtype=float)
    low_arr = sub[l].to_numpy(dtype=float)
    close_arr = sub[c].to_numpy(dtype=float)

    # =========================
    # 1. close-to-close path noise
    # =========================
    close_net_move = abs(close_arr[-1] - close_arr[0])
    close_path_length = np.sum(np.abs(np.diff(close_arr)))

    close_efficiency = close_net_move / (close_path_length + eps)
    close_noise = 1.0 - close_efficiency

    # =========================
    # 2. OHLC intrabar path noise
    # =========================
    # Two possible paths:
    # open -> high -> low -> close
    path_high_first = (
        np.abs(high_arr - open_arr)
        + np.abs(high_arr - low_arr)
        + np.abs(close_arr - low_arr)
    )

    # open -> low -> high -> close
    path_low_first = (
        np.abs(open_arr - low_arr)
        + np.abs(high_arr - low_arr)
        + np.abs(high_arr - close_arr)
    )

    # Use the shorter path as the conservative estimate
    intrabar_path = np.minimum(path_high_first, path_low_first)

    # Gap between bars
    interbar_gap = np.abs(open_arr[1:] - close_arr[:-1])

    ohlc_path_length = np.sum(intrabar_path) + np.sum(interbar_gap)
    ohlc_net_move = abs(close_arr[-1] - open_arr[0])

    ohlc_efficiency = ohlc_net_move / (ohlc_path_length + eps)
    ohlc_noise = 1.0 - ohlc_efficiency

    # =========================
    # 3. Move intensity, a supporting metric
    # =========================
    log_close = np.log(np.where(close_arr > 0, close_arr, np.nan))
    abs_log_return_path = np.nansum(np.abs(np.diff(log_close)))

    return {
        "close_efficiency": close_efficiency,
        "close_noise": close_noise,
        "ohlc_efficiency": ohlc_efficiency,
        "ohlc_noise": ohlc_noise,
        "abs_log_return_path": abs_log_return_path,
        "window_return": close_arr[-1] / open_arr[0] - 1.0,
    }


def calculate_windowed_noise(
    df: pd.DataFrame,
    window: int = 256,
    step: int | None = None,
    price_cols=("open", "high", "low", "close"),
    timestamp_col="timestamp",
) -> pd.DataFrame:
    """
    Fixed window noise over the whole dataset.

    window:
        How many bars per window.
        For M15 for example:
            64  = 16 hours
            96  = 1 day on a 24/7 market
            256 = about 2.7 days on a 24/7 market
            1024 = about 10.7 days on a 24/7 market

    step:
        Step between windows.
        step=window means non overlapping windows.
        step=window//4 means overlapping windows.
    """

    if step is None:
        step = window

    df = df.copy()

    if timestamp_col in df.columns:
        df = df.sort_values(timestamp_col)

    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(price_cols)).reset_index(drop=True)

    results = []

    for start in range(0, len(df) - window + 1, step):
        end = start + window
        sub = df.iloc[start:end]

        res = _calc_window_noise(sub, price_cols=price_cols)

        res["start_idx"] = start
        res["end_idx"] = end - 1
        res["rows"] = len(sub)

        if timestamp_col in df.columns:
            res["start_time"] = sub[timestamp_col].iloc[0]
            res["end_time"] = sub[timestamp_col].iloc[-1]

        results.append(res)

    return pd.DataFrame(results)

def compare_asset_noise(
    file_a,
    file_b,
    name_a="Asset_A",
    name_b="Asset_B",
    window=256,
    step=None,
):
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)

    noise_a = calculate_windowed_noise(df_a, window=window, step=step)
    noise_b = calculate_windowed_noise(df_b, window=window, step=step)

    noise_a["asset"] = name_a
    noise_b["asset"] = name_b

    all_noise = pd.concat([noise_a, noise_b], ignore_index=True)

    summary = all_noise.groupby("asset").agg(
        windows=("ohlc_noise", "count"),

        close_noise_mean=("close_noise", "mean"),
        close_noise_median=("close_noise", "median"),
        close_noise_p75=("close_noise", lambda x: x.quantile(0.75)),
        close_noise_p90=("close_noise", lambda x: x.quantile(0.90)),

        ohlc_noise_mean=("ohlc_noise", "mean"),
        ohlc_noise_median=("ohlc_noise", "median"),
        ohlc_noise_p75=("ohlc_noise", lambda x: x.quantile(0.75)),
        ohlc_noise_p90=("ohlc_noise", lambda x: x.quantile(0.90)),

        close_efficiency_mean=("close_efficiency", "mean"),
        ohlc_efficiency_mean=("ohlc_efficiency", "mean"),

        abs_log_return_path_mean=("abs_log_return_path", "mean"),
        window_return_abs_mean=("window_return", lambda x: x.abs().mean()),
    )

    return summary, all_noise


file_com1 = r"/home/chao/work/QuantData/Cryptocurrency/binance_public_data/um/BTCUSDT_15m.csv"
file_com2 = r"/home/chao/work/QuantData/Forex/dukascopy/spot/XAUUSD_15m.csv"

summary = compare_asset_noise(
    file_com1,
    file_com2,
    name_a=file_com1.split('/')[-1],
    name_b=file_com2.split('/')[-1],
    window=256,
    step=256,
)

print(summary)