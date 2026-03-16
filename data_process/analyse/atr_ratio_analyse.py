from pickle import FALSE
from tkinter import TRUE
import pandas as pd
import numpy as np
import datetime, os, sys, re, math, json, logging
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..','..'))
from data_process import common

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def stop_loss_atr(df: pd.DataFrame, length: int) -> pd.Series:
    length = int(length)
    if length < 2:
        length = 2

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    return atr

# df['stop_loss_atr'] = stop_loss_atr(df, length=round(A * T))

def _realized_vol_future(close: pd.Series, T: int) -> pd.Series:
    """
    Future realized volatility over next T bars:
      std of log returns in (t+1 ... t+T)
    """
    T = int(T)
    if T < 2:
        T = 2
    r = np.log(close).diff()
    # shift(-1) aligns returns to start at t+1; rolling(T) over that; then shift back
    return r.shift(-1).rolling(T).std().shift(-(T - 1))


def _realized_range_future(close: pd.Series, T: int) -> pd.Series:
    """
    Future realized range over next T bars:
      max(close[t+1:t+T]) - min(close[t+1:t+T])
    """
    T = int(T)
    if T < 2:
        T = 2
    fmax = close.shift(-1).rolling(T).max().shift(-(T - 1))
    fmin = close.shift(-1).rolling(T).min().shift(-(T - 1))
    return fmax - fmin


def _score_spearman(x: pd.Series, y: pd.Series, min_samples: int = 30) -> float:
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < min_samples:
        return np.nan
    return float(spearmanr(s.iloc[:, 0].values, s.iloc[:, 1].values).correlation)


def _score_mare_scaled(x: pd.Series, y: pd.Series, eps: float = 1e-12, min_samples: int = 30) -> float:
    """
    Median Absolute Relative Error after robust scaling y_hat = c * x,
    where c = median(y / x). Lower is better.
    """
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < min_samples:
        return np.nan
    xx = s.iloc[:, 0].values
    yy = s.iloc[:, 1].values
    c = np.median(yy / (xx + eps))
    yhat = c * xx
    return float(np.median(np.abs(yy - yhat) / (np.abs(yy) + eps)))


def plot_atr_ratio_hold_heatmap(
    df: pd.DataFrame,
    A_list=(0.25, 0.33, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0),
    T_list=(4,6,8,12,16, 20, 24,28, 32, 36,40, 44, 48,52,56 ,60, 64, 80, 96, 128, 160, 176),
    future_kind="vol",        # "vol" -> Realized Vol; "range" -> Realized Range
    score_kind="spearman",    # "spearman" (higher better) or "mare" (lower better)
    use_nonoverlap=True,      # True: sample t=0,T,2T,... to avoid overlap inflation
    min_samples=30,
    title_prefix="Heatmap",
):
    """
    Generate a heatmap: ATR ratio A + hold period T -> consistency score vs realized risk over next T bars.

    - ATR window: L = int(round(A * T)) (must be integer; if <2 then forced to 2)
    - Scoring:
        * spearman: Spearman(ATR_L(t), realized_T(t))
        * mare: median(|realized - c*ATR| / |realized|) where c is a robust scale
    - df must include: ['high','low','close'] (open is optional)
    """
    needed = {"high", "low", "close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}; expected {sorted(needed)}")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # Pre-compute future realized risk (by T)
    future_map = {}
    for T in T_list:
        T_int = int(T)
        if future_kind == "vol":
            future_map[T_int] = _realized_vol_future(close, T_int)
        elif future_kind == "range":
            future_map[T_int] = _realized_range_future(close, T_int)
        else:
            raise ValueError("future_kind must be 'vol' or 'range'")

    scorer = _score_spearman if score_kind == "spearman" else _score_mare_scaled

    # Cache ATR(L) by integer L
    atr_cache = {}

    # Output grid: rows=A, columns=T
    A_list = [float(a) for a in A_list]
    T_list = [int(t) for t in T_list]
    grid = pd.DataFrame(index=A_list, columns=T_list, dtype=float)

    n = len(df)

    for A in A_list:
        for T in T_list:
            L = int(round(A * T))      # must be integer
            if L < 2:
                L = 2

            if L not in atr_cache:
                atr_cache[L] = stop_loss_atr(df, L)

            x = atr_cache[L]
            y = future_map[T]

            if use_nonoverlap:
                idx = np.arange(0, n, T)  # step=T, avoid overlap inflation
                x_sub = x.iloc[idx]
                y_sub = y.iloc[idx]
            else:
                x_sub, y_sub = x, y

            grid.loc[A, T] = scorer(x_sub, y_sub, min_samples=min_samples)

    # Plot (matplotlib defaults)
    plt.figure(figsize=(11, 6))
    arr = grid.values.astype(float)
    im = plt.imshow(arr, aspect="auto")

    plt.xticks(range(len(T_list)), [str(t) for t in T_list], rotation=45, ha="right")
    plt.yticks(range(len(A_list)), [str(a) for a in A_list])

    plt.xlabel("Hold Period T (bars)")
    plt.ylabel("ATR Ratio A  (L = round(A*T), integer)")
    plt.title(f"{title_prefix}: {score_kind} ATR(L) vs future realized_{future_kind}(T)"
              + (" [non-overlap]" if use_nonoverlap else ""))

    plt.colorbar(im)
    plt.tight_layout()
    # plt.show()
    save_path = f"{future_kind}_{score_kind}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved to {save_path}")
    return grid

def main(para = common.BaseDefine, output_dir =common.DATA_OUT_DIR ):
    file = os.path.join(common.PROJECT_DATA_DIR, para.trading_type ,f"{para.symbol}_{para.interval}.csv")
    print(f"using file :{file}")
    # 1. Convert interval string to milliseconds
    interval_ms = common.get_interval_ms(para.interval)
    
    df = pd.read_csv(file)
    # df is loaded and contains ['open','high','low','close'] (at least high/low/close)
    grid1 = plot_atr_ratio_hold_heatmap(df, future_kind="vol", score_kind="spearman")
    grid2 = plot_atr_ratio_hold_heatmap(df, future_kind="range", score_kind="spearman")
    # Or check the error surface (lower is better)
    grid3 = plot_atr_ratio_hold_heatmap(df, future_kind="vol", score_kind="mare")


if __name__ == "__main__":
#**********column info: open_time_date_utc,open,high,low,close,volume,close_time_ms_utc,quote_asset_volume,number_of_trades,taker_buy_base_volume,taker_buy_quote_volume,ignore
    main()