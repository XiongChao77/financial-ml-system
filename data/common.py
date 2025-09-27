import logging,math
import pandas as pd
import numpy as np
#define model
candlestick_num = 120
predict_num = 16
change_rate = 0.006 # 0.2%
label_decrease = 0
# label_decrease_weak =1 
label_ignore = 1
# label_increase_weak = 3
label_increase = 2
# origin_data = "klines15m.csv"
origin_data = "data/BTCUSDT_15m.csv"
train_data = "data/train_data.csv"
test_data = "data/test_data.csv"
log_level = logging.INFO

# ====== 你可以按需要修改的默认特征列（9维）======
#只使用无量纲特征，让模型学习形态
DEFAULT_FEATURES = [
"taker_base_share", "taker_quote_share",
"price_change_pct","number_of_trades_pct","quote_asset_volume_pct","volume_pct",
"high_pct","low_pct","close_pct","EMA_25W_SLOPE_REG_4W_N"
]
# DEFAULT_FEATURES = [
#     "open","high","low","close","volume","taker_buy_base_volume","taker_buy_quote_volume", "quote_asset_volume", "number_of_trades" ,
#     "MACD_DIF","MACD_DEA","MACD", "SMA_5D","SMA_10D","SMA_10D","SMA_20D"
# ]
#"MACD_DIF","MACD_DEA","MACD"
#EMA_7W,EMA_7W_SLOPE_REG_4W,EMA_7W_SLOPE_REG_4W_N,EMA_25W,EMA_25W_SLOPE_REG_4W,EMA_25W_SLOPE_REG_4W_N 
# , "RSI_14","KDJ_K","KDJ_D","KDJ_J"
# SMA_5D,SMA_10D,SMA_20D


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    新增以下无量纲化特征：
    - price_change_pct: 当前 open 相对前一根 open 的变化率（%）
    - high_pct, low_pct, close_pct: 当前 high/low/close 相对 open 的变化率（%）
    
    参数:
        df: 包含 ['open','high','low','close'] 列的 DataFrame
    返回:
        新的 DataFrame（复制一份，不修改原始 df）
    """
    df = df.copy()

    eps = (1e-9) # 防止异常值和极端情况
    # 1. 当前 open 相对前一根 open 的变化率
    df['price_change_pct'] = (df['open'] / (df['open'].shift(1)+eps) - 1.0)
    df['number_of_trades_pct'] = (df['number_of_trades'] / (df['number_of_trades'].shift(1)+eps) - 1.0)
    df['quote_asset_volume_pct'] = (df['quote_asset_volume'] / (df['quote_asset_volume'].shift(1)+eps) - 1.0)
    df['volume_pct'] = (df['volume'] / (df['volume'].shift(1)+eps) - 1.0)
    # 平均成交价
    df['avg_price'] = df['quote_asset_volume'] / ((df['volume'] )+eps)
    # 主动买单占比
    df['taker_base_share'] = df['taker_buy_base_volume'] / ((df['volume'] )+eps)
    df['taker_quote_share'] = df['taker_buy_quote_volume'] / ((df['quote_asset_volume'] )+eps)

    # 2. 基于当前 open 计算 high/low/close 的百分比变化
    df['high_pct'] = (df['high'] / df['open'])
    df['low_pct']  = (df['low']  / df['open'])
    df['close_pct']= (df['close']/ df['open'])

    return df

def add_macd(df: pd.DataFrame,
             fast: int = 12,
             slow: int = 26,
             signal: int = 9,
             ma_windows=(9,25)) -> pd.DataFrame:
    """
    直接在原数据上添加 MACD 与多条均线（同时包含 SMA 与 EMA）。
    生成列：
      - MACD_DIF, MACD_DEA, MACD
      - MA_{w}  （简单移动均线）
      - EMA_{w} （指数移动均线，严格：窗口未满置 NaN）
    """
    if 'close' not in df.columns:
        raise ValueError("缺少列 close，无法计算 MACD/均线")

    out = df.copy()
    close = out['close'].astype(float)

    # ---- MACD ----
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = 2 * (dif - dea)

    out['MACD_DIF'] = dif
    out['MACD_DEA'] = dea
    out['MACD'] = macd

    # ---- 均线（SMA + EMA）----
    cnt = close.expanding(min_periods=1).count()  # 用于严格化 EMA
    for w in ma_windows:
        w = int(w)
        if w <= 0:
            continue
        # SMA
        out[f"MA_{w}"] = close.rolling(window=w, min_periods=w).mean()
        # # EMA（严格：窗口未满置 NaN，便于与 SMA 一致）
        # ema_w = close.ewm(span=w, adjust=False).mean()
        # out[f"EMA_{w}"] = ema_w.where(cnt >= w, np.nan)

    return out


import numpy as np
import pandas as pd

def add_weekly_mas(
    df,
    # ---- 周线均线 ----
    weeks=(7, 25),
    # ---- 日线均线（新增，可配置）----
    days=(5, 10, 20),
    price_col='close',
    method='sma',          # 'sma' 或 'ema'
    strict=True,           # 严格型：窗口未满为 NaN；宽松型：尽早给值
    kline_col='open_time_dt_utc',
    # ---- 斜率（仅对周均线，保持原行为）----
    add_slope=True,        # 是否为每条周均线计算斜率
    slope_method='diff',   # 'diff' 或 'reg'
    slope_weeks=2,         # 斜率回看窗口（单位：周）
    normalize=True         # 是否输出无量纲斜率（斜率 / MA）
):
    """
    基于时间列自动推断 1 根K线的时间长度，计算：
      * 周均线（SMA/EMA）：如 7W / 25W，并可选计算周均线斜率
      * 日均线（SMA/EMA）：如 5D / 10D / 20D（新增，天数可参数化）

    - 时间列支持：字符串/DatetimeIndex/带/不带时区；若为数字则按毫秒时间戳处理
    - 周均线列名：SMA_{w}W / EMA_{w}W
    - 日均线列名：SMA_{d}D / EMA_{d}D
    - 仅对“周均线”计算斜率，保持原函数语义不变
    """
    if kline_col not in df.columns:
        raise ValueError(f"缺少 {kline_col} 列")
    if price_col not in df.columns:
        raise ValueError(f"缺少 {price_col} 列")

    # 1) 标准化时间为 datetime64[ns, UTC]
    tmp = df[[kline_col]].dropna().copy()
    if len(tmp) < 3:
        raise ValueError("数据过少，无法稳定判断K线周期")

    s = tmp[kline_col]
    if np.issubdtype(s.dtype, np.number):
        dt = pd.to_datetime(s.astype('int64'), unit='ms', utc=True)
    else:
        dt = pd.to_datetime(s, utc=True, errors='coerce')
        if dt.isna().any():
            bad_n = int(dt.isna().sum())
            raise ValueError(f"时间列存在无法解析的值（{bad_n} 条），请清洗后再试")

    # 2) 用相邻时间差的中位数估算单K线周期（纳秒）
    dt_sorted = dt.sort_values()
    diffs_ns = np.diff(dt_sorted.astype('int64').to_numpy()).astype(float)  # ns
    kline_ns = np.median(diffs_ns)
    if not np.isfinite(kline_ns) or kline_ns <= 0:
        raise ValueError("检测到非正或无效的K线周期，请检查时间数据")

    one_day_ns  = 24 * 60 * 60 * 1e9
    one_week_ns = 7  * 24 * 60 * 60 * 1e9

    klines_per_day  = max(int(round(one_day_ns  / kline_ns)), 1)
    klines_per_week = max(int(round(one_week_ns / kline_ns)), 1)

    out = df.copy()
    close = out[price_col].astype(float)

    m = method.lower()
    if m not in ('sma', 'ema'):
        raise ValueError("method 只能为 'sma' 或 'ema'")

    # ---- 通用均线计算子函数 ----
    def _ma(series: pd.Series, window: int) -> pd.Series:
        if m == 'sma':
            min_p = window if strict else 1
            return series.rolling(window=window, min_periods=min_p).mean()
        else:
            ema = series.ewm(span=window, adjust=False).mean()
            if strict:
                counts = series.expanding(min_periods=1).count()
                ema = ema.where(counts >= window, np.nan)
            return ema

    # ---- 斜率函数（仅用于周均线）----
    def _slope_diff(series: pd.Series, steps: int) -> pd.Series:
        """用 steps 根（≈ slope_weeks 周）差分近似斜率；返回每根K线的平均变化量"""
        if steps <= 0:
            return pd.Series(np.nan, index=series.index)
        return (series - series.shift(steps)) / steps

    def _slope_reg(series: pd.Series, steps: int) -> pd.Series:
        """
        在长度=steps 的滚动窗口上对 MA 做线性回归，返回斜率（每根K线的平均变化量）。
        设窗口内 x = 0..steps-1, y = MA，斜率 = [∑(x*y) - n*x_mean*y_mean] / ∑(x-x_mean)^2
        注意：∑(x*y) 直接用 y.shift(k)*x_k 逐项叠加得到“窗口加权和”，**不要再做 rolling**。
        """
        if steps <= 1:
            return pd.Series(np.nan, index=series.index)

        x = np.arange(steps, dtype=float)
        n = float(steps)
        x_mean = (steps - 1) / 2.0
        var_x = ((x - x_mean) ** 2).sum()  # 常数 > 0

        y = series

        # ∑y（窗口和）
        sum_y = y.rolling(steps).sum()
        y_mean = sum_y / n

        # ∑(x*y)（窗口内加权和）：右端对齐的滑动窗口值
        sum_xy = pd.Series(0.0, index=y.index)
        for k in range(steps):
            sum_xy = sum_xy.add(y.shift(k) * x[k], fill_value=0.0)

        slope = (sum_xy - n * x_mean * y_mean) / var_x
        return slope

    # 3) 计算 —— 日线均线（新增）
    for d in days:
        d = int(d)
        if d <= 0:
            continue
        window_d = max(klines_per_day * d, 1)
        ma_d = _ma(close, window_d)
        if m == 'sma':
            ma_col = f"SMA_{d}D"
        else:
            ma_col = f"EMA_{d}D"
        out[ma_col] = ma_d

    # 4) 计算 —— 周线均线（原有）
    for w in weeks:
        w = int(w)
        if w <= 0:
            continue
        window_w = max(klines_per_week * w, 1)
        ma_w = _ma(close, window_w)
        ma_w_col = f"{'SMA' if m=='sma' else 'EMA'}_{w}W"
        out[ma_w_col] = ma_w

        # 周线斜率（保持原行为）
        if add_slope:
            steps = max(int(round(klines_per_week * float(slope_weeks))), 1)

            if slope_method == 'diff':
                slope = _slope_diff(ma_w, steps)
                slope_col = f"{ma_w_col}_SLOPE_{slope_weeks}W"
            elif slope_method == 'reg':
                slope = _slope_reg(ma_w, steps)
                slope_col = f"{ma_w_col}_SLOPE_REG_{slope_weeks}W"
            else:
                raise ValueError("slope_method 只能为 'diff' 或 'reg'")

            if strict:
                valid_ma = ma_w.notna()
                valid_slope = ma_w.rolling(steps).count() >= steps
                slope = slope.where(valid_ma & valid_slope, np.nan)

            out[slope_col] = slope

            if normalize:
                norm_col = f"{slope_col}_N"  # 归一化斜率
                denom = ma_w.replace(0, np.nan)
                out[norm_col] = slope / denom

    return out


def add_rsi(
    df: pd.DataFrame,
    period: int = 14,
    price_col: str = "close",
    strict: bool = True,
    prefix: str = "RSI"
) -> pd.DataFrame:
    """
    Wilder 风格 RSI（使用 EWM, alpha=1/period）
    输出列：{prefix}_{period}
    """
    if price_col not in df.columns:
        raise ValueError(f"缺少列 {price_col}")

    out = df.copy()
    close = out[price_col].astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder 平滑：alpha=1/period
    avg_gain = gain.ewm(alpha=1/float(period), adjust=False,
                        min_periods=period if strict else 1).mean()
    avg_loss = loss.ewm(alpha=1/float(period), adjust=False,
                        min_periods=period if strict else 1).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))

    col = f"{prefix}_{period}"
    out[col] = rsi

    # 严格模式：窗口未满置 NaN
    if strict:
        valid = close.expanding().count() >= period
        out[col] = out[col].where(valid, np.nan)

    return out


def add_kdj(
    df: pd.DataFrame,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    strict: bool = True,
    prefix: str = "KDJ"
) -> pd.DataFrame:
    """
    经典 KDJ：
      RSV = (C - LLV(n)) / (HHV(n) - LLV(n)) * 100
      K = EMA(RSV, alpha=1/m1)
      D = EMA(K,   alpha=1/m2)
      J = 3*K - 2*D
    输出列：{prefix}_K, {prefix}_D, {prefix}_J
    """
    for c in (high_col, low_col, close_col):
        if c not in df.columns:
            raise ValueError(f"缺少列 {c}")

    out = df.copy()
    high = out[high_col].astype(float)
    low = out[low_col].astype(float)
    close = out[close_col].astype(float)

    llv = low.rolling(window=n, min_periods=n if strict else 1).min()
    hhv = high.rolling(window=n, min_periods=n if strict else 1).max()

    rsv = (close - llv) / (hhv - llv + 1e-12) * 100.0

    # 用 EWM 实现等价的递推平滑（alpha=1/m）
    K = rsv.ewm(alpha=1/float(m1), adjust=False,
                min_periods=m1 if strict else 1).mean()
    D = K.ewm(alpha=1/float(m2), adjust=False,
              min_periods=m2 if strict else 1).mean()
    J = 3 * K - 2 * D

    k_col, d_col, j_col = f"{prefix}_K", f"{prefix}_D", f"{prefix}_J"
    out[k_col], out[d_col], out[j_col] = K, D, J

    # 严格模式：必须至少有 n 根形成 RSV，且各自平滑窗口就绪
    if strict:
        valid_rsv = close.expanding().count() >= n
        out[k_col] = out[k_col].where(valid_rsv, np.nan)
        out[d_col] = out[d_col].where(valid_rsv, np.nan)
        out[j_col] = out[j_col].where(valid_rsv, np.nan)

    return out


#####find inf
def find_float32_issues(df):
    import numpy as np, pandas as pd
    num = df.select_dtypes(include=[np.number])
    X = num.to_numpy()
    fmax = np.finfo(np.float32).max
    masks = {
        "nan": np.isnan(X),
        "pos_inf": np.isposinf(X),
        "neg_inf": np.isneginf(X),
        "abs>float32_max": np.abs(X) > fmax,
    }
    report = pd.DataFrame({k: v.sum(axis=0) for k, v in masks.items()}, index=num.columns)
    # 位置明细
    bad_mask = np.zeros_like(X, dtype=bool)
    for m in masks.values(): bad_mask |= m
    rows, cols = np.where(bad_mask)
    for r, c in zip(rows, cols):
        print(df.iloc[r].iat[c])
    positions = [(num.index[r], num.columns[c], df.loc[num.index[r], num.columns[c]]) for r, c in zip(rows, cols)]
    return report.sort_values(list(report.columns), ascending=False), positions