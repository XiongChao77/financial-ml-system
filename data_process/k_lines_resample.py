# batch_kline_resample.py
# 只依赖：common + 本文件
# 功能：读取 {PROJECT_DATA_DIR}/{symbol}_{interval}.csv
#      批量聚合：DOGEUSDT 1m -> 7m offset 1min；13m offset 0min
# 约束：
#  1) target_freq 必须能被 base_freq 整除
#  2) 检测缺失：bin 内 bar 数不够 或存在 gap -> 丢弃该 bin
#  3) offset 语义采用 pandas：只影响起点相位，后续等间隔自然排布

import os,sys
import re
import numpy as np
import pandas as pd
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common
import common  # 需要 common.PROJECT_DATA_DIR


_FREQ_RE = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$", re.IGNORECASE)
_UNIT_TO_MS = {"s": 1000, "m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 7 * 86_400_000}
_UNIT_TO_PANDAS = {"s": "S", "m": "min", "h": "H", "d": "D", "w": "W"}


def _freq_to_ms(freq: str) -> int:
    m = _FREQ_RE.match(freq)
    if not m:
        raise ValueError(f"Invalid freq: {freq!r}, expected like '1m','7m','1h'...")
    n = int(m.group(1))
    u = m.group(2).lower()
    if n <= 0:
        raise ValueError(f"freq must be positive: {freq!r}")
    return n * _UNIT_TO_MS[u]


def _freq_to_pandas(freq: str) -> str:
    m = _FREQ_RE.match(freq)
    if not m:
        raise ValueError(f"Invalid freq: {freq!r}")
    n = int(m.group(1))
    u = m.group(2).lower()
    return f"{n}{_UNIT_TO_PANDAS[u]}"


def _validate_divisible(base_freq: str, target_freq: str) -> int:
    base_ms = _freq_to_ms(base_freq)
    target_ms = _freq_to_ms(target_freq)
    if target_ms % base_ms != 0:
        raise ValueError(
            f"target_freq must be divisible by base_freq. "
            f"Got base={base_freq}, target={target_freq} "
            f"(base_ms={base_ms}, target_ms={target_ms})"
        )
    return target_ms // base_ms


def resample_klines(
    df: pd.DataFrame,
    *,
    base_freq: str,
    target_freq: str,
    offset: str | None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    适配输入列（Binance风格）：
      open_time_ms_utc, open_time_date_utc, open, high, low, close, volume,
      number_of_trades, close_time_ms_utc, quote_asset_volume,
      taker_buy_base_volume, taker_buy_quote_volume

    输出同风格字段，时间按聚合 bin 边界生成：
      open_time_ms_utc = bin_start_ms
      close_time_ms_utc = bin_end_ms - 1
    """
    expected_count = _validate_divisible(base_freq, target_freq)
    base_ms = _freq_to_ms(base_freq)
    target_ms = _freq_to_ms(target_freq)

    if "open_time_ms_utc" not in df.columns:
        raise ValueError("Missing column: open_time_ms_utc")

    df = df.copy()
    df["open_time_ms_utc"] = df["open_time_ms_utc"].astype("int64")
    idx = pd.to_datetime(df["open_time_ms_utc"], unit="ms", utc=True)
    df = df.set_index(idx).sort_index()

    # gap 标记：相邻 open_time_ms 必须严格等于 base_ms
    diff_ms = df["open_time_ms_utc"].diff()
    df["_gap"] = (diff_ms != base_ms).fillna(False).astype(int)
    df["_cnt"] = 1

    # 聚合列（存在才聚合）
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "number_of_trades": "sum",
        "quote_asset_volume": "sum",
        "taker_buy_base_volume": "sum",
        "taker_buy_quote_volume": "sum",
        "_gap": "sum",
        "_cnt": "sum",
    }
    agg = {k: v for k, v in agg.items() if k in df.columns}

    rs = df.resample(
        rule=_freq_to_pandas(target_freq),
        offset=offset,
        label="left",
        closed="left",
    ).agg(agg)

    if drop_incomplete:
        # bin内bar数量必须完整 + 无gap
        if "_cnt" in rs.columns:
            rs = rs[rs["_cnt"] == expected_count]
        if "_gap" in rs.columns:
            rs = rs[rs["_gap"] == 0]

    # 生成时间字段：bin_start / bin_end-1ms
    bin_start = rs.index  # UTC tz-aware
    bin_start_ms = (bin_start.view("int64") // 10**6).astype("int64")
    bin_end_ms = bin_start_ms + target_ms
    rs["open_time_ms_utc"] = bin_start_ms
    rs["open_time_date_utc"] = bin_start.strftime("%Y-%m-%d %H:%M:%S")
    rs["close_time_ms_utc"] = (bin_end_ms - 1).astype("int64")

    # 清理内部列
    for c in ("_gap", "_cnt"):
        if c in rs.columns:
            rs.drop(columns=[c], inplace=True)

    # 输出列顺序（尽量贴近原始）
    out_cols = [
        "open_time_ms_utc",
        "open_time_date_utc",
        "open", "high", "low", "close",
        "volume",
        "number_of_trades",
        "close_time_ms_utc",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    out_cols = [c for c in out_cols if c in rs.columns]
    rs = rs[out_cols].reset_index(drop=True)
    return rs


def input_path(symbol: str, interval: str) -> str:
    return os.path.join(common.PROJECT_DATA_DIR, f"{symbol}_{interval}.csv")


def output_path(symbol: str, src_interval: str, target_freq: str, offset: str | None) -> str:
    out_dir = os.path.join(common.PROJECT_DATA_DIR)
    os.makedirs(out_dir, exist_ok=True)
    off_tag = "offNone" if offset is None else f"off{offset}"
    off_tag = off_tag.replace(" ", "")
    return os.path.join(out_dir, f"{symbol}_{target_freq}.csv")


def batch_resample(symbol: str, interval: str, targets: list[tuple[str, str | None]]):
    src = input_path(symbol, interval)
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    print(f"[LOAD] {src}")
    df = pd.read_csv(src)

    for target_freq, offset in targets:
        print(f"[RESAMPLE] {symbol} {interval} -> {target_freq}, offset={offset}")
        out_df = resample_klines(
            df,
            base_freq=interval,
            target_freq=target_freq,
            offset=offset,
            drop_incomplete=True,
        )
        out = output_path(symbol, interval, target_freq, offset)
        out_df.to_csv(out, index=False)
        print(f"[SAVE] {out} | rows={len(out_df)}")


if __name__ == "__main__":
    symbol = "DOGEUSDT"
    interval = "1m"
    targets = [
        ("8m", "1min"),
    ]
    batch_resample(symbol, interval, targets)
