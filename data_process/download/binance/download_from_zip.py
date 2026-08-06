#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Binance Vision daily kline ZIP files by date range.

Data source:
    https://data.binance.vision/

Main behavior:
    - Download daily ZIP files and merge them into one CSV.
    - If --start is not specified, automatically fetch the Binance Vision listing
      page and infer the earliest available daily ZIP date for this market/symbol/interval.
    - If --end is not specified, default to today UTC.
    - After merging, sort and de-duplicate by open_time, then check gaps.

Examples:
    # Spot DOGEUSDT 30m, automatically find earliest available day, download to today
    python download_binance_vision_klines_v2.py --symbol DOGEUSDT --interval 30m --overwrite

    # Spot DOGEUSDT 30m, explicit range
    python download_binance_vision_klines_v2.py --symbol DOGEUSDT --interval 30m --start 2021-01-01 --end 2021-12-31 --overwrite

    # Binance USDⓈ-M futures / UM futures
    python download_binance_vision_klines_v2.py --market um --symbol DOGEUSDT --interval 30m --overwrite

Output:
    <out_dir>/<market>/<symbol>/<symbol>_<interval>.csv
"""

import argparse
import csv
import io
import os
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import requests
from requests.adapters import HTTPAdapter


BASE_URL = "https://data.binance.vision"

VALID_MARKETS = {
    "spot": "data/spot/daily/klines",
    "um": "data/futures/um/daily/klines",   # USDⓈ-M futures
    "cm": "data/futures/cm/daily/klines",   # COIN-M futures
}

# Fallback only. Normally --start=None will infer earliest date from the listing page.
FALLBACK_START = {
    "spot": date(2017, 8, 17),
    "um": date(2020, 1, 1),
    "cm": date(2020, 8, 1),
}

OUTPUT_COLUMNS = [
    "open_time_ms_utc",
    "open_time_date_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms_utc",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass
class DownloadStats:
    downloaded_files: int = 0
    missing_files: int = 0
    failed_files: int = 0
    raw_rows: int = 0
    duplicate_rows: int = 0
    final_rows: int = 0


def parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def ms_to_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])

    scales = {
        "s": 1000,
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
    }

    if unit == "M":
        raise ValueError("Monthly interval '1M' is not supported for fixed gap checking.")

    if unit not in scales:
        raise ValueError(f"Unsupported interval: {interval}")

    return value * scales[unit]


def normalize_ts_to_ms(value: str) -> int:
    """
    Normalize Binance Vision timestamps to milliseconds.

    Typical old files use milliseconds: 13 digits.
    Some newer spot files may use microseconds: 16 digits.
    """
    ts = int(float(value))
    digits = len(str(abs(ts)))

    if digits >= 16:       # microseconds
        return ts // 1000
    if digits <= 10:       # seconds, just in case
        return ts * 1000
    return ts              # milliseconds


def date_range(start: date, end: date) -> Iterable[date]:
    curr = start
    while curr <= end:
        yield curr
        curr += timedelta(days=1)


def make_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "binance-vision-kline-downloader/2.0"
    })
    return session


def build_prefix(market: str, symbol: str, interval: str) -> str:
    return f"{VALID_MARKETS[market]}/{symbol}/{interval}/"


def build_listing_url(market: str, symbol: str, interval: str) -> str:
    # Same style as:
    # https://data.binance.vision/?prefix=data/spot/daily/klines/DOGEUSDT/1m/
    return f"{BASE_URL}/?prefix={build_prefix(market, symbol, interval)}"


def build_daily_url(market: str, symbol: str, interval: str, d: date) -> str:
    day = d.strftime("%Y-%m-%d")
    filename = f"{symbol}-{interval}-{day}.zip"
    return f"{BASE_URL}/{build_prefix(market, symbol, interval)}{filename}"


def infer_available_dates_from_listing(
    session: requests.Session,
    market: str,
    symbol: str,
    interval: str,
) -> list[date]:
    """
    Fetch Binance Vision listing page and extract dates from ZIP filenames.

    The page contains links such as:
        DOGEUSDT-1m-2019-07-15.zip
        DOGEUSDT-1m-2019-07-15.zip.CHECKSUM

    We only match the real .zip files, not .zip.CHECKSUM.
    """
    url = build_listing_url(market, symbol, interval)
    print(f"🔎 Fetching listing page to infer available dates:")
    print(f"   {url}")

    r = session.get(url, timeout=30)
    r.raise_for_status()
    html = r.text

    # Match URL-escaped or raw file names in the page.
    # Use negative lookahead to avoid matching ".zip.CHECKSUM".
    pattern = re.compile(
        rf"{re.escape(symbol)}-{re.escape(interval)}-(\d{{4}}-\d{{2}}-\d{{2}})\.zip(?!\.CHECKSUM)",
        re.IGNORECASE,
    )

    dates = sorted({datetime.strptime(m.group(1), "%Y-%m-%d").date() for m in pattern.finditer(html)})

    if dates:
        print(f"✅ Listing parsed: {len(dates)} daily ZIP file(s), earliest={dates[0]}, latest={dates[-1]}")
    else:
        print("⚠️ Listing parsed but no daily ZIP filename was found.")

    return dates


def resolve_start_end(
    session: requests.Session,
    market: str,
    symbol: str,
    interval: str,
    start_arg: Optional[str],
    end_arg: Optional[str],
) -> tuple[date, date]:
    start = parse_date(start_arg)
    end = parse_date(end_arg) or datetime.now(timezone.utc).date()

    if start is not None:
        if start > end:
            raise ValueError(f"start must be <= end, got start={start}, end={end}")
        return start, end

    dates = infer_available_dates_from_listing(session, market, symbol, interval)

    if dates:
        inferred_start = dates[0]
        print(f"✅ --start not specified. Using earliest available file date: {inferred_start}")
        return inferred_start, end

    fallback = FALLBACK_START[market]
    print(f"⚠️ Failed to infer earliest date from listing. Falling back to {fallback}.")
    return fallback, end


def read_kline_rows_from_zip(zip_bytes: bytes, source_name: str) -> list[list[str]]:
    rows: list[list[str]] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV file inside ZIP: {source_name}")

        with zf.open(csv_names[0], "r") as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.reader(text)

            for row in reader:
                if not row:
                    continue

                first = row[0].strip().lower()
                if first in {"open_time", "open time", "open_time_ms_utc"}:
                    continue

                if len(row) < 12:
                    raise ValueError(f"Bad row with {len(row)} fields in {source_name}: {row[:3]}")

                open_time_ms = normalize_ts_to_ms(row[0])
                close_time_ms = normalize_ts_to_ms(row[6])

                rows.append([
                    str(open_time_ms),
                    ms_to_dt(open_time_ms),
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    str(close_time_ms),
                    row[7],
                    row[8],
                    row[9],
                    row[10],
                    row[11] if len(row) > 11 else "0",
                ])

    return rows


def download_one_day(
    session: requests.Session,
    market: str,
    symbol: str,
    interval: str,
    d: date,
    retries: int = 3,
    sleep_s: float = 1.0,
) -> tuple[str, Optional[list[list[str]]], Optional[str]]:
    url = build_daily_url(market, symbol, interval, d)

    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=30)

            if r.status_code == 404:
                return url, None, "missing"

            if r.status_code in (418, 429) or r.status_code >= 500:
                if attempt < retries:
                    time.sleep(sleep_s * attempt)
                    continue

            r.raise_for_status()
            rows = read_kline_rows_from_zip(r.content, url)
            return url, rows, None

        except Exception as e:
            if attempt < retries:
                time.sleep(sleep_s * attempt)
                continue
            return url, None, f"failed: {type(e).__name__}: {e}"

    return url, None, "failed: unknown error"


def merge_rows_by_open_time(rows: Iterable[list[str]]) -> tuple[list[list[str]], int]:
    by_open_time: dict[int, list[str]] = {}
    duplicates = 0

    for row in rows:
        open_time = int(row[0])
        if open_time in by_open_time:
            duplicates += 1
        by_open_time[open_time] = row

    merged = [by_open_time[k] for k in sorted(by_open_time)]
    return merged, duplicates


def find_gaps(rows: list[list[str]], interval_ms: int) -> list[tuple[int, int, int, int]]:
    gaps = []
    last_ms: Optional[int] = None

    for idx, row in enumerate(rows, start=1):
        curr_ms = int(row[0])

        if last_ms is not None:
            expected = last_ms + interval_ms
            if curr_ms != expected:
                missing = max(0, (curr_ms - last_ms) // interval_ms - 1)
                gaps.append((idx, last_ms, curr_ms, missing))

        last_ms = curr_ms

    return gaps


def write_output_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".temp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(rows)

    os.replace(tmp_path, path)


def print_gap_report(gaps: list[tuple[int, int, int, int]], interval_ms: int) -> None:
    if not gaps:
        print("✅ Gap check: no gaps found.")
        return

    print(f"⚠️ Gap check: found {len(gaps)} gap(s).")
    for i, (row_idx, prev_ms, curr_ms, missing) in enumerate(gaps, start=1):
        gap_start = prev_ms + interval_ms
        gap_end = curr_ms - interval_ms
        print(
            f"   [{i}] source row {row_idx}: missing {missing} candle(s), "
            f"{ms_to_dt(gap_start)} ({gap_start}) -> {ms_to_dt(gap_end)} ({gap_end}); "
            f"previous={ms_to_dt(prev_ms)} ({prev_ms}), next={ms_to_dt(curr_ms)} ({curr_ms})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Binance Vision daily kline ZIP files and merge into one CSV.")
    parser.add_argument("--market", choices=VALID_MARKETS.keys(), default="um",
                        help="Market: spot, um=USDⓈ-M futures, cm=COIN-M futures. Default: spot")
    parser.add_argument("--symbol", required=False, help="Symbol, e.g. BTCUSDT, DOGEUSDT",default="DOGEUSDT")
    parser.add_argument("--interval", required=False, help="Kline interval, e.g. 1m, 5m, 15m, 30m, 1h, 4h, 1d",default="5m")
    parser.add_argument("--start", default=None, help="Start day UTC, YYYY-MM-DD. Default: earliest conservative date for market.")
    parser.add_argument("--end", default=None, help="End day UTC, YYYY-MM-DD. Default: today UTC.")
    parser.add_argument("--dir", default="data", help="Output root directory. Default: ./data")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV if it already exists.")
    parser.add_argument("--show-missing", action="store_true", help="Print every missing ZIP URL/date, not only summary.")
    args = parser.parse_args()

    market = args.market
    symbol = args.symbol.upper()
    interval = args.interval

    interval_ms = interval_to_ms(interval)
    session = make_session()

    start, end = resolve_start_end(
        session=session,
        market=market,
        symbol=symbol,
        interval=interval,
        start_arg=args.start,
        end_arg=args.end,
    )

    if start > end:
        raise ValueError(f"start must be <= end, got start={start}, end={end}")

    out_path = Path(args.dir) / market / symbol / f"{symbol}_{interval}.csv"

    if out_path.exists() and not args.overwrite:
        print(f"❌ Output file already exists: {out_path}")
        print("   Use --overwrite to replace it.")
        return 2

    print("=" * 80)
    print("Binance Vision daily kline downloader")
    print(f"Market      : {market}")
    print(f"Symbol      : {symbol}")
    print(f"Interval    : {interval}")
    print(f"Date range  : {start} -> {end} UTC")
    print(f"Output CSV  : {out_path}")
    print("=" * 80)

    stats = DownloadStats()
    all_rows: list[list[str]] = []

    total_days = (end - start).days + 1

    for n, d in enumerate(date_range(start, end), start=1):
        url, rows, error = download_one_day(session, market, symbol, interval, d)

        if rows is not None:
            stats.downloaded_files += 1
            stats.raw_rows += len(rows)
            all_rows.extend(rows)
            print(f"[{n}/{total_days}] {d}: downloaded {len(rows)} rows")
            continue

        if error == "missing":
            stats.missing_files += 1
            if args.show_missing:
                print(f"[{n}/{total_days}] {d}: missing ZIP -> {url}")
            elif n % 50 == 0 or n == total_days:
                print(f"[{n}/{total_days}] scanning... downloaded={stats.downloaded_files}, missing={stats.missing_files}")
            continue

        stats.failed_files += 1
        print(f"[{n}/{total_days}] {d}: {error} -> {url}")

    if not all_rows:
        print("❌ No kline rows downloaded. Check symbol, interval, market, and date range.")
        return 1

    merged_rows, duplicate_rows = merge_rows_by_open_time(all_rows)
    stats.duplicate_rows = duplicate_rows
    stats.final_rows = len(merged_rows)

    write_output_csv(out_path, merged_rows)

    gaps = find_gaps(merged_rows, interval_ms)

    print("\n" + "=" * 80)
    print("Download summary")
    print(f"Downloaded ZIP files : {stats.downloaded_files}")
    print(f"Missing ZIP files    : {stats.missing_files}")
    print(f"Failed ZIP files     : {stats.failed_files}")
    print(f"Raw rows             : {stats.raw_rows}")
    print(f"Duplicate rows       : {stats.duplicate_rows}")
    print(f"Final rows           : {stats.final_rows}")
    print(f"First candle         : {ms_to_dt(int(merged_rows[0][0]))} ({merged_rows[0][0]})")
    print(f"Last candle          : {ms_to_dt(int(merged_rows[-1][0]))} ({merged_rows[-1][0]})")
    print(f"Output CSV           : {out_path}")
    print("=" * 80)

    print_gap_report(gaps, interval_ms)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
