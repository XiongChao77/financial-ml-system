#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Binance Vision daily kline ZIP files by date range.

Data source:
    https://data.binance.vision/

Main behavior:
    - Download daily ZIP files into a local ZIP cache, then merge them into one CSV.
    - If --start is not specified, automatically fetch the Binance Vision listing
      page and infer the earliest available daily ZIP date for this market/symbol/interval.
    - If --end is not specified, default to the latest completed UTC day.
    - Treat local ZIP files as the cache. download_info.json is a human-readable
      audit report only and is never used as downloader input.
    - Only dates between the first and last files advertised by Binance are
      checked; dates before the first available file are not considered missing.
    - After all expected ZIP files are present, aggregate and validate the CSV.

Examples:
    # Spot DOGEUSDT 30m, automatically find earliest available day, download to latest completed UTC day
    python download_binance_vision_klines_v2.py --symbol DOGEUSDT --interval 30m

    # Spot DOGEUSDT 30m, explicit range
    python download_binance_vision_klines_v2.py --symbol DOGEUSDT --interval 30m --start 2021-01-01 --end 2021-12-31

    # Binance USDⓈ-M futures / UM futures
    python download_binance_vision_klines_v2.py --market um --symbol DOGEUSDT --interval 30m

Output:
    <out_dir>/<market>/<symbol>/<interval>/
        download_info.json
        <symbol>_<interval>.csv
        zips/<symbol>-<interval>-YYYY-MM-DD.zip
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
import pandas as pd
from requests.adapters import HTTPAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_process import common


BASE_URL = "https://data.binance.vision"
INFO_FILENAME = "download_info.json"
ZIP_DIRNAME = "zips"

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
    cached_files: int = 0
    missing_files: int = 0
    not_available_files: int = 0
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_output_dir(base_dir: Path, market: str, symbol: str, interval: str) -> Path:
    return base_dir / market / symbol / interval


def zip_filename(symbol: str, interval: str, d: date) -> str:
    return f"{symbol}-{interval}-{d:%Y-%m-%d}.zip"


def build_zip_path(output_dir: Path, symbol: str, interval: str, d: date) -> Path:
    return output_dir / ZIP_DIRNAME / zip_filename(symbol, interval, d)


def build_csv_path(output_dir: Path, symbol: str, interval: str) -> Path:
    return output_dir / f"{symbol}_{interval}.csv"


def build_info_path(output_dir: Path) -> Path:
    return output_dir / INFO_FILENAME


def write_download_info(info_path: Path, info: dict[str, Any]) -> None:
    info_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = info_path.with_suffix(info_path.suffix + ".temp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, info_path)


def make_download_info(
    *,
    market: str,
    symbol: str,
    interval: str,
    interval_ms: int,
    output_dir: Path,
    csv_path: Path,
    start: date,
    end: date,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "purpose": "Human-readable download audit only; never used as downloader input.",
        "data_source": "binance_vision",
        "base_url": BASE_URL,
        "market": market,
        "symbol": symbol,
        "interval": interval,
        "interval_ms": interval_ms,
        "dataset_start_date": start.isoformat(),
        "dataset_end_date": end.isoformat(),
        "scan_start_date": start.isoformat(),
        "scan_end_date": end.isoformat(),
        "download_dir": str(output_dir),
        "zip_dir": ZIP_DIRNAME,
        "csv_file": csv_path.name,
        "scan_started_utc": utc_now_iso(),
        "last_updated_utc": utc_now_iso(),
        "zip_files": {},
    }


def existing_zip_path(
    output_dir: Path,
    symbol: str,
    interval: str,
    d: date,
) -> Optional[Path]:
    path = build_zip_path(output_dir, symbol, interval, d)
    if path.exists() and path.is_file():
        return path
    return None


def cached_zip_dates(
    output_dir: Path,
    symbol: str,
    interval: str,
    desired_dates: Iterable[date],
) -> set[date]:
    cached: set[date] = set()

    for d in desired_dates:
        if existing_zip_path(output_dir, symbol, interval, d) is not None:
            cached.add(d)

    return cached


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
    requested_start = parse_date(start_arg)
    requested_end = parse_date(end_arg) or (datetime.now(timezone.utc).date() - timedelta(days=1))

    if requested_start is not None and requested_start > requested_end:
        raise ValueError(
            f"start must be <= end, got start={requested_start}, end={requested_end}"
        )

    dates = infer_available_dates_from_listing(session, market, symbol, interval)

    if dates:
        latest_available = dates[-1]
        effective_end = min(requested_end, latest_available)
        requested_floor = requested_start or dates[0]
        start_candidates = [d for d in dates if requested_floor <= d <= effective_end]
        if not start_candidates:
            raise ValueError(
                "No available ZIP files in requested date range: "
                f"requested_start={requested_start or dates[0]}, requested_end={requested_end}, "
                f"source_start={dates[0]}, source_end={latest_available}"
            )

        effective_start = start_candidates[0]
        if requested_start is not None and effective_start > requested_start:
            print(
                "⚠️ Requested start date is earlier than data-source start; "
                f"requested_start={requested_start}, using_start={effective_start}"
            )
        if effective_end < requested_end:
            print(
                "⚠️ Requested end date is later than data-source end; "
                f"requested_end={requested_end}, using_end={effective_end}"
            )
        if requested_start is None:
            print(f"✅ --start not specified. Using earliest available file date: {effective_start}")
        else:
            print(f"✅ Using first available file date in requested range: {effective_start}")
        return effective_start, effective_end

    fallback = FALLBACK_START[market]
    effective_start = max(requested_start, fallback) if requested_start else fallback
    if requested_start is not None and fallback > requested_start:
        print(
            "⚠️ Requested start date is earlier than fallback data-source start; "
            f"requested_start={requested_start}, using_start={fallback}"
        )
    print(f"⚠️ Failed to infer earliest date from listing. Falling back to {effective_start}.")
    return effective_start, requested_end


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


def download_one_day_zip(
    session: requests.Session,
    market: str,
    symbol: str,
    interval: str,
    d: date,
    retries: int = 3,
    sleep_s: float = 1.0,
) -> tuple[str, Optional[bytes], Optional[str]]:
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
            read_kline_rows_from_zip(r.content, url)
            return url, r.content, None

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


def write_output_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".temp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(OUTPUT_COLUMNS)
        writer.writerows(rows)

    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Binance Vision daily kline ZIP files and merge into one CSV.")
    parser.add_argument("--market", choices=VALID_MARKETS.keys(), default="um",
                        help="Market: spot, um=USDⓈ-M futures, cm=COIN-M futures. Default: um")
    parser.add_argument("--symbol", required=False, help="Symbol, e.g. BTCUSDT, DOGEUSDT",default="DOGEUSDT")
    parser.add_argument("--interval", required=False, help="Kline interval, e.g. 1m, 5m, 15m, 30m, 1h, 4h, 1d",default="1h")
    parser.add_argument("--start", default=None, help="Start day UTC, YYYY-MM-DD. Default: earliest available date.")
    parser.add_argument(
        "--end",
        default=None,
        help="End day UTC, YYYY-MM-DD. Default: latest completed UTC day (yesterday).",
    )
    parser.add_argument(
        "--out-dir",
        "--dir",
        dest="out_dir",
        default=str(Path(__file__).resolve().parent / "data"),
        help="Base output directory. Default: ./data beside this script.",
    )
    parser.add_argument("--show-missing", action="store_true", help="Print every missing ZIP URL/date, not only summary.")
    args = parser.parse_args()

    market = args.market
    symbol = args.symbol.upper()
    interval = args.interval

    interval_ms = interval_to_ms(interval)
    session = make_session()
    base_output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir = build_output_dir(base_output_dir, market, symbol, interval)
    zip_dir = output_dir / ZIP_DIRNAME
    out_path = build_csv_path(output_dir, symbol, interval)
    info_path = build_info_path(output_dir)

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

    desired_dates = list(date_range(start, end))
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_dir.mkdir(parents=True, exist_ok=True)

    cached_dates = cached_zip_dates(
        output_dir,
        symbol,
        interval,
        desired_dates,
    )
    dates_to_download = [d for d in desired_dates if d not in cached_dates]

    # This file is deliberately rebuilt from observable state on every run.
    # It is an audit report for people, never an input to date/cache decisions.
    info = make_download_info(
        market=market,
        symbol=symbol,
        interval=interval,
        interval_ms=interval_ms,
        output_dir=output_dir,
        csv_path=out_path,
        start=start,
        end=end,
    )
    for d in desired_dates:
        zip_path = build_zip_path(output_dir, symbol, interval, d)
        info["zip_files"][d.isoformat()] = {
            "status": "cached" if d in cached_dates else "pending",
            "file": str(zip_path.relative_to(output_dir)),
            "url": build_daily_url(market, symbol, interval, d),
            **({"bytes": zip_path.stat().st_size} if d in cached_dates else {}),
        }
    write_download_info(info_path, info)

    print("=" * 80)
    print("Binance Vision daily kline downloader")
    print(f"Market      : {market}")
    print(f"Symbol      : {symbol}")
    print(f"Interval    : {interval}")
    print(f"Scan range  : {start} -> {end} UTC")
    print(f"Output dir  : {output_dir}")
    print(f"ZIP cache   : {zip_dir}")
    print(f"Info file   : {info_path}")
    print(f"Output CSV  : {out_path}")
    print(f"Cached ZIPs : {len(cached_dates)}")
    print(f"Need ZIPs   : {len(dates_to_download)}")
    print("=" * 80)

    stats = DownloadStats()
    stats.cached_files = len(cached_dates)

    total_downloads = len(dates_to_download)

    for n, d in enumerate(dates_to_download, start=1):
        url, zip_bytes, error = download_one_day_zip(session, market, symbol, interval, d)
        day = d.isoformat()

        if zip_bytes is not None:
            zip_path = build_zip_path(output_dir, symbol, interval, d)
            tmp_path = zip_path.with_suffix(zip_path.suffix + ".temp")
            with tmp_path.open("wb") as f:
                f.write(zip_bytes)
            os.replace(tmp_path, zip_path)

            stats.downloaded_files += 1
            info["zip_files"][day] = {
                "status": "downloaded",
                "file": str(zip_path.relative_to(output_dir)),
                "url": url,
                "bytes": len(zip_bytes),
                "downloaded_at_utc": utc_now_iso(),
            }
            info["last_updated_utc"] = utc_now_iso()
            write_download_info(info_path, info)
            print(f"[{n}/{total_downloads}] {d}: downloaded ZIP {len(zip_bytes)} bytes")
            continue

        if error == "missing":
            stats.missing_files += 1
            info["zip_files"][day] = {
                "status": "missing",
                "file": str(build_zip_path(output_dir, symbol, interval, d).relative_to(output_dir)),
                "url": url,
                "checked_at_utc": utc_now_iso(),
            }
            info["last_updated_utc"] = utc_now_iso()
            write_download_info(info_path, info)
            if args.show_missing:
                print(f"[{n}/{total_downloads}] {d}: missing ZIP -> {url}")
            elif n % 50 == 0 or n == total_downloads:
                print(f"[{n}/{total_downloads}] scanning... downloaded={stats.downloaded_files}, missing={stats.missing_files}")
            continue

        stats.failed_files += 1
        info["zip_files"][day] = {
            "status": "failed",
            "file": str(build_zip_path(output_dir, symbol, interval, d).relative_to(output_dir)),
            "url": url,
            "error": error,
            "checked_at_utc": utc_now_iso(),
        }
        info["last_updated_utc"] = utc_now_iso()
        write_download_info(info_path, info)
        print(f"[{n}/{total_downloads}] {d}: {error} -> {url}")

    present_dates = [
        d for d in desired_dates
        if existing_zip_path(output_dir, symbol, interval, d) is not None
    ]
    if not present_dates:
        for entry in info["zip_files"].values():
            if entry["status"] == "missing":
                entry["status"] = "not_available"
                entry["note"] = "No available ZIP boundary was found; not treated as an internal gap."
                stats.not_available_files += 1
        stats.missing_files = 0
        info["summary"] = {
            "cached": stats.cached_files,
            "downloaded": stats.downloaded_files,
            "missing": stats.missing_files,
            "not_available_at_range_edges": stats.not_available_files,
            "failed": stats.failed_files,
            "complete": False,
        }
        info["last_updated_utc"] = utc_now_iso()
        write_download_info(info_path, info)
        raise RuntimeError("No ZIP file exists in the requested date range.")

    # A run may start before the symbol was listed (especially when the remote
    # listing page was unavailable and FALLBACK_START was used). 404s before
    # the first real file or after the last real file are resource boundaries,
    # not download gaps. Only absent files between real files are gaps.
    first_present = present_dates[0]
    last_present = present_dates[-1]
    info["dataset_start_date"] = first_present.isoformat()
    info["dataset_end_date"] = last_present.isoformat()
    effective_dates = list(date_range(first_present, last_present))
    middle_missing_dates = []
    for d in desired_dates:
        if existing_zip_path(output_dir, symbol, interval, d) is not None:
            continue
        entry = info["zip_files"][d.isoformat()]
        if entry["status"] == "missing" and (d < first_present or d > last_present):
            entry["status"] = "not_available"
            entry["note"] = "Outside the first/last available ZIP range; not treated as a download gap."
            stats.not_available_files += 1
        elif entry["status"] == "missing" and first_present < d < last_present:
            middle_missing_dates.append(d)

    stats.missing_files = sum(
        info["zip_files"][d.isoformat()]["status"] == "missing"
        for d in effective_dates
    )

    if stats.failed_files or middle_missing_dates:
        info["summary"] = {
            "cached": stats.cached_files,
            "downloaded": stats.downloaded_files,
            "missing": stats.missing_files,
            "not_available_at_range_edges": stats.not_available_files,
            "failed": stats.failed_files,
            "complete": False,
        }
        info["last_updated_utc"] = utc_now_iso()
        write_download_info(info_path, info)
        raise RuntimeError(
            "Refusing to write an incomplete K-line source: "
            f"missing ZIP files inside available range={len(middle_missing_dates)}, "
            f"failed ZIP downloads={stats.failed_files}"
        )

    if first_present != start or last_present != end:
        print(
            "ℹ️ Effective available ZIP range: "
            f"{first_present} -> {last_present} UTC "
            f"({stats.not_available_files} edge date(s) ignored)"
        )

    print(f"Aggregating {len(effective_dates)} ZIP file(s) from local cache...")
    all_rows: list[list[str]] = []
    for d in effective_dates:
        zip_path = existing_zip_path(output_dir, symbol, interval, d)
        if zip_path is None:
            raise RuntimeError(f"Expected cached ZIP is missing: {build_zip_path(output_dir, symbol, interval, d)}")
        rows = read_kline_rows_from_zip(zip_path.read_bytes(), str(zip_path))
        stats.raw_rows += len(rows)
        all_rows.extend(rows)

    if not all_rows:
        print("❌ No kline rows downloaded. Check symbol, interval, market, and date range.")
        return 1

    raw_df = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
    validation = common.validate_kline_source(
        raw_df,
        interval_ms,
        source=str(out_path),
    )

    merged_rows, duplicate_rows = merge_rows_by_open_time(all_rows)
    stats.duplicate_rows = duplicate_rows
    stats.final_rows = len(merged_rows)

    # Only replace the formal CSV after all structural checks pass.
    write_output_csv(out_path, merged_rows)

    info.update({
        "csv_file": out_path.name,
        "schema_version": 2,
        "data_source": "binance_vision",
        "base_url": BASE_URL,
        "market": market,
        "symbol": symbol,
        "interval": interval,
        "interval_ms": interval_ms,
        "dataset_start_date": first_present.isoformat(),
        "dataset_end_date": last_present.isoformat(),
        "download_dir": str(output_dir),
        "zip_dir": ZIP_DIRNAME,
        "raw_rows": stats.raw_rows,
        "duplicate_rows": stats.duplicate_rows,
        "final_rows": stats.final_rows,
        "first_candle_open_time_ms_utc": int(merged_rows[0][0]),
        "first_candle_open_time_utc": ms_to_dt(int(merged_rows[0][0])),
        "last_candle_open_time_ms_utc": int(merged_rows[-1][0]),
        "last_candle_open_time_utc": ms_to_dt(int(merged_rows[-1][0])),
        "last_updated_utc": utc_now_iso(),
        "summary": {
            "cached": stats.cached_files,
            "downloaded": stats.downloaded_files,
            "missing": stats.missing_files,
            "not_available_at_range_edges": stats.not_available_files,
            "failed": stats.failed_files,
            "complete": True,
        },
    })
    write_download_info(info_path, info)

    print("\n" + "=" * 80)
    print("Download summary")
    print(f"Validated ZIP range  : {first_present} -> {last_present} UTC")
    print(f"Cached ZIP files     : {stats.cached_files}")
    print(f"Downloaded ZIP files : {stats.downloaded_files}")
    print(f"Missing ZIP files    : {stats.missing_files}")
    print(f"Unavailable at edges : {stats.not_available_files}")
    print(f"Failed ZIP files     : {stats.failed_files}")
    print(f"Raw rows             : {stats.raw_rows}")
    print(f"Duplicate rows       : {stats.duplicate_rows}")
    print(f"Final rows           : {stats.final_rows}")
    print(f"First candle         : {ms_to_dt(int(merged_rows[0][0]))} ({merged_rows[0][0]})")
    print(f"Last candle          : {ms_to_dt(int(merged_rows[-1][0]))} ({merged_rows[-1][0]})")
    print(f"Missing candles      : {validation['missing_candle_count']}")
    print(f"Overlapping candles  : {validation['overlap_count']}")
    print(f"Info file            : {info_path}")
    print(f"Output CSV           : {out_path}")
    print("=" * 80)
    print("✅ K-line source validation passed.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
