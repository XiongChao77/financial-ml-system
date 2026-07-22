#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import time
import csv
import argparse
import requests
import threading
from requests.adapters import HTTPAdapter
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

# Keep the original current_work_dir logic
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, '..'))
PROJECT_DATA_DIR = os.path.join(current_work_dir, "data")

# Configuration
BASE_URL = "https://api.binance.com"
KLINES = "/api/v3/klines"
UM_KLINES = "/api/fv3/klines"
MAX_LIMIT_PER_REQ = 1000
MAX_BATCH_SIZE = 50
SAFE_WEIGHT_LIMIT = 5400
NUM_THREADS = 8
BATCH_REQUEST_COUNT = 20

# Retry tuning
MAX_NETWORK_RETRIES = 3          # for 5xx / connection errors on a single chunk
MAX_RATE_LIMIT_RETRIES = 8       # for 418/429 before we give up on a chunk
RATE_LIMIT_BACKOFF_BASE = 10.0   # seconds, doubles each retry (capped)
RATE_LIMIT_BACKOFF_CAP = 120.0

OUTPUT_COLUMNS = [
    "open_time_ms_utc", "open_time_date_utc", "open", "high", "low", "close", "volume",
    "number_of_trades", "close_time_ms_utc", "quote_asset_volume",
    "taker_buy_base_volume", "taker_buy_quote_volume"
]


# --- Helper functions ---
def parse_date_to_ms(date_str: str) -> int:
    if not date_str:
        return 0
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_str}")


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    scales = {'s': 1000, 'm': 60000, 'h': 3600000, 'd': 86400000, 'w': 604800000, 'M': 2592000000}
    if unit in scales:
        return value * scales[unit]
    raise ValueError(f"Invalid interval: {interval}")


def ms_to_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def gap_missing_count(gap, interval_ms):
    _, previous_ms, current_ms = gap
    return max(0, (current_ms - previous_ms) // interval_ms - 1)


def format_gap_detail(gap, interval_ms):
    row_count, previous_ms, current_ms = gap
    gap_start = previous_ms + interval_ms
    gap_end = current_ms - interval_ms
    missing_count = gap_missing_count(gap, interval_ms)
    return (
        f"row {row_count}: missing {missing_count} candles "
        f"from {ms_to_dt(gap_start)} ({gap_start}) "
        f"to {ms_to_dt(gap_end)} ({gap_end}); "
        f"previous={ms_to_dt(previous_ms)} ({previous_ms}), "
        f"next={ms_to_dt(current_ms)} ({current_ms})"
    )


# --- Rate limit guard (unchanged behavior) ---
class RateLimitGuard:
    def __init__(self):
        self.lock = threading.Lock()
        self.pause_event = threading.Event()
        self.pause_event.set()

    def update(self, headers):
        try:
            weight = int(headers.get("X-MBX-USED-WEIGHT-1M", 0))
        except (TypeError, ValueError):
            return
        if weight > SAFE_WEIGHT_LIMIT:
            with self.lock:
                if self.pause_event.is_set():
                    print(f"\n⚠️ [RATE LIMIT] Weight {weight}/6000. Pausing for 30s...")
                    self.pause_event.clear()
                    threading.Timer(30.0, self.resume).start()

    def resume(self):
        print("\n✅ [RATE LIMIT] Resuming...")
        self.pause_event.set()

    def wait_if_needed(self):
        self.pause_event.wait()


# --- Core downloader class ---
class BinanceDownloader:
    def __init__(self, symbol, interval, out_dir, executor: ThreadPoolExecutor):
        self.symbol = symbol.upper()
        self.interval = interval
        self.out_dir = out_dir
        self.interval_ms = interval_to_ms(interval)
        self.step_ms = MAX_LIMIT_PER_REQ * self.interval_ms
        self.executor = executor  # shared across symbols/intervals, created once in main()

        self.session = requests.Session()
        # Pool sized to match worker-thread concurrency so threads aren't
        # fighting over (and re-creating) connections.
        adapter = HTTPAdapter(pool_connections=NUM_THREADS, pool_maxsize=NUM_THREADS)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.guard = RateLimitGuard()
        self.csv_path = os.path.join(out_dir, f"{self.symbol}_{self.interval}.csv")
        print(f"output file path:{self.csv_path}")

    def fetch_chunk(self, start_time, end_ms):
        self.guard.wait_if_needed()
        params = {
            "symbol": self.symbol, "interval": self.interval, "limit": MAX_LIMIT_PER_REQ,
            "startTime": start_time, "endTime": end_ms - 1,
        }

        network_retries = MAX_NETWORK_RETRIES
        rate_limit_retries = MAX_RATE_LIMIT_RETRIES
        backoff = RATE_LIMIT_BACKOFF_BASE

        while True:
            try:
                r = self.session.get(BASE_URL + KLINES, params=params, timeout=10)

                if r.status_code in (418, 429):
                    if rate_limit_retries <= 0:
                        print(f"\n❌ [RATE LIMIT] Giving up on chunk start={start_time} after repeated 418/429.")
                        return start_time, []
                    rate_limit_retries -= 1
                    retry_after = r.headers.get("Retry-After")
                    wait_s = float(retry_after) if retry_after else backoff
                    print(f"\n⚠️ [RATE LIMIT] HTTP {r.status_code}. Waiting {wait_s:.0f}s...")
                    time.sleep(wait_s)
                    backoff = min(backoff * 2, RATE_LIMIT_BACKOFF_CAP)
                    continue

                if r.status_code >= 500:
                    if network_retries <= 0:
                        return start_time, []
                    network_retries -= 1
                    time.sleep(1)
                    continue

                r.raise_for_status()
                self.guard.update(r.headers)
                return start_time, r.json()

            except requests.RequestException:
                if network_retries <= 0:
                    return start_time, []
                network_retries -= 1
                time.sleep(1)

    def format_kline_row(self, row):
        open_time = int(row[0])
        return [open_time, ms_to_dt(open_time), row[1], row[2], row[3], row[4], row[5], row[8], int(row[6]), row[7], row[9], row[10]]

    def download_range_generator(self, writer, start_ms, end_ms, desc="Downloading"):
        if start_ms >= end_ms:
            print(f"      {desc}: no rows needed")
            return 0

        rows_written = 0
        print(f"      {desc}: {ms_to_dt(start_ms)} ({start_ms}) -> {ms_to_dt(end_ms - self.interval_ms)} ({end_ms - self.interval_ms})")

        # Initial positioning: find the first available candle at/after start_ms == 0
        if start_ms == 0:
            _, data = self.fetch_chunk(0, end_ms)
            if not data:
                return 0
            start_ms = data[-1][0] + self.interval_ms
            rows = [self.format_kline_row(k) for k in data]
            writer.writerows(rows)
            rows_written += len(rows)
            if start_ms >= end_ms:
                print(f"      {desc}: wrote {rows_written} rows")
                return rows_written

        # Build chunk tasks. Each chunk covers up to MAX_LIMIT_PER_REQ candles:
        # [curr, curr + step_ms) i.e. candles curr, curr+interval_ms, ..., nxt-interval_ms.
        # The next chunk must start exactly at `nxt` (NOT nxt + interval_ms) or the
        # candle at `nxt` is silently skipped.
        chunk_tasks = []
        curr = start_ms
        while curr < end_ms:
            nxt = min(curr + self.step_ms, end_ms)
            chunk_tasks.append((curr, nxt))
            curr = nxt

        # Execute in batches using the shared thread pool (not recreated per batch).
        total_chunks = len(chunk_tasks)
        for i in range(0, total_chunks, MAX_BATCH_SIZE):
            batch = chunk_tasks[i:i + MAX_BATCH_SIZE]
            futures = [self.executor.submit(self.fetch_chunk, c[0], c[1]) for c in batch]

            batch_rows_by_open_time = {}
            for f in as_completed(futures):
                _, data = f.result()
                for k in data:
                    row = self.format_kline_row(k)
                    batch_rows_by_open_time[row[0]] = row  # de-dupe by open_time

            batch_rows = sorted(batch_rows_by_open_time.values(), key=lambda x: x[0])
            writer.writerows(batch_rows)
            rows_written += len(batch_rows)

            progress = min(100, (i + len(batch)) / total_chunks * 100)
            print(f"      ... {desc} Progress: {progress:.1f}% | wrote {rows_written} rows", end='\r')

        print(f"\n      {desc}: wrote {rows_written} rows")
        return rows_written

    def scan_csv_gaps(self):
        gaps, last_valid = [], None
        count = 0
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                count += 1
                curr = int(row[0])
                if last_valid is not None and curr != last_valid + self.interval_ms:
                    gaps.append([count, last_valid, curr])
                last_valid = curr
        return gaps, last_valid, count

    def print_gap_report(self, gaps, label="existing data"):
        if not gaps:
            print(f"✅ No gaps found in {label}.")
            return

        print(f"⚠️ Check gap(s) in {label}:")
        for gap in gaps:
            print(f"   - {format_gap_detail(gap, self.interval_ms)}")
        print(f"⚠️ Found {len(gaps)} gap(s) in {label}:")

    def repair_and_update(self, execute_update=False, start_time_str=None):
        print(f"\n{'='*30}\n🚀 Processing: {self.symbol} | Interval: {self.interval}\n{'='*30}")
        start_ms = parse_date_to_ms(start_time_str) if start_time_str else 0
        now = int(time.time() * 1000)

        if not os.path.exists(self.csv_path):
            if not execute_update:
                return
            print(f"📁 New File: Initializing download...")
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(OUTPUT_COLUMNS)
                self.download_range_generator(writer, start_ms, now, desc="Initial")
            gaps, last_valid, count = self.scan_csv_gaps()
            print(f"🔎 Initial download verification: rows={count}")
            if last_valid:
                print(f"   Last candle: {ms_to_dt(last_valid)} ({last_valid})")
            self.print_gap_report(gaps, label="initial download")
            return

        # Gap check
        gaps, last_valid, count = self.scan_csv_gaps()

        if not gaps and last_valid and last_valid >= now - self.interval_ms:
            print(f"✅ Data is up to date.")
            return

        self.print_gap_report(gaps, label="existing data")

        if not execute_update:
            return

        temp_csv = self.csv_path + ".temp"
        print(f"🔧 Update requested. Writing repaired data to temp file: {temp_csv}")
        print(f"   Existing rows scanned: {count}")
        if last_valid:
            print(f"   Last valid candle: {ms_to_dt(last_valid)} ({last_valid})")
        else:
            print("   Last valid candle: none")

        repaired_rows, updated_rows = 0, 0
        with open(self.csv_path, 'r') as f_in, open(temp_csv, 'w', newline='') as f_out:
            reader, writer = csv.reader(f_in), csv.writer(f_out)
            writer.writerow(next(reader))  # Header
            copied_rows = 0
            for idx, gap in enumerate(gaps, start=1):
                rows_to_copy = gap[0] - copied_rows - 1
                print(f"   [{idx}/{len(gaps)}] Copying {rows_to_copy} existing rows before gap at source row {gap[0]}...")
                writer.writerows(islice(reader, rows_to_copy))
                copied_rows += rows_to_copy

                expected_rows = gap_missing_count(gap, self.interval_ms)
                print(f"   [{idx}/{len(gaps)}] Repairing gap, expected missing candles: {expected_rows}")
                repaired_rows += self.download_range_generator(
                    writer,
                    gap[1] + self.interval_ms,
                    gap[2],
                    desc=f"Repairing gap {idx}/{len(gaps)}"
                )

            writer.writerows(reader)  # Rest
            print("   Copied remaining existing rows after repairs.")
            update_start = last_valid + self.interval_ms if last_valid else start_ms
            updated_rows = self.download_range_generator(writer, update_start, now, desc="Updating latest")

        os.replace(temp_csv, self.csv_path)
        print(f"\n✨ {self.symbol}_{self.interval} done! repaired_rows={repaired_rows}, updated_rows={updated_rows}")


# --- Entry point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance Batch Downloader")
    # Multiple symbols separated by spaces
    parser.add_argument("--symbols", nargs='+', default=["DOGEUSDT"],
                        help="List of symbols: BTCUSDT ETHUSDT ...")
    # Multiple intervals separated by spaces
    parser.add_argument("--intervals", nargs='+', default=["30m"],
                        help="List of intervals: 1m 1h 1d ...")
    parser.add_argument("--dir", default=PROJECT_DATA_DIR)
    parser.add_argument("--update", dest="update", action="store_true", default=False,
                        help="Download/repair data (default: on)")
    parser.add_argument("--no-update", dest="update", action="store_false",
                        help="Only report gaps/status, don't download anything")
    parser.add_argument("--start", default=None, help="Start Date YYYY-MM-DD")

    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    total_tasks = len(args.symbols) * len(args.intervals)
    current_task = 0

    # One thread pool shared by every symbol/interval and every gap-repair/update
    # call, instead of spinning up and tearing down a fresh pool per batch.
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as shared_executor:
        for symbol in args.symbols:
            for interval in args.intervals:
                current_task += 1
                print(f"\n[Task {current_task}/{total_tasks}]")
                try:
                    downloader = BinanceDownloader(symbol, interval, args.dir, shared_executor)
                    downloader.repair_and_update(
                        execute_update=args.update,
                        start_time_str=args.start
                    )
                except Exception as e:
                    print(f"❌ Error processing {symbol} {interval}: {e}")
                    continue

    print("\n🏁 All download tasks completed!")