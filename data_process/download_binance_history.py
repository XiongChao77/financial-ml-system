#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import time
import csv
from _csv import Writer
import argparse
import requests
import threading
import shutil
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

# Keep the original current_work_dir logic
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir, '..'))
try:
    from common import PROJECT_DATA_DIR
except ImportError:
    # If common module isn't found, default to a local data folder
    PROJECT_DATA_DIR = os.path.join(current_work_dir, "data")

# Configuration
BASE_URL = "https://api.binance.com"
KLINES = "/api/v3/klines"
MAX_LIMIT_PER_REQ = 1000
MAX_BATCH_SIZE = 50
SAFE_WEIGHT_LIMIT = 5400 
NUM_THREADS = 8 
BATCH_REQUEST_COUNT = 20 

OUTPUT_COLUMNS = [
    "open_time_ms_utc", "open_time_date_utc", "open", "high", "low", "close", "volume",
    "number_of_trades", "close_time_ms_utc", "quote_asset_volume", 
    "taker_buy_base_volume", "taker_buy_quote_volume"
]

# --- Helper functions (unchanged) ---
def parse_date_to_ms(date_str: str) -> int:
    if not date_str: return 0
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError: continue
    raise ValueError(f"Invalid date format: {date_str}")

def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    scales = {'s': 1000, 'm': 60000, 'h': 3600000, 'd': 86400000, 'w': 604800000, 'M': 2592000000}
    if unit in scales: return value * scales[unit]
    raise ValueError(f"Invalid interval: {interval}")

def ms_to_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# --- Rate limit guard (unchanged) ---
class RateLimitGuard:
    def __init__(self):
        self.lock = threading.Lock()
        self.pause_event = threading.Event()
        self.pause_event.set()

    def update(self, headers):
        try:
            weight = int(headers.get("X-MBX-USED-WEIGHT-1M", 0))
            if weight > SAFE_WEIGHT_LIMIT:
                with self.lock:
                    if self.pause_event.is_set():
                        print(f"\n⚠️ [RATE LIMIT] Weight {weight}/6000. Pausing for 30s...")
                        self.pause_event.clear()
                        threading.Timer(30.0, self.resume).start()
        except: pass

    def resume(self):
        print("\n✅ [RATE LIMIT] Resuming...")
        self.pause_event.set()

    def wait_if_needed(self):
        self.pause_event.wait()

# --- Core downloader class (unchanged) ---
class BinanceDownloader:
    def __init__(self, symbol, interval, out_dir):
        self.symbol = symbol.upper()
        self.interval = interval
        self.out_dir = out_dir
        self.interval_ms = interval_to_ms(interval)
        self.step_ms = MAX_LIMIT_PER_REQ * self.interval_ms
        self.session = requests.Session()
        self.guard = RateLimitGuard()
        self.csv_path = os.path.join(out_dir, f"{self.symbol}_{self.interval}.csv")

    def fetch_chunk(self, start_time, end_ms):
        self.guard.wait_if_needed()
        params = {"symbol": self.symbol, "interval": self.interval, "limit": MAX_LIMIT_PER_REQ, "startTime": start_time, "endTime": end_ms - 1}
        retries = 3
        while retries > 0:
            try:
                r = self.session.get(BASE_URL + KLINES, params=params, timeout=10)
                if r.status_code in (418, 429):
                    time.sleep(10); continue
                if r.status_code >= 500:
                    time.sleep(1); retries -= 1; continue
                r.raise_for_status()
                self.guard.update(r.headers)
                return start_time, r.json()
            except:
                retries -= 1; time.sleep(1)
        return start_time, []

    def format_kline_row(self, row):
        open_time = int(row[0])
        return [open_time, ms_to_dt(open_time), row[1], row[2], row[3], row[4], row[5], row[8], int(row[6]), row[7], row[9], row[10]]

    def download_range_generator(self, writer, start_ms, end_ms, desc="Downloading"):
        if start_ms >= end_ms: return
        
        # Initial positioning
        if start_ms == 0:
            _, data = self.fetch_chunk(0, end_ms)
            if not data: return
            start_ms = data[-1][0] + self.interval_ms
            writer.writerows([self.format_kline_row(k) for k in data])
            if start_ms >= end_ms: return

        # Build chunk tasks
        chunk_tasks = []
        curr = start_ms
        while curr < end_ms:
            nxt = min(curr + self.step_ms, end_ms)
            chunk_tasks.append([curr, nxt])
            curr = nxt + self.interval_ms

        # Execute in batches
        for i in range(0, len(chunk_tasks), MAX_BATCH_SIZE):
            batch = chunk_tasks[i:i+MAX_BATCH_SIZE]
            batch_rows = []
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as exec:
                futures = [exec.submit(self.fetch_chunk, t[0], t[1]) for t in batch]
                for f in as_completed(futures):
                    _, data = f.result()
                    if data: batch_rows.extend([self.format_kline_row(k) for k in data])
            
            # Sort and de-duplicate
            batch_rows = sorted(list({tuple(r): r for r in batch_rows}.values()), key=lambda x: x[0])
            writer.writerows(batch_rows)
            print(f"      ... {desc} Progress: {min(100, (i+len(batch))/len(chunk_tasks)*100):.1f}%", end='\r')

    def repair_and_update(self, execute_update=False, start_time_str=None):
        print(f"\n{'='*30}\n🚀 Processing: {self.symbol} | Interval: {self.interval}\n{'='*30}")
        start_ms = parse_date_to_ms(start_time_str) if start_time_str else 0
        now = int(time.time() * 1000)

        if not os.path.exists(self.csv_path):
            if not execute_update: return
            print(f"📁 New File: Initializing download...")
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(OUTPUT_COLUMNS)
                self.download_range_generator(writer, start_ms, now, desc="Initial")
            return

        # Gap check
        gaps, last_valid = [], None
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f); next(reader)
            count = 0
            for row in reader:
                count += 1
                curr = int(row[0])
                if last_valid and curr != last_valid + self.interval_ms:
                    gaps.append([count, last_valid, curr])
                last_valid = curr
        
        if not gaps and last_valid and last_valid >= now - self.interval_ms:
            print(f"✅ Data is up to date.")
            return

        if execute_update:
            temp_csv = self.csv_path + ".temp"
            with open(self.csv_path, 'r') as f_in, open(temp_csv, 'w', newline='') as f_out:
                reader, writer = csv.reader(f_in), csv.writer(f_out)
                writer.writerow(next(reader)) # Header
                for gap in gaps:
                    writer.writerows(islice(reader, gap[0]-1))
                    self.download_range_generator(writer, gap[1]+self.interval_ms, gap[2], desc="Repairing")
                writer.writerows(reader) # Rest
                self.download_range_generator(writer, last_valid+self.interval_ms, now, desc="Updating")
            
            shutil.move(temp_csv, self.csv_path)
            print(f"\n✨ {self.symbol}_{self.interval} done!")

# --- Entry point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance Batch Downloader")
    # Multiple symbols separated by spaces
    parser.add_argument("--symbols", nargs='+', default=["DOGEUSDT"], 
                        help="List of symbols: BTCUSDT ETHUSDT ...")    #BTCUSDT  ETHUSDT  DOGEUSDT SOLUSDT BNBUSDT TRXUSDT XRPUSDT  SUIUSDT ADAUSDT
    # Multiple intervals separated by spaces
    parser.add_argument("--intervals", nargs='+', default=["15m"], 
                        help="List of intervals: 1m 1h 1d ...")
    parser.add_argument("--dir", default=PROJECT_DATA_DIR)
    parser.add_argument("--update", action="store_true", default=True)
    parser.add_argument("--start", default=None, help="Start Date YYYY-MM-DD")
    
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)

    # Nested loops over the configuration lists
    total_tasks = len(args.symbols) * len(args.intervals)
    current_task = 0

    for symbol in args.symbols:
        for interval in args.intervals:
            current_task += 1
            print(f"\n[Task {current_task}/{total_tasks}]")
            try:
                downloader = BinanceDownloader(symbol, interval, args.dir)
                downloader.repair_and_update(
                    execute_update=args.update, 
                    start_time_str=args.start
                )
            except Exception as e:
                print(f"❌ Error processing {symbol} {interval}: {e}")
                continue

    print("\n🏁 All download tasks completed!")