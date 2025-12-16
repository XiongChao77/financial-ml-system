#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os,sys
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
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from common import PROJECT_DATA_DIR

# Configuration
BASE_URL = "https://api.binance.com"
KLINES = "/api/v3/klines"
MAX_LIMIT_PER_REQ = 1000
MAX_BATCH_SIZE = 50   # this is for a limitation for ram, approximately: 100 bytes(per klines)*MAX_LIMIT_PER_REQ*MAX_BATCH_SIZE
SAFE_WEIGHT_LIMIT = 5400 
NUM_THREADS = 8 

# --- 内存优化参数 ---
# 每次处理多少个请求后就写入磁盘。
# 20个请求 * 1000条数据 = 20,000条数据 (约2MB内存)
# 既保证了多线程的高速，又限制了内存峰值。
BATCH_REQUEST_COUNT = 20 

OUTPUT_COLUMNS = [
    "open_time_ms_utc", "open_time_date_utc", "open", "high", "low", "close", "volume",
    "number_of_trades", "close_time_ms_utc", "quote_asset_volume", 
    "taker_buy_base_volume", "taker_buy_quote_volume"
]

def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == 's': return value * 1000
    if unit == 'm': return value * 60 * 1000
    if unit == 'h': return value * 60 * 60 * 1000
    if unit == 'd': return value * 24 * 60 * 60 * 1000
    if unit == 'w': return value * 7 * 24 * 60 * 60 * 1000
    if unit == 'M': return value * 30 * 24 * 60 * 60 * 1000
    raise ValueError(f"Invalid interval: {interval}")

def ms_to_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

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
                        print(f"\n[RATE LIMIT] Weight {weight}/6000. Pausing threads for 30s...")
                        self.pause_event.clear()
                        threading.Timer(30.0, self.resume).start()
        except ValueError:
            pass

    def resume(self):
        print("\n[RATE LIMIT] Resuming download...")
        self.pause_event.set()

    def wait_if_needed(self):
        self.pause_event.wait()

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

    #klines include start_ms,, end_ms exclude
    def fetch_chunk(self, start_time, end_ms):
        """下载单个块"""
        self.guard.wait_if_needed()
        url = BASE_URL + KLINES
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": MAX_LIMIT_PER_REQ,
            "startTime": start_time,
            "endTime": end_ms - 1 #exclude end_ms
        }
        
        retries = 3
        while retries > 0:
            try:
                r = self.session.get(url, params=params, timeout=10)
                if r.status_code in (418, 429):
                    time.sleep(int(r.headers.get("Retry-After", 10)))
                    continue
                if r.status_code >= 500:
                    time.sleep(1)
                    retries -= 1
                    continue
                r.raise_for_status()
                self.guard.update(r.headers)
                # batch_rows, data_end_ms  = self.filter_data_time(r.json(), start_time, end_ms)
                print(f"fetch_chunk start_time: {start_time} end_ms: {end_ms} num: {len(r.json())}")
                return start_time, r.json()
            except requests.RequestException:
                retries -= 1
                time.sleep(1)
        return start_time, []

    def format_kline_row(self, row):
        open_time = int(row[0])
        close_time = int(row[6])
        return [
            open_time, ms_to_dt(open_time),
            row[1], row[2], row[3], row[4], row[5],
            row[8], close_time,
            row[7], row[9], row[10]
        ]

    def filter_data_time(self, data, start_ms, end_ms):
        print(type(data))
        if data[-1][0] <= end_ms:   #all data in the specific time
            return data[-1][0], data
        volid_data = []
        for item in data:
            ts = int(item[0])
            # 严格过滤范围，防止API返回多余数据
            if start_ms <= ts < end_ms:
                volid_data.append(self.format_kline_row(item))
            else:
                print(f"filter_data_time drop invalid time data ts:{ts}, start_ms:{start_ms}, end_ms{end_ms} ")
                break
        return ts, volid_data

    #klines include start_ms,, not include end_ms
    def download_range_generator(self, writer:Writer, start_ms, end_ms, desc="Downloading"):
        """
        [优化版] 生成器模式下载。
        不是一次性返回所有数据，而是分批次(Batch)返回。
        内存占用恒定，不会随下载范围增大而增大。
        """
        if start_ms >= end_ms:
            return
        
        klines_count = 0
        if start_ms == 0: # full history, get the first one
            _, data =self.fetch_chunk(start_ms, end_ms)
            if not data:
                print("fetch_chunk fail!")
                return
            start_ms = data[-1][0] + self.interval_ms
            rows = [self.format_kline_row(klines) for klines in data]
            writer.writerows(rows)
            klines_count = len(rows)
            if start_ms >= end_ms:#1764547200000 ,end_ms:1765019822492, next: 1765152000000
                print("download_range_generator download finished")
                return

        klines_count += (end_ms- start_ms + self.interval_ms -1 ) // self.interval_ms
        print(f"download_range_generator total klines counts: {klines_count}")
        req_counts = (klines_count+ MAX_LIMIT_PER_REQ -1) // MAX_LIMIT_PER_REQ
        chunk_tasks = []
        chunk_start_ms = start_ms
        while chunk_start_ms < end_ms: 
            chunk_end_ms = chunk_start_ms + MAX_LIMIT_PER_REQ * self.interval_ms
            if chunk_end_ms > end_ms :      chunk_end_ms = end_ms
            chunk_klines_count = (chunk_end_ms - chunk_start_ms) // self.interval_ms
            chunk_tasks.append([chunk_start_ms, chunk_end_ms, chunk_klines_count])
            chunk_start_ms = chunk_end_ms + self.interval_ms
        batch_counts = (len(chunk_tasks) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        for i in range(0, batch_counts):
            if i != batch_counts -1:
                batch_tasks = chunk_tasks[i*MAX_BATCH_SIZE:(i+1)*MAX_BATCH_SIZE]
            else:
                batch_tasks = chunk_tasks[i*MAX_BATCH_SIZE:]
            batch_rows = []
            with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                futures = {executor.submit(self.fetch_chunk, task[0] , task[1]): task  for task in batch_tasks}
                
                for future in as_completed(futures):
                    _, data = future.result()
                    if data:
                        for item in data:
                            ts = int(item[0])
                            # 严格过滤范围，防止API返回多余数据
                            if start_ms <= ts < end_ms:
                                batch_rows.append(self.format_kline_row(item))
            
            # 重要：多线程返回是无序的，必须在批次内排序. duplicate
            unique_rows = set(tuple(r) for r in batch_rows)
            batch_rows = sorted(unique_rows, key=lambda x: x[0])
            if batch_rows:
                # 打印进度
                last_ts = batch_rows[-1][0]
                progress = min(100, ((i*MAX_BATCH_SIZE + len(batch_tasks)) / len(chunk_tasks)) * 100)
                print(f"      ... batch {i} done. Progress: {progress:.1f}% (Last: {ms_to_dt(last_ts)})", end='\r')
            #check duplicate data and missing data    
            writer.writerows(batch_rows)
            print(f"\n      [Done] {desc} branch {i} finished.") 

        needed_starts = list(range(start_ms, end_ms, self.step_ms))
        total_chunks = len(needed_starts)
        
        print(f"   >> [{desc}] Range: {ms_to_dt(start_ms)} -> {ms_to_dt(end_ms)} (Total {total_chunks} reqs)")

    # There are cases where exchange data is missing in binance
    def repair_and_update(self, execute_update=False, replace = False):
        print(f"\n{'='*20} Start Processing: {self.symbol} {'='*20}")
        
        if not os.path.exists(self.csv_path) and execute_update == True:
            # 新文件模式
            print("File not found, starting full download...")
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(OUTPUT_COLUMNS)
                # 全量下载
                now = int(time.time() * 1000)
                # 使用生成器循环写入
                self.download_range_generator(writer, 0, now+self.interval_ms*100, desc="Full Download")
            return

        last_valid_time = None
        total_gaps_list = []
        with open(self.csv_path, 'r', encoding='utf-8') as f_in:
            reader = csv.reader(f_in)
            header = next(reader, None)
            continuous_num = 0
            line_num = 1
            for row in reader:
                continuous_num += 1
                line_num += 1
                try:
                    curr_time = int(row[0])
                    if last_valid_time is not None:
                        if curr_time != last_valid_time + self.interval_ms:
                            print(f"❌ 发现断档: line_num: {line_num} ,{ms_to_dt(last_valid_time)} -> {ms_to_dt(curr_time)}")
                            total_gaps_list.append([continuous_num, last_valid_time ,curr_time])
                            continuous_num = 0
                    last_valid_time = curr_time
                except ValueError: continue
        print(f"\n检查完成。发现 {len(total_gaps_list)} 处断档。使用 --update 参数进行修复。")
        if execute_update != True:
            return
        print("repair_and_update start repair the data")
        temp_csv = self.csv_path + ".temp"
        backup_csv = self.csv_path + ".bak"
        with open(self.csv_path, 'r', encoding='utf-8') as f_in , open(temp_csv, 'w', newline='', encoding='utf-8') as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)
            for gap in total_gaps_list:
                reader_rows = islice(reader, gap[0])
                writer.writerows(reader_rows)
                self.download_range_generator(writer, gap[1]+1, gap[2], desc="Repairing")
            #copy the rest
            writer.writerows(reader)
            #update to the now
            now = int(time.time() * 1000)
            self.download_range_generator(writer, last_valid_time+1, now, desc="Tail Update")
            if replace == True:
                # 替换文件
                print(f"Swapping files...")
                shutil.move(self.csv_path, backup_csv)
                shutil.move(temp_csv, self.csv_path)
                os.remove(backup_csv)
                print(f"🎉 Success!")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--interval", default="5m") #e.g., "1h" – supported intervals: 1s, 15s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    parser.add_argument("--dir", default=PROJECT_DATA_DIR)
    parser.add_argument("--update", default = True ,action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    downloader = BinanceDownloader(args.symbol, args.interval, args.dir)
    downloader.repair_and_update(execute_update=args.update, replace= True)