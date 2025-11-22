#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download full-history 5m/15m klines for ETHUSDT & DOGEUSDT from Binance Spot API.

- Public endpoint: https://api.binance.com/api/v3/klines
- No API key required
- Auto pagination until the latest candle
- Rate-limit aware: retries on 429/418/network with exponential backoff
- Stream-write CSV in chunks (memory friendly)
"""
import os
import time
import math
import csv
import sys
import json
import random
import requests
from datetime import datetime, timezone

BASE_URL = "https://api.binance.com"
KLINES = "/api/v3/klines"
MAX_LIMIT = 1000  # Binance per-request limit

# 目标：ETH/ DOGE 的 5m 和 15m
DOWNLOAD_PLAN = {
    "BTCUSDT": ["5m", "15m"]
    # "DOGEUSDT": ["5m", "15m"],
}

# CSV 列（Binance 固定返回顺序）
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
]

# 输出列：加上可读时间（UTC）
OUTPUT_COLUMNS = [
    "open_time_dt_utc", "open", "high", "low", "close", "volume",
    "number_of_trades", "close_time_dt_utc", "quote_asset_volume",
    "taker_buy_base_volume", "taker_buy_quote_volume"
]

def ms_to_dt(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def safe_get(params, max_retries=10, base_sleep=0.25, max_sleep=8.0):
    """
    GET 带重试与指数退避。对 429/418/5xx/网络错误重试。
    """
    url = BASE_URL + KLINES
    attempt = 0
    while True:
        try:
            r = requests.get(url, params=params, timeout=30)
            # 处理限频/封禁/服务端错误
            if r.status_code in (418, 429) or r.status_code >= 500:
                # 优先用 Retry-After
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    sleep_s = max(float(retry_after), base_sleep)
                else:
                    # 指数退避 + 抖动
                    sleep_s = min(base_sleep * (2 ** attempt) + random.random() * 0.25, max_sleep)
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError(f"Too many retries: status={r.status_code}, body={r.text[:200]}")
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return r.json(), r.headers
        except requests.RequestException as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = min(base_sleep * (2 ** (attempt - 1)) + random.random() * 0.25, max_sleep)
            time.sleep(sleep_s)

def fetch_full_history(symbol: str, interval: str, out_csv: str,
                       start_ms: int = 0, request_pause: float = 0.15):
    """
    从最早开始抓取到当前最新，写入 CSV（追加模式）。
    - start_ms=0 表示从最早可用时间开始
    - request_pause：每次成功请求后的礼貌停顿，降低触发限频的概率
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # 若文件已存在，支持断点续传：读最后一行 open_time 继续
    next_start_ms = start_ms
    if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        try:
            # 读取最后一行，取其 close_time 再 +1ms 继续
            with open(out_csv, "r", newline="", encoding="utf-8") as f:
                last_line = None
                for last_line in f:
                    pass
                if last_line:
                    parts = last_line.strip().split(",")
                    # 输出列顺序：我们第8列是 close_time_dt_utc，人类时间；原始 close_time(ms)没写入
                    # 因此断点续传需要额外保存一个 sidecar 记录最后的 close_time_ms。
                    # 为简单起见，这里改用 sidecar 文件。
                    pass
        except Exception:
            pass

    # 断点续传：使用 sidecar 保存最后 open/close ms
    sidecar = out_csv + ".state.json"
    if os.path.exists(sidecar):
        try:
            state = json.load(open(sidecar, "r", encoding="utf-8"))
            if state.get("symbol") == symbol and state.get("interval") == interval:
                next_start_ms = max(next_start_ms, int(state.get("next_start_ms", start_ms)))
        except Exception:
            pass

    # 如果是全新文件，写表头
    write_header = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
    out_f = open(out_csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(out_f)
    if write_header:
        writer.writerow(OUTPUT_COLUMNS)

    total_rows = 0
    last_open_time = None

    print(f"Start downloading {symbol} {interval} -> {out_csv}")
    while True:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": MAX_LIMIT,
        }
        # 只在需要时加时间参数（Binance 在未提供 startTime 时会从最近开始往前返回，
        # 为确保从最早向后抓，这里始终提供 startTime）
        params["startTime"] = next_start_ms

        data, headers = safe_get(params)

        if not data:
            break

        # 去重处理：当上一批最后 open_time 与本批第一根重复时，去掉重复
        if last_open_time is not None and data and data[0][0] == last_open_time:
            data = data[1:]

        if not data:
            break

        # 写入本批
        batch_rows = 0
        for row in data:
            # row: 12 列，参考 KLINE_COLUMNS
            open_time = int(row[0])
            close_time = int(row[6])
            # 选取常用列并转成字符串（避免科学计数法）
            writer.writerow([
                ms_to_dt(open_time),
                row[1],  # open
                row[2],  # high
                row[3],  # low
                row[4],  # close
                row[5],  # volume
                int(row[8]),  # number_of_trades
                ms_to_dt(close_time),
                row[7],   # quote_asset_volume
                row[9],   # taker_buy_base_volume
                row[10],  # taker_buy_quote_volume
            ])
            last_open_time = open_time
            batch_rows += 1

        total_rows += batch_rows
        out_f.flush()

        # 更新断点续传 state：下一次从当前批最后一根的 close_time + 1ms 开始
        next_start_ms = int(data[-1][6]) + 1
        with open(sidecar, "w", encoding="utf-8") as sf:
            json.dump(
                {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "next_start_ms": next_start_ms,
                    "last_batch_rows": batch_rows,
                    "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                },
                sf,
                ensure_ascii=False,
                indent=2,
            )

        # 若不足 MAX_LIMIT，说明已经到尾部
        if batch_rows < MAX_LIMIT:
            break

        # 礼貌停顿，进一步降低 429 概率
        time.sleep(request_pause)

    out_f.close()
    print(f"Done: {symbol} {interval}, rows={total_rows}, file={out_csv}")

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # 如需更换交易对（比如对 USDT 以外计价），改这里：
    plan = DOWNLOAD_PLAN

    for symbol, intervals in plan.items():
        for interval in intervals:
            out_csv = os.path.join(data_dir, f"{symbol}_{interval}.csv")
            # 从最早开始（0 ms）
            fetch_full_history(symbol=symbol, interval=interval, out_csv=out_csv, start_ms=0, request_pause=0.2)

    print("All tasks finished.")

if __name__ == "__main__":
    main()
