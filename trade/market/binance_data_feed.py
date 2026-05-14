import time
import logging
import json
import os
import sys
import requests
import pandas as pd

class BinanceDataFeed:
    """
    智能数据馈送器：
    1. 内部维护一个 self.local_cache (DataFrame)。
    2. 启动时自动计算时间并拉取全量历史。
    3. 运行时只拉取增量数据并拼接。
    4. 自动修剪过长的数据，保持内存轻量。
    """
    BASE_URL:dict[str,str] = {"spot":"https://api.binance.com/api/v3/klines",
                "um":"https://fapi.binance.com/fapi/v1/klines",
                "cm":"https://dapi.binance.com/dapi/v1/klines"}
    MAX_LIMIT_PER_REQ = 1000
    #trading_type:str ="um"             #spot  / um(USDT-M Futures) / cm    (Coin-M Futures)   
    def __init__(self, symbol, interval, trading_type:str, max_len=5000):     #"1m"/"5m"/"15m"/"1h"/"4h"/"1d"
        self.symbol = symbol
        self.interval = interval
        self.trading_type = trading_type
        self.url = self.BASE_URL[trading_type]
        self.logger = logging.getLogger("BinanceFeed")
        
        # 核心：内存中的数据缓存
        self.local_cache = None 
        
        # 内存中最多保留多少根 K 线 (防止无限增长)
        # 只要大于 feature 这里的 required_history 即可
        self.max_cache_len = max_len 

    def _process_data(self, data):
        """[内部工具] 原始 List 转 DataFrame"""
        if not data: return None
        
        cols = [
            "open_time_ms_utc", "open", "high", "low", "close", "volume", 
            "close_time_ms", "quote_asset_volume", "number_of_trades", 
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ]
        df = pd.DataFrame(data, columns=cols)
        
        # 类型转换
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume", 
                       "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # 生成时间字符串 (仅用于日志或调试，计算尽量用 ms)
        df["open_time_date_utc"] = pd.to_datetime(df["open_time_ms_utc"], unit="ms", utc=True)\
                                    .dt.strftime("%Y-%m-%d %H:%M:%S")
        return df

    def _fetch_range_api(self, start_ts, end_ts=None):
        """[内部工具] 纯粹的 API 分页抓取逻辑"""
        if end_ts is None:
            end_ts = int(time.time() * 1000)
            
        all_dfs = []
        curr = start_ts
        
        while True:
            if curr >= end_ts: break
            
            params = {
                "symbol": self.symbol, 
                "interval": self.interval,
                "startTime": curr,
                "endTime": end_ts,
                "limit": self.MAX_LIMIT_PER_REQ
            }
            
            try:
                resp = requests.get(self.url, params=params, timeout=5)
                data = resp.json()
            except Exception as e:
                self.logger.error(f"Network error: {e}")
                break

            if not isinstance(data, list) or not data:
                break
                
            df_batch = self._process_data(data)
            all_dfs.append(df_batch)
            
            # 更新游标：最后一根的收盘时间 + 1ms
            last_close = int(data[-1][6])
            curr = last_close + 1
            
            if len(data) < self.MAX_LIMIT_PER_REQ:
                break # 抓完了
            
            time.sleep(0.1)

        if not all_dfs: return None
        return pd.concat(all_dfs, ignore_index=True)

    def initialize_cache(self, required_bars, interval_ms):
        """
        [启动预热] 计算需要的时间，拉取数据并初始化 local_cache
        """
        self.logger.info("Initializing local data cache...")
        
        # 1. 计算回溯时间 (加 1.2 倍 Buff)
        duration_ms = required_bars * interval_ms * 1.2 + 100
        start_time = int(time.time() * 1000) - int(duration_ms)
        
        # 2. 拉取全量
        df = self._fetch_range_api(start_time)
        
        if df is not None and not df.empty:
            # 3. 去重并存入缓存
            df.drop_duplicates("open_time_ms_utc", inplace=True)
            df.sort_values("open_time_ms_utc", inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            self.local_cache = df
            self.logger.info(f"Cache initialized with {len(df)} bars. Last: {df.iloc[-1]['open_time_date_utc']}. Old: {df.iloc[0]['open_time_date_utc']}")
        else:
            raise RuntimeError("Failed to initialize data cache!")

    def get_latest_data(self):
        """
        [增量更新] 获取最新数据，更新缓存，并返回完整的 DataFrame 供特征工程使用
        """
        if self.local_cache is None:
            self.logger.warning("Cache is empty, running initialization...")
            # 这是一个保底，实际上应该在 start 显式调用 initialize
            return None

        # 1. 确定增量抓取的起点
        # 起点 = 缓存中最后一根的收盘时间 + 1ms
        last_k = self.local_cache.iloc[-1]
        start_time = int(last_k["close_time_ms"]) + 1
        
        # 2. 抓取增量 (通常这里只会抓到 0 条或 1-2 条)
        new_df = self._fetch_range_api(start_time)
        
        if new_df is not None and not new_df.empty:
            self.logger.info(f"Updates found: {len(new_df)} new bars.")
            
            # 3. 拼接更新
            # concat 是 pandas 比较昂贵的操作，但对于 (5000行 + 1行) 来说非常快
            self.local_cache = pd.concat([self.local_cache, new_df], ignore_index=True)
            
            # 4. 安全清洗 (去重 + 排序)
            self.local_cache.drop_duplicates("open_time_ms_utc", inplace=True)
            
            # 5. 内存管理 (剪枝)
            # 如果缓存超过最大长度，切掉头部的旧数据
            if len(self.local_cache) > self.max_cache_len:
                self.local_cache = self.local_cache.iloc[-self.max_cache_len:].reset_index(drop=True)
        
        # 6. 返回**副本**给策略使用 (防止外部修改污染缓存)
        # 此时需要剔除未走完的 K 线（Binance API 总是返回最新的一根未闭合 K 线）
        
        # 获取当前系统时间
        current_time = int(time.time() * 1000)
        
        # 这里的 copy 很重要，特征工程会修改 df
        export_df = self.local_cache.copy() 
        
        if not export_df.empty:
            last_close_time = export_df.iloc[-1]["close_time_ms"]
            # 如果最后一根 K 线的收盘时间在未来，说明它没走完 -> 剔除
            if last_close_time > current_time:
                export_df = export_df.iloc[:-1]
        
        return export_df