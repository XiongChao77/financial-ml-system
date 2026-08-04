import time
import logging
import json
import os
import sys
import requests
import pandas as pd

class BinanceDataFeed:
    """
    Smart data feeder:
    1. Keeps an internal self.local_cache (DataFrame).
    2. Works out the time range on start-up and pulls the full history.
    3. While running it only pulls the increment and appends it.
    4. Trims the data automatically to keep memory light.
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
        
        # Core: the in-memory data cache
        self.local_cache = None 
        
        # Maximum number of klines kept in memory (prevents unbounded growth)
        # Anything above the feature side required_history is fine
        self.max_cache_len = max_len 

    def _process_data(self, data):
        """[internal] raw list -> DataFrame"""
        if not data: return None
        
        cols = [
            "open_time_ms_utc", "open", "high", "low", "close", "volume", 
            "close_time_ms", "quote_asset_volume", "number_of_trades", 
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ]
        df = pd.DataFrame(data, columns=cols)
        
        # Type conversion
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume", 
                       "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Build the time string (for logs/debugging only, computations stay in ms)
        df["open_time_date_utc"] = pd.to_datetime(df["open_time_ms_utc"], unit="ms", utc=True)\
                                    .dt.strftime("%Y-%m-%d %H:%M:%S")
        return df

    def _fetch_range_api(self, start_ts, end_ts=None):
        """[internal] pure API pagination logic"""
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
            
            # Move the cursor: close time of the last kline + 1ms
            last_close = int(data[-1][6])
            curr = last_close + 1
            
            if len(data) < self.MAX_LIMIT_PER_REQ:
                break # done
            
            time.sleep(0.1)

        if not all_dfs: return None
        return pd.concat(all_dfs, ignore_index=True)

    def initialize_cache(self, required_bars, interval_ms):
        """
        [start-up warm-up] work out the required time range, pull the data and initialize local_cache
        """
        self.logger.info("Initializing local data cache...")
        
        # 1. Lookback time (1.2x buffer added)
        duration_ms = required_bars * interval_ms * 1.2 + 100
        start_time = int(time.time() * 1000) - int(duration_ms)
        
        # 2. Pull everything
        df = self._fetch_range_api(start_time)
        
        if df is not None and not df.empty:
            # 3. Deduplicate and store in the cache
            df.drop_duplicates("open_time_ms_utc", inplace=True)
            df.sort_values("open_time_ms_utc", inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            self.local_cache = df
            self.logger.info(f"Cache initialized with {len(df)} bars. Last: {df.iloc[-1]['open_time_date_utc']}. Old: {df.iloc[0]['open_time_date_utc']}")
        else:
            raise RuntimeError("Failed to initialize data cache!")

    def get_latest_data(self):
        """
        [incremental update] fetch the newest data, refresh the cache and return the full DataFrame for feature engineering
        """
        if self.local_cache is None:
            self.logger.warning("Cache is empty, running initialization...")
            # A fallback; initialize should really be called explicitly in start
            return None

        # 1. Where the incremental fetch starts
        # start = close time of the last cached kline + 1ms
        last_k = self.local_cache.iloc[-1]
        start_time = int(last_k["close_time_ms"]) + 1
        
        # 2. Fetch the increment (usually 0 or 1-2 klines)
        new_df = self._fetch_range_api(start_time)
        
        if new_df is not None and not new_df.empty:
            self.logger.info(f"Updates found: {len(new_df)} new bars.")
            
            # 3. Append and update
            # concat is one of the more expensive pandas operations, but (5000 rows + 1 row) is very fast
            self.local_cache = pd.concat([self.local_cache, new_df], ignore_index=True)
            
            # 4. Safety cleanup (deduplicate + sort)
            self.local_cache.drop_duplicates("open_time_ms_utc", inplace=True)
            
            # 5. Memory management (pruning)
            # Cut the old head off once the cache exceeds the maximum length
            if len(self.local_cache) > self.max_cache_len:
                self.local_cache = self.local_cache.iloc[-self.max_cache_len:].reset_index(drop=True)
        
        # 6. Return a **copy** to the strategy (so outside edits cannot pollute the cache)
        # The unfinished kline must be dropped here (the Binance API always returns the latest unclosed kline)
        
        # Current system time
        current_time = int(time.time() * 1000)
        
        # The copy matters here, feature engineering modifies the df
        export_df = self.local_cache.copy() 
        
        if not export_df.empty:
            last_close_time = export_df.iloc[-1]["close_time_ms"]
            # A close time in the future means the last kline is not finished -> drop it
            if last_close_time > current_time:
                export_df = export_df.iloc[:-1]
        
        return export_df