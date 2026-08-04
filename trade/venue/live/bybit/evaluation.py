import requests
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

class BybitMarketScanner:
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        self.category = "linear"  # USDT margined perpetuals only

    def request(self, endpoint, params):
        url = f"{self.base_url}{endpoint}"
        try:
            res = requests.get(url, params=params, timeout=10)
            return res.json()
        except Exception as e:
            print(f"网络异常: {e}")
            return None

    def get_all_symbols(self):
        """Every online USDT margined symbol"""
        res = self.request("/v5/market/instruments-info", {"category": self.category})
        if res and res['retCode'] == 0:
            # Drop non USDT settled and delisted symbols
            return [i['symbol'] for i in res['result']['list'] if i['quoteCoin'] == 'USDT' and i['status'] == 'Trading']
        return []

    def get_klines(self, symbol, interval, limit=1000):
        """Historical kline data"""
        params = {
            "category": self.category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        res = self.request("/v5/market/kline", params)
        if res and res['retCode'] == 0:
            data = res['result']['list']
            # Return the close prices in chronological order [oldest -> newest]
            closes = [float(k[4]) for k in data]
            closes.reverse()
            highs = [float(k[2]) for k in data]
            lows = [float(k[3]) for k in data]
            return np.array(closes), np.array(highs), np.array(lows)
        return None, None, None

    def calculate_hurst(self, ts):
        """
        Hurst exponent
        H < 0.5: mean reverting (choppy)
        H > 0.5: trending
        H = 0.5: random walk
        """
        if len(ts) < 100: return 0.5
        lags = range(2, 100)
        # Standard deviation at different lags
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        # Slope from the regression
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    def calculate_path_volatility(self, closes):
        """
        Path volatility: standard deviation of the log returns
        Measures how violently the price jitters along the way
        """
        if len(closes) < 2: return 0
        # Log returns
        log_returns = np.diff(np.log(closes))
        # Return their standard deviation (could be annualized; the raw std is used as a relative metric)
        return np.std(log_returns) * 100

    def calculate_efficiency_ratio(self, closes):
        """
        Kaufman efficiency ratio
        ER = directional displacement / total path length
        The lower the ER, the more the price 'wanders' (noisy), which suits market making
        """
        direction = abs(closes[-1] - closes[0])
        volatility = np.sum(np.abs(np.diff(closes)))
        return direction / volatility if volatility != 0 else 1

    def scan(self):
        print(f"🚀 开始扫描 Bybit (侧重路径波动与噪音分析)...")
        symbols = self.get_all_symbols()
        results = []

        for symbol in symbols[:80]:
            # 1. Fetch 1 minute klines (the last 1440, i.e. 24 hours of fine grained moves)
            closes_1m, _, _ = self.get_klines(symbol, "1", limit=1440)
            if closes_1m is None or len(closes_1m) < 100: continue

            # 2. Hurst exponent (long term character)
            h_val = self.calculate_hurst(closes_1m)

            # 3. Path volatility (jitter intensity)
            path_vol = self.calculate_path_volatility(closes_1m)

            # 4. Efficiency ratio (lower means noisier)
            er_val = self.calculate_efficiency_ratio(closes_1m)

            results.append({
                "symbol": symbol,
                "hurst": h_val,
                "path_vol": path_vol,
                "er": er_val
            })
            
            print(f"🔍 分析中: {symbol:10} | Hurst: {h_val:.3f} | 路径波动: {path_vol:.4f} | 效率比: {er_val:.4f}")
            time.sleep(0.1)

        # Ranking: we want a low Hurst, high path volatility and a low efficiency ratio
        # Combined score = (1/hurst) * path_vol * (1/er)
        sorted_list = sorted(results, key=lambda x: (1/x['hurst'] if x['hurst'] > 0 else 0) * x['path_vol'] * (1/x['er']), reverse=True)

        print("\n" + "🏆 最终推荐：高噪音、高跳动、均值回归标的")
        for i, item in enumerate(sorted_list[:5]):
            print(f"TOP {i+1}: {item['symbol']} | 路径波动(抖动度): {item['path_vol']:.4f} | 噪音水平(1/ER): {1/item['er']:.2f}")

if __name__ == "__main__":
    scanner = BybitMarketScanner()
    scanner.scan()