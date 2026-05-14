import requests
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

class BybitMarketScanner:
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        self.category = "linear"  # 只看 U 本位永续合约

    def request(self, endpoint, params):
        url = f"{self.base_url}{endpoint}"
        try:
            res = requests.get(url, params=params, timeout=10)
            return res.json()
        except Exception as e:
            print(f"网络异常: {e}")
            return None

    def get_all_symbols(self):
        """获取所有在线的 U 本位交易对"""
        res = self.request("/v5/market/instruments-info", {"category": self.category})
        if res and res['retCode'] == 0:
            # 过滤掉非 USDT 结算和已经下线的币
            return [i['symbol'] for i in res['result']['list'] if i['quoteCoin'] == 'USDT' and i['status'] == 'Trading']
        return []

    def get_klines(self, symbol, interval, limit=1000):
        """获取历史 K 线数据"""
        params = {
            "category": self.category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        res = self.request("/v5/market/kline", params)
        if res and res['retCode'] == 0:
            data = res['result']['list']
            # 返回正序收盘价 [最旧 -> 最新]
            closes = [float(k[4]) for k in data]
            closes.reverse()
            highs = [float(k[2]) for k in data]
            lows = [float(k[3]) for k in data]
            return np.array(closes), np.array(highs), np.array(lows)
        return None, None, None

    def calculate_hurst(self, ts):
        """
        计算赫斯特指数 (Hurst Exponent)
        H < 0.5: 均值回归 (震荡)
        H > 0.5: 趋势延续
        H = 0.5: 随机游走
        """
        if len(ts) < 100: return 0.5
        lags = range(2, 100)
        # 计算不同滞后下的标准差
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        # 回归计算斜率
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    def calculate_path_volatility(self, closes):
        """
        计算过程波动率：对数收益率的标准差
        反映的是价格在过程中的'抖动'剧烈程度
        """
        if len(closes) < 2: return 0
        # 计算对数收益率
        log_returns = np.diff(np.log(closes))
        # 返回收益率的标准差 (并进行年化或周期化处理，这里直接取标准差作为相对指标)
        return np.std(log_returns) * 100

    def calculate_efficiency_ratio(self, closes):
        """
        卡夫曼效率比 (Efficiency Ratio)
        ER = 方向性位移 / 路径总长度
        ER 越小，说明价格在'乱走' (噪音多)，更适合做市
        """
        direction = abs(closes[-1] - closes[0])
        volatility = np.sum(np.abs(np.diff(closes)))
        return direction / volatility if volatility != 0 else 1

    def scan(self):
        print(f"🚀 开始扫描 Bybit (侧重路径波动与噪音分析)...")
        symbols = self.get_all_symbols()
        results = []

        for symbol in symbols[:80]:
            # 1. 获取 1 分钟 K 线 (获取最近 1440 根，即 24 小时的精细跳动)
            closes_1m, _, _ = self.get_klines(symbol, "1", limit=1440)
            if closes_1m is None or len(closes_1m) < 100: continue

            # 2. 计算赫斯特指数 (长期基因)
            h_val = self.calculate_hurst(closes_1m)

            # 3.  计算过程波动率 (跳动剧烈度)
            path_vol = self.calculate_path_volatility(closes_1m)

            # 4.  计算效率比 (越低代表噪音越大)
            er_val = self.calculate_efficiency_ratio(closes_1m)

            results.append({
                "symbol": symbol,
                "hurst": h_val,
                "path_vol": path_vol,
                "er": er_val
            })
            
            print(f"🔍 分析中: {symbol:10} | Hurst: {h_val:.3f} | 路径波动: {path_vol:.4f} | 效率比: {er_val:.4f}")
            time.sleep(0.1)

        # 排序：我们寻找 Hurst 小、路径波动高、效率比低的标的
        # 综合得分 = (1/hurst) * path_vol * (1/er)
        sorted_list = sorted(results, key=lambda x: (1/x['hurst'] if x['hurst'] > 0 else 0) * x['path_vol'] * (1/x['er']), reverse=True)

        print("\n" + "🏆 最终推荐：高噪音、高跳动、均值回归标的")
        for i, item in enumerate(sorted_list[:5]):
            print(f"TOP {i+1}: {item['symbol']} | 路径波动(抖动度): {item['path_vol']:.4f} | 噪音水平(1/ER): {1/item['er']:.2f}")

if __name__ == "__main__":
    scanner = BybitMarketScanner()
    scanner.scan()