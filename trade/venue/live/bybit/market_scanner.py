import time
import os,sys
import math
import pandas as pd
from datetime import datetime
import logging

# Import your Bybit engine (only needs an API key for read-only queries)
from bybit_engine import BybitEngine 
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", "..", "..", ".."))
from data_process import common
# Logging setup
class MarketScanner:
    def __init__(self, engine: BybitEngine):
        self.engine = engine
        self.logger, _ = common.setup_session_logger(
            sub_folder=self.__class__.__name__, 
            console_level=logging.INFO, 
            file_level=logging.INFO
        )

    def get_top_liquid_coins(self, top_n=50):
        """
        Top N USDT contracts by turnover
        """
        self.logger.info(f"🔍 正在扫描全市场，寻找成交额 Top {top_n} 的标的...")
        all_tickers = []
        cursor = ""
        
        try:
            while True:
                # Fetch the tickers of every linear contract
                res = self.engine.http.get_tickers(category="linear", limit=100, cursor=cursor)
                if res.get('retCode') != 0: break
                
                data = res.get('result', {})
                all_tickers.extend(data.get('list', []))
                cursor = data.get('nextPageCursor', "")
                if not cursor: break
            
            # Keep the USDT perpetuals
            usdt_tickers = [
                t for t in all_tickers 
                if t['symbol'].endswith('USDT') and 'turnover24h' in t
            ]
            
            # Sort by turnover, descending
            sorted_tickers = sorted(
                usdt_tickers, 
                key=lambda x: float(x.get('turnover24h', 0)), 
                reverse=True
            )
            
            # Return the top N
            return sorted_tickers[:top_n]
            
        except Exception as e:
            self.logger.error(f"❌ 扫描市场失败: {e}")
            return []
        
    def analyze_coin_v3(self, ticker_info):
        """
        Deep evaluation V3: with per-segment consistency check
        """
        symbol = ticker_info['symbol']
        res = self.engine.http.get_kline(category="linear", symbol=symbol, interval=60, limit=72)
        
        if res.get('retCode') != 0 or not res.get('result', {}).get('list'):
            return None

        k_data = res['result']['list']
        closes = [float(k[4]) for k in k_data]
        closes.reverse() 

        # Returns
        diffs = [abs((closes[i] - closes[i-1]) / closes[i-1]) * 100 for i in range(1, len(closes))]
        
        # Stability per segment
        segment_stabilities = []
        for i in range(3):
            seg = sorted(diffs[i*24 : (i+1)*24])
            if seg:
                s_50, s_95 = seg[int(len(seg)*0.5)], seg[int(len(seg)*0.95)]
                segment_stabilities.append(s_50 / s_95 if s_95 > 0 else 0)

        avg_stability = sum(segment_stabilities) / 3
        min_stability = min(segment_stabilities)
        # Consistency error: how much the daily behaviour fluctuates
        cons_err = math.sqrt(sum((s - avg_stability)**2 for s in segment_stabilities) / 3)

        #  Uniform key names: the keys defined here must be found when printing later
        return {
            "Symbol": symbol,
            "Vol_50": round(sorted(diffs)[int(len(diffs)*0.5)], 3), # median of the everyday moves
            "Min_Stab": round(min_stability, 2),                   # worst stability across the 3 days
            "Avg_Stab": round(avg_stability, 2),                   # average stability
            "Cons_Err": round(cons_err, 3),                        # style consistency error
            "Z-Score": round((closes[-1] - (sum(closes)/len(closes))) / (math.sqrt(sum((c - (sum(closes)/len(closes)))**2 for c in closes)/len(closes))), 2),
            "Turnover(M)": round(float(ticker_info.get('turnover24h', 0)) / 1_000_000, 2)
        }
  
    def analyze_coin_v2(self, ticker_info):
        symbol = ticker_info['symbol']
        # More klines, so there are enough samples for the quantiles (e.g. 200)
        res = self.engine.http.get_kline(category="linear", symbol=symbol, interval=60, limit=200)
        
        if res.get('retCode') != 0 or not res.get('result', {}).get('list'):
            return None

        k_data = res['result']['list']
        closes = [float(k[4]) for k in k_data]
        closes.reverse() 

        # 1. Absolute return of every minute/hour
        returns = []
        for i in range(1, len(closes)):
            # The absolute rate of change directly reflects the size of the "jitter"
            ret = abs((closes[i] - closes[i-1]) / closes[i-1]) * 100
            returns.append(ret)
        
        # 2. Quantiles (percentage returns)
        returns.sort()
        n = len(returns)
        
        vol_50 = returns[int(n * 0.50)]  # typical move (median)
        vol_95 = returns[int(n * 0.95)]  # extreme move (95th percentile)
        vol_avg = sum(returns) / n        # average move
        
        # 3. Stability score (ratio)
        # The higher the ratio, the closer the extreme 95% is to the ordinary 50%, i.e. a smoother market
        stability = (vol_50 / vol_95) if vol_95 > 0 else 0

        # 4. Z-score logic unchanged
        ma = sum(closes) / n
        std_dev = math.sqrt(sum((c - ma)**2 for c in closes) / n)
        z_score = (closes[-1] - ma) / std_dev if std_dev > 0 else 0

        return {
            "Symbol": symbol,
            "Vol_Avg(%)": round(vol_avg, 3),
            "Vol_50(%)": round(vol_50, 3),  #  what we care about most: everyday activity
            "Vol_95(%)": round(vol_95, 3),  # risk warning: strength of the extreme wicks
            "Stability": round(stability, 2), # stability: the higher the better for a grid
            "Z-Score": round(z_score, 2),
            "Turnover(M)": round(float(ticker_info.get('turnover24h', 0)) / 1_000_000, 2)
        }

    def analyze_coin(self, ticker_info):
        """
        Deep evaluation of how well one symbol suits a grid
        """
        symbol = ticker_info['symbol']
        turnover = float(ticker_info.get('turnover24h', 0))
        
        # 1. Fetch the last 3 days of 1 hour klines (for the short term volatility)
        # 72 klines
        res = self.engine.http.get_kline(category="linear", symbol=symbol, interval=60, limit=72)
        
        if res.get('retCode') != 0 or not res.get('result', {}).get('list'):
            return None

        # Bybit returns the data in reverse order (index 0 is the newest)
        k_data = res['result']['list']
        # Convert into a list of floats [Open, High, Low, Close, Volume, ...]
        closes = [float(k[4]) for k in k_data]
        highs = [float(k[2]) for k in k_data]
        lows = [float(k[3]) for k in k_data]
        
        # Reverse the list into chronological order (old -> new) to compute the indicators
        closes.reverse() 
        highs.reverse()
        lows.reverse()
        
        # --- indicators ---
        
        # A. Volatility: standard deviation of the log returns
        # A statistical measure: the larger it is, the more violently the price moves and the more a grid earns
        returns = []
        for i in range(1, len(closes)):
            r = math.log(closes[i] / closes[i-1])
            returns.append(r)
        
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret)**2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        # Annualized volatility (reference) or simply the per-period percentage
        volatility_pct = std_dev * 100 

        # B. Amplitude: (high - low) / low
        # How far the price travelled over the last 3 days
        period_high = max(highs)
        period_low = min(lows)
        amplitude_pct = ((period_high - period_low) / period_low) * 100

        # C. Z-score (mean reversion potential)
        # How many standard deviations the price is away from the mean
        curr_price = closes[-1]
        ma = sum(closes) / len(closes)
        price_std = math.sqrt(sum((c - ma)**2 for c in closes) / len(closes))
        z_score = (curr_price - ma) / price_std if price_std > 0 else 0
        
        # D. Quick trend assessment
        # Simple SMA slope check
        trend = "震荡"
        if z_score > 2.0: trend = "超买/拉升"
        elif z_score < -2.0: trend = "超卖/暴跌"
        elif volatility_pct < 0.5: trend = "dead fish" # too little movement

        return {
            "Symbol": symbol,
            "Price": curr_price,
            "Turnover(M)": round(turnover / 1_000_000, 2), # in millions of USDT
            "Volatility(%)": round(volatility_pct, 3),     # higher is better
            "Amplitude(%)": round(amplitude_pct, 2),       # higher is better
            "Z-Score": round(z_score, 2),                  # the closer to 0 the safer, a large absolute value means risk
            "Trend": trend
        }

    def run_report(self):
        # 1. Get the list
        top_coins = self.get_top_liquid_coins(50)
        self.logger.info(f"✅ 锁定 Top 50 流动性标的，开始逐一评估...")
        
        results = []
        
        for i, coin in enumerate(top_coins):
            # Print the progress
            print(f"\r⏳ 正在分析 [{i+1}/50]: {coin['symbol']} ...", end="", flush=True)
            
            metrics = self.analyze_coin_v3(coin)
            if metrics:
                results.append(metrics)
            
            # Stay under the API rate limit
            time.sleep(0.1)
        
        print("\n") # newline
        
        df = pd.DataFrame(results)
        
        #  Fix: sort and filter with the new key names defined above
        # Sort by Min_Stab (worst case stability) first, to find the symbols that stay stable every day
        df_sorted = df.sort_values(by="Min_Stab", ascending=False)
        
        print("\n" + "="*100)
        print("📊 市场多维扫描报告 (分段一致性版)")
        print("="*100)
        
        #  Make sure these column names match the keys returned by analyze_coin exactly
        display_cols = ["Symbol", "Turnover(M)", "Vol_50", "Min_Stab", "Avg_Stab", "Cons_Err", "Z-Score"]
        print(df_sorted[display_cols].to_string(index=False))
        print("="*100)

        # 5. Save the file
        filename = f"crypto_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df_sorted.to_csv(filename, index=False)
        self.logger.info(f"💾 报告已保存至: {filename}")
        
        return df_sorted

if __name__ == "__main__":
    # Put your API key here (read-only market data needs no key, but the Engine class expects one)
    # or change BybitEngine to accept empty credentials
    BASE = os.path.dirname(os.path.abspath(__file__))
    # A real key is recommended here to avoid rate limiting
    API_K = os.path.join(BASE, "keys", "hmac_api_key")
    API_S = os.path.join(BASE, "keys", "hmac_secret")
    RSA_K = os.path.join(BASE, "keys", "api_key")     
    RSA_P = os.path.join(BASE, "keys", "bybit_rsa.pem")
    
    # Initialize
    try:
        engine = BybitEngine(API_K, API_S, RSA_K, RSA_P)
        scanner = MarketScanner(engine)
        scanner.run_report()
    except Exception as e:
        print(f"请检查配置: {e}")