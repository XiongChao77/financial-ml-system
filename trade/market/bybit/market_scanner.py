import time
import os,sys
import math
import pandas as pd
from datetime import datetime
import logging

# 引入你的 Bybit 引擎 (只需要 API Key 做只读查询)
from bybit_engine import BybitEngine 
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..' , '..'))
from data_process import common
# 配置日志
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
        获取成交额最大的前 N 个 USDT 合约
        """
        self.logger.info(f"🔍 正在扫描全市场，寻找成交额 Top {top_n} 的标的...")
        all_tickers = []
        cursor = ""
        
        try:
            while True:
                # 获取所有 Linear 合约行情
                res = self.engine.http.get_tickers(category="linear", limit=100, cursor=cursor)
                if res.get('retCode') != 0: break
                
                data = res.get('result', {})
                all_tickers.extend(data.get('list', []))
                cursor = data.get('nextPageCursor', "")
                if not cursor: break
            
            # 筛选 USDT 永续
            usdt_tickers = [
                t for t in all_tickers 
                if t['symbol'].endswith('USDT') and 'turnover24h' in t
            ]
            
            # 按成交额降序排序
            sorted_tickers = sorted(
                usdt_tickers, 
                key=lambda x: float(x.get('turnover24h', 0)), 
                reverse=True
            )
            
            # 返回前 N 个
            return sorted_tickers[:top_n]
            
        except Exception as e:
            self.logger.error(f"❌ 扫描市场失败: {e}")
            return []
        
    def analyze_coin_v3(self, ticker_info):
        """
        深度评估 V3：分段一致性校验版本
        """
        symbol = ticker_info['symbol']
        res = self.engine.http.get_kline(category="linear", symbol=symbol, interval=60, limit=72)
        
        if res.get('retCode') != 0 or not res.get('result', {}).get('list'):
            return None

        k_data = res['result']['list']
        closes = [float(k[4]) for k in k_data]
        closes.reverse() 

        # 计算收益率
        diffs = [abs((closes[i] - closes[i-1]) / closes[i-1]) * 100 for i in range(1, len(closes))]
        
        # 分段计算稳定性
        segment_stabilities = []
        for i in range(3):
            seg = sorted(diffs[i*24 : (i+1)*24])
            if seg:
                s_50, s_95 = seg[int(len(seg)*0.5)], seg[int(len(seg)*0.95)]
                segment_stabilities.append(s_50 / s_95 if s_95 > 0 else 0)

        avg_stability = sum(segment_stabilities) / 3
        min_stability = min(segment_stabilities)
        # 一致性误差：每天表现的波动程度
        cons_err = math.sqrt(sum((s - avg_stability)**2 for s in segment_stabilities) / 3)

        #  统一键名：确保这里定义的键，后面打印时能找得到
        return {
            "Symbol": symbol,
            "Vol_50": round(sorted(diffs)[int(len(diffs)*0.5)], 3), # 日常波动中位数
            "Min_Stab": round(min_stability, 2),                   # 3天中表现最差的稳定性
            "Avg_Stab": round(avg_stability, 2),                   # 平均稳定性
            "Cons_Err": round(cons_err, 3),                        # 风格一致性误差
            "Z-Score": round((closes[-1] - (sum(closes)/len(closes))) / (math.sqrt(sum((c - (sum(closes)/len(closes)))**2 for c in closes)/len(closes))), 2),
            "Turnover(M)": round(float(ticker_info.get('turnover24h', 0)) / 1_000_000, 2)
        }
  
    def analyze_coin_v2(self, ticker_info):
        symbol = ticker_info['symbol']
        # 增加 K 线数量，以便获得足够的分位样本 (例如 200 根)
        res = self.engine.http.get_kline(category="linear", symbol=symbol, interval=60, limit=200)
        
        if res.get('retCode') != 0 or not res.get('result', {}).get('list'):
            return None

        k_data = res['result']['list']
        closes = [float(k[4]) for k in k_data]
        closes.reverse() 

        # 1. 计算每一分钟/小时的绝对收益率
        returns = []
        for i in range(1, len(closes)):
            # 使用绝对值变化率，直接反应“震动”幅度
            ret = abs((closes[i] - closes[i-1]) / closes[i-1]) * 100
            returns.append(ret)
        
        # 2. 计算分位数 (百分比收益)
        returns.sort()
        n = len(returns)
        
        vol_50 = returns[int(n * 0.50)]  # 典型波动 (中位数)
        vol_95 = returns[int(n * 0.95)]  # 极端波动 (95%分位)
        vol_avg = sum(returns) / n        # 平均波动
        
        # 3. 计算稳定性评分 (Ratio)
        # 比值越高，说明 95% 的极端情况离 50% 的普通情况不远，走势更温和
        stability = (vol_50 / vol_95) if vol_95 > 0 else 0

        # 4. Z-Score 逻辑保持不变
        ma = sum(closes) / n
        std_dev = math.sqrt(sum((c - ma)**2 for c in closes) / n)
        z_score = (closes[-1] - ma) / std_dev if std_dev > 0 else 0

        return {
            "Symbol": symbol,
            "Vol_Avg(%)": round(vol_avg, 3),
            "Vol_50(%)": round(vol_50, 3),  #  我们最看重的：日常活跃度
            "Vol_95(%)": round(vol_95, 3),  # ⚠️ 风险预警：极端插针强度
            "Stability": round(stability, 2), # 💎 稳定性：越高越适合网格
            "Z-Score": round(z_score, 2),
            "Turnover(M)": round(float(ticker_info.get('turnover24h', 0)) / 1_000_000, 2)
        }

    def analyze_coin(self, ticker_info):
        """
        深度评估单个币种的网格适性
        """
        symbol = ticker_info['symbol']
        turnover = float(ticker_info.get('turnover24h', 0))
        
        # 1. 获取最近 3 天的 1小时 K线 (用于计算短期波动率)
        # 72根K线
        res = self.engine.http.get_kline(category="linear", symbol=symbol, interval=60, limit=72)
        
        if res.get('retCode') != 0 or not res.get('result', {}).get('list'):
            return None

        # Bybit 返回数据是反序的 (index 0 是当前最新)
        k_data = res['result']['list']
        # 转换为浮点数列表 [Open, High, Low, Close, Volume, ...]
        closes = [float(k[4]) for k in k_data]
        highs = [float(k[2]) for k in k_data]
        lows = [float(k[3]) for k in k_data]
        
        # 将列表反转为正序 (旧 -> 新) 以便计算指标
        closes.reverse() 
        highs.reverse()
        lows.reverse()
        
        # --- 计算指标 ---
        
        # A. 波动率 (Volatility): 使用对数收益率的标准差
        # 这是一个统计学指标，数值越大，说明价格跳动越剧烈，网格利润越高
        returns = []
        for i in range(1, len(closes)):
            r = math.log(closes[i] / closes[i-1])
            returns.append(r)
        
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret)**2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        # 年化波动率 (参考) 或直接用周期波动率百分比
        volatility_pct = std_dev * 100 

        # B. 区间振幅 (Amplitude): (最高-最低)/最低
        # 判断过去3天价格跑了多远
        period_high = max(highs)
        period_low = min(lows)
        amplitude_pct = ((period_high - period_low) / period_low) * 100

        # C. Z-Score (均值回归潜力)
        # 当前价格偏离均线多少个标准差
        curr_price = closes[-1]
        ma = sum(closes) / len(closes)
        price_std = math.sqrt(sum((c - ma)**2 for c in closes) / len(closes))
        z_score = (curr_price - ma) / price_std if price_std > 0 else 0
        
        # D. 趋势状态简评
        # 简单的 SMA 斜率判定
        trend = "震荡"
        if z_score > 2.0: trend = "超买/拉升"
        elif z_score < -2.0: trend = "超卖/暴跌"
        elif volatility_pct < 0.5: trend = "死鱼" # 波动太小

        return {
            "Symbol": symbol,
            "Price": curr_price,
            "Turnover(M)": round(turnover / 1_000_000, 2), # 百万 U 单位
            "Volatility(%)": round(volatility_pct, 3),     # 越高越好
            "Amplitude(%)": round(amplitude_pct, 2),       # 越高越好
            "Z-Score": round(z_score, 2),                  # 越接近 0 越安全，绝对值大代表风险
            "Trend": trend
        }

    def run_report(self):
        # 1. 获取名单
        top_coins = self.get_top_liquid_coins(50)
        self.logger.info(f"✅ 锁定 Top 50 流动性标的，开始逐一评估...")
        
        results = []
        
        for i, coin in enumerate(top_coins):
            # 打印进度
            print(f"\r⏳ 正在分析 [{i+1}/50]: {coin['symbol']} ...", end="", flush=True)
            
            metrics = self.analyze_coin_v3(coin)
            if metrics:
                results.append(metrics)
            
            # 防 API 限频
            time.sleep(0.1)
        
        print("\n") # 换行
        
        df = pd.DataFrame(results)
        
        #  修复：使用上面定义的新键名进行排序和筛选
        # 我们优先按 Min_Stab (保底稳定性) 排序，选出每天都稳的币
        df_sorted = df.sort_values(by="Min_Stab", ascending=False)
        
        print("\n" + "="*100)
        print("📊 市场多维扫描报告 (分段一致性版)")
        print("="*100)
        
        #  确保这里的列名与 analyze_coin 返回的 Key 完全一致
        display_cols = ["Symbol", "Turnover(M)", "Vol_50", "Min_Stab", "Avg_Stab", "Cons_Err", "Z-Score"]
        print(df_sorted[display_cols].to_string(index=False))
        print("="*100)

        # 5. 保存文件
        filename = f"crypto_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df_sorted.to_csv(filename, index=False)
        self.logger.info(f"💾 报告已保存至: {filename}")
        
        return df_sorted

if __name__ == "__main__":
    # 填入你的 API Key (如果只是查行情，其实不需要 Key，但为了复用 Engine 类还是填一下)
    # 或者修改 BybitEngine 支持空 Key 初始化
    BASE = os.path.dirname(os.path.abspath(__file__))
    # 这里建议填入真实的 key 以防触发限频
    API_K = os.path.join(BASE, "keys", "hmac_api_key")
    API_S = os.path.join(BASE, "keys", "hmac_secret")
    RSA_K = os.path.join(BASE, "keys", "api_key")     
    RSA_P = os.path.join(BASE, "keys", "bybit_rsa.pem")
    
    # 初始化
    try:
        engine = BybitEngine(API_K, API_S, RSA_K, RSA_P)
        scanner = MarketScanner(engine)
        scanner.run_report()
    except Exception as e:
        print(f"请检查配置: {e}")