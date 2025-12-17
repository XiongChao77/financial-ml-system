import pandas as pd
import numpy as np
import scipy.stats as stats
import os, sys
"""
对市场数据进行解剖，量化多空差异
"""
# 引入您的数据加载逻辑
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, '..'))
from data_process import common #

def get_market_anatomy(df):
    """
    对市场数据进行解剖，量化多空差异
    """
    # 1. 基础预处理
    df['ret'] = df['close'].pct_change()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 定义上涨K线和下跌K线
    up_candles = df[df['ret'] > 0]
    down_candles = df[df['ret'] < 0]
    
    # ----------------------------------------------------
    # 维度 1: 力度 (Magnitude) - 谁更猛？
    # ----------------------------------------------------
    avg_up_ret = up_candles['ret'].mean()
    avg_down_ret = down_candles['ret'].mean() # 负数
    
    # ----------------------------------------------------
    # 维度 2: 速度 (Velocity) - 极值有多大？
    # ----------------------------------------------------
    max_up_candle = up_candles['ret'].max()
    max_down_candle = down_candles['ret'].min()
    
    # ----------------------------------------------------
    # 维度 3: 波动率 (Volatility) - 谁更不稳定？
    # ----------------------------------------------------
    # 计算下行波动率 (Downside Deviation) vs 上行波动率
    up_vol = up_candles['ret'].std()
    down_vol = down_candles['ret'].std()
    
    # ----------------------------------------------------
    # 维度 4: 成交量 (Volume) - 钱去哪了？
    # ----------------------------------------------------
    avg_vol_up = up_candles['volume'].mean()
    avg_vol_down = down_candles['volume'].mean()
    
    # ----------------------------------------------------
    # 维度 5: 连贯性 (Serial Correlation) - 趋势能持续多久？
    # ----------------------------------------------------
    # 简单的游程检验逻辑：计算连续上涨/下跌的平均长度
    price_diff = df['close'].diff()
    df['direction'] = np.sign(price_diff)
    
    # 这是一个非常巧妙的计算连续次数的方法
    # 当方向变化时，cumsum 会增加，从而分组
    g = df['direction'].ne(df['direction'].shift()).cumsum()
    streaks = df.groupby(g)['direction'].agg(['mean', 'count']) # mean即方向(1/-1), count即长度
    
    avg_up_streak = streaks[streaks['mean'] == 1]['count'].mean()
    avg_down_streak = streaks[streaks['mean'] == -1]['count'].mean()

    # ----------------------------------------------------
    # 维度 6: 极端行情解剖 (Top 5% Volatility)
    # ----------------------------------------------------
    # 取波动率最大的 5% K线
    threshold_top5 = df['ret'].abs().quantile(0.95)
    extreme_df = df[df['ret'].abs() > threshold_top5]
    
    ex_up = extreme_df[extreme_df['ret'] > 0]
    ex_down = extreme_df[extreme_df['ret'] < 0]
    
    print("-" * 70)
    print(f"🔥 极端行情解剖 (Top 5% 剧烈波动时刻):")
    print(f"   • 阈值: > {threshold_top5*100:.2f}% (绝对涨跌幅)")
    print(f"   • 暴涨次数: {len(ex_up)}")
    print(f"   • 暴跌次数: {len(ex_down)} (Ratio: x{len(ex_down)/len(ex_up):.2f})")
    print(f"   • 暴涨均量: {ex_up['volume'].mean():.2f}")
    print(f"   • 暴跌均量: {ex_down['volume'].mean():.2f} (Ratio: x{ex_down['volume'].mean()/ex_up['volume'].mean():.2f})")
    print("="*60)

    # ----------------------------------------------------
    # 输出报告
    # ----------------------------------------------------
    print("="*60)
    print(f"📊 市场多空解剖报告 (Market Anatomy Report)")
    print("="*60)
    print(f"{'Metric':<25} | {'📈 Up (Long)':<15} | {'📉 Down (Short)':<15} | {'Diff Ratio':<10}")
    print("-" * 70)
    
    print(f"{'Avg Return (K线均幅)':<25} | {avg_up_ret*100:.4f}%          | {avg_down_ret*100:.4f}%          | x{abs(avg_down_ret/avg_up_ret):.2f}")
    print(f"{'Max Candle (最大单根)':<25} | {max_up_candle*100:.2f}%           | {max_down_candle*100:.2f}%           | -")
    print(f"{'Volatility (波动率)':<25} | {up_vol:.5f}          | {down_vol:.5f}          | x{down_vol/up_vol:.2f}")
    print(f"{'Avg Volume (均量)':<25} | {avg_vol_up:.2f}          | {avg_vol_down:.2f}          | x{avg_vol_down/avg_vol_up:.2f}")
    print(f"{'Avg Streak (平均连涨/跌)':<25} | {avg_up_streak:.2f} bars         | {avg_down_streak:.2f} bars         | -")
    
    # 统计偏度
    skew = df['ret'].skew()
    kurt = df['ret'].kurtosis()
    print("-" * 70)
    print(f"📐 整体统计特征:")
    print(f"   • Skewness (偏度): {skew:.4f} ({'左偏/暴跌多' if skew < 0 else '右偏/暴涨多'})")
    print(f"   • Kurtosis (峰度): {kurt:.4f} (>{3} 说明是肥尾分布，极端行情多)")
    print("="*60)

if __name__ == "__main__":
    # 读取您的原始数据
    df = pd.read_csv(common.origin_data_path)
    # 简单清洗
    df = df.dropna()
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    get_market_anatomy(df)