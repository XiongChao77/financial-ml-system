import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import datetime,os,sys, re, math, json, logging
current_work_dir = os.path.dirname(__file__) 
sys.path.append(os.path.join(current_work_dir,'..'))
from data_process import common
from data_process.regime_discovery import LabelRegimeAnalyzer

# 模拟你项目中的 EPS 定义
EPS = 1e-9

def plot_doge_vol_variance():
    # 1. 读取数据 (参考 train_old.py 的加载方式)
    file = common.origin_data_path
    # 1. 获取周期字符串并转为毫秒
    interval_str = get_interval_from_filename(file)
    interval_ms = get_interval_ms(interval_str)
    
    # 2. 存入元数据，方便 attach_label_v2 和后续模型使用
    metadata = {
        "symbol_interval": interval_str,
        "interval_ms": interval_ms, # <--- 新增
        "candlestick_num": common.CANDLESTICK_NUM,
        "predict_num": common.PREDICT_NUM,
        "vol_multiplier": common.VOL_MULTIPLIER,
        "stop_multiplier_rate": common.STOP_MULTIPLIER_RATE,
    }

    df = pd.read_csv(file)
    
    # 将时间列转换为 datetime 并设为索引
    if 'open_time_date_utc' in df.columns:
        df['time'] = pd.to_datetime(df['open_time_date_utc'])
        df.set_index('time', inplace=True)
    
    # 转换为浮点数确保计算精度
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    prev_close = close.shift(1)

    # 2. 计算基础波动率 (NATR - Normalized ATR)
    # TR = max(H-L, |H-PC|, |L-PC|)
    tr = np.maximum(high - low, 
                    np.maximum((high - prev_close).abs(), 
                               (low - prev_close).abs()))
    
    # 计算百分比波动率 (当前波幅 / 当前价格)
    natr_raw = tr / (close + EPS)
    
    # 定义窗口：短期看震感，长期看环境方差
    short_w = 14
    long_w = 200
    
    # 短期平均波动 (震感强度)
    short_natr = natr_raw.rolling(short_w).mean()
    
    # 3. 🌟 计算“波动的方差” (CV - 变异系数)
    # 逻辑：长期波动的标准差 / 长期波动的均值
    long_natr_mean = natr_raw.rolling(long_w).mean()
    long_natr_std = natr_raw.rolling(long_w).std()
    
    # CV 越低 = 长期波动越均匀 (你想要的趋势背景)
    # CV 越高 = 波动率本身在剧烈震荡 (市场极其混乱)
    vol_cv = long_natr_std / (long_natr_mean + EPS)
    
    # 4. 计算综合趋势评分 (Burst / Consistency)
    # 爆发比：短期波动 / 长期平均
    burst_ratio = short_natr / (long_natr_mean + EPS)
    # 趋势得分：爆发比越高 且 CV 越低（1/CV越高），得分越高
    trend_vol_score = burst_ratio * (1.0 / (vol_cv + EPS))

    # 5. 开始绘图
    plt.style.use('dark_background') # 使用深色背景更像交易终端
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1, 1, 1.5]})
    
    # Panel 0: 价格
    axes[0].plot(close, label='DOGE Price', color='#F7931A', alpha=0.9)
    axes[0].set_title('DOGE/USDT Price (Trend Discovery)')
    axes[0].legend(loc='upper left')
    axes[0].grid(alpha=0.2)

    # Panel 1: 短期波动率 (震感)
    axes[1].plot(short_natr * 100, label=f'Short NATR {short_w} (%)', color='cyan')
    axes[1].set_title('Short-term Volatility Intensity')
    axes[1].legend(loc='upper left')
    axes[1].grid(alpha=0.2)

    # Panel 2: 波动一致性 (你关心的“方差”)
    # 我们画 1/CV，值越高代表波动越“均匀”，越容易出趋势
    consistency = 1.0 / (vol_cv + EPS)
    axes[2].plot(consistency, label='Volatility Consistency (1/CV)', color='magenta')
    axes[2].axhline(y=np.nanmedian(consistency), color='white', linestyle='--', alpha=0.5, label='Median')
    axes[2].set_title('Volatility Variance Filter (Higher = More Uniform/Predictable)')
    axes[2].legend(loc='upper left')
    axes[2].grid(alpha=0.2)

    # Panel 3: 最终策略执行参考值
    # 当此分数达到峰值时，代表“在极其均匀的背景下突然爆发”
    axes[3].fill_between(trend_vol_score.index, trend_vol_score, 0, 
                         where=(trend_vol_score > np.nanpercentile(trend_vol_score, 75)),
                         color='green', alpha=0.5, label='High Confidence Trend Area')
    axes[3].plot(trend_vol_score, color='lime', linewidth=0.8, alpha=0.3)
    axes[3].set_title('Final Execution Score: Burst / Consistency')
    axes[3].set_yscale('log') # 对数轴看爆发更明显
    axes[3].legend(loc='upper left')
    axes[3].grid(alpha=0.2)

    plt.tight_layout()
    # 自动保存
    output_name = "doge_trend_vol_analysis.png"
    plt.savefig(output_name)
    print(f"✅ 绘图完成，分析结果已保存至: {output_name}")
    plt.show()


def get_interval_from_filename(path: str) -> str:
    """
    从路径中提取时间周期 (如 ETHUSDT_3m.csv -> 3m)
    """
    filename = os.path.basename(path)
    # 匹配 1s, 15s, 1m, 3m... 1M 等格式
    match = re.search(r'_(\d+[smhdwM])\.csv', filename)
    if match:
        return match.group(1)
    return "unknown"

def get_interval_ms(interval_str: str) -> int:
    """
    将周期字符串转换为毫秒数
    支持: 1s, 15s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """
    # 定义基础单位（毫秒）
    units = {
        's': 1000,
        'm': 60 * 1000,
        'h': 60 * 60 * 1000,
        'd': 24 * 60 * 60 * 1000,
        'w': 7 * 24 * 60 * 60 * 1000,
        'M': 30 * 24 * 60 * 60 * 1000  # 按照标准 30 天计算
    }
    
    # 使用正则表达式拆分数字和单位
    match = re.match(r'(\d+)([smhdwM])', interval_str)
    if not match:
        return 0
    
    value, unit = match.groups()
    return int(value) * units[unit]

# 执行示例
plot_doge_vol_variance()
