from enum import IntEnum
import logging,math
import pandas as pd
import numpy as np
import os, colorlog , logging
from datetime import datetime
from data_process.feature import *
from data_process.logger    import setup_logger,setup_session_logger

class Signal(IntEnum):
    INVALID = -1
    SHORT = 0
    NEUTRAL = 1
    LONG = 2

#define model
CANDLESTICK_NUM = 120
PREDICT_NUM = 16
# 波动率系数 (0.5 ~ 1.0 之间调整)
'''
乘数 (Multiplier),阈值位置,含义
VOL_MULTIPLIER=1.0,1σ,约 31.8% 的价格变动会超出这个阈值（上下尾部）。
VOL_MULTIPLIER=0.5,0.5σ,约 61.7% 的价格变动会超出这个阈值。信号数量适中。
VOL_MULTIPLIER=1.5,1.5σ,仅约 13.4% 的价格变动会超出这个阈值。
VOL_MULTIPLIER=2.0,2σ,仅约 4.6% 的价格变动会超出这个阈值。    
''' 
VOL_MULTIPLIER = 0.9
# 最小硬阈值 (覆盖手续费+滑点)
MIN_THRESHOLD = 0.01  # 0.25%
STOP_MULTIPLIER_RATE = 0.4
# label_decrease_weak =1 
# label_increase_weak = 3
model_train_rate = 0.8
DATA_PROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DATA_PROCESS_DIR)
TEMPORARY_DIR = os.path.join(PROJECT_DIR , 'output')
PROJECT_DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR),'QuantData','Cryptocurrency')
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "BTCUSDT_15m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "BTCUSDT_15m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "BTCUSDT_5m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "ETHUSDT_15m.csv")
origin_data_path = os.path.join(PROJECT_DATA_DIR, "ETHUSDT_5m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "ETHUSDT_1m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "ETHUSDT_3m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "BNBUSDT_15m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "DOGEUSDT_15m.csv")
# origin_data_path = os.path.join(PROJECT_DATA_DIR, "DOGEUSDT_5m.csv")
DATA_PROCESS_OUT_DIR = os.path.join(DATA_PROCESS_DIR, 'output')
train_data_path = os.path.join(TEMPORARY_DIR, "train_data.csv")
test_data_path  = os.path.join(TEMPORARY_DIR, "test_data.csv")
log_level = logging.INFO

FEATURE_CONFIG: list[FeatureBase] = [
    # 【动量趋势】通过快慢均线差值捕捉价格运行的速度与拐点
    FeatureMACD(fast=12, slow=26, signal=9),
    
    # 【多维均线】构建基于K线根数、天数、周数的多尺度价格支撑/压力位锚点
    FeatureMA(weeks=[7,25], days=[5, 10, 20], bars=[], method='sma', strict=True),
    # 【均价基准】计算特定窗口内成交量加权平均价，提供比收盘价更稳定的价值中轴
    FeatureWAP(vwap_windows=(20, 48, 96)),  #VWAP vs SMA
    
    # 【超买超卖】衡量近期涨跌幅的相对强弱，判断价格是否偏离统计均值过远
    # FeatureRsi(period=14, price_col='close', strict=True, prefix='RSI'),
    
    # 【灵敏摆动】结合最高/最低价在窗口内的相对位置，捕捉极短期的价格反转动能
    FeatureKdj(n=9, m1=3, m2=3, high_col='high', low_col='low', close_col='close', strict=True),
    
    # 【量能活跃】对比当前成交量与历史均值，识别市场参与度的爆发或萎缩
    FeatureVolMa(vol_ma_windows=(5, 10, 20)),
    
    # 【资金额热度】从成交金额（Quote Asset）维度观察市场热度，补充单纯量能的不足
    # FeatureQavMa(vol_ma_windows=(5, 10, 20)),   #VolMa vs QavMa 二选一
    
    # 【累积能量】通过价格涨跌方向累加成交量，观察资金的长线流入流出趋势
    # FeatureOBV(),
    # 【价量强度】根据价格变化百分比加权成交量，更细腻地衡量价量配合的紧密度
    FeaturePVT(),   #OBV/PVT 二选一
    
    # 【资金流向】利用收盘价在极值间的相对位置衡量主动买/卖盘的净流量
    FeatureCFM(cmf_window=20),
    
    # 【资金指数】结合典型价(TP)与成交量，判断资金在当前周期内的真实驱动力
    FeatureMFI(mfi_window=14),  #MFI vs RSI 二选一
    
    # 【交易力度】统计平均每笔成交的量能（ATS），区分散户行为与机构大单活动
    FeatureATS(),
    
    # 【微观结构】解构单根K线的实体、影线比例及跳空，捕捉最细微的多空博弈痕迹
    FeatureCandle(),
]

def attach_attr(df):
    # 1. 基础处理
    # df.drop('ignore', axis=1, inplace=True)
    # --- 2. 指标计算 (生成所有原始、未缩放的特征列) ---
    # df = add_relative_features(df)
    FeatureFactory(FEATURE_CONFIG).generate(df)

def attach_label(df, 
                                 candlestick_num:int = CANDLESTICK_NUM, 
                                 predict_num:int = PREDICT_NUM, 
                                 vol_multiplier = VOL_MULTIPLIER,
                                 stop_multiplier_rate = STOP_MULTIPLIER_RATE, # 这里传入比例，如 0.3 或 0.5
                                 min_threshold = MIN_THRESHOLD):
    """
    打标签逻辑 (盈亏比绑定版):
    止损阈值不再独立计算，而是止盈阈值的一个百分比。
    
    例如: vol_multiplier=2.0 (预期涨2倍波动), stop_multiplier_rate=0.5
    => 目标涨幅 = 2.0 * Vol
    => 容忍跌幅 = 1.0 * Vol (即目标的 50%)
    """
    # 0. 基础检查
    assert 'close' in df.columns and 'high' in df.columns and 'low' in df.columns, "缺少 OHLC 数据"
    assert predict_num > 0, "predict_num 必须 > 0"
    print(f"[Labeling] Target Vol: {vol_multiplier}, Stop Rate: {stop_multiplier_rate*100}%")
    
    vol_window = candlestick_num
    df = calculate_thresholds(df, candlestick_num, predict_num, vol_multiplier, stop_multiplier_rate, min_threshold)
    # 3. 获取未来数据 (含路径依赖检查)
    # -------------------------------------------------------------------------
    # 最终收益率
    future_close = df['close'].shift(-predict_num)
    pct_final = (future_close - df['close']) / df['close']
    
    # 期间最低价 (检查多单止损)
    future_low_min = df['low'].rolling(window=predict_num).min().shift(-predict_num)
    # 期间最高价 (检查空单止损)
    future_high_max = df['high'].rolling(window=predict_num).max().shift(-predict_num)

    # 计算期间最大回撤/反向波动
    max_drawdown = (future_low_min - df['close']) / df['close'] 
    max_runup = (future_high_max - df['close']) / df['close']

    # 4. 打标签 (Labeling)
    # -------------------------------------------------------------------------
    
    # === Label 2: 做多 ===
    # 1. 最终涨幅 > 目标
    # 2. 期间最大跌幅 > -止损 (即没有跌破止损线)
    cond_long = (pct_final > df['threshold']) & \
                (max_drawdown > -df['stop_threshold'])

    # === Label 0: 做空 ===
    # 1. 最终跌幅 < -目标
    # 2. 期间最大涨幅 < 止损 (即没有涨破止损线)
    cond_short = (pct_final < -df['threshold']) & \
                 (max_runup < df['stop_threshold'])

    # 应用标签
    df['label'] = np.select(
        [cond_short, cond_long],
        [Signal.SHORT, Signal.LONG],
        default=Signal.NEUTRAL
    )

    # 5. 清洗
    df.iloc[:vol_window, df.columns.get_loc('label')] = Signal.NEUTRAL
    
    if predict_num > 0:
        df = df.iloc[:-predict_num].reset_index(drop=True)

    df['label'] = df['label'].astype(int)

    df['return_rate'] = pct_final   #must be drop when train!!!
        
    return df

def attach_label_v2(df, 
                    candlestick_num=CANDLESTICK_NUM, 
                    predict_num=PREDICT_NUM, 
                    vol_multiplier=VOL_MULTIPLIER,
                    stop_multiplier_rate=STOP_MULTIPLIER_RATE,
                    min_threshold=MIN_THRESHOLD):
    # 0. 基础检查
    time_col = 'open_time_ms_utc'
    assert all(c in df.columns for c in ['close', 'high', 'low', time_col]), "数据缺少 OHLC 或时间戳"
    
    # 1. 计算标准时间步长和目标时间跨度
    time_values = df[time_col].values
    time_diffs = np.diff(time_values)
    expected_interval = np.median(time_diffs[time_diffs > 0])
    target_timespan = predict_num * expected_interval
    
    # 2. 计算动态阈值 (保持你原有的逻辑)
    df = calculate_thresholds(df, candlestick_num, predict_num, vol_multiplier, stop_multiplier_rate, min_threshold)

    # ==============================================================================
    # 3. 【核心进化】：物理时间锚定搜索
    # ==============================================================================
    # 我们要找每一行对应的：当前时间 + target_timespan
    target_times = time_values + target_timespan
    
    # 使用 searchsorted 找到“最接近或刚刚超过”目标时间的索引
    # side='left' 确保我们找到的是物理时间上 >= 目标时间的第一个点
    target_indices = np.searchsorted(time_values, target_times, side='left')
    
    # 边界处理：超过长度的设为无效
    invalid_mask = target_indices >= len(df)
    target_indices[invalid_mask] = 0 # 临时占位，稍后用 mask 过滤
    
    # 验证找到的 K 线与目标时间的偏差 (容忍度 0.5 个周期)
    actual_found_times = time_values[target_indices]
    time_gap = np.abs(actual_found_times - target_times)
    time_valid_mask = (~invalid_mask) & (time_gap < (expected_interval * 0.5))

    # 4. 获取物理对齐的未来价格
    # 即使在第 14 行，只要时间戳对，我们就取它的 close
    future_close = np.where(time_valid_mask, df['close'].values[target_indices], np.nan)
    pct_final = (future_close - df['close']) / df['close']

    # 5. 路径依赖检查 (止损检查)
    # 虽然你提到行号会偏，但在断层不严重的情况下，
    # 物理后 16 行的 rolling 依然是捕捉“这段时间内是否爆仓”的最快方式。
    # 严谨起见，我们依然使用原本的 rolling，但最终标签由 time_valid_mask 守护。
    future_low_min = df['low'].rolling(window=predict_num).min().shift(-predict_num)
    future_high_max = df['high'].rolling(window=predict_num).max().shift(-predict_num)
    
    max_drawdown = (future_low_min - df['close']) / df['close']
    max_runup = (future_high_max - df['close']) / df['close']

    # 6. 打标签逻辑
    # -------------------------------------------------------------------------
    # 条件 1: 时间必须对齐 (没有掉进时空裂缝)
    # 条件 2: 满足涨跌幅和止损限制
    cond_long = time_valid_mask & \
                (pct_final > df['threshold']) & \
                (max_drawdown > -df['stop_threshold'])

    cond_short = time_valid_mask & \
                 (pct_final < -df['threshold']) & \
                 (max_runup < df['stop_threshold'])
    
    # 定义标签：
    # -1: 废弃 (时间不对齐或数据不足)
    #  1: 震荡 (连续但无显著趋势)
    #  0/2: 趋势
    conditions = [
        ~time_valid_mask, # 优先级最高，时间断层直接标记为 -1
        cond_short,
        cond_long
    ]
    choices = [Signal.INVALID, Signal.SHORT, Signal.LONG]
    
    df['label'] = np.select(conditions, choices, default=Signal.NEUTRAL)
    
    # 7. 最终清洗
    # 预热期设为震荡或无效
    df.iloc[:candlestick_num, df.columns.get_loc('label')] = Signal.INVALID
    
    # 记录收益率用于调试 (注意此时 pct_final 是物理对齐的)
    df['return_rate'] = pct_final 
    
    # 过滤掉 -1 的行 (可选，也可以留给 Dataset 处理)
    # 如果你想在这里保持 df 长度不变，就不要执行 drop
    # 但建议在 Dataset 侧利用 label != -1 进行过滤
    
    df['label'] = df['label'].astype(int)
    return df

def compare_labeling_quality(df_raw, candlestick_num = CANDLESTICK_NUM, predict_num = PREDICT_NUM):
    """
    比较两种打标签方法的差异并输出审计报告
    """
    print("🚀 Starting Labeling Audit...")
    
    # 1. 分别运行两个版本
    # 注意：V1 内部可能会 reset_index 或 drop 最后几行，所以我们用副本
    df_v1 = attach_label(df_raw.copy(), candlestick_num, predict_num)
    df_v2 = attach_label_v2(df_raw.copy(), candlestick_num, predict_num)
    
    # 2. 对齐数据
    # 因为 V1 和 V2 对尾部处理可能不同，我们以时间戳作为 Key 进行合并
    # 这样可以确保我们比较的是同一个物理时刻
    time_col = 'open_time_ms_utc'
    comparison = pd.merge(
        df_v1[[time_col, 'label']].rename(columns={'label': 'v1_label'}),
        df_v2[[time_col, 'label']].rename(columns={'label': 'v2_label'}),
        on=time_col,
        how='inner'
    )
    
    total_common = len(comparison)
    
    # 3. 计算基础统计
    matches = (comparison['v1_label'] == comparison['v2_label']).sum()
    match_rate = matches / total_common * 100
    
    print(f"\n📊 --- Overall Statistics ---")
    print(f"Total Aligned Rows: {total_common}")
    print(f"Exact Matches:      {matches} ({match_rate:.2f}%)")
    print(f"Mismatches:         {total_common - matches}")

    # 4. 深度分析：V1 认为有效但 V2 认为无效 (纠偏分析)
    # 这部分通常是 V2 发现时间断层后主动放弃的噪声
    v2_invalid_but_v1_active = comparison[
        (comparison['v2_label'] == -1) & (comparison['v1_label'] != -1)
    ]
    
    print(f"\n🛡️  --- Correction Analysis ---")
    print(f"V1 labels invalidated by V2: {len(v2_invalid_but_v1_active)}")
    if len(v2_invalid_but_v1_active) > 0:
        print(f"Breakdown of V1 labels that were actually 'Time Gaps':")
        print(v2_invalid_but_v1_active['v1_label'].value_counts())

    # 5. 信号转换分析 (0, 1, 2 之间的漂移)
    # 排除掉 -1 之后，看剩下的有效信号是否一致
    active_mask = (comparison['v1_label'] != -1) & (comparison['v2_label'] != -1)
    active_comp = comparison[active_mask]
    
    if len(active_comp) > 0:
        active_matches = (active_comp['v1_label'] == active_comp['v2_label']).sum()
        print(f"\n🎯 --- Signal Consistency (Excluding Invalids) ---")
        print(f"Active Signal Match Rate: {active_matches / len(active_comp) * 100:.2f}%")
        
        # 交叉表：直观查看 0->2 或者 1->0 这种逻辑漂移
        cross_tab = pd.crosstab(active_comp['v1_label'], active_comp['v2_label'], 
                                rownames=['V1 (Row-based)'], colnames=['V2 (Time-based)'])
        print("\nConfusion Matrix (Active Signals Only):")
        print(cross_tab)

    return comparison

def calculate_thresholds(df, 
                         candlestick_num: int = CANDLESTICK_NUM, 
                         predict_num: int = PREDICT_NUM, 
                         vol_multiplier = VOL_MULTIPLIER,
                         stop_multiplier_rate = STOP_MULTIPLIER_RATE, 
                         min_threshold = MIN_THRESHOLD):
    """
    【核心逻辑提取】计算动态止盈 (threshold) 和 止损 (stop_threshold)
    该函数同时用于:
    1. 训练前的数据预处理 (attach_label)
    2. 回测/实盘中的信号生成 (simulation/production)
    """
    # 基础检查
    assert 'close' in df.columns, "缺少 Close 数据"
    
    # 1. 计算波动率基准 (Rolling Standard Deviation)
    vol_window = candlestick_num
    returns = df['close'].pct_change()
    rolling_std = returns.rolling(window=vol_window).std()
    
    # 2. 时间扩充波动率 (Time Scaling: sigma * sqrt(T))
    expected_vol = rolling_std * np.sqrt(predict_num)
    
    # 3. 计算动态阈值
    # A. 目标止盈阈值 (基于波动率)
    target_threshold = (expected_vol * vol_multiplier).clip(lower=min_threshold)
    
    # B. 目标止损阈值 (基于止盈的百分比，保持盈亏比)
    stop_threshold = target_threshold * stop_multiplier_rate
    
    # 4. 写入 DataFrame (处理 NaN)
    # 使用 copy 防止 SettingWithCopyWarning (视调用情况而定，这里直接赋值通常没问题)
    df['threshold'] = target_threshold.fillna(min_threshold)
    df['stop_threshold'] = stop_threshold.fillna(min_threshold * stop_multiplier_rate)
    
    return df

def data_analyze(df, candlestick_num:int = CANDLESTICK_NUM, predict_num:int= PREDICT_NUM , vol_multiplier = VOL_MULTIPLIER,
                 min_threshold = MIN_THRESHOLD):
    # ============================================================
    # === 【新增功能】 计算未来窗口内的最大上涨和最大下跌比例 ===
    # ============================================================
    # 逻辑说明：
    # 1. rolling(window=N).max() 计算的是 [t-N+1, t] 的最大值
    # 2. shift(-N) 将数据向上平移，使得索引 t 处的数据变成原索引 t+N 处的数据
    # 3. 结合起来：在索引 t 处，我们要的是 [t+1, t+N] 的极值
    
    # 未来的最高价序列 (窗口: t+1 到 t+predict_num)
    future_rolling_max = df['high'].rolling(window=predict_num).max().shift(-predict_num)
    # 未来的最低价序列 (窗口: t+1 到 t+predict_num)
    future_rolling_min = df['low'].rolling(window=predict_num).min().shift(-predict_num)
    
    # 计算相对于当前收盘价的涨跌幅
    # max_up: 未来窗口内能卖到的最高收益率
    df['max_up'] = (future_rolling_max - df['close']) / df['close']
    # max_down: 未来窗口内可能承受的最大亏损率
    df['max_down'] = (future_rolling_min - df['close']) / df['close']

def float_range(start, end, step):
    """
    Like range(), but for float.
    Inclusive of end (with tolerance).
    """
    values = []
    v = start
    eps = step / 10

    while v <= end + eps:
        values.append(round(v, 10))
        v += step

    return values

class FeatureFactory:
    def __init__(self, config_list):
        self.features = config_list
        self._X = None
        self._feature_index = None
        self._base_stats_pool = None

    def generate(self,df):
        for f in self.features:     f.generate(df)

    def _prepare_normalize_context(self, X, feature_cols):
        self._X = X
        self._feature_cols = tuple(feature_cols)
        self._feature_index = {f: i for i, f in enumerate(feature_cols)}
        self._base_stats_pool = {}

    def get_base_stats(self, base_feature):
        if self._X is None or self._feature_index is None:
            raise RuntimeError("prepare_normalize_context() must be called first")

        base_idx = (
            self._feature_index[base_feature]
            if isinstance(base_feature, str)
            else base_feature
        )

        if base_idx not in self._base_stats_pool:
            base = self._X[:, :, base_idx]
            mu = np.nanmean(base, axis=1, keepdims=True)
            sigma = np.nanstd(base, axis=1, keepdims=True)
            denom = sigma + 0.1 * np.abs(mu) + EPS
            self._base_stats_pool[base_idx] = (mu, denom)

        return self._base_stats_pool[base_idx]

    def normalize(self, X: np.ndarray, feature_cols: list[str]):
        self._prepare_normalize_context(X, feature_cols)
        for f in self.features:
            f.normalize(X, feature_cols, self)

    def get_global_min_history(self, kline_interval_ms:int) -> int:
        """遍历所有已注册特征，返回其中最大的历史需求"""
        return max([f.min_history_request(kline_interval_ms) for f in self.features])