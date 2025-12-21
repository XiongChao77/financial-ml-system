from enum import IntEnum
from functools import lru_cache
import logging,math
import pandas as pd
import numpy as np
import os, colorlog , logging, json,platform
from datetime import datetime
from data_process.feature import *

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
# --- 核心判断逻辑 ---
TEMPORARY_DIR = os.path.join(PROJECT_DIR, 'output')
if platform.system().lower() != 'windows':
    os.makedirs('/dev/shm/quant', exist_ok=True)
    if not os.path.islink(TEMPORARY_DIR):   os.symlink('/dev/shm/quant', TEMPORARY_DIR)  # Linux/Ubuntu 环境：直接映射到共享内存
PERSISTENCE_DIR = os.path.join(os.path.dirname(PROJECT_DIR),'quant_output')
os.makedirs(PERSISTENCE_DIR, exist_ok=True)

PROJECT_DATA_DIR = os.path.join(os.path.dirname(PROJECT_DIR),'QuantData','Cryptocurrency')
symbol = 'ETHUSDT' # option: 'BTCUSDT' 'ETHUSDT' 'BNBUSDT' 'DOGEUSDT'
interval = '3m' # option: 1s, 15s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
origin_data_path = os.path.join(PROJECT_DATA_DIR, f"{symbol}_{interval}.csv")
train_data_path = os.path.join(TEMPORARY_DIR, "train_data.csv")
test_data_path  = os.path.join(TEMPORARY_DIR, "test_data.csv")
data_config_path  = os.path.join(TEMPORARY_DIR, "data_config_meta.json")
log_level = logging.INFO

CONF_DF = 'to_feather'#/'to_feather'/'to_csv'

def save_train_df(df):
    if CONF_DF == 'to_csv':
        df.to_csv(train_data_path, index=False, encoding="utf-8")
    else:
        # Feather 格式需确保列名为字符串，否则会报错
        df.columns = df.columns.astype(str)
        df.to_feather(train_data_path)

def load_train_df():
    if CONF_DF == 'to_csv':
        return pd.read_csv(train_data_path, encoding="utf-8")
    else:
        return pd.read_feather(train_data_path)

def save_test_df(df):
    if CONF_DF == 'to_csv':
        df.to_csv(test_data_path, index=False, encoding="utf-8")
    else:
        df.columns = df.columns.astype(str)
        df.to_feather(test_data_path)

def load_test_df():
    if CONF_DF == 'to_csv':
        return pd.read_csv(test_data_path, encoding="utf-8")
    else:
        return pd.read_feather(test_data_path)

def attach_attr(df, kline_interval_ms):
    # 1. 基础处理
    # df.drop('ignore', axis=1, inplace=True)
    # --- 2. 指标计算 (生成所有原始、未缩放的特征列) ---
    # df = add_relative_features(df)
    FeatureFactory(FEATURE_CONFIG_LIST, kline_interval_ms).generate(df)

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
                    interval_ms,
                    candlestick_num=CANDLESTICK_NUM, 
                    predict_num=PREDICT_NUM, 
                    vol_multiplier=VOL_MULTIPLIER,
                    stop_multiplier_rate=STOP_MULTIPLIER_RATE,
                    min_threshold=MIN_THRESHOLD):
    time_col = 'open_time_ms_utc'
    time_values = df[time_col].values
    
    # 1. 动态阈值预计算
    df = calculate_thresholds(df, candlestick_num, predict_num, vol_multiplier, stop_multiplier_rate, min_threshold)

    # 2. 物理时间锚定
    target_times = time_values + (predict_num * interval_ms)
    target_indices = np.searchsorted(time_values, target_times, side='left')
    
    # 3. 构建“时空对齐”掩码
    in_bounds = target_indices < len(df)
    safe_idx = np.where(in_bounds, target_indices, 0)
    actual_times = time_values[safe_idx]
    
    # 【核心修正】：只有时间戳完全相等，才算有效
    is_exact_match = in_bounds & (actual_times == target_times)
    final_valid_mask = is_exact_match 

    # 4. 计算收益率与极致检查
    pct_final = (np.where(final_valid_mask, df['close'].values[safe_idx], np.nan) - df['close']) / df['close']
    steps = (target_indices - np.arange(len(df))).clip(1, predict_num)
    
    low_mtx = np.column_stack([df['low'].shift(-i).values for i in range(1, predict_num + 1)])
    high_mtx = np.column_stack([df['high'].shift(-i).values for i in range(1, predict_num + 1)])
    
    future_low_min = np.minimum.accumulate(low_mtx, axis=1)[np.arange(len(df)), steps - 1]
    future_high_max = np.maximum.accumulate(high_mtx, axis=1)[np.arange(len(df)), steps - 1]
    
    max_dd = (future_low_min - df['close']) / df['close']
    max_ru = (future_high_max - df['close']) / df['close']

    # 5. 打标签逻辑
    cond_long = final_valid_mask & (pct_final > df['threshold']) & (max_dd > -df['stop_threshold'])
    cond_short = final_valid_mask & (pct_final < -df['threshold']) & (max_ru < df['stop_threshold'])
    
    # 将 ~final_valid_mask 放在第一位，确保断层直接变 -1
    conditions = [~final_valid_mask, cond_short, cond_long]
    choices = [Signal.INVALID, Signal.SHORT, Signal.LONG]
    
    df['label'] = np.select(conditions, choices, default=Signal.NEUTRAL).astype(int)
    
    # ==============================================================================
    # 🔍 【深度审计】打印原始 ms 与 物理跨度检查
    # ==============================================================================
    invalid_rows = df[df['label'] == Signal.INVALID]
    num_invalid = len(invalid_rows)

    print(f"\n📊 [Label Audit] 发现 {num_invalid} 个无效标签 (Signal.INVALID):")
    
    if num_invalid > 0:
        sample_size = min(30, num_invalid)
        print(f"📍 前 {sample_size} 个无效行明细 (含原始 ms):")
        # 增加一列：物理差值 (Gap_ms)，用来定位到底跳了多少时间
        print(f"{'Index':>8} | {'Raw MS':>15} | {'Target MS':>15} | {'Gap_ms':>10} | {'Reason'}")
        print("-" * 85)
        
        # 预先获取整个时间序列的 numpy 数组用于快速计算
        time_values = df['open_time_ms_utc'].values
        
        for idx, row in invalid_rows.head(sample_size).iterrows():
            curr_ms = int(row['open_time_ms_utc'])
            # 理想中，这根 K 线对应的预测终点时间
            ideal_target_ms = curr_ms + (predict_num * interval_ms)
            
            # 使用 searchsorted 找出的实际索引
            target_idx = np.searchsorted(time_values, ideal_target_ms, side='left')
            
            if target_idx < len(df):
                actual_target_ms = time_values[target_idx]
                gap_ms = actual_target_ms - ideal_target_ms
                # 如果 gap_ms > 0，说明理想的时间点在数据里是空的
                reason = f"Gap: {gap_ms}ms" if gap_ms > 0 else "Continuous"
            else:
                reason = "Out-of-Bounds"
                gap_ms = "N/A"

            print(f"{idx:8d} | {curr_ms:15d} | {ideal_target_ms:15d} | {str(gap_ms):>10} | {reason}")

    df['return_rate'] = pct_final 
    return df

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

def clean_data_quality_auto(df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    自动化检查所有列：
    1. 丢弃任何含有 NaN 的行。
    2. 探测数值列，并丢弃其中含有 0 的行（排除日期等非数值列）。
    """
    logger.info("启动自动化数据质量扫描...")
    initial_count = len(df)

    # 1. 统计并丢弃 NaN (全局检查)
    na_rows = df.isna().any(axis=1).sum()
    if na_rows > 0:
        logger.warning(f"检测到 {na_rows} 行数据包含空值 (NaN)，准备丢弃。")

    # 2. 识别数值型列进行零值检查
    # 我们只对 float 和 int 类型的列检查 0，避免对字符串时间列误判
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 统计数值列中的零值
    zero_mask = (df[numeric_cols] == 0).any(axis=1)
    zero_rows = zero_mask.sum()
    
    if zero_rows > 0:
        # 找出具体是哪些列含有 0，打印详细日志
        zero_stats = (df[numeric_cols] == 0).sum()
        logger.warning(f"检测到 {zero_rows} 行数据包含零值，分布如下:\n{zero_stats[zero_stats > 0]}")

    # 3. 综合清洗：剔除 NaN 或 0 的行
    # 我们保留数值不为 0 且非空的数据
    condition = df.isna().any(axis=1) | zero_mask
    df_cleaned = df[~condition].copy()

    # 4. 重置索引 (非常重要，防止后续 rolling 计算因为索引断裂出问题)
    df_cleaned.reset_index(drop=True, inplace=True)

    final_count = len(df_cleaned)
    dropped_count = initial_count - final_count

    if dropped_count > 0:
        logger.info(f"✅ 清洗完毕: 原始 {initial_count} 行 -> 剩余 {final_count} 行 (丢弃了 {dropped_count} 行)")
    else:
        logger.info("✅ 扫描完毕: 未发现空值或零值，数据质量完美。")

    return df_cleaned

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

@lru_cache(maxsize=1)   #only available when tarining
def load_interval_ms(config_path = data_config_path):
    """从元数据文件中安全读取间隔毫秒数"""
    if not os.path.exists(config_path):
        raise RuntimeError(f"❌ 找不到配置文件: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        # 提取 interval_ms
        interval_ms = meta.get("interval_ms")
        if interval_ms is None:
            raise RuntimeError(f"⚠️ 配置文件中缺失 'interval_ms' 字段！")
        return interval_ms
    except Exception as e:
        raise RuntimeError(f"💥 读取 JSON 时发生意外错误: {e}")

def setup_session_logger(sub_folder: str, symbol: str = symbol, console_level: int = logging.DEBUG, file_level: int = logging.INFO):
    """
    配置实盘会话日志：
    1. 生成带时间戳的文件名 (session_SYMBOL_Time.log)
    2. 将 FileHandler 挂载到 Root Logger (捕获所有模块日志)
    3. 将 StreamHandler (彩色) 挂载到 Root Logger (控制台显示所有日志)
    
    :param log_root: 日志根目录 (例如 common.TEMPORARY_DIR)
    :param sub_folder: 子文件夹名 (例如 'market_ftmo_sessions')
    :param symbol: 交易品种
    :return: (root_logger, log_file_path)
    """
    # 1. 准备目录
    log_dir = os.path.join(PERSISTENCE_DIR, sub_folder)
    os.makedirs(log_dir, exist_ok=True)

    # 2. 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sym_str = f"_{symbol}" if symbol else ""
    log_filename = f"session{sym_str}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    # 3. 获取 Root Logger
    # 注意：我们配置 Root，这样 self.logger = logging.getLogger("AnyName") 都会被捕获
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # 开启所有级别，具体过滤交给 Handler

    # 清除旧的 Handlers (防止重复打印，特别是 Jupyter 或多次调用时)
    if root_logger.handlers:
        root_logger.handlers = []

    # 4. 配置控制台输出 (复用之前的彩色格式)
    log_format_console = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    
    color_formatter = colorlog.ColoredFormatter(
        log_format_console,
        datefmt="%H:%M:%S", # 控制台时间短一点，方便看
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'RECORD':   'blue',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red,bg_yellow',
        }
    )
    ch.setFormatter(color_formatter)
    root_logger.addHandler(ch)

    # 5. 配置文件输出 (全量记录)
    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setLevel(file_level) # 文件通常记录 INFO 以上
    
    # 文件格式包含完整日期
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(file_formatter)
    root_logger.addHandler(fh)

    # 打印一条初始化消息
    root_logger.info(f"Session Logger Initialized. Log file: {log_file_path}")

    return root_logger, log_file_path