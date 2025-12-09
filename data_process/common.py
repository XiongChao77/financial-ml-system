import logging,math
import pandas as pd
import numpy as np
import os, colorlog , logging
from data_process.feature import *
#define model
CANDLESTICK_NUM = 120
PREDICT_NUM = 16
change_rate = 0.006 # 0.2%
weak_change = change_rate / 5.0
# 波动率系数 (0.5 ~ 1.0 之间调整)
'''
乘数 (Multiplier),阈值位置,含义
VOL_MULTIPLIER=1.0,1σ,约 31.8% 的价格变动会超出这个阈值（上下尾部）。
VOL_MULTIPLIER=0.5,0.5σ,约 61.7% 的价格变动会超出这个阈值。信号数量适中。
VOL_MULTIPLIER=1.5,1.5σ,仅约 13.4% 的价格变动会超出这个阈值。
VOL_MULTIPLIER=2.0,2σ,仅约 4.6% 的价格变动会超出这个阈值。    
''' 
VOL_MULTIPLIER = 0.6
# 最小硬阈值 (覆盖手续费+滑点)
MIN_THRESHOLD = 0.007  # 0.25%

label_decrease = 0
# label_decrease_weak =1 
label_ignore = 1
# label_increase_weak = 3
label_increase = 2
model_train_rate = 0.8
DATA_PROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(DATA_PROCESS_DIR)
TEMPORARY_DIR = os.path.join(PROJECT_DIR , 'temporary')
origin_data_path = os.path.join(os.path.dirname(PROJECT_DIR),'QuantData','Cryptocurrency', "BTCUSDT_15m.csv")
DATA_PROCESS_OUT_DIR = os.path.join(DATA_PROCESS_DIR, 'output')
train_data_path = os.path.join(TEMPORARY_DIR, "train_data.csv")
test_data_path  = os.path.join(TEMPORARY_DIR, "test_data.csv")
log_level = logging.INFO

def attach_attr(df):
    # 1. 基础处理
    df.rename({'ignore':'label'},axis=1, inplace=True) 

    # --- 2. 指标计算 (生成所有原始、未缩放的特征列) ---
    # df = add_relative_features(df)
    FeatureFactory(FEATURE_CONFIG).generate(df)

def attach_label(df, candlestick_num:int = CANDLESTICK_NUM, predict_num:int= PREDICT_NUM , vol_multiplier = VOL_MULTIPLIER,
                 min_threshold = MIN_THRESHOLD, keep_rate = False):
    """
    依据未来收益率与当前波动率的动态关系分3类,并将计算出的动态阈值保存到 'threshold' 列。
    
    Label 0: 下跌 (收益率 < -动态阈值)
    Label 1: 震荡 (绝对值 <= 动态阈值)
    Label 2: 上涨 (收益率 > 动态阈值)
    """
    assert 'close' in df.columns, "缺少列 close"
    assert predict_num > 0, "predict_num 必须 > 0"
    print(f"[attach_label] VOL_MULTIPLIER:{vol_multiplier},MIN_THRESHOLD:{min_threshold}")
    # ---------------- 参数设置 ----------------
    # 波动率参考窗口
    vol_window = candlestick_num 
    
    # 1. 计算未来收益率 (Target)
    future_close = df['close'].shift(-predict_num)
    pct = (future_close - df['close']) / df['close']

    # 2. 计算动态阈值 (Dynamic Threshold)
    returns = df['close'].pct_change()
    rolling_std = returns.rolling(window=vol_window).std()
    expected_vol = rolling_std * np.sqrt(predict_num)
    
    # 计算阈值序列
    dynamic_threshold = (expected_vol * vol_multiplier).clip(lower=min_threshold)
    
    # === 【新增】 将阈值写入 DataFrame ===
    # 填充前部的 NaN (预热期)，避免保存出来的 CSV 这一列前面是空的
    # 我们可以用 min_threshold 填充，或者用第一个有效值填充
    df['threshold'] = dynamic_threshold.fillna(min_threshold)
    
    # 3. 打标签 (Labeling)
    # 注意：这里直接使用列 df['threshold'] 进行比较
    cond_decrease = (pct < -df['threshold'])
    cond_increase = (pct > df['threshold'])
    
    df['label'] = np.select(
        [cond_decrease, cond_increase],
        [label_decrease, label_increase],
        default=label_ignore
    )

    # 4. 数据清洗
    # 前 vol_window 行波动率计算不准，强制忽略
    df.iloc[:vol_window, df.columns.get_loc('label')] = label_ignore

    # 删尾（未来价格不可用的部分）
    if predict_num > 0:
        df = df.iloc[:-predict_num].reset_index(drop=True)

    df['label'] = df['label'].astype(int)

    if keep_rate == True:
        df['return_rate'] = pct

#console: All
#file: above Info
def setup_logger(log_name="app_logger", log_path="logs"):
    """
    设置日志记录器：控制台彩色输出 (INFO+)，文件输出 (INFO+)。
    :return: 配置好的日志记录器对象。
    """
    # 确保日志目录存在
    os.makedirs(log_path, exist_ok=True)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG) # 确保所有消息都能进入处理流程

    # 避免重复添加 handlers (重要，防止多次调用函数时重复记录)
    if logger.handlers:
        logger.handlers = []

    log_format = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"

    # --- 1. 控制台处理程序 (StreamHandler) ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG) # 控制台输出：INFO 及以上
    
    # 彩色格式化器
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s" + log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red,bg_yellow',
        }
    )
    ch.setFormatter(color_formatter)
    logger.addHandler(ch)

    # --- 2. 文件处理程序 (FileHandler) ---
    log_file = os.path.join(log_path, f"{log_name}.log")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    
    # 设置文件最低输出等级：INFO
    fh.setLevel(logging.INFO) 
    
    # 普通格式化器
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    return logger

FEATURE_CONFIG:list[FeatureBase] = [
    FeatureMACD(fast=12, slow=26, signal=9),
    FeatureMA(weeks=[7,25], days=[5, 10, 20], method='sma', strict=True, add_slope = False, slope_method='reg', slope_weeks= 2),
    FeatureRsi(period=14, price_col='close', strict=True, prefix='RSI'),
    FeatureKdj(n=9, m1=3, m2=3, high_col='high', low_col='low',  close_col='close', strict=True, prefix='KDJ'),
    FeatureVolMa(vol_ma_windows = (5, 10, 20)),
    FeatureQavMa(vol_ma_windows = (5, 10, 20)),
    FeatureOBV(),
    FeaturePVT(),
    FeatureWAP(vwap_windows = (20, 48, 96)),
    FeatureCFM(cmf_window = 20),
    FeatureMFI(mfi_window=14),
    FeatureATS(), #FeatureContainer(FeatureATS),
    FeatureCandle(),
]

class FeatureFactory:
    def __init__(self, config_list:list[FeatureBase] = FEATURE_CONFIG):
        self.features:list[FeatureBase] = config_list
        # for f in config_list:
        #     self.features.append(f.feature(f.parameters))
    def generate(self,df):
        for f in self.features:     f.generate(df)
    def normalize(self, X: np.ndarray, feature_cols: list[str]):
        for f in self.features:
            f.normalize(X, feature_cols)