import logging
#define model
candlestick_num = 120
predict_num = 16
change_rate = 0.004 # 0.2%
label_decrease = 0
# label_decrease_weak =1 
label_ignore = 1
# label_increase_weak = 3
label_increase = 2
origin_data = "klines15m.csv"
log_level = logging.INFO

def add_macd(df, fast=12, slow=26, signal=9):
    """
    在数据中加入MACD指标：
      - MACD_DIF: EMA(fast) - EMA(slow)
      - MACD_DEA: DIF的EMA(signal)
      - MACD:     2 * (DIF - DEA)
    计算基于 Close_price 列。
    """
    if 'Close_price' not in df.columns:
        raise ValueError("缺少列 Close_price，无法计算MACD")
    close = df['Close_price']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd = 2 * (dif - dea)
    df['MACD_DIF'] = dif
    df['MACD_DEA'] = dea
    df['MACD'] = macd
    return df