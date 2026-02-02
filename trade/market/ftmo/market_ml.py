import time
import logging
import json
import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# 添加项目路径以导入自定义模块
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..' , '..'))

# 引入自定义模块
from data_process import common
from data_process.common import FeatureFactory
from model import model_loader
from trade.strategy.strategy_ml import FtmoBrain, MarketState, PositionDir, ActionType, Signal
from trade.market.ftmo import mt5_executor
from trade.market.binance_data_feed import  BinanceDataFeed
pd.set_option("display.max_columns", None)   # 不限制列数
pd.set_option("display.width", None)         # 自动宽度（别强行换行）
pd.set_option("display.max_colwidth", None)  # 单元格内容不截断
# ============================================================
# 配置区域
# ============================================================
class LiveConfig:
    # 交易品种映射
    SYMBOL_BINANCE = "DOGEUSDT"  # 数据源品种
    SYMBOL_FTMO = "DOGEUSD"      # 交易执行品种 (FTMO通常是 BTCUSD)
    
    # 时间周期 (分钟)
    TIMEFRAME = common.CommonDefine.interval
    allow_short = True
    allow_long = True
    holdbar = common.CommonDefine.predict_num#CommonDefine.predict_num
    thresh: float =None#0.5#None#0.45
    commission = 0.05   # 0.1 = 0.1%  .can't be 0
    cash = 10000
    stop_loss_long = 0.03  # 0-1
    stop_loss_short = 0.015  # 0-1
    atr_sl_mult_long = 5 #2.5
    atr_sl_mult_short = 2.5 #2.5
    take_profit = 0.99 #止盈. 0 - n倍
    trade_risk = 0.5     #0-1
    max_daily_loss_pct = 0.025

    mt5_path = r"C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe"
    max_layers = 1
    # MT5 魔法数字
    MAGIC_NUMBER = 888888
    # 轮询间隔 (秒)
    POLL_INTERVAL = 5
# ============================================================
# 3. 主控程序：LiveBot
# ============================================================
class LiveBot:
    def __init__(self):
        self.logger, log_path = common.setup_session_logger(
                    sub_folder=f'market_ml',
                    symbol=LiveConfig.SYMBOL_FTMO
                )
        
        self.logger.info("Initializing Live Bot...")

        self._log_startup_info(log_path)
        self.executor = mt5_executor.MT5Executor(LiveConfig.mt5_path,LiveConfig.SYMBOL_FTMO, LiveConfig.MAGIC_NUMBER,logger= self.logger)
        self.model_handler = model_loader.ModelHandler() # 自动加载训练好的模型

        # 1. 设置参数
        self.interval_ms = common.get_interval_ms(LiveConfig.TIMEFRAME) 
        self.factory = FeatureFactory(self.interval_ms)
        
        # 2. 计算历史需求 (数量)
        self.min_bars_needed = self.factory.get_global_min_history() + common.CommonDefine.predict_num
        self.logger.info(f"History Required: {self.min_bars_needed} bars")
        
        # 3. 初始化数据源 (带缓存)
        # max_len 设置得比 min_bars_needed 大一些，比如 +500，留有余地
        self.data_feed = BinanceDataFeed(
            LiveConfig.SYMBOL_BINANCE, 
            LiveConfig.TIMEFRAME, 
            max_len = self.min_bars_needed + 500
        )
        #strategy
        self.brain = FtmoBrain(
            self,
            trade_risk=LiveConfig.trade_risk,
            max_layers=LiveConfig.max_layers,
            holdbar=LiveConfig.holdbar,
            allow_long=LiveConfig.allow_long,
            allow_short=LiveConfig.allow_short,
            thresh=LiveConfig.thresh,
            stop_loss_long = LiveConfig.stop_loss_long,
            stop_loss_short = LiveConfig.stop_loss_short,
            atr_sl_mult_long = LiveConfig.atr_sl_mult_long,
            atr_sl_mult_short = LiveConfig.atr_sl_mult_short,
            max_daily_loss_pct = LiveConfig.max_daily_loss_pct,
        )

        # 4. 执行数据预热 (Warmup) -> 填充内存
        self.data_feed.initialize_cache(self.min_bars_needed, self.interval_ms)
        
        # 记录一下初始化后的最后一根时间
        initial_df = self.data_feed.get_latest_data()
        self.last_candle_time = initial_df.iloc[-1]["open_time_date_utc"] if not initial_df.empty else None

    def _log_startup_info(self, log_path):
        """
        [新增] 打印本次运行的详细环境信息
        """
        self.logger.info("=" * 60)
        self.logger.info(f"🚀 LIVE BOT SESSION STARTED")
        self.logger.info(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"📂 Log File: {log_path}")
        self.logger.info(f"📊 Target Symbol: {LiveConfig.SYMBOL_FTMO}")
        self.logger.info(f"🔗 Data Source: {LiveConfig.SYMBOL_BINANCE}")
        self.logger.info("-" * 20 + " PARAMETERS " + "-" * 20)
        
        # 自动遍历 Config 类的所有参数
        for key in dir(LiveConfig):
            if not key.startswith("__"):
                val = getattr(LiveConfig, key)
                self.logger.info(f"{key.ljust(20)}: {val}")
        
        self.logger.info("=" * 60)

    def run_step(self):
        """
        每隔几秒运行一次
        """
        # 1. 获取最新数据 (DataFeed 内部自动处理增量更新)
        # 这里的 df 已经是清洗好、长度足够、且剔除了未闭合 K 线的完美数据
        df = self.data_feed.get_latest_data()
        
        if df is None or df.empty:
            return

        # 2. 检查是否有新 K 线产生
        # 比较这一轮拿到的最新时间 vs 上一轮处理的时间
        current_candle_time = df.iloc[-1]["open_time_date_utc"]
        
        if self.last_candle_time == current_candle_time:
            # 时间没变，说明没有新 K 线收盘 -> 跳过
            pass#return 
            
        self.logger.info(f"✨ New Candle Closed: {current_candle_time} | Buffer Size: {len(df)}")
        
        # 2. 特征工程 & 模型预测
        try:
            self.factory.generate(df)

            # C. 模型推理
            # ModelHandler 内部会进行 TimeSeriesWindowDataset 处理和归一化
            # 注意：predict 返回的是包含 pred 和 pred_prob 的 DataFrame
            inference_df = df.iloc[-(self.model_handler.window + 200):]
            df_pred, _ = self.model_handler.predict(inference_df, kline_interval_ms= self.interval_ms, is_live = True, diff_thresh = None)
            
            # 获取最新一根 K 线的预测结果
            last_row = df_pred.iloc[-1]
            pred = last_row["pred"]
            pred_prob = last_row["pred_prob"]
            current_price = last_row["close"]
            
            self.logger.info(f"Predict: Signal={pred}, Prob={pred_prob:.4f}, Price={current_price}")
            
        except Exception as e:
            self.logger.error(f"Prediction Pipeline Error: {e}")
            import traceback
            traceback.print_exc()
            return

        # 3. 获取 MT5 当前状态
        curr_dir, curr_layers, curr_vol = self.executor.get_current_state() #sync state here
        self.logger.info(f"MT5 State: Dir={curr_dir}, Layers={curr_layers}, Vol={curr_vol}")

        # 数据有效性检查
        current_signal = Signal.INVALID if np.isnan(pred) else Signal(int(pred))
        current_prob = 0.0 if np.isnan(pred_prob) else float(pred_prob)

        state = MarketState(
            price= current_price,
            signal= current_signal,
            pred_prob= float(current_prob),
            position_dir= curr_dir,
            layers= curr_layers,
            current_time= self.executor.get_server_time(),
            account_balance= self.executor.get_account_equity(),
            atr= last_row["atr_14"],
            slow_atr = None,
            vol_regime = None,
        )

        self.brain.decide(state)

        # 更新时间戳，防止重复执行
        self.last_candle_time = current_candle_time

    def start(self):
        while True:
            try:
                self.run_step()
                time.sleep(LiveConfig.POLL_INTERVAL)
            except KeyboardInterrupt:
                self.logger.info("Bot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Main Loop Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = LiveBot()
    bot.start()