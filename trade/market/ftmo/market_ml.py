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
from data_process.common import FeatureFactory, FEATURE_CONFIG_LIST
from model import model_loader
from trade.strategy.strategy_ml import FtmoBrain, MarketState, PositionDir, ActionType, Signal
from trade.market.ftmo import ftmo_executor
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
    TIMEFRAME = common.interval
    allow_short = True
    allow_long = True
    holdbar = common.PREDICT_NUM#PREDICT_NUM
    thresh: float =None#0.5#None#0.45
    commission = 0.05   # 0.1 = 0.1%  .can't be 0
    cash = 10000
    stop_loss = 0.5  # 0-1
    stop_loss_long = 0.03  # 0-1
    stop_loss_short = 0.015  # 0-1
    atr_sl_mult_long = 5 #2.5
    atr_sl_mult_short = 2.5 #2.5
    take_profit = 0.99 #止盈. 0 - n倍
    trade_risk = 0.5     #0-1
    max_daily_loss_pct = 0.025

    MAX_LAYERS = 1
    
    # MT5 魔法数字
    MAGIC_NUMBER = 888888
    
    # 轮询间隔 (秒)
    POLL_INTERVAL = 1

# ============================================================
# 3. 主控程序：LiveBot
# ============================================================
class LiveBot:
    def __init__(self):
        self.logger, log_path = common.setup_session_logger(
                    sub_folder=f'{__file__}',
                    symbol=LiveConfig.SYMBOL_FTMO
                )
        
        self.logger.info("Initializing Live Bot...")

        self._log_startup_info(log_path)
        self.executor = ftmo_executor.MT5Executor(LiveConfig.SYMBOL_FTMO, LiveConfig.MAGIC_NUMBER, sl_scale = LiveConfig.stop_loss)
        self.model_handler = model_loader.ModelHandler() # 自动加载训练好的模型

        # 1. 设置参数
        self.interval_ms = LiveConfig.TIMEFRAME * 60 * 1000 
        self.factory = FeatureFactory(FEATURE_CONFIG_LIST, self.interval_ms)
        
        # 2. 计算历史需求 (数量)
        self.min_bars_needed = self.factory.get_global_min_history()
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
            executor= self.executor,
            trade_risk=LiveConfig.trade_risk, 
            max_layers=LiveConfig.MAX_LAYERS,
            holdbar=LiveConfig.holdbar,
            allow_long=LiveConfig.allow_long,
            allow_short=LiveConfig.allow_short,
            thresh=LiveConfig.thresh
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
            return 
            
        self.logger.info(f"✨ New Candle Closed: {current_candle_time} | Buffer Size: {len(df)}")
        
        # 2. 特征工程 & 模型预测
        try:
            # A. 特征计算 (FeatureFactory)
            # 使用 FeatureFactory 生成特征
            self.factory.generate(df)

            # B. 计算动态阈值 (复用 common.py 逻辑)
            # 这会生成 threshold 和 stop_threshold 列
            df = common.calculate_thresholds(df)
            
            # C. 模型推理
            # ModelHandler 内部会进行 TimeSeriesWindowDataset 处理和归一化
            # 注意：predict 返回的是包含 pred 和 pred_prob 的 DataFrame
            inference_df = df.iloc[-(self.model_handler.window + 200):]
            df_pred, _ = self.model_handler.predict(inference_df, self.interval_ms)
            
            # 获取最新一根 K 线的预测结果
            last_row = df_pred.iloc[-1]
            pred_signal = last_row["pred"]
            pred_prob = last_row["pred_prob"]
            current_price = last_row["close"]
            stop_threshold_pct = last_row["stop_threshold"]
            
            self.logger.info(f"Predict: Signal={pred_signal}, Prob={pred_prob:.4f}, Price={current_price}")
            
        except Exception as e:
            self.logger.error(f"Prediction Pipeline Error: {e}")
            import traceback
            traceback.print_exc()
            return

        self.executor.update_context(stop_threshold_pct)
        # 3. 获取 MT5 当前状态
        curr_dir, curr_layers, curr_vol = self.executor.get_current_state() #sync state here
        self.logger.info(f"MT5 State: Dir={curr_dir}, Layers={curr_layers}, Vol={curr_vol}")

        # 4. 构建 MarketState
        state = MarketState(
            price=current_price,
            signal=Signal(int(pred_signal)), # 转换为 Enum
            pred_prob=float(pred_prob),
            position_dir=curr_dir,
            layers=curr_layers
        )

        # 5. BrainBase 决策
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