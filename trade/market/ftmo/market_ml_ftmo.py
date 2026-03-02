import time
import logging
import json
import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import argparse

# 添加项目路径以导入自定义模块
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..' , '..'))

# 引入自定义模块
from data_process import common
from data_process.common import FeatureFactory
from model import model_loader
from model.train_2head import TrainConfig
from trade.strategy.strategy_ml import FtmoBrain, MarketState, PositionDir, ActionType, Signal
from trade.market.ftmo import mt5_executor
from trade.market.bybit.bybit_executor import BybitExecutor 
from trade.market.binance_data_feed import  BinanceDataFeed
from trade.bt.simulation import StrategyPara
pd.set_option("display.max_columns", None)   # 不限制列数
pd.set_option("display.width", None)         # 自动宽度（别强行换行）
pd.set_option("display.max_colwidth", None)  # 单元格内容不截断
# ============================================================
# 配置区域
# ============================================================
ADDITIONAL_FEATURES = ["atr_14"]

class LiveConfig:
    ftmo_mt5_path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
    the5ers_mt5_path = r"C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe"
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
        parser = argparse.ArgumentParser(description="LiveBot Configuration")
        parser.add_argument("-t", "--tarin_out_path", required=True, help="Path to the training output directory")
        parser.add_argument(
            "-m", "--mt5_option", required=True, type=int, choices=[0, 1],
            help="MT5 terminal option: 0 for FTMO, 1 for The5ers (default: 0), 2 for bybit"
        )
        parser.add_argument("-k", "--key", required=False, help="Key path for Bybit (if mt5_option is 2)")

        args = parser.parse_args()

        self.tarin_out_path = args.tarin_out_path
        self.pre_para,self.train_para ,self.strategy_para = self._load_strategy()
        self.logger, log_path = common.setup_session_logger(
                    sub_folder=f'market_ftmo',
                    symbol=self.pre_para.symbol
                )
        
        if args.mt5_option == 0:
            self.mt5_path = LiveConfig.ftmo_mt5_path
            self.executor = mt5_executor.MT5Executor(self.mt5_path, self.pre_para.symbol, LiveConfig.MAGIC_NUMBER, logger=self.logger)
        elif args.mt5_option == 1:
            self.mt5_path = LiveConfig.the5ers_mt5_path
            self.executor = mt5_executor.MT5Executor(self.mt5_path, self.pre_para.symbol, LiveConfig.MAGIC_NUMBER, logger=self.logger)
        elif args.mt5_option == 2:
            if not args.key:
                print("Error: Key path is required for Bybit option")
                sys.exit(1)
            self.executor = BybitExecutor(args.key, self.pre_para.symbol)
        # self.tarin_out_path = os.path.join(common.PERSISTENCE_DIR, r"market_prepare/ETH/market_59_20")
        # self.mt5_path = LiveConfig.the5ers_mt5_path

        self.logger.info("Initializing Live Bot...")

        self._log_startup_info(log_path)
        self.model_handler = model_loader.ModelHandler(tarin_out_path = self.tarin_out_path, device= 'cpu') # 自动加载训练好的模型

        # 1. 设置参数
        self.interval_ms = common.get_interval_ms(self.pre_para.interval)
        full_feature_list = self.train_para.feature_conf_list + ADDITIONAL_FEATURES
        self.factory = FeatureFactory(self.interval_ms,feature_conf_list= full_feature_list)
        
        # 2. 计算历史需求 (数量)
        self.min_bars_needed = self.factory.get_global_min_history() + common.BaseDefine.predict_num
        self.logger.info(f"History Required: {self.min_bars_needed} bars")
        
        # 3. 初始化数据源 (带缓存)
        # max_len 设置得比 min_bars_needed 大一些，比如 +500，留有余地
        self.data_feed = BinanceDataFeed(
            self.pre_para.symbol, 
            self.pre_para.interval, 
            self.pre_para.trading_type,
            max_len = self.min_bars_needed + 500
        )
        #strategy
        self.brain = FtmoBrain(
            self.executor,
            trade_risk=self.strategy_para.trade_risk,
            max_layers=LiveConfig.max_layers,
            holdbar=self.strategy_para.holdbar,
            allow_long=self.strategy_para.allow_long,
            allow_short=self.strategy_para.allow_short,
            thresh=self.strategy_para.thresh,
            stop_loss_long = self.strategy_para.stop_loss_long,
            stop_loss_short = self.strategy_para.stop_loss_short,
            atr_sl_mult_long = self.strategy_para.atr_sl_mult_long,
            atr_sl_mult_short = self.strategy_para.atr_sl_mult_short,
            max_daily_loss_pct = self.strategy_para.max_daily_loss_pct,
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
        self.logger.info(f"🔗 symbol: {self.pre_para.symbol} | interval: {self.pre_para.interval}")
        self.logger.info("-" * 20 + " PARAMETERS " + "-" * 20)
        
        # 自动遍历 Config 类的所有参数
        for key in dir(LiveConfig):
            if not key.startswith("__"):
                val = getattr(LiveConfig, key)
                self.logger.info(f"{key.ljust(20)}: {val}")
        
        self.logger.info("=" * 60)

    def _load_strategy(self)-> tuple[common.BaseDefine,TrainConfig, StrategyPara]:
        r = common.load_selected_configs(os.path.join(self.tarin_out_path,'market.jsonl'))[0]  # just to validate file and format
        params = r["short"] if "short" in r else r
        return common.BaseDefine(**params["params"]["common"]), TrainConfig(**params["params"]["train"]), StrategyPara(**params["params"]["strategy"])

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
            df = self.factory.generate(df)

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

    def test_executor(self):
        """
        测试 self.executor 的接口功能，包括 user_order 和 user_close
        """
        try:
            self.logger.info("Testing MT5 Executor...")

            # 测试获取当前状态
            curr_dir, curr_layers, curr_vol = self.executor.get_current_state()
            self.logger.info(f"MT5 State: Dir={curr_dir}, Layers={curr_layers}, Vol={curr_vol}")

            # 测试获取服务器时间
            server_time = self.executor.get_server_time()
            self.logger.info(f"Server Time: {server_time}")

            # 测试获取账户权益
            account_equity = self.executor.get_account_equity()
            self.logger.info(f"Account Equity: {account_equity}")

            # 测试 user_order 接口
            self.logger.info("Placing test order...")
            self.executor.user_order(size=0.01, is_buy=True, stop_loss=0.05)
            self.logger.info("Test order placed successfully.")

            # 测试 user_close 接口
            self.logger.info("Closing test order...")
            self.executor.user_close()
            self.logger.info("Test order closed successfully.")

            self.logger.info("MT5 Executor test completed successfully.")
        except Exception as e:
            self.logger.error(f"MT5 Executor test failed: {e}")
            import traceback
            traceback.print_exc()

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