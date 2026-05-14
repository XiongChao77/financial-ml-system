import time
import logging
import pandas as pd
from datetime import datetime
import os, sys

# 路径处理
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..', '..'))

# 引入模块
from Quant.trade.market.ftmo.market_ml import BinanceDataFeed
from trade.market.ftmo.mt5_executor import MT5Executor # 刚才写的那个类
from trade.strategy.strategy_turtle import TurtleBrain
from trade.strategy.strategy_ml import PositionDir
from data_process import common

# ================= 配置区域 =================
class LiveConfig:
    # 账号配置
    MAGIC_NUMBER = 20260118
    
    # 品种映射
    SYMBOL_BINANCE = "DOGEUSDT"
    SYMBOL_FTMO    = "DOGEUSD" # ⚠️ 请检查 MT5 里的实际名字
    
    # 策略参数 (FTMO 稳健版)
    TIMEFRAME      = "4h"   # 4小时 "1m"/"5m"/"15m"/"1h"/"4h"/"1d"
    ENTRY_PERIOD   = 15    # 入场周期
    EXIT_PERIOD    = 10    # 离场周期
    RISK_PER_UNIT  = 0.01  # 单笔风险 1%
    MAX_DAILY_LOSS = 0.045 # FTMO 4.5% 预警线
    
    # 轮询
    POLL_INTERVAL  = 5    # 秒
    INTERVAL_MAP = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "1h": 60,
        "4h": 240,  # 确保 240 对应 4h
        "1d": 1440
    }

# ================= 主程序 =================
class TurtleLiveBot:
    def __init__(self):
        # 日志设置
        self.logger , _= common.setup_session_logger(sub_folder='ftmo_turtle_live',console_level= logging.DEBUG, file_level = logging.DEBUG)
        self.logger.info("🚀 Turtle Strategy Live Bot Starting...")

        # 1. 初始化 MT5 执行器
        try:
            self.executor = MT5Executor(
                symbol=LiveConfig.SYMBOL_FTMO, 
                magic=LiveConfig.MAGIC_NUMBER
            )
        except Exception as e:
            self.logger.critical(f"Executor Init Failed: {e}")
            sys.exit(1)

        # 2. 初始化大脑
        self.brain = TurtleBrain(
            executor=self.executor,
            entry_period=LiveConfig.ENTRY_PERIOD,
            exit_period=LiveConfig.EXIT_PERIOD,
            atr_period=20,
            max_layers=1, # 强制单层，符合你的优化
            risk_per_unit=LiveConfig.RISK_PER_UNIT,
            max_daily_loss_pct=LiveConfig.MAX_DAILY_LOSS
        )
        
        # 3. 初始化数据源 (Binance)
        self.data_feed = BinanceDataFeed(
            symbol=LiveConfig.SYMBOL_BINANCE, 
            interval=LiveConfig.TIMEFRAME,
            max_len=1000 # 只需要最近几百根计算 ATR 和 通道
        )
        
        self.last_candle_time = None

    # 在你的 TurtleLiveBot 类中修改 run_step
    def run_step(self):
        # 1. 获取最新 K 线
        df = self.data_feed.get_latest_data()
        if df is None or df.empty: return

        # 2. 只有新 K 线闭合才触发（4H 周期）
        current_candle_time = df.iloc[-1]["open_time_date_utc"]
        if self.last_candle_time == current_candle_time: return
        
        # 3. 同步实盘上下文（账户净值与当前方向）
        # 注意：这里我们不需要判断层数，因为 max_layers=1
        curr_dir, _, last_price = self.executor.get_current_state()
        equity = self.executor.get_account_equity()
        
        # 计算当前持仓价值占比 (如果没有持仓就是 0)
        current_price = df.iloc[-1]['close']
        curr_pos_size_pct = 0.0
        if curr_dir != PositionDir.FLAT:
            # 假设当前只有 1 层，那么 size_pct 只要是非 0 即可
            # 大脑内部会根据 curr_pos_size > 0 来判定是否正在持仓
            curr_pos_size_pct = 0.01 # 给一个标称值，代表“有仓位”
        
        # 4. 喂给大脑
        # 大脑内部会自动执行 _check_gaps 和决定 ActionType.OPEN 或 CLOSE
        self.brain.decide(
            df=df,
            current_time=pd.to_datetime(current_candle_time),
            account_balance=equity,
            curr_dir=curr_dir,
            curr_pos_size=curr_pos_size_pct, # 关键：告诉大脑我们现在有没有仓
            last_entry_price=last_price
        )
        
        self.last_candle_time = current_candle_time

    def start(self):
        self.logger.info("📡 Pre-fetching history data...")
        self.data_feed.initialize_cache(500,LiveConfig.INTERVAL_MAP[LiveConfig.TIMEFRAME] * 60 * 1000)
        
        self.logger.info("🟢 System Live. Polling for new candles...")
        while True:
            try:
                self.run_step()
                time.sleep(LiveConfig.POLL_INTERVAL)
            except KeyboardInterrupt:
                self.logger.info("🛑 Bot stopped by user.")
                break
            except Exception as e:
                self.logger.error(f"❌ Main Loop Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = TurtleLiveBot()
    bot.start()