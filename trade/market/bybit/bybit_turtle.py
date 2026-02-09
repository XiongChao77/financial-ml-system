import time
import logging
import pandas as pd
import argparse
import sys
import os
import signal
from datetime import datetime

# 路径适配
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..', '..'))

from bybit_engine import BybitEngine
from data_process.common import setup_session_logger
from trade.strategy.strategy_ml import PositionDir, ActionType
# 假设 TurtleBrain 在此路径，请确保该文件存在
from trade.strategy.strategy_turtle import TurtleBrain 
from trade.market.bybit.bybit_executor import BybitExecutor 
# ================= 配置区域 =================
class TurtleConfig:
    # 策略参数
    SYMBOL         = "DOGEUSDT"
    TIMEFRAME      = 240       # 分钟 (Bybit: 1, 3, 5, 15, 60, 240, D)
    ENTRY_PERIOD   = 15        # 唐奇安入场周期
    EXIT_PERIOD    = 10        # 唐奇安离场周期
    ATR_PERIOD     = 20
    MAX_LAYERS     = 1         # 最大加仓层数
    RISK_PER_UNIT  = 0.01      # 单笔风险 1%
    MAX_DAILY_LOSS = 0.5      # 最大回撤限制
    UPPER_LIMIT    = 0.7
    UNIT_PCT_SCALE = 2
    
    # 轮询间隔 (秒)
    POLL_INTERVAL  = 10

# ================= 数据源适配器 =================
class BybitDataFeed:
    def __init__(self, engine: BybitEngine, symbol: str, interval: int):
        self.engine = engine
        self.symbol = symbol
        self.interval = str(interval) # Bybit API 需要字符串
        self.logger = logging.getLogger("BybitDataFeed")

    def get_latest_data(self, limit=200) -> pd.DataFrame:
        """从 Bybit 获取 K 线并转换为 DataFrame"""
        try:
            # 调用 engine 的 http 接口
            res = self.engine.http.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.interval,
                limit=limit
            )
            
            if res.get('retCode') != 0:
                self.logger.error(f"获取 K 线失败: {res.get('retMsg')}")
                return None

            # Bybit 返回数据格式: [startTime, open, high, low, close, volume, turnover]
            # 且顺序是倒序（最新在最前）
            raw_list = res['result']['list']
            data = []
            for row in raw_list:
                data.append({
                    "open_time": int(row[0]), # 毫秒时间戳
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5])
                })
            
            # 转为 DataFrame 并按时间正序排列
            df = pd.DataFrame(data)
            df = df.sort_values("open_time").reset_index(drop=True)
            
            # 转换时间索引
            df['open_time_date_utc'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time_date_utc', inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"数据处理异常: {e}")
            return None
# ================= 主机器人逻辑 =================
class BybitTurtleBot:
    def __init__(self, engine, is_long_account):
        self.logger, _ = setup_session_logger(
            sub_folder="BybitTurtle", 
            console_level=logging.DEBUG, 
            file_level=logging.DEBUG
        )
        self.engine = engine
        self.symbol = TurtleConfig.SYMBOL
        
        # 初始化组件
        self.data_feed = BybitDataFeed(engine, self.symbol, TurtleConfig.TIMEFRAME)
        self.executor = BybitExecutor(engine, self.symbol)
        
        # 初始化大脑
        self.brain = TurtleBrain(
            executor=self.executor,
            entry_period=TurtleConfig.ENTRY_PERIOD,
            exit_period=TurtleConfig.EXIT_PERIOD,
            atr_period=TurtleConfig.ATR_PERIOD,
            max_layers=TurtleConfig.MAX_LAYERS,
            risk_per_unit=TurtleConfig.RISK_PER_UNIT,
            max_daily_loss_pct=TurtleConfig.MAX_DAILY_LOSS,
            upper_limit = TurtleConfig.UPPER_LIMIT,
            unit_pct_scale = TurtleConfig.UNIT_PCT_SCALE,
        )
        
        self.last_candle_time = None
        self.stop_signal = False
        
        # 设置杠杆
        self.engine.set_leverage(self.symbol, "10")
        try:
            # 尝试切换为单向持仓模式
            res = self.engine.http.switch_position_mode(
                category="linear", 
                symbol=self.symbol, 
                mode=0 # 0: 单向持仓
            )
            if res.get('retCode') == 0:
                self.logger.info(f"✅ [{self.symbol}] 成功切换为单向持仓模式")
        except Exception as e:
            # 如果报错代码是 110025，说明已经是目标模式，直接忽略即可
            if "110025" in str(e):
                self.logger.info(f"ℹ️ [{self.symbol}] 已经是单向持仓模式，无需修改")
            else:
                self.logger.error(f"⚠️ 切换持仓模式失败: {e}")

    def run_step(self):
        # 1. 获取数据
        df = self.data_feed.get_latest_data()
        if df is None or df.empty: return

        # 2. 检查 K 线闭合 (基于 open_time)
        current_candle_time = df.iloc[-1].name
        if self.last_candle_time == current_candle_time:
            # 只有在时间变动时才运行逻辑
            return 
        
        self.logger.info(f"📊 新 K 线闭合: {current_candle_time} | Close: {df.iloc[-1]['close']}")
        
        # 3. 获取实盘状态
        curr_dir, _, last_price = self.executor.get_current_state()
        equity = self.executor.get_account_equity()
        
        # 计算持仓占比 (简单模拟，如需精确需计算名义价值)
        current_price = df.iloc[-1]['close']
        curr_pos_size_pct = 0.0
        if curr_dir != PositionDir.FLAT:
            curr_pos_size_pct = 0.1 # 只要有持仓，给个非0值让Brain知道

        # 4. 大脑决策
        # 注意：BrainBase 内部会调用 executor.user_order 进行下单
        self.brain.decide(
            df=df,
            current_time=pd.to_datetime(datetime.now()), # 或使用 server time
            account_balance=equity,
            curr_dir=curr_dir,
            curr_pos_size=curr_pos_size_pct,
            last_entry_price=last_price
        )
        
        self.last_candle_time = current_candle_time

    def run(self):
        self.logger.info("🚀 Bybit Turtle Strategy Started...")
        last_heartbeat = 0
        
        while not self.stop_signal:
            try:
                # 增加心跳：每 5 分钟打印一次，证明循环没卡死
                if time.time() - last_heartbeat > 300:
                    self.logger.info("💓 Heartbeat: Bot is still alive and cycling...")
                    last_heartbeat = time.time()
                
                self.run_step()
                time.sleep(TurtleConfig.POLL_INTERVAL)
            except Exception as e:
                self.logger.error(f"主循环异常: {e}", exc_info=True)
                time.sleep(10)

if __name__ == "__main__":
    # 解析参数，复用 Martingale 的逻辑
    parser = argparse.ArgumentParser(description="Bybit Turtle Bot")
    parser.add_argument("-t", "--testnet", action="store_true", help="Run on Testnet")
    args = parser.parse_args()

    # 路径配置
    keypath = 'Maringale'
    side_path = 'Long' # 默认用 Long 文件夹下的 key
    BASE = os.path.dirname(os.path.abspath(__file__))
    
    # 假设你的 Key 文件结构如下
    # keys/Maringale/Long/hmac_api_key
    API_K = os.path.join(BASE, "keys", keypath, side_path, "hmac_api_key")
    API_S = os.path.join(BASE, "keys", keypath, side_path, "hmac_secret")
    RSA_K = os.path.join(BASE, "keys", keypath, side_path, "api_key")
    RSA_P = os.path.join(BASE, "keys", "bybit_rsa.pem")

    if not os.path.exists(API_K):
        print(f"❌ Key file not found: {API_K}")
        sys.exit(1)

    # 初始化引擎
    engine = BybitEngine(API_K, API_S, RSA_K, RSA_P, testnet=args.testnet)
    
    # 启动机器人
    bot = BybitTurtleBot(engine, is_long_account=True)
    
    # 信号处理
    def signal_handler(sig, frame):
        print("\n👋 Stop signal received...")
        bot.stop_signal = True
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    bot.run()