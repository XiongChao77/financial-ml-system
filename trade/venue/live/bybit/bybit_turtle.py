import time
import logging
import pandas as pd
import argparse
import sys
import os
import signal
from datetime import datetime

# Path setup
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", "..", "..", ".."))

from bybit_engine import BybitEngine
from data_process.common import setup_session_logger
from trade.core.protocol import PositionDir, ActionType
# TurtleStrategy is expected at this path, make sure the file exists
from trade.strategy.strategy_turtle import TurtleStrategy 
from trade.venue.live.bybit.bybit_venue import BybitVenue 
# ================= configuration =================
class TurtleConfig:
    # Strategy parameters
    SYMBOL         = "DOGEUSDT"
    TIMEFRAME      = 240       # minutes (Bybit: 1, 3, 5, 15, 60, 240, D)
    ENTRY_PERIOD   = 15        # Donchian entry period
    EXIT_PERIOD    = 10        # Donchian exit period
    ATR_PERIOD     = 20
    MAX_LAYERS     = 1         # maximum number of layers
    RISK_PER_UNIT  = 0.01      # 1% risk per trade
    MAX_DAILY_LOSS = 0.5      # maximum drawdown limit
    UPPER_LIMIT    = 0.7
    UNIT_PCT_SCALE = 2
    
    # Polling interval (seconds)
    POLL_INTERVAL  = 10

# ================= data source adapter =================
class BybitDataFeed:
    def __init__(self, engine: BybitEngine, symbol: str, interval: int):
        self.engine = engine
        self.symbol = symbol
        self.interval = str(interval) # the Bybit API expects a string
        self.logger = logging.getLogger("BybitDataFeed")

    def get_latest_data(self, limit=200) -> pd.DataFrame:
        """Fetch klines from Bybit and convert them into a DataFrame"""
        try:
            # Call the engine's http interface
            res = self.engine.http.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.interval,
                limit=limit
            )
            
            if res.get('retCode') != 0:
                self.logger.error(f"获取 K 线失败: {res.get('retMsg')}")
                return None

            # Bybit returns: [startTime, open, high, low, close, volume, turnover]
            # in reverse order (newest first)
            raw_list = res['result']['list']
            data = []
            for row in raw_list:
                data.append({
                    "open_time": int(row[0]), # millisecond timestamp
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5])
                })
            
            # Convert to a DataFrame and sort chronologically
            df = pd.DataFrame(data)
            df = df.sort_values("open_time").reset_index(drop=True)
            
            # Convert the time index
            df['open_time_date_utc'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time_date_utc', inplace=True)
            
            return df
        except Exception as e:
            self.logger.error(f"数据处理异常: {e}")
            return None
# ================= main bot logic =================
class BybitTurtleBot:
    def __init__(self, engine, is_long_account):
        self.logger, _ = setup_session_logger(
            sub_folder="BybitTurtle", 
            console_level=logging.DEBUG, 
            file_level=logging.DEBUG
        )
        self.engine = engine
        self.symbol = TurtleConfig.SYMBOL
        
        # Initialize the components
        self.data_feed = BybitDataFeed(engine, self.symbol, TurtleConfig.TIMEFRAME)
        self.venue = BybitVenue(engine, self.symbol)
        
        # Initialize the decision engine
        self.strategy = TurtleStrategy(
            venue=self.venue,
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
        
        # Set the leverage
        self.engine.set_leverage(self.symbol, "10")
        try:
            # Try to switch to one-way position mode
            res = self.engine.http.switch_position_mode(
                category="linear", 
                symbol=self.symbol, 
                mode=0 # 0: one-way position
            )
            if res.get('retCode') == 0:
                self.logger.info(f"✅ [{self.symbol}] 成功切换为单向持仓模式")
        except Exception as e:
            # Error code 110025 means it is already in the target mode, just ignore it
            if "110025" in str(e):
                self.logger.info(f"ℹ️ [{self.symbol}] 已经是单向持仓模式，无需修改")
            else:
                self.logger.error(f"⚠️ 切换持仓模式失败: {e}")

    def run_step(self):
        # 1. Fetch the data
        df = self.data_feed.get_latest_data()
        if df is None or df.empty: return

        # 2. Check whether the kline closed (based on open_time)
        current_candle_time = df.iloc[-1].name
        if self.last_candle_time == current_candle_time:
            # Only run the logic when the timestamp changed
            return 
        
        self.logger.info(f"📊 新 K 线闭合: {current_candle_time} | Close: {df.iloc[-1]['close']}")
        
        # 3. Read the live state
        curr_dir, _, last_price = self.venue.get_current_state()
        equity = self.venue.get_account_equity()
        
        # Position share (rough estimate; the exact value needs the notional)
        current_price = df.iloc[-1]['close']
        curr_pos_size_pct = 0.0
        if curr_dir != PositionDir.FLAT:
            curr_pos_size_pct = 0.1 # any non-zero value tells the strategy layer there is a position

        # 4. Let the engine decide
        # Note: StrategyBase calls venue.submit_order internally to place orders
        self.strategy.process(
            df=df,
            current_time=pd.to_datetime(datetime.now()), # or use the server time
            account_equity=equity,
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
                # Heartbeat: print every 5 minutes to show the loop is alive
                if time.time() - last_heartbeat > 300:
                    self.logger.info("💓 Heartbeat: Bot is still alive and cycling...")
                    last_heartbeat = time.time()
                
                self.run_step()
                time.sleep(TurtleConfig.POLL_INTERVAL)
            except Exception as e:
                self.logger.error(f"主循环异常: {e}", exc_info=True)
                time.sleep(10)

if __name__ == "__main__":
    # Parse the arguments, reusing the martingale logic
    parser = argparse.ArgumentParser(description="Bybit Turtle Bot")
    parser.add_argument("-t", "--testnet", action="store_true", help="Run on Testnet")
    args = parser.parse_args()

    # Path configuration
    keypath = 'Maringale'
    side_path = 'Long' # use the key under the Long folder by default
    BASE = os.path.dirname(os.path.abspath(__file__))
    
    # Your key files are assumed to be laid out like this
    # keys/Maringale/Long/hmac_api_key
    API_K = os.path.join(BASE, "keys", keypath, side_path, "hmac_api_key")
    API_S = os.path.join(BASE, "keys", keypath, side_path, "hmac_secret")
    RSA_K = os.path.join(BASE, "keys", keypath, side_path, "api_key")
    RSA_P = os.path.join(BASE, "keys", "bybit_rsa.pem")

    if not os.path.exists(API_K):
        print(f"❌ Key file not found: {API_K}")
        sys.exit(1)

    # Initialize the engine
    engine = BybitEngine(API_K, API_S, RSA_K, RSA_P, testnet=args.testnet)
    
    # Start the bot
    bot = BybitTurtleBot(engine, is_long_account=True)
    
    # Signal handling
    def signal_handler(sig, frame):
        print("\n👋 Stop signal received...")
        bot.stop_signal = True
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    bot.run()