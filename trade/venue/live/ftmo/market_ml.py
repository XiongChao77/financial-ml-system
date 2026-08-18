import time
import logging
import json
import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# Add the project path so custom modules can be imported
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", "..", "..", ".."))

# Import project modules
from data_process import common
from data_process.common import FeatureFactory
from model import model_loader
from model import data_loader
from trade.strategy.strategy_ml import MlSignalStrategy, MlStrategyConfig
from trade.core.protocol import (
    Observation, MarketView, PositionView, AccountView, PositionDir, ActionType, Signal,
)
from trade.venue.live.ftmo import mt5_venue
from trade.venue.live.binance_data_feed import  BinanceDataFeed
pd.set_option("display.max_columns", None)   # no column limit
pd.set_option("display.width", None)         # automatic width (no forced wrapping)
pd.set_option("display.max_colwidth", None)  # do not truncate cell content
# ============================================================
# Configuration
# ============================================================
class LiveConfig:
    # Symbol mapping
    SYMBOL_BINANCE = "DOGEUSDT"  # data source symbol
    SYMBOL_FTMO = "DOGEUSD"      # execution symbol (FTMO usually uses BTCUSD)
    
    # Timeframe (minutes)
    TIMEFRAME = common.BaseDefine.interval
    allow_short = True
    allow_long = True
    fixed_hold_bars = common.BaseDefine.predict_num#BaseDefine.predict_num
    prob_thresh: float =None#0.5#None#0.45
    commission = 0.05   # 0.1 = 0.1%  .can't be 0
    cash = 10000
    atr_sl_long_mult = 5 #2.5
    atr_sl_short_mult = 2.5 #2.5
    atr_tp_mult = 0.99 # take profit. 0 - n multiples
    risk_per_trade_pct = 0.5     #0-1
    max_daily_loss_pct = 0.025

    mt5_path = r"C:\Program Files\Five Percent Online MetaTrader 5\terminal64.exe"
    max_layers = 1
    # MT5 magic number
    MAGIC_NUMBER = 888888
    # Polling interval (seconds)
    POLL_INTERVAL = 5
# ============================================================
# 3. Main controller: LiveBot
# ============================================================
class LiveBot:
    def __init__(self):
        self.logger, log_path = common.setup_session_logger(
                    sub_folder=f'market_ml',
                    symbol=LiveConfig.SYMBOL_FTMO
                )
        
        self.logger.info("Initializing Live Bot...")

        self._log_startup_info(log_path)
        self.venue = mt5_venue.MT5Venue(LiveConfig.mt5_path,LiveConfig.SYMBOL_FTMO, LiveConfig.MAGIC_NUMBER,logger= self.logger)
        self.model_handler = model_loader.ModelHandler() # loads the trained model automatically

        # 1. Set the parameters
        self.interval_ms = common.get_interval_ms(LiveConfig.TIMEFRAME) 
        self.factory = FeatureFactory(self.interval_ms)
        
        # 2. Work out how much history is needed (count)
        self.min_bars_needed = self.factory.get_global_min_history() + common.BaseDefine.predict_num
        self.logger.info(f"History Required: {self.min_bars_needed} bars")
        
        # 3. Initialize the data source (with cache)
        # max_len is set a bit above min_bars_needed, e.g. +500, to leave headroom
        self.data_feed = BinanceDataFeed(
            LiveConfig.SYMBOL_BINANCE, 
            LiveConfig.TIMEFRAME, 
            max_len = self.min_bars_needed + 500
        )
        #strategy
        self.strategy = MlSignalStrategy(
            self,
            config=MlStrategyConfig(
                risk_per_trade_pct=LiveConfig.risk_per_trade_pct,
                fixed_hold_bars=LiveConfig.fixed_hold_bars,
                allow_long=LiveConfig.allow_long,
                allow_short=LiveConfig.allow_short,
                prob_thresh=LiveConfig.prob_thresh,
                atr_sl_long_mult=LiveConfig.atr_sl_long_mult,
                atr_sl_short_mult=LiveConfig.atr_sl_short_mult,
                atr_tp_mult=LiveConfig.atr_tp_mult,
                max_daily_loss_pct=LiveConfig.max_daily_loss_pct,
            ),
            init_equity=self.venue.get_account_equity(),
        )

        # 4. Warm up the data -> fill the memory cache
        self.data_feed.initialize_cache(self.min_bars_needed, self.interval_ms)
        
        # Remember the last timestamp after initialization
        initial_df = self.data_feed.get_latest_data()
        self.last_candle_time = initial_df.iloc[-1]["open_time_date_utc"] if not initial_df.empty else None

    def _log_startup_info(self, log_path):
        """
        [added] print the detailed environment of this run
        """
        self.logger.info("=" * 60)
        self.logger.info(f"🚀 LIVE BOT SESSION STARTED")
        self.logger.info(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"📂 Log File: {log_path}")
        self.logger.info(f"📊 Target Symbol: {LiveConfig.SYMBOL_FTMO}")
        self.logger.info(f"🔗 Data Source: {LiveConfig.SYMBOL_BINANCE}")
        self.logger.info("-" * 20 + " PARAMETERS " + "-" * 20)
        
        # Walk every parameter of the Config class automatically
        for key in dir(LiveConfig):
            if not key.startswith("__"):
                val = getattr(LiveConfig, key)
                self.logger.info(f"{key.ljust(20)}: {val}")
        
        self.logger.info("=" * 60)

    def run_step(self):
        """
        Runs every few seconds
        """
        # 1. Get the latest data (DataFeed handles the incremental update internally)
        # This df is already cleaned, long enough, and free of the unclosed kline
        df = self.data_feed.get_latest_data()
        
        if df is None or df.empty:
            return

        # 2. Check whether a new kline appeared
        # Compare the newest timestamp of this round with the one processed last round
        current_candle_time = df.iloc[-1]["open_time_date_utc"]
        
        if self.last_candle_time == current_candle_time:
            # Unchanged timestamp means no kline closed -> skip
            pass#return 
            
        self.logger.info(f"✨ New Candle Closed: {current_candle_time} | Buffer Size: {len(df)}")
        
        # 2. Feature engineering & model prediction
        try:
            df = self.factory.generate(df)

            # C. Model inference
            # ModelHandler handles the TimeSeriesWindowDataset and the normalization internally
            # Note: predict returns a DataFrame carrying pred and pred_prob
            inference_df = df.iloc[-(self.model_handler.seq_len + 200):]
            ds = data_loader.TimeSeriesWindowDataset(
                df=df,
                kline_interval_ms = _interval_ms,
                feature_cols=handler.feature_cols,
                label_col=handler.label_col,
                seq_len=handler.seq_len,
                is_live=False,
            )
            df_pred, _ = self.model_handler.predict(inference_df, kline_interval_ms= self.interval_ms, is_live = True, diff_thresh = None)
            
            # Prediction of the most recent kline
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

        # 3. Read the current MT5 state
        curr_dir, curr_layers, curr_vol = self.venue.get_current_state() #sync state here
        self.logger.info(f"MT5 State: Dir={curr_dir}, Layers={curr_layers}, Vol={curr_vol}")

        # Data sanity check
        current_signal = Signal.INVALID if np.isnan(pred) else Signal(int(pred))
        current_prob = 0.0 if np.isnan(pred_prob) else float(pred_prob)

        state = Observation(
            market=MarketView(
                price=current_price,
                signal=current_signal,
                pred_prob=float(current_prob),
                atr_pct=last_row["atr_14"],
            ),
            position=PositionView(dir=curr_dir, layers=curr_layers),
            account=AccountView(equity=self.venue.get_account_equity()),
            current_time=self.venue.get_server_time(),
        )

        self.strategy.process(state)

        # Update the timestamp so the same bar is not processed twice
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
