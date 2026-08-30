import backtrader as bt
import pandas as pd
import logging
from trade.venue.bt.bt_venue_base import BtVenue
from trade.strategy.strategy_turtle import TurtleStrategy
from trade.core.protocol import PositionDir

class TurtleBtVenue(BtVenue):
    def __init__(self):
        super().__init__()
        strategy_config = self.p.strategy_config
        # === core: build the decision engine ===
        # self (i.e. the BtVenue) is passed in as the executor
        self.strategy = TurtleStrategy(
            venue=self,
            config=strategy_config,
            bar_interval_ms=self.p.bar_interval_ms,
        )
        
        # Audit-only variables
        self.atr = bt.ind.ATR(period=strategy_config.atr_period)

    def next(self):
        """
        Fires once per kline: data conversion -> drive the decision
        """
        # 1. Risk audit
        self.collect_bar_metrics()
        current_atr = self.atr[0]

        # 2. Convert the backtrader series into a pandas DataFrame
        lookback = max(self.strategy.config.entry_period, self.strategy.config.atr_period * 4) + 10
        if len(self.data) < lookback:
            return
            
        dt_list = [self.data.datetime.datetime(-i) for i in range(lookback-1, -1, -1)]

        df = pd.DataFrame({
            'open': self.data.open.get(size=lookback),
            'high': self.data.high.get(size=lookback),
            'low': self.data.low.get(size=lookback),
            'close': self.data.close.get(size=lookback),
            'volume': self.data.volume.get(size=lookback)
        }, index=dt_list) # key: set the time index

        # 3. Read the live environment parameters
        current_time = self.data.datetime.datetime(0)
        account_equity = self.broker.getvalue() # current total account equity
        current_price = self.data.close[0]

        # --- added: build the position state handed to StrategyBase ---
        # A. Position direction (PositionDir)
        if self.position.size > 0:
            curr_dir = PositionDir.POSITIVE 
        elif self.position.size < 0:
            curr_dir = PositionDir.NEGATIVE
        else:
            curr_dir = PositionDir.FLAT

        # B. Notional share of the position (curr_pos_size)
        # Formula: (abs(position size) * current price) / total equity
        pos_value = abs(self.position.size) * current_price
        curr_pos_size = pos_value / account_equity if account_equity > 0 else 0

        # Detect current position state from Backtrader internals
        if not self.position:
            last_entry_price = 0.0
        else:
            if self.live_trades:
                # Get the price of the most recent order added
                last_trade = self.live_trades[-1]
                last_entry_price = last_trade['main'].created.price
            else:
                last_entry_price = current_price # Fallback

        # 4. Drive the decision: pass the 5 reworked parameters
        self.strategy.process(
            df=df, 
            current_time=current_time, 
            account_equity=account_equity,
            curr_dir=curr_dir,
            curr_pos_size=curr_pos_size,
            last_entry_price = last_entry_price,
            # atr = current_atr,
        )
