import backtrader as bt
import pandas as pd
from trade.venue.bt.bt_venue import BtVenue
from trade.strategy.strategy_rules import RulesStrategy
from trade.core.protocol import PositionDir

class RulesBtVenue(BtVenue):
    def __init__(self):
        super().__init__()

        # 3. Build the decision engine
        self.strategy = RulesStrategy(
            venue=self,
        )
        # 1. Turtle strategy parameters (must stay in sync with RulesStrategy)
        self.entry_period = self.strategy.entry_period
        self.exit_period = self.strategy.exit_period
        self.atr_period = self.strategy.atr_period

        # 2. Pre-create the backtrader indicators in init (computed vectorized)
        # Note: the (-1) offset emulates shift(1) and prevents look-ahead bias
        self.entry_high = bt.indicators.Highest(self.data.high(-1), period=self.entry_period)
        self.entry_low = bt.indicators.Lowest(self.data.low(-1), period=self.entry_period)
        self.exit_high = bt.indicators.Highest(self.data.high(-1), period=self.exit_period)
        self.exit_low = bt.indicators.Lowest(self.data.low(-1), period=self.exit_period)
        
        # The standard turtle uses Wilder's ATR, backtrader defaults to a smoothed moving average
        self.atr = bt.indicators.ATR(self.data, period=self.atr_period)


    def next(self):
        # 1. Risk audit
        self.collect_bar_metrics()

        # 2. Determine the lookback length dynamically
        # At least 2 rows are needed for gap detection; entry_period + 1 if the strategy recomputes indicators
        lookback = self.strategy.entry_period*10
        
        # Skip until enough data points are available
        if len(self.data) < lookback:
            return

        # Take the historical slice
        # backtrader's get(size=N) returns the N values counting back from the current point
        times = [self.data.datetime.datetime(-i) for i in range(lookback)][::-1]
        
        df = pd.DataFrame({
            'open': self.data.open.get(size=lookback),
            'high': self.data.high.get(size=lookback),
            'low': self.data.low.get(size=lookback),
            'close': self.data.close.get(size=lookback),
            
            # Same for the historical values of the pre-computed indicators
            'atr': self.atr.get(size=lookback),
            'entry_high': self.entry_high.get(size=lookback),
            'entry_low': self.entry_low.get(size=lookback),
            'exit_high': self.exit_high.get(size=lookback),
            'exit_low': self.exit_low.get(size=lookback)
        }, index=times)

        # 3. Read the live position state
        if self.position.size > 0:
            curr_dir = PositionDir.POSITIVE 
        elif self.position.size < 0:
            curr_dir = PositionDir.NEGATIVE
        else:
            curr_dir = PositionDir.FLAT

        # 4. Drive the decision
        # Note: the df passed in is lookback long, which satisfies iloc[-1] and iloc[-2]
        current_time = times[-1]
        self.strategy.process(
            df=df, 
            current_time=current_time, 
            account_equity=self.broker.getvalue(),
            curr_dir=curr_dir,
            curr_pos_qty=self.position.size,
        )
    # ----------------------------------------------------------------
    # Audit helpers
    # ----------------------------------------------------------------

    def stop(self):
        self.logger.info(f"🚩 回测结束 | 最大保证金占用: {self.max_margin_level:.2%}")