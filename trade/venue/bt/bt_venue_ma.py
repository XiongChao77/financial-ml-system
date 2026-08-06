import backtrader as bt
from trade.venue.bt.bt_venue_base import BtVenue
from trade.strategy.strategy_ma import MaCrossoverStrategy, MaObservation
from trade.core.protocol import PositionDir
from trade.core.protocol import ActionType

class MaBtVenue(BtVenue):
    def __init__(self):
        super().__init__()
        strategy_config = self.p.strategy_config
        # Declare the indicators: backtrader handles their warm-up automatically
        self.fast_ma = bt.ind.SMA(period=strategy_config.fast_period)
        self.slow_ma = bt.ind.SMA(period=strategy_config.slow_period)
        
        self.strategy = MaCrossoverStrategy(config=strategy_config)

    def next(self):
        # Warm-up guard
        if len(self) < self.strategy.config.slow_period:
            return

        # Assembled by the venue layer, then extended with the MA specific fields
        base = self.observe()
        state = MaObservation(
            market=base.market,           # the MA strategy does not rely on the pred signal column
            position=base.position,
            account=base.account,
            current_time=base.current_time,
            fast_ma=self.fast_ma[0],
            slow_ma=self.slow_ma[0],
        )

        action = self.strategy.process(state)
        self.execute_action(action)

    def execute_action(self, action):
        if action.action == ActionType.HOLD:
            return
