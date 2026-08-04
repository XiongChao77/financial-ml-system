import backtrader as bt
from trade.venue.bt.bt_venue import BtVenue
from trade.strategy.strategy_ma import MaCrossoverStrategy, MaObservation
from trade.core.protocol import PositionDir
from trade.core.protocol import ActionType

class MaBtVenue(BtVenue):
    params = dict(
        fast_period=50,
        slow_period=200,
        risk_per_trade_pct=0.95,
        stop_loss=0.05,  # fixed 5% stop loss
    )

    def __init__(self):
        super().__init__()
        # Declare the indicators: backtrader handles their warm-up automatically
        self.fast_ma = bt.ind.SMA(period=self.params.fast_period)
        self.slow_ma = bt.ind.SMA(period=self.params.slow_period)
        
        self.strategy = MaCrossoverStrategy(risk_per_trade_pct=self.params.risk_per_trade_pct)

    def next(self):
        # Warm-up guard
        if len(self) < self.params.slow_period:
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