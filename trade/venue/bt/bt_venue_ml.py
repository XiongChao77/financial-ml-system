from trade.venue.bt.bt_venue_base import BtVenue
from trade.strategy.strategy_ml import MlSignalStrategy, Observation,TradeIntent,ActionType,PositionDir,Signal
# --- Strategy ---
class MlBtVenue(BtVenue):
    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        strategy_config = self.p.strategy_config
        self.strategy = MlSignalStrategy(
            self,
            config=strategy_config,
            init_equity=self.p.initial_equity,
            exist_hold_bars=0,
            leverage=self.leverage,
        )

    def next(self):
        # Margin usage + input signal collection + label alignment audit (generic venue side)
        self.collect_bar_metrics()

        state = self.observe()
        self.strategy.process(state)
