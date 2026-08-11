from trade.strategy.strategy_bbm import BbmSignalStrategy
from trade.venue.bt.bt_venue_base import BtVenue


class BbmBtVenue(BtVenue):
    def __init__(self):
        super().__init__()
        self.strategy = BbmSignalStrategy(
            self,
            config=self.p.strategy_config,
            init_equity=self.p.initial_equity,
            leverage=self.leverage,
        )

    def next(self):
        self.collect_bar_metrics()
        self.strategy.process(self.observe())
