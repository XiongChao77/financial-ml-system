from trade.strategy.strategy_bbm import BbmSignalStrategy
from trade.venue.bt.bt_venue_base import BtVenue


class BbmBtVenue(BtVenue):
    def __init__(self):
        super().__init__()
        self.strategy = BbmSignalStrategy(
            config=self.p.strategy_config,
            init_equity=self.p.initial_equity,
            data_interval_ms=self.p.data_interval_ms,
            leverage=self.leverage,
        )

    def next(self):
        self.collect_bar_metrics()
        venue_observation = self.observe()
        intent = self.strategy.process(venue_observation)
        self.execute_action(intent)
