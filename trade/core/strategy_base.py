"""
Strategy layer base class: consumes an Observation, produces a TradeIntent, executed through the venue.

The strategy layer imports neither backtrader nor any exchange SDK -- the same strategy instance
can run a backtest on BtVenue and trade live on BybitVenue / MT5Venue.
"""

from abc import ABC, abstractmethod

from trade.core.protocol import TradeIntent
from trade.core.venue_base import VenueBase


class StrategyBase(ABC):

    def __init__(self, venue: VenueBase):
        super().__init__()
        self.venue = venue

    @abstractmethod
    def process(self) -> TradeIntent:
        """
        Turn an Observation into a TradeIntent
        """
        pass

    def finalize(self):
        """
        Wrap-up hook: called once when the venue life cycle ends, to print the strategy side summary.
        (Distinct from venue.stop(): venue.stop() is the venue life cycle,
         finalize() is the strategy settling its own books.)
        """
        pass

    def report(self) -> dict:
        """
        Single exit for strategy specific statistics (content differs per strategy, channel does not).

        Returns a dict that the venue automatically splits in two for the layer above:
          - scalars (int/float/str/bool/None) -> report["strategy"], directly jsonl-able
          - everything else (list/dict details) -> report_additional["strategy_detail"]
        You may also return {"summary": {...}, "detail": {...}} to split it explicitly.
        Defaults to an empty dict, i.e. no strategy specific statistics.
        """
        return {}
