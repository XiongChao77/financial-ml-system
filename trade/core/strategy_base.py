"""
Strategy layer base class: consumes an Observation, produces a TradeIntent, executed through the venue.

The strategy layer imports neither backtrader nor any exchange SDK -- the same strategy instance
can run a backtest on BtVenue and trade live on BybitVenue / MT5Venue.
"""

from abc import ABC, abstractmethod

from trade.core.protocol import TradeIntent,ActionType,PositionDir,Observation
from trade.core.venue_base import VenueBase

class StrategyBase(ABC):

    def __init__(self, venue: VenueBase):
        super().__init__()
        self.venue = venue
        self.last_action:TradeIntent = None
        self.last_state:Observation = None

    def process(self, state: Observation) -> TradeIntent:
        action = self._process(state)
        self.last_state = state
        if action.action != ActionType.NOOP:
            self.last_action = action
        return action

    @abstractmethod
    def _process(self, state: Observation) -> TradeIntent:
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

    def execute_action(self, action: TradeIntent):
        """Reworked to use the submit_order interface and pass the stop loss parameters"""
        if action.action == ActionType.NOOP:
            return

        if action.action == ActionType.CLOSE:
            self.venue.close_position()
        else:
            is_buy = (action.target_dir == PositionDir.POSITIVE )
            
            # Execute the order
            if action.action == ActionType.REVERSE:
                # On a reverse, close everything first
                self.venue.close_position()
                # then open the first layer in the new direction
                self.venue.submit_order(action.order_qty, is_buy=is_buy, stop_loss_pct=action.stop_loss_pct, take_profit_pct=action.take_profit_pct)
                
            elif action.action == ActionType.OPEN:
                self.venue.submit_order(action.order_qty, is_buy=is_buy, stop_loss_pct=action.stop_loss_pct, take_profit_pct=action.take_profit_pct)