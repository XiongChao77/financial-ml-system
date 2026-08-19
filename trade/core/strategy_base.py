"""
Strategy layer base class: consumes an Observation, produces a TradeIntent, executed through the venue.

The strategy layer imports neither backtrader nor any exchange SDK -- the same strategy instance
can run a backtest on BtVenue and trade live on BybitVenue / MT5Venue.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional

from trade.core.protocol import TradeIntent,ActionType,PositionDir,Observation
from trade.core.venue_base import VenueBase

class StrategyBase(ABC):

    def __init__(self, venue: Optional[VenueBase]):
        super().__init__()
        self.venue = venue
        self.last_action:TradeIntent = None
        self.last_state:Observation = None
        self._last_bar_time = None
        self.bar_interval_ms = None

    def process(self, *args, **kwargs) -> TradeIntent:
        """Validate the incoming bar, then dispatch to the strategy decision logic.

        Most strategies receive an ``Observation``.  The older dataframe based
        strategies pass ``current_time`` as a keyword argument; keeping both
        forms here gives every strategy the same continuity check and warning
        hook.
        """
        state = (
            args[0]
            if args and isinstance(args[0], Observation)
            else kwargs.get("state")
        )
        current_time = (
            state.current_time
            if isinstance(state, Observation)
            else kwargs.get("current_time")
        )
        expected_interval_ms = self._expected_bar_interval_ms(state)
        self._check_bar_continuity(current_time, expected_interval_ms)

        action = self._process(*args, **kwargs)
        self._last_bar_time = current_time
        if isinstance(state, Observation):
            self.last_state = state
        if action.action != ActionType.NOOP:
            self.last_action = action
        return action

    def strategy_warning(self, message: str) -> None:
        """Single replaceable warning channel for strategy-level data problems.

        Live trading can override this method with email/alert delivery without
        changing the continuity check itself.
        """
        logger = getattr(self, "logger", logging.getLogger("trade"))
        logger.error(message)

    def _expected_bar_interval_ms(self, state) -> Optional[int]:
        if isinstance(state, Observation):
            interval_ms = getattr(state.market, "bar_interval_ms", None)
            if interval_ms is not None:
                return interval_ms

        if self.bar_interval_ms is not None:
            return self.bar_interval_ms

        venue_params = getattr(getattr(self, "venue", None), "p", None)
        return getattr(venue_params, "bar_interval_ms", None)

    def _check_bar_continuity(self, current_time, expected_interval_ms) -> None:
        if current_time is None or self._last_bar_time is None:
            return
        if expected_interval_ms is None or expected_interval_ms <= 0:
            return

        try:
            actual_interval_ms = int(
                round((current_time - self._last_bar_time).total_seconds() * 1000)
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self.strategy_warning(
                "K-line continuity check failed: "
                f"previous={self._last_bar_time!r}, current={current_time!r}, error={exc}"
            )
            return

        expected_interval_ms = int(expected_interval_ms)
        if actual_interval_ms == expected_interval_ms:
            return

        if actual_interval_ms <= 0:
            detail = "duplicate or out-of-order bar"
        elif actual_interval_ms > expected_interval_ms:
            if actual_interval_ms % expected_interval_ms == 0:
                missing = actual_interval_ms // expected_interval_ms - 1
                detail = f"missing_bars={missing}"
            else:
                detail = "unaligned gap or wrong-period bar"
        else:
            detail = "overlapping or wrong-period bar"

        self.strategy_warning(
            "K-line discontinuity detected: "
            f"previous={self._last_bar_time}, current={current_time}, "
            f"expected_interval_ms={expected_interval_ms}, "
            f"actual_interval_ms={actual_interval_ms}, {detail}"
        )

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
        elif action.action == ActionType.OPEN:
            is_buy = (action.target_dir == PositionDir.POSITIVE )
            self.venue.submit_order(
                action.order_qty,
                is_buy=is_buy,
                stop_loss_pct=action.stop_loss_pct,
                take_profit_pct=action.take_profit_pct,
            )
