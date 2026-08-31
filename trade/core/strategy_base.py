"""
Strategy layer base class: consumes an Observation, produces a TradeIntent, executed through the venue.

The strategy layer imports neither backtrader nor any exchange SDK -- the same strategy instance
can run a backtest on BtVenue and trade live on BybitVenue / MT5Venue.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional

from trade.core.protocol import TradeIntent, ActionType, PositionDir, Observation
from trade.core.venue_base import VenueBase


class StrategyBase(ABC):

    def __init__(self, data_interval_ms: int):
        super().__init__()
        if data_interval_ms is None or data_interval_ms <= 0:
            raise ValueError("data_interval_ms must be positive")
        self.last_action: TradeIntent = None
        self.last_state: Observation = None
        self._last_candle_open_time_utc = None
        self.data_interval_ms = int(data_interval_ms)

    def process(self, *args, **kwargs) -> TradeIntent:
        """Validate the incoming bar, then dispatch to the strategy decision logic.

        Most strategies receive an ``Observation``.  The older dataframe based
        strategies pass ``candle_open_time_utc`` as a keyword argument; keeping both
        forms here gives every strategy the same continuity check and warning
        hook.
        """
        state = args[0] if args and isinstance(args[0], Observation) else kwargs.get("state")
        candle_open_time_utc = (
            state.candle_open_time_utc
            if isinstance(state, Observation)
            else kwargs.get("candle_open_time_utc")
        )
        self._check_data_continuity(candle_open_time_utc)

        action = self._process(*args, **kwargs)
        self._last_candle_open_time_utc = candle_open_time_utc
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

    def _check_data_continuity(self, candle_open_time_utc) -> None:
        if candle_open_time_utc is None or self._last_candle_open_time_utc is None:
            return

        try:
            actual_interval_ms = int(
                round(
                    (
                        candle_open_time_utc
                        - self._last_candle_open_time_utc
                    ).total_seconds()
                    * 1000
                )
            )
        except (AttributeError, TypeError, ValueError) as exc:
            self.strategy_warning(
                "K-line continuity check failed: "
                f"previous={self._last_candle_open_time_utc!r}, "
                f"current={candle_open_time_utc!r}, error={exc}"
            )
            return

        expected_interval_ms = self.data_interval_ms
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
            f"previous={self._last_candle_open_time_utc}, "
            f"current={candle_open_time_utc}, "
            f"expected_interval_ms={expected_interval_ms}, "
            f"actual_interval_ms={actual_interval_ms}, {detail}"
        )

    @abstractmethod
    def _process(self, state: Observation) -> TradeIntent:
        pass

    def finalize(self):
        pass

    def report(self) -> tuple[dict, dict]:
        """Return ``(report_summary, report_details)`` for this strategy."""
        return {}, {}
