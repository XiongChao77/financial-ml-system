import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from trade.core.protocol import ActionType, Observation, PositionDir, Signal, TradeIntent
from trade.core.strategy_base import StrategyBase
from trade.core.venue_base import VenueBase


@dataclass
class BbmStrategyConfig:
    """Strategy parameters for the Binary Barrier Model label backtest."""

    strategy_type: str = "bbm"
    risk_per_trade_pct: float = 0.02
    allow_long: bool = True
    allow_short: bool = True
    prob_thresh: Optional[float] = None
    min_expected_move_pct: float = 0.01 
    max_daily_loss_pct: float = 0.025


class BbmSignalStrategy(StrategyBase):
    """
    Enter from model signal and let the bracket order exit by BBM barriers.

    There is intentionally no holding-duration rule here: once a position is
    opened, later model signals do not close or reverse it. The trade exits only
    through the attached take-profit/stop-loss bracket or the daily loss guard.

    Backtrader note: bracket exits are simulated from OHLC bars, so if one bar
    contains both the stop-loss and take-profit prices, the engine cannot know
    the real intrabar order. Backtrader resolves the first executable child
    order according to its broker/order matching flow, then OCO-cancels the
    sibling. That is different from BBM labeling, where same-bar dual touches
    are marked INVALID because the forward path is ambiguous.
    """

    def __init__(
        self,
        venue: VenueBase,
        config: BbmStrategyConfig,
        init_equity: float,
        leverage: float = 1.0,
    ):
        super().__init__(venue)
        self.logger = logging.getLogger("trade")
        self.config = config
        self.init_equity = float(init_equity)
        self.leverage = max(1.0, float(leverage))
        self.day_start_equity: Optional[float] = None
        self.last_trade_date = None
        self.is_halted_today = False
        self.meltdown_days = 0
        self.entries = 0
        self.skipped_missing_threshold = 0
        self.skipped_small_threshold = 0
        self.skipped_size = 0

    def _update_daily_state(self, current_time: datetime, account_equity: float):
        current_date = current_time.date()
        if self.last_trade_date != current_date:
            self.day_start_equity = account_equity
            self.last_trade_date = current_date
            self.is_halted_today = False

    def _barrier_pcts(self, target_dir: PositionDir, state: Observation) -> tuple[float, float]:
        long_threshold = state.market.threshold_long
        short_threshold = state.market.threshold_short
        if not valid_pct(long_threshold) or not valid_pct(short_threshold):
            self.skipped_missing_threshold += 1
            return 0.0, 0.0
        if (
            long_threshold < self.config.min_expected_move_pct
            or short_threshold < self.config.min_expected_move_pct
        ):
            self.skipped_small_threshold += 1
            return 0.0, 0.0

        if target_dir == PositionDir.POSITIVE:
            return float(short_threshold), float(long_threshold)
        if target_dir == PositionDir.NEGATIVE:
            return float(long_threshold), float(short_threshold)
        return 0.0, 0.0

    def _calculate_order_qty(
        self,
        state: Observation,
        stop_loss_pct: float,
        remaining_risk_budget: float,
    ) -> float:
        if not valid_pct(stop_loss_pct):
            return 0.0

        risk_equity = max(self.init_equity, state.account.equity)
        intended_qty = (
            self.config.risk_per_trade_pct * risk_equity
        ) / (state.market.price * stop_loss_pct)
        max_risk_qty = (
            remaining_risk_budget * 0.8
        ) / (state.market.price * stop_loss_pct)
        final_qty = min(intended_qty, max_risk_qty)

        required_margin = final_qty * state.market.price / self.leverage
        if required_margin > state.account.equity:
            final_qty = state.account.equity * self.leverage / state.market.price

        if final_qty <= 0 or not math.isfinite(final_qty):
            self.skipped_size += 1
            return 0.0
        return final_qty

    def process(self, state: Observation) -> TradeIntent:
        signal = state.market.signal
        if signal == Signal.INVALID:
            signal = Signal.NEUTRAL
        if self.config.prob_thresh is not None and state.market.pred_prob < self.config.prob_thresh:
            signal = Signal.NEUTRAL

        self._update_daily_state(state.current_time, state.account.equity)
        daily_loss_abs = max(0.0, self.day_start_equity - state.account.equity)
        daily_max_loss_allowed_abs = self.day_start_equity * self.config.max_daily_loss_pct

        if daily_loss_abs >= daily_max_loss_allowed_abs:
            if not self.is_halted_today:
                self.logger.warning(
                    "Daily loss guard triggered: "
                    f"{daily_loss_abs / self.day_start_equity:.2%}"
                )
                self.is_halted_today = True
                self.meltdown_days += 1
            if state.position.dir != PositionDir.FLAT:
                action = TradeIntent(ActionType.CLOSE)
                self.execute_action(action)
                return action
            return TradeIntent(ActionType.NOOP)

        if state.position.dir != PositionDir.FLAT or self.is_halted_today:
            return TradeIntent(ActionType.NOOP)

        target_dir = PositionDir.FLAT
        if signal == Signal.POSITIVE and self.config.allow_long:
            target_dir = PositionDir.POSITIVE
        elif signal == Signal.NEGATIVE and self.config.allow_short:
            target_dir = PositionDir.NEGATIVE

        if target_dir == PositionDir.FLAT:
            return TradeIntent(ActionType.NOOP)

        stop_loss_pct, take_profit_pct = self._barrier_pcts(target_dir, state)
        if not valid_pct(stop_loss_pct) or not valid_pct(take_profit_pct):
            return TradeIntent(ActionType.NOOP)

        remaining_risk_budget = max(0.0, daily_max_loss_allowed_abs - daily_loss_abs)
        order_qty = self._calculate_order_qty(state, stop_loss_pct, remaining_risk_budget)
        if order_qty <= 0:
            return TradeIntent(ActionType.NOOP)

        action = TradeIntent(
            action=ActionType.OPEN,
            target_dir=target_dir,
            target_layers=1,
            order_qty=order_qty,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            reason="bbm_signal_entry",
        )
        self.entries += 1
        self.execute_action(action)
        return action

    def execute_action(self, action: TradeIntent):
        if action.action == ActionType.NOOP:
            return
        if action.action == ActionType.CLOSE:
            self.venue.close_position()
            return
        if action.action == ActionType.OPEN:
            self.venue.submit_order(
                action.order_qty,
                is_buy=action.target_dir == PositionDir.POSITIVE,
                stop_loss_pct=action.stop_loss_pct,
                take_profit_pct=action.take_profit_pct,
            )

    def report(self) -> dict:
        return {
            "meltdown_days": self.meltdown_days,
            "entries": self.entries,
            "min_expected_move_pct": self.config.min_expected_move_pct,
            "skipped_missing_threshold": self.skipped_missing_threshold,
            "skipped_small_threshold": self.skipped_small_threshold,
            "skipped_size": self.skipped_size,
        }


def valid_pct(value) -> bool:
    return (
        value is not None
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
    )
