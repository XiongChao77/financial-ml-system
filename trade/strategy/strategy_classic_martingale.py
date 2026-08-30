"""Classic price-martingale decision logic.

This is the small, conventional trading variant of martingale:

1. Open one base order.
2. Whenever price moves against the latest entry by ``price_deviation_pct``,
   add the next fixed 2x level (1x, 2x, 4x, ...).
3. Close when estimated net profit reaches ``take_profit_pct`` of entry notional.

With OHLC input the default pessimistic mode replays Open -> Low -> High -> Close
in an internal deterministic ledger. Without OHLC it falls back to event-driven intents.
"""

from dataclasses import dataclass
import logging
from typing import Optional

from trade.core.protocol import (
    ActionType,
    Observation,
    PositionDir,
    TradeIntent,
)
from trade.core.strategy_base import StrategyBase
from trade.core.venue_base import VenueBase


@dataclass(frozen=True)
class ClassicMartingaleConfig:
    """Parameters for one classic martingale cycle."""

    base_order_pct: float = 0.02
    price_deviation_pct: float = 0.01
    max_layers: int = 7
    take_profit_pct: float = 0.01  # target net profit / total entry notional

    def __post_init__(self):
        if not 0.0 < self.base_order_pct <= 1.0:
            raise ValueError("base_order_pct must be in (0, 1]")
        if self.price_deviation_pct <= 0.0:
            raise ValueError("price_deviation_pct must be positive")
        if self.max_layers < 1:
            raise ValueError("max_layers must be >= 1")
        if self.take_profit_pct <= 0.0:
            raise ValueError("take_profit_pct must be positive")


class ClassicMartingaleStrategy(StrategyBase):
    """Long-only 2x martingale with pessimistic OHLC execution."""

    VOLUME_MULT = 2.0

    def __init__(
        self,
        venue: VenueBase,
        config: ClassicMartingaleConfig,
        bar_interval_ms: int,
        leverage: float = 1.0,
        commission_pct: float = 0.0,
        maintenance_margin_pct: float = 0.0,
        margin_usage_cap_pct: float = 0.80,
        pessimistic: bool = True,
    ):
        super().__init__(venue, bar_interval_ms)
        self.config = config
        self.leverage = max(1.0, float(leverage))
        self.commission_rate = max(0.0, float(commission_pct)) / 100.0
        self.maintenance_margin_rate = max(
            0.0, float(maintenance_margin_pct)
        ) / 100.0
        if not 0.0 < margin_usage_cap_pct <= 1.0:
            raise ValueError("margin_usage_cap_pct must be in (0, 1]")
        self.margin_usage_cap_pct = float(margin_usage_cap_pct)
        self.pessimistic = pessimistic
        self.logger = logging.getLogger("trade")

        self.cycle_dir = PositionDir.FLAT
        self.cycle_layers = 0
        self.cycle_qty = 0.0
        self.cycle_cost = 0.0
        self.cycle_avg_price = 0.0
        self.last_entry_price = 0.0
        self.cycle_base_notional = 0.0

        self.pending_entry = False
        self.pending_close_reason: Optional[str] = None

        self.cycles_started = 0
        self.cycles_closed = 0
        self.safety_orders_filled = 0
        self.max_layers_seen = 0
        self.exit_counts = {"take_profit": 0, "external": 0}

        # Pessimistic OHLC simulator ledger.  It is intentionally separate
        # from Backtrader's asynchronous order state.
        self.initial_equity: Optional[float] = None
        self.realized_equity: Optional[float] = None
        self.marked_equity: Optional[float] = None
        self.cycle_entry_fees = 0.0
        self.cycle_start_bar: Optional[int] = None
        self.cycle_start_time = None
        self.max_fills_in_one_bar = 0
        self.trade_records = []
        self.bar_index = 0
        self.bankrupt = False

    def on_fill(self, price: float, size: float, is_buy: bool):
        """Update the weighted average after a base or safety order fills."""
        qty = abs(float(size))
        if qty <= 0.0 or price <= 0.0:
            return

        direction = PositionDir.POSITIVE if is_buy else PositionDir.NEGATIVE
        if self.cycle_dir == PositionDir.FLAT:
            self.cycle_dir = direction
        elif direction != self.cycle_dir:
            # Closing fills do not belong in the entry-price calculation.
            return

        if self.cycle_layers == 0:
            self.cycles_started += 1
            if self.cycle_base_notional <= 0.0:
                self.cycle_base_notional = qty * price
        else:
            self.safety_orders_filled += 1

        self.cycle_qty += qty
        self.cycle_cost += qty * price
        self.cycle_avg_price = self.cycle_cost / self.cycle_qty
        self.last_entry_price = price
        self.cycle_layers += 1
        self.max_layers_seen = max(self.max_layers_seen, self.cycle_layers)
        self.pending_entry = False

    def on_order_failed(self, *, is_entry: bool):
        """Let a venue release pending state after a rejected/cancelled order."""
        if is_entry:
            self.pending_entry = False
        else:
            self.pending_close_reason = None

    def _entry_direction(self, state: Observation) -> PositionDir:
        return PositionDir.POSITIVE

    def _capped_qty(
        self,
        desired_notional: float,
        state: Observation,
    ) -> float:
        """Cap the next order so used margin stays below the configured limit."""
        price = float(state.market.price)
        equity = max(0.0, float(state.account.equity))
        if price <= 0.0 or equity <= 0.0:
            return 0.0

        used_margin = self.cycle_qty * price / self.leverage
        margin_budget = equity * self.margin_usage_cap_pct
        free_margin = max(0.0, margin_budget - used_margin)
        max_notional = free_margin * self.leverage
        return min(desired_notional, max_notional) / price

    def _adverse_move_from_last_entry(self, price: float) -> float:
        if self.last_entry_price <= 0.0:
            return 0.0
        if self.cycle_dir == PositionDir.POSITIVE:
            return (self.last_entry_price - price) / self.last_entry_price
        return (price - self.last_entry_price) / self.last_entry_price

    def _return_from_average(self, price: float) -> float:
        if self.cycle_avg_price <= 0.0:
            return 0.0
        if self.cycle_dir == PositionDir.POSITIVE:
            return (price - self.cycle_avg_price) / self.cycle_avg_price
        return (self.cycle_avg_price - price) / self.cycle_avg_price

    def _finish_observed_cycle(self):
        """Settle local state after the venue reports that it is flat."""
        if self.cycle_layers > 0:
            self.cycles_closed += 1
            reason = self.pending_close_reason or "external"
            self.exit_counts[reason] = self.exit_counts.get(reason, 0) + 1

        self.cycle_dir = PositionDir.FLAT
        self.cycle_layers = 0
        self.cycle_qty = 0.0
        self.cycle_cost = 0.0
        self.cycle_avg_price = 0.0
        self.last_entry_price = 0.0
        self.cycle_base_notional = 0.0
        self.pending_entry = False
        self.pending_close_reason = None

    def _has_ohlc(self, state: Observation) -> bool:
        values = (
            state.market.open,
            state.market.high,
            state.market.low,
            state.market.close,
        )
        return all(value is not None and float(value) == float(value) for value in values)

    def _initialize_pessimistic_equity(self, state: Observation):
        if self.initial_equity is None:
            equity = float(state.account.equity)
            self.initial_equity = equity
            self.realized_equity = equity
            self.marked_equity = equity

    def _marked_equity_at(self, price: float) -> float:
        if self.realized_equity is None:
            return 0.0
        if self.cycle_qty <= 0.0:
            return self.realized_equity
        gross_pnl = self.cycle_qty * price - self.cycle_cost
        return self.realized_equity + gross_pnl - self.cycle_entry_fees

    def _liquidation_price(self) -> Optional[float]:
        """Long liquidation price from equity == maintenance margin."""
        if self.cycle_qty <= 0.0 or self.realized_equity is None:
            return None
        denominator = self.cycle_qty * (1.0 - self.maintenance_margin_rate)
        if denominator <= 0.0:
            return None
        price = (
            self.cycle_cost
            + self.cycle_entry_fees
            - self.realized_equity
        ) / denominator
        return price if price > 0.0 else None

    def _can_fill_at(self, notional: float, price: float) -> bool:
        if notional <= 0.0 or price <= 0.0 or self.realized_equity is None:
            return False
        added_qty = notional / price
        qty = self.cycle_qty + added_qty
        cost = self.cycle_cost + notional
        fees = self.cycle_entry_fees + notional * self.commission_rate
        equity = self.realized_equity + qty * price - cost - fees
        required_margin = qty * price / self.leverage
        return (
            equity > 0.0
            and required_margin
            <= equity * self.margin_usage_cap_pct + 1e-12
        )

    def _synthetic_fill(self, price: float, notional: float):
        qty = notional / price
        if self.cycle_layers == 0:
            self.cycle_start_bar = self.bar_index
            self.cycle_base_notional = notional
            self.cycle_start_time = self._current_bar_time
        self.on_fill(price=price, size=qty, is_buy=True)
        self.cycle_entry_fees += notional * self.commission_rate

    def _take_profit_price(self) -> Optional[float]:
        """Price that realizes the configured net profit after both-side fees."""
        if self.cycle_qty <= 0.0:
            return None
        target_profit = self.cycle_cost * self.config.take_profit_pct
        denominator = self.cycle_qty * (1.0 - self.commission_rate)
        if denominator <= 0.0:
            return None
        return (
            self.cycle_cost + self.cycle_entry_fees + target_profit
        ) / denominator

    def _close_synthetic(self, price: float, reason: str) -> dict:
        exit_notional = self.cycle_qty * price
        gross_pnl = exit_notional - self.cycle_cost
        exit_fee = exit_notional * self.commission_rate
        total_fees = self.cycle_entry_fees + exit_fee
        net_pnl = gross_pnl - total_fees
        entry_notional = self.cycle_cost
        same_bar = self.cycle_start_bar == self.bar_index
        record = {
            "index": len(self.trade_records) + 1,
            "opened_at": self.cycle_start_time,
            "closed_at": self._current_bar_time,
            "entry_bar": self.cycle_start_bar,
            "exit_bar": self.bar_index,
            "bars_held": self.bar_index - self.cycle_start_bar + 1,
            "layers": self.cycle_layers,
            "entry_notional": entry_notional,
            "avg_entry_price": self.cycle_avg_price,
            "exit_price": price,
            "gross_pnl": gross_pnl,
            "fees": total_fees,
            "net_pnl": net_pnl,
            "profit_pct": net_pnl / entry_notional if entry_notional else 0.0,
            "same_bar": same_bar,
            "multiple_fills_same_bar": self.max_fills_in_one_bar > 1,
            "max_fills_in_one_bar": self.max_fills_in_one_bar,
            "reason": reason,
        }
        self.trade_records.append(record)
        self.cycles_closed += 1
        self.exit_counts[reason] = self.exit_counts.get(reason, 0) + 1
        self.realized_equity += net_pnl
        self.marked_equity = self.realized_equity
        if reason == "liquidation":
            self.bankrupt = True

        self.cycle_dir = PositionDir.FLAT
        self.cycle_layers = 0
        self.cycle_qty = 0.0
        self.cycle_cost = 0.0
        self.cycle_avg_price = 0.0
        self.last_entry_price = 0.0
        self.cycle_base_notional = 0.0
        self.cycle_entry_fees = 0.0
        self.cycle_start_bar = None
        self.cycle_start_time = None
        self.max_fills_in_one_bar = 0
        self.pending_entry = False
        self.pending_close_reason = None
        return record

    def _process_pessimistic_bar(self, state: Observation) -> TradeIntent:
        """Replay one completed long-only bar as Open -> Low -> High -> Close."""
        self.bar_index += 1
        self._current_bar_time = state.current_time
        self._initialize_pessimistic_equity(state)
        if self.bankrupt:
            return TradeIntent(ActionType.NOOP, reason="bankrupt")

        open_price = float(state.market.open)
        high_price = float(state.market.high)
        low_price = float(state.market.low)
        close_price = float(state.market.close)
        if min(open_price, high_price, low_price, close_price) <= 0.0:
            raise ValueError("OHLC prices must be positive")
        if high_price < max(open_price, low_price, close_price):
            raise ValueError("high must be >= open, low and close")
        if low_price > min(open_price, high_price, close_price):
            raise ValueError("low must be <= open, high and close")

        fills_this_bar = 0
        if self.cycle_layers == 0:
            if (
                state.market.bars_to_close is not None
                and state.market.bars_to_close <= 2
            ):
                self.marked_equity = self.realized_equity
                return TradeIntent(ActionType.NOOP, reason="near_close")
            base_notional = self.realized_equity * self.config.base_order_pct
            if not self._can_fill_at(base_notional, open_price):
                self.marked_equity = self.realized_equity
                return TradeIntent(ActionType.NOOP, reason="base_order_no_margin")
            self._synthetic_fill(open_price, base_notional)
            fills_this_bar = 1
        else:
            liquidation_price = self._liquidation_price()
            if liquidation_price is not None and open_price <= liquidation_price:
                self._close_synthetic(open_price, "liquidation")
                return TradeIntent(ActionType.CLOSE, reason="liquidation_at_open")

        # Pessimistic leg: price falls first and crosses every reachable layer.
        while self.cycle_layers < self.config.max_layers:
            next_entry_price = self.last_entry_price * (
                1.0 - self.config.price_deviation_pct
            )
            liquidation_price = self._liquidation_price()
            if (
                liquidation_price is not None
                and low_price <= liquidation_price
                and liquidation_price >= next_entry_price
            ):
                self.max_fills_in_one_bar = max(
                    self.max_fills_in_one_bar, fills_this_bar
                )
                self._close_synthetic(liquidation_price, "liquidation")
                return TradeIntent(ActionType.CLOSE, reason="liquidation_before_layer")
            if low_price > next_entry_price:
                break

            desired_notional = (
                self.cycle_base_notional
                * self.VOLUME_MULT ** self.cycle_layers
            )
            if not self._can_fill_at(desired_notional, next_entry_price):
                break
            self._synthetic_fill(next_entry_price, desired_notional)
            fills_this_bar += 1

        self.max_fills_in_one_bar = max(
            self.max_fills_in_one_bar, fills_this_bar
        )
        liquidation_price = self._liquidation_price()
        if liquidation_price is not None and low_price <= liquidation_price:
            self._close_synthetic(liquidation_price, "liquidation")
            return TradeIntent(ActionType.CLOSE, reason="liquidation_at_low")

        # Only after surviving Low may the rebound towards High take profit.
        take_profit_price = self._take_profit_price()
        if take_profit_price is not None and high_price >= take_profit_price:
            self._close_synthetic(take_profit_price, "take_profit")
            return TradeIntent(ActionType.CLOSE, reason="take_profit")

        self.marked_equity = self._marked_equity_at(close_price)
        return TradeIntent(ActionType.NOOP, reason="carry_to_next_bar")

    def _layer_statistics(self) -> dict:
        result = {}
        total_trades = len(self.trade_records)
        for trade in self.trade_records:
            key = str(trade["layers"])
            bucket = result.setdefault(
                key,
                {
                    "trades": 0,
                    "profitable_trades": 0,
                    "entry_notional": 0.0,
                    "net_profit_abs": 0.0,
                },
            )
            bucket["trades"] += 1
            bucket["profitable_trades"] += int(trade["net_pnl"] > 0.0)
            bucket["entry_notional"] += trade["entry_notional"]
            bucket["net_profit_abs"] += trade["net_pnl"]

        for bucket in result.values():
            bucket["trade_pct"] = (
                bucket["trades"] / total_trades if total_trades else 0.0
            )
            bucket["win_rate"] = (
                bucket["profitable_trades"] / bucket["trades"]
            )
            bucket["net_profit_pct"] = (
                bucket["net_profit_abs"] / bucket["entry_notional"]
                if bucket["entry_notional"]
                else 0.0
            )
        return result

    def _process(self, state: Observation) -> TradeIntent:
        if self.pessimistic and self._has_ohlc(state):
            return self._process_pessimistic_bar(state)

        price = float(state.market.price)

        # A close is considered complete only after the venue reports FLAT.
        if state.position.dir == PositionDir.FLAT and self.cycle_layers > 0:
            self._finish_observed_cycle()

        if state.position.dir == PositionDir.FLAT:
            if self.pending_entry:
                return TradeIntent(ActionType.NOOP, reason="awaiting_entry_fill")

            direction = self._entry_direction(state)
            if direction == PositionDir.FLAT:
                return TradeIntent(ActionType.NOOP, reason="no_entry_signal")
            if state.market.bars_to_close is not None and state.market.bars_to_close <= 2:
                return TradeIntent(ActionType.NOOP, reason="near_close")

            base_notional = state.account.equity * self.config.base_order_pct
            qty = self._capped_qty(base_notional, state)
            if qty <= 0.0:
                return TradeIntent(ActionType.NOOP, reason="no_margin")

            # Future levels use the order actually allowed by the margin cap.
            self.cycle_base_notional = qty * price
            self.pending_entry = True
            return self._emit(
                TradeIntent(
                    ActionType.OPEN,
                    target_dir=direction,
                    order_qty=qty,
                    layer=1,
                    reason="base_order",
                )
            )

        if self.pending_close_reason is not None:
            return TradeIntent(ActionType.NOOP, reason="awaiting_close_fill")
        if self.pending_entry or self.cycle_qty <= 0.0:
            return TradeIntent(ActionType.NOOP, reason="awaiting_entry_fill")
        if state.position.dir != self.cycle_dir:
            self.pending_close_reason = "external"
            return self._emit(TradeIntent(ActionType.CLOSE, reason="direction_mismatch"))

        cycle_return = self._return_from_average(price)
        if cycle_return + 1e-12 >= self.config.take_profit_pct:
            self.pending_close_reason = "take_profit"
            return self._emit(TradeIntent(ActionType.CLOSE, reason="take_profit"))

        if (
            self.cycle_layers < self.config.max_layers
            and self._adverse_move_from_last_entry(price)
            >= self.config.price_deviation_pct
        ):
            desired_notional = (
                self.cycle_base_notional
                * self.VOLUME_MULT ** self.cycle_layers
            )
            qty = self._capped_qty(desired_notional, state)
            if qty > 0.0:
                self.pending_entry = True
                return self._emit(
                    TradeIntent(
                        ActionType.PYRAMID,
                        target_dir=self.cycle_dir,
                        order_qty=qty,
                        layer=self.cycle_layers + 1,
                        reason=f"safety_order_{self.cycle_layers + 1}",
                    )
                )
            return TradeIntent(ActionType.NOOP, reason="no_margin")

        return TradeIntent(ActionType.NOOP, reason="hold")

    def _emit(self, intent: TradeIntent) -> TradeIntent:
        if intent.action == ActionType.CLOSE:
            self.venue.close_position()
        elif intent.action in (ActionType.OPEN, ActionType.PYRAMID):
            self.venue.submit_order(
                intent.order_qty,
                is_buy=intent.target_dir == PositionDir.POSITIVE,
            )
        return intent

    def report(self) -> tuple[dict, dict]:
        total_trades = len(self.trade_records)
        profitable_trades = sum(
            trade["net_pnl"] > 0.0 for trade in self.trade_records
        )
        total_net_profit = sum(
            trade["net_pnl"] for trade in self.trade_records
        )
        same_bar_trades = sum(
            trade["same_bar"] for trade in self.trade_records
        )
        multiple_fills_same_bar = sum(
            trade["multiple_fills_same_bar"] for trade in self.trade_records
        )
        summary = {
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "losing_trades": total_trades - profitable_trades,
            "win_rate": profitable_trades / total_trades if total_trades else 0.0,
            "total_net_profit_abs": total_net_profit,
            "total_net_profit_pct": (
                total_net_profit / self.initial_equity
                if self.initial_equity
                else 0.0
            ),
            "same_bar_trades": same_bar_trades,
            "same_bar_trade_pct": (
                same_bar_trades / total_trades if total_trades else 0.0
            ),
            "multiple_fills_same_bar_trades": multiple_fills_same_bar,
            "liquidations": self.exit_counts.get("liquidation", 0),
            "bankrupt": self.bankrupt,
            "cycles_started": self.cycles_started,
            "cycles_closed": self.cycles_closed,
            "safety_orders_filled": self.safety_orders_filled,
            "max_layers_seen": self.max_layers_seen,
            "current_layers": self.cycle_layers,
            "current_avg_price": self.cycle_avg_price,
            "marked_equity": self.marked_equity,
            "exit_counts": dict(self.exit_counts),
        }
        summary.update(
            {
                "layer_stats": self._layer_statistics(),
                "trades": list(self.trade_records),
            }
        )
        return summary, {}
