"""
Venue layer base class: the only boundary between a strategy and "one concrete trading venue".

A venue works in two directions:
  inbound  observe : assemble the feed's market signals plus the account/position only it knows into an Observation
  outbound execute : take a TradeIntent and turn it into real orders (submit_order / close_position)

Backtest (BtVenue) and live trading (BybitVenue / MT5Venue) are its two families of implementation;
switching backtest framework or exchange should only touch this layer.
"""

from abc import abstractmethod
from datetime import UTC, date, datetime, timedelta, timezone
import math
from zoneinfo import ZoneInfo

from trade.core.protocol import Firm, OrderType, PositionView
from trade.core.protocol import TradeIntent, ActionType, PositionDir


class VenueBase:
    FIRM_DAILY_RESET_TIMEZONES = {
        Firm.FTMO: ZoneInfo("Europe/Prague"),
        # The5ers defines its reset as a fixed UTC+3 boundary without DST.
        Firm.THE5ERS: timezone(timedelta(hours=3)),
    }

    # ---------------- inbound: queries ----------------
    @abstractmethod
    def get_account_equity(self):
        pass

    @abstractmethod
    def get_current_state(self) -> PositionView:
        pass

    @staticmethod
    def _normalized_candle_time_utc(candle_open_time_utc: datetime) -> datetime:
        if candle_open_time_utc is None or candle_open_time_utc.tzinfo is None:
            raise ValueError("candle_open_time_utc must be timezone-aware")
        return candle_open_time_utc.astimezone(UTC)

    def get_daily_reset_date(self, candle_open_time_utc: datetime) -> date:
        """Return the UTC candle date used by ordinary venues and backtests."""

        return self._normalized_candle_time_utc(candle_open_time_utc).date()

    def get_firm_daily_reset_date(
        self,
        candle_open_time_utc: datetime,
        firm: Firm,
    ) -> date:
        reset_timezone = self.FIRM_DAILY_RESET_TIMEZONES[firm]
        return self._normalized_candle_time_utc(
            candle_open_time_utc
        ).astimezone(reset_timezone).date()

    @abstractmethod
    def get_last_position_open_time(self):  # return utc
        pass

    # ---------------- outbound: orders ----------------
    @abstractmethod
    def close_position(self, size=None, **kwargs):
        pass

    @staticmethod
    def normalize_order_request(order_type=OrderType.MARKET, price=None):
        """Validate an entry order type and its optional limit price."""
        try:
            normalized_type = order_type if isinstance(order_type, OrderType) else OrderType(str(order_type).strip().casefold())
        except ValueError as exc:
            raise ValueError("order_type must be 'market' or 'limit'") from exc

        if normalized_type == OrderType.MARKET:
            if price is not None:
                raise ValueError("price must be omitted for market orders")
            return normalized_type, None

        try:
            normalized_price = float(price)
        except (TypeError, ValueError) as exc:
            raise ValueError("A positive finite price is required for limit orders") from exc
        if not math.isfinite(normalized_price) or normalized_price <= 0:
            raise ValueError("A positive finite price is required for limit orders")
        return normalized_type, normalized_price

    # size: Order quantity denominated in units of the base asset, not its notional value.
    @abstractmethod
    def submit_order(
        self,
        size,
        is_buy,
        stop_loss_pct=None,
        take_profit_pct=None,
        *,
        order_type=OrderType.MARKET,
        price=None,
    ):
        pass

    def execute_action(self, action: TradeIntent):
        """Reworked to use the submit_order interface and pass the stop loss parameters"""
        if action.action == ActionType.NOOP:
            return
        if action.action == ActionType.CLOSE:
            self.close_position()
        elif action.action == ActionType.OPEN:
            is_buy = action.target_dir == PositionDir.POSITIVE
            self.submit_order(
                action.order_qty,
                is_buy=is_buy,
                stop_loss_pct=action.stop_loss_pct,
                take_profit_pct=action.take_profit_pct,
            )

    # ---------------- outbound: cash and life cycle (optional) ----------------
    def withdraw_cash(self, amount):
        """Move cash out of the account (used by the restartable martingale restart cost). Unsupported by default."""
        raise NotImplementedError

    def request_halt(self):
        """The strategy asks to stop this run (e.g. both accounts wiped out). Ignored by default."""
        pass
