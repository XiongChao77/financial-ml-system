"""
Venue layer base class: the only boundary between a strategy and "one concrete trading venue".

A venue works in two directions:
  inbound  observe : assemble the feed's market signals plus the account/position only it knows into an Observation
  outbound execute : take a TradeIntent and turn it into real orders (submit_order / close_position)

Backtest (BtVenue) and live trading (BybitVenue / MT5Venue) are its two families of implementation;
switching backtest framework or exchange should only touch this layer.
"""

from abc import abstractmethod
import math

from trade.core.protocol import OrderType


class VenueBase:
    # ---------------- inbound: queries ----------------
    @abstractmethod
    def get_account_equity(self):
        pass

    @abstractmethod
    def get_current_state(self):
        pass

    @abstractmethod
    def get_server_time(self):
        pass

    @abstractmethod
    def get_last_position_open_time(self):  # return UTC time
        pass

    # ---------------- outbound: orders ----------------
    @abstractmethod
    def close_position(self, size=None, **kwargs):
        pass

    @staticmethod
    def normalize_order_request(order_type=OrderType.MARKET, price=None):
        """Validate an entry order type and its optional limit price."""
        try:
            normalized_type = (
                order_type
                if isinstance(order_type, OrderType)
                else OrderType(str(order_type).strip().casefold())
            )
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

    # ---------------- outbound: cash and life cycle (optional) ----------------
    def withdraw_cash(self, amount):
        """Move cash out of the account (used by the restartable martingale restart cost). Unsupported by default."""
        raise NotImplementedError

    def request_halt(self):
        """The strategy asks to stop this run (e.g. both accounts wiped out). Ignored by default."""
        pass
