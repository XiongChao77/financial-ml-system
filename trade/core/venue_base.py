"""
Venue layer base class: the only boundary between a strategy and "one concrete trading venue".

A venue works in two directions:
  inbound  observe : assemble the feed's market signals plus the account/position only it knows into an Observation
  outbound execute : take a TradeIntent and turn it into real orders (submit_order / close_position)

Backtest (BtVenue) and live trading (BybitVenue / MT5Venue) are its two families of implementation;
switching backtest framework or exchange should only touch this layer.
"""

from abc import abstractmethod


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

    @abstractmethod
    def submit_order(self, size, is_buy, stop_loss_pct=None, take_profit_pct=None):
        pass

    # ---------------- outbound: cash and life cycle (optional) ----------------
    def withdraw_cash(self, amount):
        """Move cash out of the account (used by the restartable martingale restart cost). Unsupported by default."""
        raise NotImplementedError

    def request_halt(self):
        """The strategy asks to stop this run (e.g. both accounts wiped out). Ignored by default."""
        pass
