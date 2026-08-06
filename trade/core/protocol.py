"""
Shared protocol between the three layers (feed / strategy / venue depend on this file only).

  feed     produces market data and model signals -> packed into an Observation by the venue
  strategy consumes the Observation and produces a TradeIntent
  venue    executes the TradeIntent and feeds account/position back into the next Observation

This file depends on neither backtrader nor any exchange SDK.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional
from datetime import datetime
from math import inf

from data_process.common import Signal


# ============================================================
# Enums (system protocol layer)
# ============================================================
class PositionDir(IntEnum):
    """
    Current / target position direction
    """
    NEGATIVE = -1
    FLAT = 0
    POSITIVE  = 1


class ActionType(Enum):
    """
    Trade action emitted by the strategy layer
    """
    NOOP = "noop"   # do nothing, leave every position state untouched
    HOLD = "hold"         # do nothing
    OPEN = "open"         # open a position
    CLOSE = "close"       # close the position
    REVERSE = "reverse"   # reverse the position
    PYRAMID = "pyramid"   # add to the position


@dataclass
class TradeIntent:
    """
    Output of the strategy layer: what to do.

    The single intent type of the whole system (strategy_base / strategy_ml /
    strategy_martingale each used to define their own, with inconsistent fields).
    Every strategy fills in what it needs; the venue only reads the fields it knows:
      - direction / layer semantics only (turtle, ...): target_dir / target_layers / target_pct
      - explicit order size (ML / martingale)        : order_qty / stop_loss_pct / take_profit_pct
      - martingale extras                            : layer (which safety order) / reason (intent origin, for tracing)
    """
    action: ActionType
    target_dir: PositionDir = PositionDir.FLAT
    target_layers: int = 0
    target_pct: Optional[float] = None
    order_qty: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    layer: int = 0
    reason: str = ""


@dataclass
class MarketView:
    """From the feed: market data + model signal; account independent and reproducible offline."""
    price: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    signal: Optional[Signal] = None
    pred_prob: float = 0.0
    atr_pct: float = 0.0            # current ATR (as a fraction of price)
    slow_atr: Optional[float] = None
    vol_regime: Optional[float] = None
    # Optional feed value: infinity means a continuous market with no close boundary.
    bars_to_close: Optional[float] = inf


@dataclass
class PositionView:
    """From the venue: the current position."""
    dir: PositionDir = PositionDir.FLAT
    layers: int = 0


@dataclass
class AccountView:
    """From the venue: account equity."""
    equity: float = 0.0


@dataclass
class Observation:
    """
    Input of the strategy layer: everything the strategy can see on one bar.

    Kept in three blocks so the boundaries stay obvious:
      market   <- what the feed can compute on its own
      position <- the position, known only to the venue
      account  <- the equity, known only to the venue
    """
    market: MarketView
    position: PositionView = field(default_factory=PositionView)
    account: AccountView = field(default_factory=AccountView)
    current_time: Optional[datetime] = None
