from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


class MarginMode(Enum):
    CROSS = "cross"
    ISOLATED = "isolated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AccountBalance:
    balance: float
    equity: float


@dataclass(frozen=True)
class AccountPosition:
    symbol: str
    side: PositionSide
    quantity: float

    entry_price: float
    mark_price: float
    notional: Optional[float]

    unrealized_pnl: float
    unrealized_pnl_pct: Optional[float]

    leverage: Optional[float]
    liquidation_price: Optional[float]

    margin_mode: MarginMode = MarginMode.UNKNOWN


class AccountDashboard(ABC):
    """
    Read-only account information for logging, monitoring, and frontend display.

    Dashboard methods must not participate in strategy decisions or order
    execution. Failures in this interface should not affect the trading path.
    """

    @abstractmethod
    def get_dashboard_balance(self) -> AccountBalance:
        """Return the current account balance and equity."""
        raise NotImplementedError

    @abstractmethod
    def get_dashboard_position(self) -> Optional[AccountPosition]:
        """Return the single logical strategy position, if one exists."""
        raise NotImplementedError
