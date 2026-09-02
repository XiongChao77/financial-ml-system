"""Venue-neutral execution measurements for live orders."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional


@dataclass(frozen=True)
class ExecutionFill:
    """One fill, or one venue-provided aggregate fill when details are absent."""

    price: float
    quantity: float
    order_id: str = ""
    deal_id: str = ""
    executed_at_utc: Optional[datetime] = None
    is_aggregate: bool = False


@dataclass(frozen=True)
class ExecutionReport:
    """Normalized result of one strategy entry intent."""

    side: str
    requested_quantity: float
    decision_price: float
    decision_at_utc: datetime
    bid: float
    ask: float
    spread_pct: float
    status: str
    reason: str = ""
    fills: tuple[ExecutionFill, ...] = ()
    completed_at_utc: datetime = field(default_factory=lambda: datetime.now(UTC))
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def filled_quantity(self) -> float:
        return sum(fill.quantity for fill in self.fills)

    @property
    def fill_vwap(self) -> Optional[float]:
        quantity = self.filled_quantity
        if quantity <= 0:
            return None
        return sum(fill.price * fill.quantity for fill in self.fills) / quantity

    @property
    def best_fill_price(self) -> Optional[float]:
        prices = [fill.price for fill in self.fills if not fill.is_aggregate]
        if not prices:
            return None
        return min(prices) if self.side == "buy" else max(prices)

    @property
    def arrival_price(self) -> Optional[float]:
        """Best detailed fill, used as the market price when the order arrived."""

        return self.best_fill_price

    @property
    def worst_fill_price(self) -> Optional[float]:
        prices = [fill.price for fill in self.fills if not fill.is_aggregate]
        if not prices:
            return None
        return max(prices) if self.side == "buy" else min(prices)

    @property
    def latency_slippage_bps(self) -> Optional[float]:
        arrival = self.arrival_price
        if arrival is None or not math.isfinite(self.decision_price) or self.decision_price <= 0:
            return None
        direction = 1.0 if self.side == "buy" else -1.0
        return direction * (arrival - self.decision_price) / self.decision_price * 10_000.0

    @property
    def execution_slippage_vwap_bps(self) -> Optional[float]:
        best = self.best_fill_price
        vwap = self.fill_vwap
        if best is None or vwap is None or best <= 0:
            return None
        direction = 1.0 if self.side == "buy" else -1.0
        return direction * (vwap - best) / best * 10_000.0

    @property
    def execution_slippage_worst_bps(self) -> Optional[float]:
        best = self.best_fill_price
        worst = self.worst_fill_price
        if best is None or worst is None or best <= 0:
            return None
        direction = 1.0 if self.side == "buy" else -1.0
        return direction * (worst - best) / best * 10_000.0
