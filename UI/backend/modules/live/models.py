"""Validated wire models for ephemeral live runner snapshots."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class AccountSnapshot(StrictModel):
    balance: float
    equity: float


class PositionSnapshot(StrictModel):
    symbol: str
    side: Literal["long", "short"]
    quantity: float
    entry_price: float
    mark_price: float
    notional: float | None = None
    unrealized_pnl: float
    unrealized_pnl_pct: float | None = None
    leverage: float | None = None
    liquidation_price: float | None = None
    margin_mode: Literal["cross", "isolated", "unknown"] = "unknown"


class SignalSnapshot(StrictModel):
    model_output: str
    raw_model_output: float | None = None
    probability: float | None = None
    net_score: float | None = None
    decision: str
    target_side: str
    quantity: float | None = None
    reason: str = ""
    updated_at: AwareDatetime


class AvailabilitySnapshot(StrictModel):
    account: bool
    position: bool
    latest_signal: bool


class SnapshotError(StrictModel):
    component: str
    message: str


class StrategySnapshot(StrictModel):
    strategy_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    interval: str = Field(min_length=1)
    status: Literal["running", "stopped"]
    account: AccountSnapshot | None = None
    position: PositionSnapshot | None = None
    latest_signal: SignalSnapshot | None = None
    availability: AvailabilitySnapshot
    errors: list[SnapshotError] = Field(default_factory=list)


class RunnerSnapshot(StrictModel):
    runner_id: str = Field(min_length=1)
    runner_instance_id: str = Field(min_length=1)
    runner_started_at: AwareDatetime
    sequence: int = Field(ge=1)
    sent_at: AwareDatetime
    strategies: list[StrategySnapshot]
