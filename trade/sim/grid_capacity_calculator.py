from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd


LONG = 1
SHORT = -1


@dataclass(frozen=True)
class GridParams:
    # --- strategy ---
    price_deviation_pct: float = 0.01   # first-layer adverse/favourable distance
    take_profit_pct: float = 0.0008       # fixed net cash target / cycle-start balance
    deviation_step_mult: float = 1.5
    direction: Literal["long", "short"] = "long"

    # --- account / fees ---
    initial_equity: float = 10_000.0
    leverage: float = 50.0
    taker_fee_pct: float = 0.05         # base entry and final stop
    maker_fee_pct: float = 0.02         # safety entries and take-profit exit
    maintenance_margin_pct: float = 0.5
    margin_usage_cap_pct: float = 0.80

    # Normalized starting price. Change it only when absolute prices are useful.
    initial_price: float = 1.0

    # Safety bound for the search, not a strategy layer setting.
    search_limit: int = 100

    def __post_init__(self) -> None:
        if not 0 < self.price_deviation_pct < 1:
            raise ValueError("price_deviation_pct must be in (0, 1)")
        if not 0 < self.take_profit_pct < 1:
            raise ValueError("take_profit_pct must be in (0, 1)")
        if self.deviation_step_mult <= 0:
            raise ValueError("deviation_step_mult must be positive")
        if self.initial_equity <= 0 or self.initial_price <= 0:
            raise ValueError("initial_equity and initial_price must be positive")
        if self.leverage < 1:
            raise ValueError("leverage must be >= 1")
        if not 0 < self.margin_usage_cap_pct <= 1:
            raise ValueError("margin_usage_cap_pct must be in (0, 1]")
        if not 0 <= self.maintenance_margin_pct < 100:
            raise ValueError("maintenance_margin_pct must be in [0, 100)")
        if min(self.taker_fee_pct, self.maker_fee_pct) < 0:
            raise ValueError("fee rates must be non-negative")
        if self.search_limit < 1:
            raise ValueError("search_limit must be >= 1")


def side_of(direction: str) -> int:
    return LONG if direction == "long" else SHORT


def target_notional(
    *,
    params: GridParams,
    side: int,
    fill_price: float,
    layer_index: int,
    target_cash: float,
    qty: float,
    cost: float,
    entry_fees: float,
) -> float:
    """
    Additional entry notional required so that closing the whole position at the
    current layer's favourable boundary earns target_cash net of entry/exit fees.
    """
    deviation = params.price_deviation_pct * (
        params.deviation_step_mult ** layer_index
    )
    if deviation >= 1:
        return 0.0

    exit_ratio = 1.0 + side * deviation
    exit_price = fill_price * exit_ratio

    maker = params.maker_fee_pct / 100.0
    entry_rate = (
        params.taker_fee_pct / 100.0 if layer_index == 0 else maker
    )

    incremental_return = (
        side * (exit_ratio - 1.0)
        - exit_ratio * maker
        - entry_rate
    )
    if incremental_return <= 0:
        return 0.0

    existing_net_at_exit = (
        side * (qty * exit_price - cost)
        - entry_fees
        - qty * exit_price * maker
    )

    required = (target_cash - existing_net_at_exit) / incremental_return
    return max(0.0, required)


def liquidation_price(
    *,
    side: int,
    balance_after_fees: float,
    qty: float,
    cost: float,
    maintenance_rate: float,
) -> Optional[float]:
    if qty <= 0:
        return None

    if side == LONG:
        denominator = qty * (1.0 - maintenance_rate)
        if denominator <= 0:
            return None
        price = (cost - balance_after_fees) / denominator
    else:
        price = (
            cost + balance_after_fees
        ) / (qty * (1.0 + maintenance_rate))

    return price if price > 0 else None


def analyze_grid(params: GridParams) -> tuple[pd.DataFrame, str]:
    side = side_of(params.direction)
    maker = params.maker_fee_pct / 100.0
    taker = params.taker_fee_pct / 100.0
    maintenance = params.maintenance_margin_pct / 100.0

    target_cash = params.initial_equity * params.take_profit_pct

    balance = params.initial_equity
    qty = 0.0
    cost = 0.0
    entry_fees = 0.0
    fill_price = params.initial_price

    rows: list[dict] = []
    stop_reason = "search limit reached"

    for layer_index in range(params.search_limit):
        layer = layer_index + 1
        deviation = params.price_deviation_pct * (
            params.deviation_step_mult ** layer_index
        )
        if deviation >= 1:
            stop_reason = f"layer {layer}: deviation reached 100%"
            break

        if layer_index > 0:
            previous_deviation = params.price_deviation_pct * (
                params.deviation_step_mult ** (layer_index - 1)
            )
            fill_price *= 1.0 - side * previous_deviation
            if fill_price <= 0:
                stop_reason = f"layer {layer}: non-positive fill price"
                break

        notional = target_notional(
            params=params,
            side=side,
            fill_price=fill_price,
            layer_index=layer_index,
            target_cash=target_cash,
            qty=qty,
            cost=cost,
            entry_fees=entry_fees,
        )
        if notional <= 0:
            stop_reason = f"layer {layer}: target-derived notional is non-positive"
            break

        entry_rate = taker if layer_index == 0 else maker
        fee = notional * entry_rate

        equity_at_fill_before_fee = balance + side * (qty * fill_price - cost)
        equity_after_fill = equity_at_fill_before_fee - fee
        new_cost = cost + notional
        new_qty = qty + notional / fill_price
        new_entry_fees = entry_fees + fee

        initial_margin = new_cost / params.leverage
        margin_usage = (
            initial_margin / equity_after_fill
            if equity_after_fill > 0 else float("inf")
        )

        if (
            equity_after_fill <= 0
            or margin_usage > params.margin_usage_cap_pct + 1e-12
        ):
            stop_reason = (
                f"layer {layer}: margin cap exceeded "
                f"({margin_usage:.2%} > {params.margin_usage_cap_pct:.2%})"
            )
            break

        tp_price = fill_price * (1.0 + side * deviation)
        next_or_stop_price = fill_price * (1.0 - side * deviation)

        # Equity after the current full ladder is pierced at the next adverse boundary
        # and the whole position is closed there with a taker fee.
        stop_exit_notional = new_qty * next_or_stop_price
        gross_stop_pnl = side * (stop_exit_notional - new_cost)
        stop_exit_fee = stop_exit_notional * taker
        equity_after_stop = balance - fee + gross_stop_pnl - stop_exit_fee
        drawdown_after_stop = (
            1.0 - equity_after_stop / params.initial_equity
        )

        liq_price = liquidation_price(
            side=side,
            balance_after_fees=balance - fee,
            qty=new_qty,
            cost=new_cost,
            maintenance_rate=maintenance,
        )

        liquidation_before_stop = False
        if liq_price is not None:
            if side == LONG:
                liquidation_before_stop = liq_price >= next_or_stop_price
            else:
                liquidation_before_stop = liq_price <= next_or_stop_price

        cumulative_move = side * (
            fill_price / params.initial_price - 1.0
        )
        stop_move = side * (
            next_or_stop_price / params.initial_price - 1.0
        )

        rows.append(
            {
                "layer": layer,
                "layer_deviation_pct": deviation * 100,
                "entry_price": fill_price,
                "cumulative_adverse_to_entry_pct": -cumulative_move * 100,
                "tp_price": tp_price,
                "next_or_stop_price": next_or_stop_price,
                "grid_total_adverse_range_pct": -stop_move * 100,
                "added_notional": notional,
                "cumulative_notional": new_cost,
                "equity_at_entry": equity_after_fill,
                "initial_margin": initial_margin,
                "margin_usage_pct": margin_usage * 100,
                "liquidation_price": liq_price,
                "liquidation_before_stop": liquidation_before_stop,
                "equity_after_grid_stop": equity_after_stop,
                "drawdown_after_grid_stop_pct": drawdown_after_stop * 100,
            }
        )

        # A layer is counted as usable only when its planned stop is reached before
        # liquidation. Otherwise the previous row is the maximum complete grid.
        if liquidation_before_stop:
            stop_reason = f"layer {layer}: liquidation occurs before the planned grid stop"
            rows.pop()
            break

        balance -= fee
        qty = new_qty
        cost = new_cost
        entry_fees = new_entry_fees

    frame = pd.DataFrame(rows)
    return frame, stop_reason


def main() -> None:
    params = GridParams()

    table, stop_reason = analyze_grid(params)

    if table.empty:
        print("No executable grid layer.")
        print("Reason:", stop_reason)
        return

    display_columns = [
        "layer",
        "layer_deviation_pct",
        "entry_price",
        "grid_total_adverse_range_pct",
        "added_notional",
        "cumulative_notional",
        "margin_usage_pct",
        "equity_after_grid_stop",
        "drawdown_after_grid_stop_pct",
        "liquidation_price",
    ]

    print(f"Maximum complete grid layers: {len(table)}")
    print(f"Stop reason: {stop_reason}")
    print()
    print(
        table[display_columns].to_string(
            index=False,
            formatters={
                "layer_deviation_pct": "{:.4f}".format,
                "entry_price": "{:.8f}".format,
                "grid_total_adverse_range_pct": "{:.4f}".format,
                "added_notional": "{:.2f}".format,
                "cumulative_notional": "{:.2f}".format,
                "margin_usage_pct": "{:.2f}".format,
                "equity_after_grid_stop": "{:.2f}".format,
                "drawdown_after_grid_stop_pct": "{:.2f}".format,
                "liquidation_price": lambda x: "" if pd.isna(x) else f"{x:.8f}",
            },
        )
    )

    table.to_csv("grid_capacity_table.csv", index=False)
    print("\nSaved: grid_capacity_table.csv")


if __name__ == "__main__":
    main()
