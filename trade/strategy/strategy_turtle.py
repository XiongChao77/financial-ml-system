from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional
from data_process.common import Signal
from trade.strategy.strategy_ftmo import Brain, TradingAction, ActionType, PositionDir

@dataclass
class TurtleMarketState:
    """
    Turtle Brain specific state
    """
    price: float
    high_band: float    # Donchian High (Entry)
    low_band: float     # Donchian Low (Exit)
    atr: float          # N (Volatility)
    
    # Portfolio State
    position_dir: PositionDir
    layers: int
    last_entry_price: float # Price of the last added unit

class TurtleBrain(Brain):
    def __init__(
        self, 
        max_layers: int = 4,   # Turtle usually caps at 4 units
        risk_per_unit: float = 0.01 # 1% risk per unit
    ):
        self.max_layers = max_layers
        self.risk_per_unit = risk_per_unit

    def decide(self, state: TurtleMarketState) -> TradingAction:
        action = TradingAction(ActionType.HOLD)
        
        # === 1. Calculate Unit Size (Target %) ===
        # Turtle Formula: Unit = (1% of Account) / (N * Dollars_per_point)
        # In % terms: Target% = (1% Risk * Price) / ATR
        # We clamp ATR to avoid division by zero
        safe_atr = state.atr if state.atr > 0 else state.price * 0.01
        
        # Calculate how much % of total equity ONE unit represents at current volatility
        unit_pct = (self.risk_per_unit * state.price) / safe_atr
        
        # Cap unit size for safety (e.g., never more than 50% of equity per unit)
        unit_pct = min(unit_pct, 0.50)

        # === 2. Entry Logic (Breakout) ===
        if state.position_dir == PositionDir.FLAT:
            # Long Breakout
            if state.price > state.high_band:
                return TradingAction(
                    action=ActionType.OPEN,
                    target_dir=PositionDir.LONG,
                    target_layers=1,
                    target_pct=unit_pct * 1 # 1 Unit
                )
            # Short Breakout
            elif state.price < state.low_band:
                return TradingAction(
                    action=ActionType.OPEN,
                    target_dir=PositionDir.SHORT,
                    target_layers=1,
                    target_pct=unit_pct * 1 # 1 Unit
                )

        # === 3. Pyramiding (Adding Units) ===
        elif state.layers < self.max_layers:
            # Pyramiding Threshold: 0.5 * N (ATR)
            threshold = 0.5 * state.atr
            
            if state.position_dir == PositionDir.LONG:
                if state.price > state.last_entry_price + threshold:
                    new_layers = state.layers + 1
                    return TradingAction(
                        action=ActionType.PYRAMID,
                        target_dir=PositionDir.LONG,
                        target_layers=new_layers,
                        target_pct=unit_pct * new_layers # Target Total Units
                    )
            
            elif state.position_dir == PositionDir.SHORT:
                if state.price < state.last_entry_price - threshold:
                    new_layers = state.layers + 1
                    return TradingAction(
                        action=ActionType.PYRAMID,
                        target_dir=PositionDir.SHORT,
                        target_layers=new_layers,
                        target_pct=unit_pct * new_layers
                    )

        # === 4. Exit Logic (Touch opposite band) ===
        # Note: Stop Loss is handled by the Executor (2N), this is the "Profit/Breakout Exit"
        if state.position_dir == PositionDir.LONG:
            if state.price < state.low_band: # System 1: 10-day low, System 2: 20-day low
                return TradingAction(ActionType.CLOSE)
        
        elif state.position_dir == PositionDir.SHORT:
            if state.price > state.high_band:
                return TradingAction(ActionType.CLOSE)

        return action