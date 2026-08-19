from dataclasses import dataclass
from trade.core.protocol import Observation, TradeIntent, ActionType, PositionDir
from trade.core.strategy_base import StrategyBase

@dataclass
class MaObservation(Observation):
    """Moving average specific observation (the generic Observation plus two moving averages)"""
    fast_ma: float = 0.0
    slow_ma: float = 0.0

@dataclass(frozen=True)
class MaStrategyConfig:
    fast_period: int = 50
    slow_period: int = 200
    risk_per_trade_pct: float = 0.95


class MaCrossoverStrategy(StrategyBase):
    def __init__(self, config: MaStrategyConfig):
        super().__init__(venue=None)
        self.config = config

    def _process(self, state: MaObservation) -> TradeIntent:
        # 1. Detect the crossover direction
        # Golden cross: fast > slow
        if state.fast_ma > state.slow_ma:
            target_dir = PositionDir.POSITIVE 
        # Death cross: fast < slow
        else:
            target_dir = PositionDir.NEGATIVE

        # 2. State machine
        action = TradeIntent(ActionType.NOOP)

        # Currently flat -> open
        if state.position.dir == PositionDir.FLAT:
            action = TradeIntent(
                action=ActionType.OPEN,
                target_dir=target_dir,
                target_layers=1,
                target_pct=self.config.risk_per_trade_pct * target_dir
            )
        # Currently in position with the opposite direction -> reverse
        elif state.position.dir != target_dir:
            action = TradeIntent(
                action=ActionType.REVERSE,
                target_dir=target_dir,
                target_layers=1,
                target_pct=self.config.risk_per_trade_pct * target_dir
            )

        return action
