from dataclasses import dataclass
from trade.strategy.strategy_ftmo import BrainBase, MarketState, TradingAction, ActionType, PositionDir

@dataclass
class MaMarketState(MarketState):
    """均线策略特定的市场状态"""
    fast_ma: float
    slow_ma: float

class MaCrossoverBrain(BrainBase):
    def __init__(self, trade_risk: float = 0.95):
        self.trade_risk = trade_risk

    def decide(self, state: MaMarketState) -> TradingAction:
        # 1. 判断交叉方向
        # 金叉：快线 > 慢线
        if state.fast_ma > state.slow_ma:
            target_dir = PositionDir.LONG
        # 死叉：快线 < 慢线
        else:
            target_dir = PositionDir.SHORT

        # 2. 状态机逻辑
        action = TradingAction(ActionType.HOLD)

        # 当前无持仓 -> 开仓
        if state.position_dir == PositionDir.FLAT:
            action = TradingAction(
                action=ActionType.OPEN,
                target_dir=target_dir,
                target_layers=1,
                target_pct=self.trade_risk * target_dir
            )
        # 当前有持仓且方向相反 -> 反手 (Reverse)
        elif state.position_dir != target_dir:
            action = TradingAction(
                action=ActionType.REVERSE,
                target_dir=target_dir,
                target_layers=1,
                target_pct=self.trade_risk * target_dir
            )

        return action