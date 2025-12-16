from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from data_process.common import Signal
from trade.strategy.base_executor import BaseExecutor
# ============================================================
# 枚举定义（系统协议层）
# ============================================================
class PositionDir(IntEnum):
    """
    当前/目标持仓方向
    """
    SHORT = -1
    FLAT = 0
    LONG = 1


class ActionType(Enum):
    """
    Brain 输出的交易动作
    """
    HOLD = "hold"         # 什么都不做
    OPEN = "open"         # 开仓
    CLOSE = "close"       # 平仓
    REVERSE = "reverse"   # 反手
    PYRAMID = "pyramid"   # 加仓


# ============================================================
# Brain 输入 / 输出数据结构
# ============================================================

@dataclass
class MarketState:
    """
    Brain 的输入：当前市场 + 策略状态
    """
    price: float
    signal: Signal
    pred_prob: float

    position_dir: PositionDir
    layers: int

@dataclass
class TradingAction:
    """
    Brain 的输出：要做什么
    """
    action: ActionType
    target_dir: PositionDir = PositionDir.FLAT
    target_layers: int = 0
    target_pct: Optional[float] = None


# ============================================================
# Brain 抽象基类
# ============================================================

class Brain(ABC):

    @abstractmethod
    def decide(self, state: MarketState) -> TradingAction:
        """
        根据 MarketState 输出 TradingAction
        """
        pass


# ============================================================
# FtmoBrain：你的策略逻辑（已枚举化）
# ============================================================

class FtmoBrain(Brain):

    def __init__(
        self,
        executor: BaseExecutor,
        trade_risk: float,
        max_layers: int,
        holdbar: int,
        allow_long: bool = True,
        allow_short: bool = True,
        thresh: Optional[float] = None,
    ):
        self.executor = executor
        self.trade_risk = trade_risk
        self.max_layers = max_layers
        self.holdbar = holdbar
        self.allow_long = allow_long
        self.allow_short = allow_short
        self.thresh = thresh

    def decide(self, state: MarketState) -> TradingAction:

        # -------------------------------
        # 1. 置信度过滤
        # -------------------------------
        signal = state.signal
        if self.thresh is not None and state.pred_prob < self.thresh:
            signal = Signal.NEUTRAL
        # -------------------------------
        # 2. 信号 -> 目标方向
        # -------------------------------
        target_dir = PositionDir.FLAT

        if signal == Signal.LONG and self.allow_long:
            target_dir = PositionDir.LONG
        elif signal == Signal.SHORT and self.allow_short:
            target_dir = PositionDir.SHORT

        action = TradingAction(ActionType.HOLD)

        if state.position_dir == PositionDir.FLAT:  #当前无持仓
            if target_dir != PositionDir.FLAT:
                action = TradingAction(
                    action=ActionType.OPEN,
                    target_dir=target_dir,
                    target_layers=1,
                    target_pct=self.trade_risk * target_dir,
                )
        else:   #当前有持仓
            if target_dir == PositionDir.FLAT:  #震荡 -> 平仓
                action = TradingAction(ActionType.CLOSE)
            elif target_dir != state.position_dir:  #方向反转 -> 反手
                action = TradingAction(
                    action=ActionType.REVERSE,
                    target_dir=target_dir,
                    target_layers=1,
                    target_pct=self.trade_risk * target_dir,
                )
            elif state.layers < self.max_layers:  #同向 -> 加仓
                new_layers = state.layers + 1
                action = TradingAction(
                    action=ActionType.PYRAMID,
                    target_dir=state.position_dir,
                    target_layers=new_layers,
                    target_pct=self.trade_risk * new_layers * state.position_dir,
                )
        self.execute_action(action)


    def execute_action(self, action: TradingAction):

        if action.action == ActionType.HOLD:
            return

        if action.action == ActionType.CLOSE:
            self.executor.user_close()
            self.dir = 0
            self.layers = 0
            return

        if action.action in (ActionType.OPEN, ActionType.REVERSE, ActionType.PYRAMID):
            self.dir = action.target_dir
            self.layers = action.target_layers
            self.executor.user_order_target_percent(target_pct=action.target_pct)