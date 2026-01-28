from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional
from trade.strategy.base_executor import BaseExecutor

# ============================================================
# 枚举定义（系统协议层）
# ============================================================
class PositionDir(IntEnum):
    """
    当前/目标持仓方向
    """
    NEGATIVE = -1
    FLAT = 0
    POSITIVE  = 1


class ActionType(Enum):
    """
    BrainBase 输出的交易动作
    """
    HOLD = "hold"         # 什么都不做
    OPEN = "open"         # 开仓
    CLOSE = "close"       # 平仓
    REVERSE = "reverse"   # 反手
    PYRAMID = "pyramid"   # 加仓

@dataclass
class TradingAction:
    """
    BrainBase 的输出：要做什么
    """
    action: ActionType
    target_dir: PositionDir = PositionDir.FLAT
    target_layers: int = 0
    target_pct: Optional[float] = None

# ============================================================
# BrainBase 抽象基类
# ============================================================

class BrainBase(ABC):

    def __init__(self,executor: BaseExecutor,):
        super().__init__()
        self.executor = executor

    @abstractmethod
    def decide(self) -> TradingAction:
        """
        根据 MarketState 输出 TradingAction
        """
        pass