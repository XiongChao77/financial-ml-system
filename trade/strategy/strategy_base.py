from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional
from datetime import datetime
from trade.strategy.base_executor import BaseExecutor
from data_process.common import Signal
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
    NOOP = "noop"   # 无操作,不改变任何仓位状态
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

@dataclass
class MarketState:
    """
    BrainBase 的输入：当前市场 + 策略状态 + 账户元数据
    """
    price: float
    signal: Signal
    pred_prob: float
    atr_pct: float              #  由 DataProvider 提供的当前 ATR
    slow_atr : float
    vol_regime: float

    position_dir: PositionDir   #仓位方向
    layers: int

    #  用于风控审计的元数据
    current_time: datetime
    account_equity: float
    bars_to_close: int

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