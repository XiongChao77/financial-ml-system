from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Optional
from data_process.common import Signal,PREDICT_NUM
from trade.strategy.base_executor import BaseExecutor
from trade.strategy.strategy_base import *
import numpy as  np
import logging 
# ============================================================
# BrainBase 输入 / 输出数据结构
# ============================================================

@dataclass
class MarketState:
    """
    BrainBase 的输入：当前市场 + 策略状态
    """
    price: float
    signal: Signal
    pred_prob: float

    position_dir: PositionDir
    layers: int

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
# FtmoBrain：你的策略逻辑（已枚举化）
# ============================================================

class FtmoBrain(BrainBase):

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
        self.logger = logging.getLogger("trade")
        self.bars_held = 0
        self.max_hold_num = holdbar   #PREDICT_NUM
        self.executor = executor
        self.trade_risk = trade_risk
        self.max_layers = max_layers
        self.allow_long = allow_long
        self.allow_short = allow_short
        self.thresh = thresh
        self.all_durations = []        # 存储所有已完成交易的持仓时长
        self.current_trade_bars = 0    # 当前交易已持续的 K 线数
        self.was_in_position = False   # 记录上一根 K 线是否有持仓
        self.current_signal_streak = 0
        self.all_signal_streaks = []

    def decide(self, state: MarketState) -> TradingAction:
        # ---------------------------------------------------------
        # 1. 信号预处理与连击统计 (Signal Streak Tracking)
        # ---------------------------------------------------------
        signal = state.signal
        
        if signal == Signal.INVALID:
            signal = Signal.NEUTRAL
            
        if self.thresh is not None and state.pred_prob < self.thresh:
            signal = Signal.NEUTRAL

        # --- 核心统计：记录连续趋势信号的长度 ---
        if signal != Signal.NEUTRAL:
            # 只要不是震荡信号，连击数增加，有效期重置
            self.current_signal_streak += 1
            self.bars_held = 0 
        else:
            # 一旦变成震荡信号，记录上一段连击的长度并归零
            if self.current_signal_streak > 0:
                self.all_signal_streaks.append(self.current_signal_streak)
            self.current_signal_streak = 0
            self.bars_held += 1

        # ---------------------------------------------------------
        # 2. 信号映射至目标方向 (Target Direction)
        # ---------------------------------------------------------
        target_dir = PositionDir.FLAT

        if signal == Signal.LONG and self.allow_long:
            target_dir = PositionDir.LONG
        elif signal == Signal.SHORT and self.allow_short:
            target_dir = PositionDir.SHORT
        
        if state.position_dir != PositionDir.FLAT :#and signal == Signal.NEUTRAL:
            if self.bars_held < self.max_hold_num:
                target_dir = state.position_dir
            else:
                target_dir = PositionDir.FLAT

        action = TradingAction(ActionType.HOLD)

        # ---------------------------------------------------------
        # 3. 动作决策与时长统计
        # ---------------------------------------------------------
        if state.position_dir == PositionDir.FLAT:
            if target_dir != PositionDir.FLAT:
                self.current_trade_bars = 1
                action = TradingAction(
                    action=ActionType.OPEN,
                    target_dir=target_dir,
                    target_layers=1,
                    target_pct=self.trade_risk * target_dir,
                )
        else:
            self.current_trade_bars += 1

            if target_dir == PositionDir.FLAT:
                self.all_durations.append(self.current_trade_bars)
                self.current_trade_bars = 0
                action = TradingAction(ActionType.CLOSE)

            elif target_dir != state.position_dir:
                self.all_durations.append(self.current_trade_bars)
                self.current_trade_bars = 1
                action = TradingAction(
                    action=ActionType.REVERSE,
                    target_dir=target_dir,
                    target_layers=1,
                    target_pct=self.trade_risk * target_dir,
                )

            elif state.layers < self.max_layers and signal != Signal.NEUTRAL:
                action = TradingAction(
                    action=ActionType.PYRAMID,
                    target_dir=state.position_dir,
                    target_layers=state.layers + 1,
                    target_pct=self.trade_risk * (state.layers + 1) * state.position_dir,
                )

        self.execute_action(action)
        return action

    def stop(self):
        """
        回测结束时的终局审计
        """
        if self.current_trade_bars > 0:
            self.all_durations.append(self.current_trade_bars)
        self.logger.info("=== 正在生成持仓时长分布报告 ===")
        
        if not self.all_durations:
            self.logger.info("❌ 回测期间未产生完成的交易信号。")
            return

        durations = np.array(self.all_durations)
        max_hold_num = self.max_hold_num # 默认 16
        
        # 核心统计指标
        avg_dur = np.mean(durations)
        median_dur = np.median(durations)
        max_dur = np.max(durations)
        # 续期率：持仓超过 max_hold_num 的比例
        renewal_count = np.sum(durations > max_hold_num)
        renewal_rate = renewal_count / len(durations)

        self.logger.info(f"\n" + "="*40)
        self.logger.info(f"📊 持仓延续性审计报告")
        self.logger.info(f"总计完成交易: {len(durations)} 笔")
        self.logger.info(f"平均持仓时长: {avg_dur:.2f} 根 K 线")
        self.logger.info(f"最长持仓时长: {max_dur} 根 K 线")
        self.logger.info(f"信号续期次数: {renewal_count} (持仓 > {max_hold_num} bars)")
        self.logger.info(f"有效续期比例: {renewal_rate:.2%}")
        self.logger.info(f"="*40 + "\n")

        # 打印分布直方图 (ASCII 简易版)
        self.log_histogram(durations)

        if self.all_signal_streaks:
            streaks = np.array(self.all_signal_streaks)
            self.logger.info(f"\n🎯 信号连击深度审计 (Consecutive Trend Signals)")
            self.logger.info(f"平均连击长度: {np.mean(streaks):.2f} 根 K 线")
            self.logger.info(f"最大连击长度: {np.max(streaks)} 根 K 线")
            self.logger.info(f"单点爆发比例 (Length=1): {np.sum(streaks == 1) / len(streaks):.2%}")
            
            # 打印分布
            counts, bins = np.histogram(streaks, bins=[1, 2, 5, 10, 20, 50, 100])
            for i in range(len(counts)):
                self.logger.info(f"  连击区间 [{bins[i]:>2}-{bins[i+1]:>2}]: {counts[i]} 次")

    def log_histogram(self, data):
        """打印一个简单的控制台直方图，观察分布"""
        counts, bins = np.histogram(data, bins=10)
        for i in range(len(counts)):
            bar = "█" * int(counts[i] / len(data) * 40)
            self.logger.info(f"[{bins[i]:>3.0f} - {bins[i+1]:>3.0f} bars]: {bar} {counts[i]}")

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