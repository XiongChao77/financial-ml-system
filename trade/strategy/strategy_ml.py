from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from trade.strategy.base_executor import BaseExecutor
from trade.strategy.strategy_base import *
import numpy as np
import logging,math

# ============================================================
# BrainBase 输入 / 输出数据结构 (扩展版)
# ============================================================

@dataclass
class TradingAction:
    """
    BrainBase 的输出：包含具体的订单信息
    """
    action: ActionType
    target_dir: PositionDir = PositionDir.FLAT
    target_layers: int = 0
    #  新增字段以支持 user_order 接口
    order_qty: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0

# ============================================================
# FtmoBrain：强化风控与动态仓位版
# ============================================================

class FtmoBrain(BrainBase):

    def __init__(
        self,
        executor: BaseExecutor,
        init_equity :float,
        trade_risk: float = 0.01,    # 单层风险比例
        min_hold_num: int = 16,
        exist_hold_num: int = 0,
        allow_long: bool = True,
        allow_short: bool = True,
        thresh: Optional[float] = None,
        #  仿照 RulesBrain 新增的参数
        max_daily_loss_pct: float = 0.03, # 日内熔断阈值
        atr_sl_mult_long:float = 3,
        atr_sl_mult_short:float = 3,
        atr_tp:float = 5,
        leverage:int = 1,
        decide_version: int = 0
    ):
        self.logger = logging.getLogger("trade")
        self.executor = executor
        self.init_equity = init_equity
        self.trade_risk = trade_risk
        self.min_hold_num = min_hold_num
        self.allow_long = allow_long
        self.allow_short = allow_short
        self.thresh = thresh
        self.atr_sl_mult_long = atr_sl_mult_long
        self.atr_sl_mult_short = atr_sl_mult_short
        self.atr_tp = atr_tp
        self.leverage = leverage
        self.pre_signal = Signal.NEUTRAL
        self.pre_position_dir:PositionDir = PositionDir.FLAT
        self.decide_version = decide_version
        
        # ---  风控状态管理 ---
        self.max_daily_loss_pct = max_daily_loss_pct
        
        self.day_start_equity = None
        self.last_trade_date = None
        self.is_halted_today = False
        
        # --- 统计指标 ---
        self.bars_since_confirming_signal = exist_hold_num
        self.all_durations = []
        self.current_trade_bars = 0
        self.current_signal_streak = 0
        self.all_signal_streaks = []

    def _update_daily_equity(self, current_time: datetime, account_equity: float):
        """日内净值更新与熔断重置"""
        current_date = current_time.date()
        if self.last_trade_date != current_date:
            self.day_start_equity = account_equity
            self.last_trade_date = current_date
            self.is_halted_today = False

    def _calculate_unit_pct(self, target_dir: PositionDir, state: MarketState, remaining_risk_budget: float) -> tuple[float, float, float]:
        if state.account_equity < self.init_equity:
            risk_equity = self.init_equity  #design for fTMO challenge
        else:
            risk_equity = state.account_equity
        if target_dir == PositionDir.POSITIVE:
            sl_pct = state.atr_pct * self.atr_sl_mult_long
            tp_pct = state.atr_pct * self.atr_tp
            intended_qty = (self.trade_risk * risk_equity) / (state.price * sl_pct)
            max_risk_qty  = (remaining_risk_budget * 0.8) / (state.price * sl_pct)
        elif target_dir == PositionDir.NEGATIVE:
            sl_pct = state.atr_pct * self.atr_sl_mult_short
            tp_pct = state.atr_pct * self.atr_tp
            intended_qty = (self.trade_risk * risk_equity) / (state.price * sl_pct)
            max_risk_qty  = (remaining_risk_budget * 0.8) / (state.price * sl_pct)
        else:
            return 0,0,0
        final_order_qty = min(intended_qty, max_risk_qty )

        required_margin = final_order_qty * state.price / self.leverage
        free_margin = state.account_equity # Because are no position before open a new one, reserve not consider yet
        if required_margin > free_margin:
            self.logger.info(
                f"❌ [MARGIN NOT ENOUGH] required_margin={required_margin:.2f}, "
                f"free_margin={free_margin:.2f}, "
                f"qty={final_order_qty:.4f}, price={state.price:.2f}, leverage={self.leverage}"
            )
            return 0, 0, 0

        if final_order_qty < intended_qty:
            self.logger.debug(f"🛡️ [BUDGET CUT] 原始建议Quantity {intended_qty:.4f} 因预算限制削减至 {final_order_qty:.4f}")
        if tp_pct > 0.9:
            self.logger.debug(f"🛡️ [TP CUT] original tp_pct {tp_pct:.4f} ,adjust to 0.9")
            tp_pct = 0.9
        return final_order_qty , sl_pct,tp_pct

    def decide(self, state: MarketState) -> TradingAction:
        # singal preprocesss
        signal = state.signal
        if signal == Signal.INVALID:
            signal = Signal.NEUTRAL
        if self.thresh is not None and state.pred_prob < self.thresh:
            signal = Signal.NEUTRAL
        
        # record signal
        if signal == self.pre_signal:
            if signal != Signal.NEUTRAL:
                self.current_signal_streak += 1 
        else:
            if self.pre_signal != Signal.NEUTRAL:
                if self.current_signal_streak > 0:
                    self.all_signal_streaks.append(self.current_signal_streak)
                self.current_signal_streak = 1
            else:
                self.current_signal_streak = 1
            self.pre_signal = signal

        if state.position_dir != self.pre_position_dir:
            if state.position_dir == PositionDir.FLAT: # close detected
                self.all_durations.append(self.current_trade_bars)
                self.current_trade_bars = 0
            else: # open or reserve detected
                if self.pre_position_dir == PositionDir.FLAT: # open detected
                    self.current_trade_bars = 1
                else: #reserve
                    self.all_durations.append(self.current_trade_bars)
                    self.current_trade_bars = 1
            self.bars_since_confirming_signal = 0
            self.pre_position_dir = state.position_dir
        elif state.position_dir != PositionDir.FLAT:
            self.bars_since_confirming_signal += 1
            self.current_trade_bars += 1 # hold detected
            
        # 1. 每日风险审计与熔断检查
        """update daily equity"""
        current_date = state.current_time.date()
        if self.last_trade_date != current_date:
            self.day_start_equity = state.account_equity
            self.last_trade_date = current_date
            self.is_halted_today = False

        # trade action
        target_dir = PositionDir.FLAT
        if signal == Signal.POSITIVE  and self.allow_long:
            target_dir = PositionDir.POSITIVE 
        elif signal == Signal.NEGATIVE and self.allow_short:
            target_dir = PositionDir.NEGATIVE
        
        if state.position_dir != PositionDir.FLAT:
            if target_dir == state.position_dir:
                self.bars_since_confirming_signal = 0 # reset
            elif self.bars_since_confirming_signal < self.min_hold_num:
                target_dir = state.position_dir
            else:
                pass
            
        # force check , in those conditions the position should be close immediately & open forbidden
        daily_loss_abs = max(0.0, self.day_start_equity - state.account_equity)
        daily_max_loss_allowed_abs = self.day_start_equity * self.max_daily_loss_pct
        # total_max_loss_allowed_abs = self.init_equity * self.max_daily_loss_pct
        if daily_loss_abs >= daily_max_loss_allowed_abs:
            if self.is_halted_today == False:
                self.logger.warning(f"🚨 [MELTDOWN] 日亏损触及上限! 亏损率: {daily_loss_abs/self.day_start_equity:.2%}")
                self.is_halted_today = True
            target_dir = PositionDir.FLAT
        elif state.position_dir == PositionDir.FLAT :
            if target_dir != PositionDir.FLAT and state.bars_to_close <= self.min_hold_num:
                self.logger.info(
                    f"⏳ [NO OPEN] bars_to_close={state.bars_to_close} < min_hold_num={self.min_hold_num}"
                )
                target_dir = PositionDir.FLAT
        # 2. 收盘前 2 根 bar，若仍有仓位，强制平仓
        else:
            if state.bars_to_close <= 2:
                self.logger.info(
                    f"🔚 [CLOSE BEFORE MARKET CLOSE] bars_to_close={state.bars_to_close}, force close position."
                )
                target_dir = PositionDir.FLAT

        action = TradingAction(ActionType.NOOP)

        #new order
        if target_dir != PositionDir.FLAT and target_dir != state.position_dir:
            remaining_risk_budget = max(0.0, daily_max_loss_allowed_abs - daily_loss_abs)
            final_order_qty , sl_pct, tp_pct = self._calculate_unit_pct(target_dir, state, remaining_risk_budget)
            if final_order_qty == 0:
                target_dir = PositionDir.FLAT

        # 5. 执行决策逻辑 (封装订单信息)
        if state.position_dir == PositionDir.FLAT:
            if target_dir != PositionDir.FLAT:

                action = TradingAction(
                    action=ActionType.OPEN,
                    target_dir=target_dir,
                    target_layers=1,
                    order_qty=final_order_qty,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                )
        else:
            if target_dir == PositionDir.FLAT:
                action = TradingAction(ActionType.CLOSE)
            elif target_dir != state.position_dir:
                action = TradingAction(
                    action=ActionType.REVERSE,
                    target_dir=target_dir,
                    target_layers=1,
                    order_qty=final_order_qty,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                )

        self.execute_action(action)
        return action

    def execute_action(self, action: TradingAction):
        """修改为使用 user_order 接口，并传递止损参数"""
        if action.action == ActionType.NOOP:
            return

        if action.action == ActionType.CLOSE:
            self.executor.user_close()
            return

        is_buy = (action.target_dir == PositionDir.POSITIVE )
        
        # 处理订单执行
        if action.action == ActionType.REVERSE:
            # 反手时先平掉当前所有仓位
            self.executor.user_close()
            # 然后开立新方向的第一层仓位
            self.executor.user_order(action.order_qty, is_buy=is_buy, stop_loss_pct=action.stop_loss_pct, take_profit_pct=action.take_profit_pct)
            
        elif action.action == ActionType.OPEN:
            self.executor.user_order(action.order_qty, is_buy=is_buy, stop_loss_pct=action.stop_loss_pct, take_profit_pct=action.take_profit_pct)

    def stop(self):
        """
        回测结束时的终局审计
        """
        if True:
            if self.current_trade_bars > 0:
                self.all_durations.append(self.current_trade_bars)
            self.logger.info("=== 正在生成持仓时长分布报告 ===")
            
            if not self.all_durations:
                self.logger.info("❌ 回测期间未产生完成的交易信号。")
                return

            durations = np.array(self.all_durations)
            min_hold_num = self.min_hold_num # 默认 16
            
            # 核心统计指标
            avg_dur = np.mean(durations)
            median_dur = np.median(durations)
            max_dur = np.max(durations)
            min_dur = np.min(durations)
            # 续期率：持仓超过 min_hold_num 的比例
            renewal_count = np.sum(durations > min_hold_num)
            renewal_rate = renewal_count / len(durations)

            self.logger.info(f"[Hold] count={len(durations)} min_hold_num={min_hold_num} avg={avg_dur:.1f} med={median_dur:.1f} min={min_dur} max={max_dur} renew rate={renewal_rate:.1%}")
            # 打印分布直方图 (ASCII 简易版)
            # self.log_histogram(durations)

            if self.all_signal_streaks:
                s = np.array(self.all_signal_streaks)
                self.logger.info(
                    f"[Streak] n={len(s)} min={np.min(s)} avg={np.mean(s):.1f} med={np.median(s):.1f} p95={np.percentile(s,95):.1f} max={np.max(s)}"
                )

    def log_histogram(self, data):
        """打印一个简单的控制台直方图，观察分布"""
        counts, bins = np.histogram(data, bins=10)
        for i in range(len(counts)):
            bar = "█" * int(counts[i] / len(data) * 40)
            self.logger.info(f"[{bins[i]:>3.0f} - {bins[i+1]:>3.0f} bars]: {bar} {counts[i]}")


def valid_number(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)
