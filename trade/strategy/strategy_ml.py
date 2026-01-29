from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from data_process.common import Signal, CommonDefine
from trade.strategy.base_executor import BaseExecutor
from trade.strategy.strategy_base import *
import numpy as np
import logging,math

# ============================================================
# BrainBase 输入 / 输出数据结构 (扩展版)
# ============================================================

@dataclass
class MarketState:
    """
    BrainBase 的输入：当前市场 + 策略状态 + 账户元数据
    """
    price: float
    signal: Signal
    pred_prob: float
    atr: float              #  由 DataProvider 提供的当前 ATR
    slow_atr : float
    vol_regime: float

    position_dir: PositionDir   #仓位方向
    layers: int
    
    #  用于风控审计的元数据
    current_time: datetime
    account_balance: float

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
    stop_loss: float = 0.0

# ============================================================
# FtmoBrain：强化风控与动态仓位版
# ============================================================

class FtmoBrain(BrainBase):

    def __init__(
        self,
        executor: BaseExecutor,
        trade_risk: float = 0.1,    # 单层风险比例 (10%)
        max_layers: int = 1,
        holdbar: int = 16,
        allow_long: bool = True,
        allow_short: bool = True,
        thresh: Optional[float] = None,
        #  仿照 RulesBrain 新增的参数
        max_daily_loss_pct: float = 0.5, # 日内熔断阈值 (3.5%)
        unit_pct_scale: float = 1.9,      # 仓位利用率缩放系数
        stop_loss_long: float = 0.05,      # 做多止损百分比
        stop_loss_short: float = 0.05,     # 做空止损百分比
        atr_sl_mult_long:float = 3,
        atr_sl_mult_short:float = 3,
    ):
        self.logger = logging.getLogger("trade")
        self.executor = executor
        self.trade_risk = trade_risk
        self.max_layers = max_layers
        self.max_hold_num = holdbar
        self.allow_long = allow_long
        self.allow_short = allow_short
        self.thresh = thresh
        self.stop_loss_long = stop_loss_long
        self.stop_loss_short = stop_loss_short
        self.atr_sl_mult_long = atr_sl_mult_long
        self.atr_sl_mult_short = atr_sl_mult_short
        
        # ---  风控状态管理 ---
        self.max_daily_loss_pct = max_daily_loss_pct
        self.unit_pct_scale = unit_pct_scale
        
        self.day_start_equity = None
        self.last_trade_date = None
        self.is_halted_today = False
        
        # --- 统计指标 ---
        self.bars_held = 0
        self.all_durations = []
        self.current_trade_bars = 0
        self.current_signal_streak = 0
        self.all_signal_streaks = []

    def _update_daily_equity(self, current_time: datetime, account_balance: float):
        """日内净值更新与熔断重置"""
        current_date = current_time.date()
        if self.last_trade_date != current_date:
            self.day_start_equity = account_balance
            self.last_trade_date = current_date
            self.is_halted_today = False

    def _calculate_dynamic_unit_pct(self, state: MarketState) -> float:
        """基于 ATR 计算理论仓位百分比"""
        if state.atr <= 0:
            return 0.0
        # 基础海龟比例 (Risk / (2 * ATR_Pct))
        atr_pct = state.atr / state.price
        raw_nominal_pct = self.trade_risk / (2.0 * atr_pct)
        return self.trade_risk

    def decide(self, state: MarketState) -> TradingAction:
        # 1. 每日风险审计与熔断检查
        self._update_daily_equity(state.current_time, state.account_balance)
        if self.is_halted_today:
            return TradingAction(ActionType.HOLD)

        daily_loss_abs = max(0.0, self.day_start_equity - state.account_balance)
        max_loss_allowed_abs = self.day_start_equity * self.max_daily_loss_pct
        remaining_budget = max(0.0, max_loss_allowed_abs - daily_loss_abs)

        if daily_loss_abs >= max_loss_allowed_abs:
            self.is_halted_today = True
            self.logger.warning(f"🚨 [MELTDOWN] 日亏损触及上限! 亏损率: {daily_loss_abs/self.day_start_equity:.2%}")
            self.executor.user_close()
            return TradingAction(ActionType.CLOSE)

        # 2. 信号预处理
        signal = state.signal
        if signal == Signal.INVALID:
            signal = Signal.NEUTRAL
        if self.thresh is not None and state.pred_prob < self.thresh:
            signal = Signal.NEUTRAL

        if signal != Signal.NEUTRAL:
            self.current_signal_streak += 1
            self.bars_held = 0 
        else:
            if self.current_signal_streak > 0:
                self.all_signal_streaks.append(self.current_signal_streak)
            self.current_signal_streak = 0
            self.bars_held += 1

        # 4. 信号映射与出场逻辑
        target_dir = PositionDir.FLAT
        if signal == Signal.POSITIVE  and self.allow_long:
            target_dir = PositionDir.POSITIVE 
        elif signal == Signal.NEGATIVE and self.allow_short:
            target_dir = PositionDir.NEGATIVE
        
        if state.position_dir != PositionDir.FLAT:
            if self.bars_held < self.max_hold_num:
                target_dir = state.position_dir
            else:
                target_dir = PositionDir.FLAT

        action = TradingAction(ActionType.HOLD)

        #new order limit
        # if state.position_dir == PositionDir.FLAT and target_dir != PositionDir.FLAT:
        #     if valid_number(state.atr) and valid_number(state.slow_atr):
        #         # if state.atr/state.slow_atr < 0.7: 
        #         #     target_dir = PositionDir.FLAT
        #         if target_dir != PositionDir.FLAT and  state.slow_atr * math.sqrt(CommonDefine.predict_num) < 0.02:
        #             target_dir = PositionDir.FLAT
        #             self.logger.debug(f"filter signal {target_dir} by slow_atr:{state.slow_atr}")
            # if target_dir != PositionDir.FLAT and state.vol_regime < 1 :
            #     target_dir = PositionDir.FLAT
            # if target_dir != PositionDir.FLAT and  state.atr * math.sqrt(CommonDefine.predict_num) < 0.02:
            #     target_dir = PositionDir.FLAT

        if target_dir != PositionDir.FLAT:
            # 3. 计算下单参数
            # 使用 trade_risk 作为固定百分比，或切换回 _calculate_dynamic_unit_pct
            sl_pct = 0.05
            if self.atr_sl_mult_long!=None and target_dir == PositionDir.POSITIVE  and state.atr > 0:
                sl_pct = state.atr * self.atr_sl_mult_long
            elif self.atr_sl_mult_short!=None and target_dir == PositionDir.NEGATIVE and state.atr > 0:
                sl_pct = state.atr * self.atr_sl_mult_short
            elif self.atr_sl_mult_long==None and  self.atr_sl_mult_short==None:
                sl_pct = self.stop_loss_long if target_dir == PositionDir.POSITIVE  else self.stop_loss_short
            
            if target_dir == PositionDir.FLAT or sl_pct <= 0:
                final_order_qty = 0.0
            else:
                base_pct = self.trade_risk 
                
                intended_qty = (base_pct * state.account_balance) / state.price
                
                max_budget_qty = (remaining_budget * 0.8) / (state.price * sl_pct)
                
                # 最终取最小值
                final_order_qty = min(intended_qty, max_budget_qty)
                
                if final_order_qty < intended_qty:
                    self.logger.debug(f"🛡️ [BUDGET CUT] 原始建议股数 {intended_qty:.4f} 因预算限制削减至 {final_order_qty:.4f}")

        # 5. 执行决策逻辑 (封装订单信息)
        if state.position_dir == PositionDir.FLAT:
            if target_dir != PositionDir.FLAT:
                self.current_trade_bars = 1
                action = TradingAction(
                    action=ActionType.OPEN,
                    target_dir=target_dir,
                    target_layers=1,
                    order_qty=final_order_qty,
                    stop_loss=sl_pct
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
                    order_qty=final_order_qty,
                    stop_loss=sl_pct
                )
            elif state.layers < self.max_layers and signal != Signal.NEUTRAL:
                new_layers = state.layers + 1
                action = TradingAction(
                    action=ActionType.PYRAMID,
                    target_dir=state.position_dir,
                    target_layers=new_layers,
                    order_qty=final_order_qty,
                    stop_loss=sl_pct
                )

        self.execute_action(action)
        return action

    def execute_action(self, action: TradingAction):
        """修改为使用 user_order 接口，并传递止损参数"""
        if action.action == ActionType.HOLD:
            return

        if action.action == ActionType.CLOSE:
            self.executor.user_close()
            self.dir = 0
            self.layers = 0
            return

        is_buy = (action.target_dir == PositionDir.POSITIVE )
        
        # 处理订单执行
        if action.action == ActionType.REVERSE:
            # 反手时先平掉当前所有仓位
            self.executor.user_close()
            # 然后开立新方向的第一层仓位
            self.executor.user_order(action.order_qty, is_buy=is_buy, stop_loss=action.stop_loss)
            self.dir = action.target_dir
            self.layers = 1
            
        elif action.action in (ActionType.OPEN, ActionType.PYRAMID):
            # 开仓或加仓均直接下单
            self.executor.user_order(action.order_qty, is_buy=is_buy, stop_loss=action.stop_loss)
            self.dir = action.target_dir
            self.layers = action.target_layers

    def stop(self):
        """
        回测结束时的终局审计
        """
        if False:
            pass
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

def valid_number(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)
