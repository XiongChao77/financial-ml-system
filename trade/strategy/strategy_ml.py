from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from trade.core.venue_base import VenueBase
from trade.core.protocol import *
from trade.core.strategy_base import StrategyBase
import numpy as np
import logging,math

# ============================================================
# MlSignalStrategy: hardened risk control and dynamic sizing
# ============================================================

@dataclass
class MlStrategyConfig:
    """Static parameters owned by the ML decision strategy."""

    risk_per_trade_pct: float = 0.01
    min_hold_bars: int = 16
    allow_long: bool = True
    allow_short: bool = True
    prob_thresh: Optional[float] = None
    max_daily_loss_pct: float = 0.03
    atr_sl_long_mult: float = 3.0
    atr_sl_short_mult: float = 3.0
    atr_tp_mult: float = 5.0
    decide_version: int = 0


class MlSignalStrategy(StrategyBase):

    def __init__(
        self,
        venue: VenueBase,
        config: MlStrategyConfig,
        init_equity: float,
        exist_hold_bars: int = 0,
        leverage: float = 1.0,
    ):
        super().__init__(venue)
        self.logger = logging.getLogger("trade")
        self.config = config
        self.init_equity = float(init_equity)
        self.leverage = max(1.0, float(leverage))
        self.pre_signal = Signal.NEUTRAL
        self.pre_position_dir: PositionDir = PositionDir.FLAT

        # --- risk control state ---
        self.day_start_equity = None
        self.last_trade_date = None
        self.is_halted_today = False
        self.meltdown_days = 0

        # --- statistics ---
        self.bars_since_confirming_signal = exist_hold_bars
        self.all_durations = []
        self.current_trade_bars = 0
        self.current_signal_streak = 0
        self.all_signal_streaks = []

    def _update_daily_equity(self, current_time: datetime, account_equity: float):
        """Daily equity update and circuit breaker reset"""
        current_date = current_time.date()
        if self.last_trade_date != current_date:
            self.day_start_equity = account_equity
            self.last_trade_date = current_date
            self.is_halted_today = False

    def _calculate_unit_pct(self, target_dir: PositionDir, state: Observation, remaining_risk_budget: float) -> tuple[float, float, float]:
        atr_pct = state.market.atr_pct
        if atr_pct is None or not math.isfinite(atr_pct) or atr_pct <= 0:
            self.logger.warning(
                "ATR is unavailable; skipping the ATR-sized entry for this bar"
            )
            return 0, 0, 0
        if state.account.equity < self.init_equity:
            risk_equity = self.init_equity  #design for fTMO challenge
        else:
            risk_equity = state.account.equity
        if target_dir == PositionDir.POSITIVE:
            sl_pct = state.market.atr_pct * self.config.atr_sl_long_mult
            tp_pct = state.market.atr_pct * self.config.atr_tp_mult
            intended_qty = (self.config.risk_per_trade_pct * risk_equity) / (state.market.price * sl_pct)
            max_risk_qty  = (remaining_risk_budget * 0.8) / (state.market.price * sl_pct)
        elif target_dir == PositionDir.NEGATIVE:
            sl_pct = state.market.atr_pct * self.config.atr_sl_short_mult
            tp_pct = state.market.atr_pct * self.config.atr_tp_mult
            intended_qty = (self.config.risk_per_trade_pct * risk_equity) / (state.market.price * sl_pct)
            max_risk_qty  = (remaining_risk_budget * 0.8) / (state.market.price * sl_pct)
        else:
            return 0,0,0
        final_order_qty = min(intended_qty, max_risk_qty )

        required_margin = final_order_qty * state.market.price / self.leverage
        free_margin = state.account.equity # Because are no position before open a new one, reserve not consider yet
        if required_margin > free_margin:
            self.logger.info(
                f"❌ [MARGIN NOT ENOUGH] required_margin={required_margin:.2f}, "
                f"free_margin={free_margin:.2f}, "
                f"qty={final_order_qty:.4f}, price={state.market.price:.2f}, leverage={self.leverage}"
            )
            return 0, 0, 0

        if final_order_qty < intended_qty:
            self.logger.debug(f"🛡️ [BUDGET CUT] 原始建议Quantity {intended_qty:.4f} 因预算限制削减至 {final_order_qty:.4f}")
        if tp_pct > 0.9:
            self.logger.debug(f"🛡️ [TP CUT] original tp_pct {tp_pct:.4f} ,adjust to 0.9")
            tp_pct = 0.9
        return final_order_qty , sl_pct,tp_pct

    def process(self, state: Observation) -> TradeIntent:
        # singal preprocesss
        signal = state.market.signal
        if signal == Signal.INVALID:
            signal = Signal.NEUTRAL
        if self.config.prob_thresh is not None and state.market.pred_prob < self.config.prob_thresh:
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

        if state.position.dir != self.pre_position_dir:
            if state.position.dir == PositionDir.FLAT: # close detected
                self.all_durations.append(self.current_trade_bars)
                self.current_trade_bars = 0
            else: # open or reserve detected
                if self.pre_position_dir == PositionDir.FLAT: # open detected
                    self.current_trade_bars = 1
                else: #reserve
                    self.all_durations.append(self.current_trade_bars)
                    self.current_trade_bars = 1
            self.bars_since_confirming_signal = 0
            self.pre_position_dir = state.position.dir
        elif state.position.dir != PositionDir.FLAT:
            self.bars_since_confirming_signal += 1
            self.current_trade_bars += 1 # hold detected
            
        # 1. Daily risk audit and circuit breaker check
        """update daily equity"""
        current_date = state.current_time.date()
        if self.last_trade_date != current_date:
            self.day_start_equity = state.account.equity
            self.last_trade_date = current_date
            self.is_halted_today = False

        # trade action
        target_dir = PositionDir.FLAT
        if signal == Signal.POSITIVE  and self.config.allow_long:
            target_dir = PositionDir.POSITIVE 
        elif signal == Signal.NEGATIVE and self.config.allow_short:
            target_dir = PositionDir.NEGATIVE

        bars_to_close = state.market.bars_to_close
        if bars_to_close is None or math.isnan(bars_to_close):
            bars_to_close = math.inf
        
        if state.position.dir != PositionDir.FLAT:
            if target_dir == state.position.dir:
                self.bars_since_confirming_signal = 0 # reset
            elif self.bars_since_confirming_signal < self.config.min_hold_bars:
                target_dir = state.position.dir
            else:
                pass
            
        # force check , in those conditions the position should be close immediately & open forbidden
        daily_loss_abs = max(0.0, self.day_start_equity - state.account.equity)
        daily_max_loss_allowed_abs = self.day_start_equity * self.config.max_daily_loss_pct
        # total_max_loss_allowed_abs = self.init_equity * self.config.max_daily_loss_pct
        if daily_loss_abs >= daily_max_loss_allowed_abs:
            if self.is_halted_today == False:
                self.logger.warning(f"🚨 [MELTDOWN] 日亏损触及上限! 亏损率: {daily_loss_abs/self.day_start_equity:.2%}")
                self.is_halted_today = True
                self.meltdown_days += 1
            target_dir = PositionDir.FLAT
        elif state.position.dir == PositionDir.FLAT :
            if target_dir != PositionDir.FLAT and bars_to_close <= self.config.min_hold_bars:
                self.logger.info(
                    f"⏳ [NO OPEN] bars_to_close={bars_to_close} < min_hold_bars={self.config.min_hold_bars}"
                )
                target_dir = PositionDir.FLAT
        # 2. Two bars before the close, force flat if still in position
        else:
            if bars_to_close <= 2:
                self.logger.info(
                    f"🔚 [CLOSE BEFORE MARKET CLOSE] bars_to_close={bars_to_close}, force close position."
                )
                target_dir = PositionDir.FLAT

        action = TradeIntent(ActionType.NOOP)

        #new order
        if target_dir != PositionDir.FLAT and target_dir != state.position.dir:
            remaining_risk_budget = max(0.0, daily_max_loss_allowed_abs - daily_loss_abs)
            final_order_qty , sl_pct, tp_pct = self._calculate_unit_pct(target_dir, state, remaining_risk_budget)
            if final_order_qty == 0:
                target_dir = PositionDir.FLAT

        # 5. Run the decision logic (packs the order information)
        if state.position.dir == PositionDir.FLAT:
            if target_dir != PositionDir.FLAT:

                action = TradeIntent(
                    action=ActionType.OPEN,
                    target_dir=target_dir,
                    target_layers=1,
                    order_qty=final_order_qty,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                )
        else:
            if target_dir == PositionDir.FLAT:
                action = TradeIntent(ActionType.CLOSE)
            elif target_dir != state.position.dir:
                action = TradeIntent(
                    action=ActionType.REVERSE,
                    target_dir=target_dir,
                    target_layers=1,
                    order_qty=final_order_qty,
                    stop_loss_pct=sl_pct,
                    take_profit_pct=tp_pct,
                )

        self.execute_action(action)
        return action

    def execute_action(self, action: TradeIntent):
        """Reworked to use the submit_order interface and pass the stop loss parameters"""
        if action.action == ActionType.NOOP:
            return

        if action.action == ActionType.CLOSE:
            self.venue.close_position()
            return

        is_buy = (action.target_dir == PositionDir.POSITIVE )
        
        # Execute the order
        if action.action == ActionType.REVERSE:
            # On a reverse, close everything first
            self.venue.close_position()
            # then open the first layer in the new direction
            self.venue.submit_order(action.order_qty, is_buy=is_buy, stop_loss_pct=action.stop_loss_pct, take_profit_pct=action.take_profit_pct)
            
        elif action.action == ActionType.OPEN:
            self.venue.submit_order(action.order_qty, is_buy=is_buy, stop_loss_pct=action.stop_loss_pct, take_profit_pct=action.take_profit_pct)

    def finalize(self):
        """
        Final audit at the end of the backtest
        """
        if True:
            if self.current_trade_bars > 0:
                self.all_durations.append(self.current_trade_bars)
            self.logger.info("=== 正在生成持仓时长分布报告 ===")
            
            if not self.all_durations:
                self.logger.info("❌ 回测期间未产生完成的交易信号。")
                return

            durations = np.array(self.all_durations)
            min_hold_bars = self.config.min_hold_bars # default 16
            
            # Core statistics
            avg_dur = np.mean(durations)
            median_dur = np.median(durations)
            max_dur = np.max(durations)
            min_dur = np.min(durations)
            # Renewal rate: share of positions held longer than min_hold_bars
            renewal_count = np.sum(durations > min_hold_bars)
            renewal_rate = renewal_count / len(durations)

            self.logger.info(f"[Hold] count={len(durations)} min_hold_bars={min_hold_bars} avg={avg_dur:.1f} med={median_dur:.1f} min={min_dur} max={max_dur} renew rate={renewal_rate:.1%}")
            # Print the distribution histogram (simple ASCII)
            # self.log_histogram(durations)

            if self.all_signal_streaks:
                s = np.array(self.all_signal_streaks)
                self.logger.info(
                    f"[Streak] n={len(s)} min={np.min(s)} avg={np.mean(s):.1f} med={np.median(s):.1f} p95={np.percentile(s,95):.1f} max={np.max(s)}"
                )

    def report(self) -> dict:
        """
        MlSignalStrategy specific statistics (holding time distribution / signal streaks / intraday breaker).
        The scalar part goes into report["strategy"], the list details into report_additional.
        """
        metrics = {
            'meltdown_days': self.meltdown_days,
            'min_hold_bars': self.config.min_hold_bars,
        }
        if self.all_durations:
            d = np.array(self.all_durations)
            metrics.update({
                'hold_count': int(len(d)),
                'hold_avg_bars': float(d.mean()),
                'hold_med_bars': float(np.median(d)),
                'hold_min_bars': int(d.min()),
                'hold_max_bars': int(d.max()),
                'hold_p95_bars': float(np.percentile(d, 95)),
                'hold_renewal_rate': float((d > self.config.min_hold_bars).mean()),
                'hold_durations': self.all_durations,          # detail
            })
        if self.all_signal_streaks:
            s = np.array(self.all_signal_streaks)
            metrics.update({
                'streak_count': int(len(s)),
                'streak_avg': float(s.mean()),
                'streak_med': float(np.median(s)),
                'streak_p95': float(np.percentile(s, 95)),
                'streak_max': int(s.max()),
                'signal_streaks': self.all_signal_streaks,     # detail
            })
        return metrics

    def log_histogram(self, data):
        """Print a simple console histogram to inspect the distribution"""
        counts, bins = np.histogram(data, bins=10)
        for i in range(len(counts)):
            bar = "█" * int(counts[i] / len(data) * 40)
            self.logger.info(f"[{bins[i]:>3.0f} - {bins[i+1]:>3.0f} bars]: {bar} {counts[i]}")


def valid_number(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)
