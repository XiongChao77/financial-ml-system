import pandas as pd
import logging
import math
from datetime import datetime
from trade.strategy.strategy_ml import BrainBase, TradingAction, ActionType, PositionDir
from trade.strategy.base_executor import BaseExecutor

class RulesBrain(BrainBase):
    def __init__(self, executor: BaseExecutor, **kwargs):
        super().__init__(executor)
        # --- 基础参数 ---
        self.entry_period = kwargs.get('entry_period', 20)
        self.exit_period = kwargs.get('exit_period', 10)
        self.atr_period = kwargs.get('atr_period', 20)
        self.max_layers = kwargs.get('max_layers', 1)
        
        # --- 核心风控参数 (同步自 TurtleBrain) ---
        self.risk_per_trade = kwargs.get('risk_per_trade', 0.01)     # 单层风险 (1%)
        self.max_daily_loss_pct = kwargs.get('max_daily_loss_pct', 0.035) # 日亏损上限 (4.5%)
        self.unit_pct_scale = kwargs.get('unit_pct_scale', 1.9)      # 资金利用率缩放
        self.upper_limit = kwargs.get('upper_limit', 0.6)            # 单层名义价值上限
        self.pyramid_gap_atr = kwargs.get('pyramid_gap_atr', 0.5)

        # --- 状态管理 ---
        self.day_start_equity = None
        self.last_trade_date = None
        self.is_halted_today = False
        self.curr_layers = 0 
        self.layer_sizes = []  
        self.last_order_price = 0.0
        
        self.logger = logging.getLogger("RulesBrain")

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """保持原有计算逻辑，确保 ATR 与唐奇安通道对齐"""
        required = ['atr', 'entry_high', 'entry_low', 'exit_high', 'exit_low']
        if all(col in df.columns for col in required): return df

        df = df.copy()
        df['entry_high'] = df['high'].shift(1).rolling(window=self.entry_period).max()
        df['entry_low'] = df['low'].shift(1).rolling(window=self.entry_period).min()
        df['exit_high'] = df['high'].shift(1).rolling(window=self.exit_period).max()
        df['exit_low'] = df['low'].shift(1).rolling(window=self.exit_period).min()
        
        tr = pd.concat([df['high']-df['low'], 
                       (df['high']-df['close'].shift(1)).abs(), 
                       (df['low']-df['close'].shift(1)).abs()], axis=1).max(axis=1)
        
        atr_vals = [0.0] * len(df)
        if len(df) >= self.atr_period:
            atr_vals[self.atr_period-1] = tr[:self.atr_period].mean()
            for i in range(self.atr_period, len(df)):
                atr_vals[i] = (atr_vals[i-1] * (self.atr_period-1) + tr.iloc[i]) / self.atr_period
        df['atr'] = atr_vals
        return df

    def _update_daily_equity(self, current_time: datetime, account_balance: float):
        """日内净值更新与熔断重置"""
        current_date = current_time.date()
        if self.last_trade_date != current_date:
            self.day_start_equity = account_balance
            self.last_trade_date = current_date
            self.is_halted_today = False

    def decide(self, df: pd.DataFrame, current_time: datetime, account_balance: float, 
               curr_dir: PositionDir, curr_pos_qty: float) -> TradingAction:
        
        if len(df) < self.entry_period: return TradingAction(ActionType.HOLD)

        # 1. 每日风险审计与熔断机制 (Sync from TurtleBrain)
        self._update_daily_equity(current_time, account_balance)
        if self.is_halted_today:
            return TradingAction(ActionType.HOLD)

        daily_loss_abs = max(0.0, self.day_start_equity - account_balance)
        max_loss_allowed_abs = self.day_start_equity * self.max_daily_loss_pct
        remaining_budget = max_loss_allowed_abs - daily_loss_abs

        if daily_loss_abs >= max_loss_allowed_abs:
            self.is_halted_today = True
            self.logger.warning(f"🚨 [MELTDOWN] 日亏损触及上限! 亏损: {daily_loss_abs/self.day_start_equity:.2f}, 强平离场。| Price: {df['close'].iloc[-1]}")
            self.executor.user_close()
            return TradingAction(ActionType.CLOSE)

        # 2. 状态对齐 (保持 RulesBrain 特有的 Precision Reconciliation)
        df = self._calculate_indicators(df)
        self._check_gaps(df, current_time) # 调用封装的检测函数
        curr_row = df.iloc[-1]
        current_price = curr_row['close']
        atr = curr_row['atr']
        abs_qty = abs(curr_pos_qty)
        
        if abs_qty < 1e-8:
            self.curr_layers, self.layer_sizes, self.last_order_price = 0, [], 0.0
            curr_dir = PositionDir.FLAT
        elif len(self.layer_sizes) > 0:
            matched_count, is_forward = self._find_matched_layers(abs_qty)
            if matched_count > 0:
                new_sizes = self.layer_sizes[:matched_count] if is_forward else self.layer_sizes[-matched_count:]
                self.layer_sizes, self.curr_layers = new_sizes, len(new_sizes)
            else:
                if self.curr_layers != self.max_layers:
                    self.logger.error(f"⚠️ [仓位脱节] 无法匹配层级！实际:{abs_qty:.4f}")
                    self.curr_layers = self.max_layers 

        # 3. 核心数学锚定：Unit 计算与动态止损 (Optimized)
        # 计算理论 Unit (基于风险百分比和 2*ATR)
        # Unit Shares = (Balance * Risk) / (2 * ATR)
        if atr <= 0: return TradingAction(ActionType.HOLD)
        
        raw_unit_shares = (account_balance * self.risk_per_trade) / (2.0 * atr)
        
        # 名义价值约束 (Nominal Value Constraint)
        unit_nominal_pct = (raw_unit_shares * current_price) / account_balance
        if unit_nominal_pct > self.upper_limit:
            unit_nominal_pct = self.upper_limit
        
        # 最终执行股数 (应用 scale)
        final_unit_shares = (unit_nominal_pct * account_balance * self.unit_pct_scale) / current_price
        
        # 预算约束止损 (Budget-Constrained Stop Loss)
        # 计算加仓后预估的总仓位名义价值占比
        target_layers = min(self.curr_layers + 1, self.max_layers)
        total_nominal_val = (unit_nominal_pct * self.unit_pct_scale) * target_layers * account_balance
        
        # 基于剩余日内预算计算出的最大允许止损比例 (0.8 为滑点安全系数)
        max_sl_ratio = (remaining_budget / total_nominal_val) * 0.8 if total_nominal_val > 0 else 0.05
        turtle_sl_ratio = (2.0 * atr) / current_price
        
        # 取两者最小值，确保止损既符合海龟法则，又不突破日内熔断预算
        final_sl_pct = min(turtle_sl_ratio, max_sl_ratio)

        # 4. 出场判断
        if (curr_dir == PositionDir.LONG and current_price < curr_row['exit_low']) or \
           (curr_dir == PositionDir.SHORT and current_price > curr_row['exit_high']):
            self.executor.user_close()
            return TradingAction(ActionType.CLOSE)

        # 5. 进场与加仓判定 (使用优化后的 Unit 和 SL)
        if curr_dir == PositionDir.FLAT:
            is_long = current_price > curr_row['entry_high']
            is_short = current_price < curr_row['entry_low']
            if is_long or is_short:
                self.layer_sizes = [final_unit_shares]
                self.last_order_price, self.curr_layers = current_price, 1
                direction = 'long' if  is_long else 'short'
                self.logger.debug(f"🐢 [ENTRY] SL_Pct: {final_sl_pct:.2%} | {direction} | Shares: {final_unit_shares:.4f}")
                self.executor.user_order(final_unit_shares, is_buy=is_long, stop_loss=final_sl_pct)
                return TradingAction(ActionType.OPEN)

        elif self.curr_layers < self.max_layers:
            threshold = self.pyramid_gap_atr * atr
            if (curr_dir == PositionDir.LONG and current_price > self.last_order_price + threshold) or \
               (curr_dir == PositionDir.SHORT and current_price < self.last_order_price - threshold):
                
                self.layer_sizes.append(final_unit_shares)
                self.last_order_price, self.curr_layers = current_price, len(self.layer_sizes)
                direction = 'long' if  curr_dir == PositionDir.LONG else 'short'
                self.logger.info(f"➕ [PYRAMID] Layer: {self.curr_layers} | SL_Pct: {final_sl_pct:.2%} | {direction} ")
                self.executor.user_order(final_unit_shares, is_buy=(curr_dir == PositionDir.LONG), stop_loss=final_sl_pct)
                return TradingAction(ActionType.PYRAMID)

        return TradingAction(ActionType.HOLD)

    def _find_matched_layers(self, abs_qty):
        """保持原有的双向层级匹配逻辑"""
        if not self.layer_sizes: return 0, True
        cum_forward = 0.0
        for i, size in enumerate(self.layer_sizes):
            cum_forward += size
            if math.isclose(abs_qty, cum_forward, rel_tol=1e-5): return i + 1, True
        cum_backward = 0.0
        for i, size in enumerate(reversed(self.layer_sizes)):
            cum_backward += size
            if math.isclose(abs_qty, cum_backward, rel_tol=1e-5): return i + 1, False
        return 0, True
    
    def _check_gaps(self, df: pd.DataFrame, current_time: datetime):
        """
        检测价格跳空（Price Gap）和时间跳空（Time Gap）
        """
        if len(df) < 2:
            return

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # --- 1. 价格跳空检测 (Price Gap) ---
        current_open = last_row['open']
        prev_close = prev_row['close']
        gap_price_pct = (current_open - prev_close) / prev_close if prev_close > 0 else 0
        
        # 获取时间戳
        # last_row.name 通常是当前 K 线的时间，prev_row.name 是上一根的时间
        curr_time_str = last_row.name.strftime('%Y-%m-%d %H:%M') if hasattr(last_row.name, 'strftime') else str(last_row.name)
        prev_time_str = prev_row.name.strftime('%Y-%m-%d %H:%M') if hasattr(prev_row.name, 'strftime') else str(prev_row.name)

        atr = last_row.get('atr', 0)
        price_gap_threshold = (0.5 * atr / current_open) if current_open > 0 else 0.01

        if abs(gap_price_pct) > price_gap_threshold:
            self.logger.warning(
                f"⚠️ [价格跳空] 出现于 {curr_time_str} | 缺口: {gap_price_pct:.2%} | "
                f"时间跨度: {prev_time_str} -> {curr_time_str} | "
                f"价格跳变: {prev_close:.4f} (前收) -> {current_open:.4f} (今开)"
            )

        if False:
            # --- 2. 时间跳空检测 (Time Gap) ---
            # 定义：当前 K 线时间与前一根 K 线时间之间的间隔是否符合预期（如 4h）
            # 注意：df 的 index 必须是 datetime 类型
            current_ts = last_row.name if isinstance(last_row.name, datetime) else current_time
            prev_ts = prev_row.name
            
            if isinstance(current_ts, datetime) and isinstance(prev_ts, datetime):
                time_delta = current_ts - prev_ts
                # 根据 simulation_typical.py 中的设置，预期间隔通常为 4小时
                expected_delta = pd.Timedelta(hours=4) 
                
                if time_delta > expected_delta:
                    self.logger.error(
                        f"⏰ [时间跳空/数据缺失] {current_ts} 与上一根 K 线相差 {time_delta} "
                        f"(预期: {expected_delta})"
                    )