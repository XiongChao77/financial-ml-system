import pandas as pd
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from trade.core.protocol import TradeIntent, ActionType, PositionDir
from trade.core.strategy_base import StrategyBase
from trade.core.venue_base import VenueBase

@dataclass(frozen=True)
class RulesStrategyConfig:
    entry_period: int = 20
    exit_period: int = 10
    atr_period: int = 20
    max_layers: int = 1
    risk_per_trade: float = 0.01
    max_daily_loss_pct: float = 0.035
    unit_pct_scale: float = 1.9
    upper_limit: float = 0.6
    pyramid_gap_atr: float = 0.5


class RulesStrategy(StrategyBase):
    def __init__(
        self,
        venue: VenueBase,
        config: RulesStrategyConfig,
        bar_interval_ms: int,
    ):
        super().__init__(venue, bar_interval_ms)
        self.config = config

        # --- state ---
        self.day_start_equity = None
        self.last_trade_date = None
        self.is_halted_today = False
        self.curr_layers = 0
        self.layer_sizes = []
        self.last_order_price = 0.0
        self.logger = logging.getLogger("RulesStrategy")

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keeps the original computation so ATR stays aligned with the Donchian channel"""
        required = ['atr', 'entry_high', 'entry_low', 'exit_high', 'exit_low']
        if all(col in df.columns for col in required): return df

        df = df.copy()
        df['entry_high'] = df['high'].shift(1).rolling(window=self.config.entry_period).max()
        df['entry_low'] = df['low'].shift(1).rolling(window=self.config.entry_period).min()
        df['exit_high'] = df['high'].shift(1).rolling(window=self.config.exit_period).max()
        df['exit_low'] = df['low'].shift(1).rolling(window=self.config.exit_period).min()
        
        tr = pd.concat([df['high']-df['low'], 
                       (df['high']-df['close'].shift(1)).abs(), 
                       (df['low']-df['close'].shift(1)).abs()], axis=1).max(axis=1)
        
        atr_vals = [0.0] * len(df)
        if len(df) >= self.config.atr_period:
            atr_vals[self.config.atr_period-1] = tr[:self.config.atr_period].mean()
            for i in range(self.config.atr_period, len(df)):
                atr_vals[i] = (atr_vals[i-1] * (self.config.atr_period-1) + tr.iloc[i]) / self.config.atr_period
        df['atr'] = atr_vals
        return df

    def _update_daily_equity(self, current_time: datetime, account_equity: float):
        """Daily equity update and circuit breaker reset"""
        current_date = current_time.date()
        if self.last_trade_date != current_date:
            self.day_start_equity = account_equity
            self.last_trade_date = current_date
            self.is_halted_today = False

    def _process(self, df: pd.DataFrame, current_time: datetime, account_equity: float,
               curr_dir: PositionDir, curr_pos_qty: float) -> TradeIntent:
        
        if len(df) < self.config.entry_period: return TradeIntent(ActionType.NOOP)

        # 1. Daily risk audit and circuit breaker (sync from TurtleStrategy)
        self._update_daily_equity(current_time, account_equity)
        if self.is_halted_today:
            return TradeIntent(ActionType.NOOP)

        daily_loss_abs = max(0.0, self.day_start_equity - account_equity)
        max_loss_allowed_abs = self.day_start_equity * self.config.max_daily_loss_pct
        remaining_budget = max_loss_allowed_abs - daily_loss_abs

        if daily_loss_abs >= max_loss_allowed_abs:
            self.is_halted_today = True
            self.logger.warning(f"🚨 [MELTDOWN] Daily loss limit reached; forced exit | Loss: {daily_loss_abs/self.day_start_equity:.2f} | Price: {df['close'].iloc[-1]}")
            self.venue.close_position()
            return TradeIntent(ActionType.CLOSE)

        # 2. State reconciliation (RulesStrategy specific precision reconciliation)
        df = self._calculate_indicators(df)
        self._check_gaps(df, current_time) # call the gap detection helper
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
                if self.curr_layers != self.config.max_layers:
                    self.logger.error(f"⚠️ [POSITION MISMATCH] Unable to match layer | Actual: {abs_qty:.4f}")
                    self.curr_layers = self.config.max_layers

        # 3. Core anchoring: unit sizing and dynamic stop loss (optimized)
        # Theoretical unit (from the risk percentage and 2*ATR)
        # Unit Shares = (Balance * Risk) / (2 * ATR)
        if atr <= 0: return TradeIntent(ActionType.NOOP)
        
        raw_unit_shares = (account_equity * self.config.risk_per_trade) / (2.0 * atr)
        
        # Nominal value constraint
        unit_nominal_pct = (raw_unit_shares * current_price) / account_equity
        if unit_nominal_pct > self.config.upper_limit:
            unit_nominal_pct = self.config.upper_limit
        
        # Final order size (scale applied)
        final_unit_shares = (unit_nominal_pct * account_equity * self.config.unit_pct_scale) / current_price
        
        # Budget-constrained stop loss
        # Estimated total notional share after adding this layer
        target_layers = min(self.curr_layers + 1, self.config.max_layers)
        total_nominal_val = (unit_nominal_pct * self.config.unit_pct_scale) * target_layers * account_equity
        
        # Largest stop loss share allowed by the remaining daily budget (0.8 = slippage safety factor)
        max_sl_ratio = (remaining_budget / total_nominal_val) * 0.8 if total_nominal_val > 0 else 0.05
        turtle_sl_ratio = (2.0 * atr) / current_price
        
        # Take the smaller of the two so the stop obeys the turtle rule and the daily breaker budget
        final_sl_pct = min(turtle_sl_ratio, max_sl_ratio)

        # 4. Exit check
        if (curr_dir == PositionDir.POSITIVE  and current_price < curr_row['exit_low']) or \
           (curr_dir == PositionDir.NEGATIVE and current_price > curr_row['exit_high']):
            self.venue.close_position()
            return TradeIntent(ActionType.CLOSE)

        # 5. Entry and pyramiding check (with the optimized unit and SL)
        if curr_dir == PositionDir.FLAT:
            is_long = current_price > curr_row['entry_high']
            is_short = current_price < curr_row['entry_low']
            if is_long or is_short:
                self.layer_sizes = [final_unit_shares]
                self.last_order_price, self.curr_layers = current_price, 1
                direction = 'long' if  is_long else 'short'
                self.logger.debug(f"🐢 [ENTRY] SL_Pct: {final_sl_pct:.2%} | {direction} | Shares: {final_unit_shares:.4f}")
                self.venue.submit_order(
                    final_unit_shares,
                    is_buy=is_long,
                    stop_loss_pct=final_sl_pct,
                )
                return TradeIntent(ActionType.OPEN)

        elif self.curr_layers < self.config.max_layers:
            threshold = self.config.pyramid_gap_atr * atr
            if (curr_dir == PositionDir.POSITIVE  and current_price > self.last_order_price + threshold) or \
               (curr_dir == PositionDir.NEGATIVE and current_price < self.last_order_price - threshold):
                
                self.layer_sizes.append(final_unit_shares)
                self.last_order_price, self.curr_layers = current_price, len(self.layer_sizes)
                direction = 'long' if  curr_dir == PositionDir.POSITIVE  else 'short'
                self.logger.info(f"➕ [PYRAMID] Layer: {self.curr_layers} | SL_Pct: {final_sl_pct:.2%} | {direction} ")
                self.venue.submit_order(
                    final_unit_shares,
                    is_buy=(curr_dir == PositionDir.POSITIVE),
                    stop_loss_pct=final_sl_pct,
                )
                return TradeIntent(ActionType.PYRAMID)

        return TradeIntent(ActionType.NOOP)

    def _find_matched_layers(self, abs_qty):
        """Keeps the original two-sided layer matching logic"""
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
        """Detect price gaps; StrategyBase checks timestamp continuity."""
        if len(df) < 2:
            return

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        # --- 1. Price gap detection ---
        current_open = last_row['open']
        prev_close = prev_row['close']
        gap_price_pct = (current_open - prev_close) / prev_close if prev_close > 0 else 0
        
        # Timestamps
        # last_row.name is usually the time of the current kline, prev_row.name the previous one
        curr_time_str = last_row.name.strftime('%Y-%m-%d %H:%M') if hasattr(last_row.name, 'strftime') else str(last_row.name)
        prev_time_str = prev_row.name.strftime('%Y-%m-%d %H:%M') if hasattr(prev_row.name, 'strftime') else str(prev_row.name)

        atr = last_row.get('atr', 0)
        price_gap_threshold = (0.5 * atr / current_open) if current_open > 0 else 0.01

        if abs(gap_price_pct) > price_gap_threshold:
            self.logger.warning(
                f"⚠️ [PRICE GAP] Detected at {curr_time_str} | Gap: {gap_price_pct:.2%} | "
                f"Time span: {prev_time_str} -> {curr_time_str} | "
                f"Price jump: {prev_close:.4f} (previous close) -> {current_open:.4f} (current open)"
            )
