import pandas as pd
import logging
from dataclasses import dataclass
from datetime import datetime
from trade.core.protocol import TradeIntent, ActionType, PositionDir
from trade.core.strategy_base import StrategyBase
from trade.core.venue_base import VenueBase

@dataclass(frozen=True)
class TurtleStrategyConfig:
    entry_period: int = 20
    exit_period: int = 10
    atr_period: int = 20
    max_layers: int = 4
    risk_per_unit: float = 0.01
    max_daily_loss_pct: float = 0.045
    soft_limit_ratio: float = 0.6
    upper_limit: float = 0.7
    unit_pct_scale: float = 0.7


class TurtleStrategy(StrategyBase):
    def __init__(self, venue: VenueBase, config: TurtleStrategyConfig):
        super().__init__(venue)
        self.config = config

        self.curr_layers = 0
        self.day_start_equity = None
        self.last_trade_date = None
        self.is_halted_today = False
        self.logger = logging.getLogger("TurtleStrategy")

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 1. Compute TR (True Range)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 2. Strict Wilder's ATR
        atr_period = self.config.atr_period
        df['atr'] = 0.0
        
        if len(df) >= atr_period:
            # A. Seed value: simple average (SMA) of the first n TRs
            sma_seed = df['tr'].iloc[:atr_period].mean()
            df.loc[df.index[atr_period-1], 'atr'] = sma_seed
            
            # B. Recursion: Current ATR = (Prior ATR * (n-1) + Current TR) / n
            # Pandas' ewm cannot pick up a manually patched seed in the middle of a vectorized run,
            # so we loop to make sure every row builds on the correct previous value
            tr_values = df['tr'].values
            atr_values = df['atr'].values
            for i in range(atr_period, len(df)):
                atr_values[i] = (atr_values[i-1] * (atr_period - 1) + tr_values[i]) / atr_period
            df['atr'] = atr_values

        # 3. Donchian channel
        df['entry_high'] = df['high'].shift(1).rolling(window=self.config.entry_period).max()
        df['entry_low'] = df['low'].shift(1).rolling(window=self.config.entry_period).min()
        df['exit_high'] = df['high'].shift(1).rolling(window=self.config.exit_period).max()
        df['exit_low'] = df['low'].shift(1).rolling(window=self.config.exit_period).min()
        
        return df

    def _update_daily_equity(self, current_time: datetime, account_equity: float):
        current_date = current_time.date()
        if self.last_trade_date != current_date:
            self.day_start_equity = account_equity
            self.last_trade_date = current_date
            self.is_halted_today = False

    def _process(self, df: pd.DataFrame, current_time: datetime, account_equity: float,
               curr_dir: PositionDir, curr_pos_size: float, last_entry_price: float) -> TradeIntent:
        
        if len(df) < self.config.entry_period:
            return TradeIntent(ActionType.NOOP)

        self._update_daily_equity(current_time, account_equity)
        if curr_dir == PositionDir.FLAT:
            self.curr_layers = 0

        if self.is_halted_today:
            return TradeIntent(ActionType.NOOP)

        # 1. Risk budget audit
        daily_loss_abs = max(0, self.day_start_equity - account_equity)
        max_loss_allowed_abs = self.day_start_equity * self.config.max_daily_loss_pct
        remaining_budget = max_loss_allowed_abs - daily_loss_abs

        if daily_loss_abs > max_loss_allowed_abs:
            self.is_halted_today = True
            self.logger.warning(f"🚨 [MELTDOWN] Daily loss limit breached {(daily_loss_abs/self.day_start_equity)*100:.4f}% ! time:{df.iloc[-1].name} Closing all positions.")
            self.venue.close_position()
            return TradeIntent(ActionType.CLOSE) # force a flatten-everything signal
        
        # 2. Indicators
        df = self._calculate_indicators(df)
        self._check_gaps(df, current_time) # call the gap detection helper
        last_row = df.iloc[-1]
        current_price = last_row['close']
        atr = last_row['atr']
        
        # 3. Anchor sizing to the stop loss (fixes the false sense of safety)
        # Estimated total notional share after adding this layer
        target_layers = (self.curr_layers + 1) if self.curr_layers < self.config.max_layers else self.config.max_layers
        
        # Unit size from the risk formula
        unit_size = (account_equity * self.config.risk_per_unit) / atr if atr > 0 else 0
        # unit_size = unit_size*0.2
        unit_pct = (unit_size * current_price) / account_equity
        # 3. Enforce the share cap (e.g. 50%)
        if unit_pct > self.config.upper_limit:
            unit_pct = self.config.upper_limit
        # === key: keep the actual order size in sync ===
        unit_pct = unit_pct* self.config.unit_pct_scale
        unit_size = (unit_pct * account_equity) / current_price
        # unit_pct = unit_pct*0.2
        
        # Target total notional of the position
        total_target_pct = unit_pct * target_layers

        # === core fix: the stop loss was mathematically disconnected ===
        # Formula: stop loss share <= remaining budget / total notional
        # The 0.8 factor covers slippage, so even a stop out keeps today's loss under 4.8%
        max_sl_ratio = (remaining_budget / (total_target_pct * account_equity if total_target_pct > 0 else 1)) * 0.8
        
        turtle_sl = (2.0 * atr) / current_price
        # Take the smaller of the two, safety first
        final_sl_ratio = min(turtle_sl, max_sl_ratio)

        # 4. Decision logic (submit_order places each order on its own)
        action = TradeIntent(ActionType.NOOP)
        # self.logger.info(f"entry_high {last_row['entry_high']}, entry_low: {last_row['entry_low']}, exit_low: {last_row['exit_low']}, exit_high: {last_row['exit_high']}")
        if curr_dir == PositionDir.FLAT:
            if current_price > last_row['entry_high']:
                action = TradeIntent(
                    action=ActionType.OPEN,
                    price=current_price,
                    target_dir=PositionDir.POSITIVE,
                    target_layers=1,
                    target_pct=unit_pct,
                )
            elif current_price < last_row['entry_low']:
                action = TradeIntent(
                    action=ActionType.OPEN,
                    price=current_price,
                    target_dir=PositionDir.NEGATIVE,
                    target_layers=1,
                    target_pct=unit_pct,
                )
        elif self.curr_layers < self.config.max_layers:
            threshold = 0.5 * atr
            if curr_dir == PositionDir.POSITIVE  and current_price > last_entry_price + threshold:
                action = TradeIntent(
                    action=ActionType.PYRAMID,
                    price=current_price,
                    target_dir=PositionDir.POSITIVE,
                    target_layers=target_layers,
                    target_pct=unit_pct,
                )
            elif curr_dir == PositionDir.NEGATIVE and current_price < last_entry_price - threshold:
                action = TradeIntent(
                    action=ActionType.PYRAMID,
                    price=current_price,
                    target_dir=PositionDir.NEGATIVE,
                    target_layers=target_layers,
                    target_pct=unit_pct,
                )

        # Exit
        if (curr_dir == PositionDir.POSITIVE  and current_price < last_row['exit_low']) or \
           (curr_dir == PositionDir.NEGATIVE and current_price > last_row['exit_high']):
            action = TradeIntent(ActionType.CLOSE)

        # 5. Execution: handled by venue.submit_order
        if action.action != ActionType.NOOP:
            if action.action == ActionType.CLOSE:
                self.curr_layers = 0
                self.venue.close_position()
            else:
                self.curr_layers = action.target_layers
                is_buy = action.target_dir == PositionDir.POSITIVE 
                self.logger.info(f"🐢 Order: {action.action} | is_buy: {is_buy} | Layer: {self.curr_layers} | Size_Pct: {unit_pct:.2%} | SL: {final_sl_ratio:.2%}")
                # change: use submit_order instead of target_percent
                self.venue.submit_order(
                    unit_size,
                    is_buy,
                    stop_loss_pct=final_sl_ratio,
                )
        
        return action
    
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
