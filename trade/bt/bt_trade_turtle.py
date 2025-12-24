import backtrader as bt
import logging
from trade.bt.bt_executor import BtExecutor
from trade.strategy.strategy_turtle import TurtleBrain, TurtleMarketState
from trade.strategy.strategy_ftmo import PositionDir
from trade.strategy.strategy_ftmo import ActionType

class TurtleStrategy(BtExecutor):
    params = dict(
        # Turtle System 1 defaults
        entry_period=20,  # Enter on 20-day High
        exit_period=10,   # Exit on 10-day Low
        atr_period=20,
        max_layers=4,
        risk_per_unit=0.01, # 1% Risk
    )

    def __init__(self):
        super().__init__()
        # Entry Channels (e.g., 20 days)
        self.entry_high = bt.ind.Highest(self.data.high(-1), period=self.params.entry_period)
        self.entry_low = bt.ind.Lowest(self.data.low(-1), period=self.params.entry_period)
        
        # Exit Channels (e.g., 10 days)
        self.exit_high = bt.ind.Highest(self.data.high(-1), period=self.params.exit_period)
        self.exit_low = bt.ind.Lowest(self.data.low(-1), period=self.params.exit_period)
        
        self.atr = bt.ind.ATR(period=self.params.atr_period)
        
        # === Brain ===
        self.brain = TurtleBrain(
            max_layers=self.params.max_layers,
            risk_per_unit=self.params.risk_per_unit
        )
        
        # State tracking
        self.dir = PositionDir.FLAT
        self.layers = 0
        self.last_entry_price = 0.0

    def next(self):
        # 1. Update State
        current_price = self.data.close[0]
        current_atr = self.atr[0]
        
        # Detect current position state from Backtrader internals
        if not self.position:
            self.dir = PositionDir.FLAT
            self.layers = 0
            self.last_entry_price = 0.0
        else:
            # Determine direction
            self.dir = PositionDir.LONG if self.position.size > 0 else PositionDir.SHORT
            
            # Determine layers (approximate based on live_trades count)
            # This relies on BtExecutor's live_trades list
            self.layers = len(self.live_trades)
            
            # Determine last entry price
            if self.live_trades:
                # Get the price of the most recent order added
                last_trade = self.live_trades[-1]
                self.last_entry_price = last_trade['main'].created.price
            else:
                self.last_entry_price = current_price # Fallback

        # 2. Dynamic Stop Loss Adjustment for BtExecutor
        # Turtle Stop = 2 * N
        # BtExecutor expects stop_loss as a percentage (e.g., 0.05 for 5%)
        # So: % = (2 * ATR) / Price
        if current_price > 0:
            implied_stop_pct = (2.0 * current_atr) / current_price
            self.params.stop_loss = implied_stop_pct
            # Note: This only affects NEW orders sent by executor._open_bracket
            # It does not update existing stops (which is a limitation of simple implementations)
        # 计算海龟法则的 2N 止损比例
        # Stop Ratio = (2 * ATR) / Price
        current_stop_ratio = (2.0 * current_atr) / current_price if current_price > 0 else 0.05

        # 3. Brain Decision
        state = TurtleMarketState(
            price=current_price,
            high_band=self.exit_high[0],
            low_band=self.exit_low[0], # Use exit band for Longs
            # For Shorts, we use the specific exit bands
            # But to keep state simple, we might need to swap them based on context logic 
            # or pass all bands. Let's pass the relevant Breakout bands for entry:
            atr=current_atr,
            position_dir=self.dir,
            layers=self.layers,
            last_entry_price=self.last_entry_price
        )
        
        # Fix logic for bands passed to brain to support asymmetric exit
        # If flat, we pass Entry bands.
        if self.dir == PositionDir.FLAT:
            # Use Entry 20 bands
            state.high_band = self.entry_high[0] 
            state.low_band = self.entry_low[0]
        elif self.dir == PositionDir.LONG:
            # Use Exit 10 band
            state.low_band = self.exit_low[0]
        elif self.dir == PositionDir.SHORT:
            # Use Exit 10 band
            state.high_band = self.exit_high[0]

        action = self.brain.decide(state)
        
        # 4. Execute
        self.brain_execute(action, current_stop_ratio)

    def brain_execute(self, action, stop_loss_ratio):        
        if action.action == ActionType.HOLD:
            return

        if action.action == ActionType.CLOSE:
            self.logger.info("🐢 Turtle Exit Triggered")
            self.user_close()
            return

        if action.action in (ActionType.OPEN, ActionType.PYRAMID, ActionType.REVERSE):
            self.logger.info(f"🐢 Turtle Action: {action.action} | Target Pct: {action.target_pct:.2%} (ATR Stop: {self.params.stop_loss:.2%})")
            
            # Since BtExecutor.user_order_target_percent handles the math of 
            # "Target - Current = Order Size", we just pass the cumulative target.
            
            if action.target_dir == PositionDir.SHORT:
                # Executor expects negative percent for short
                self.user_order_target_percent(-abs(action.target_pct), stop_loss=stop_loss_ratio)
            else:
                self.user_order_target_percent(abs(action.target_pct), stop_loss=stop_loss_ratio)