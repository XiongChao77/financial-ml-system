import pandas as pd
import logging
from datetime import datetime
from trade.strategy.strategy_ftmo import Brain, TradingAction, ActionType, PositionDir
from trade.strategy.base_executor import BaseExecutor

class TurtleBrain(Brain):
    def __init__(
        self, 
        executor: BaseExecutor,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        max_layers: int = 4,
        risk_per_unit: float = 0.01,
        max_daily_loss_pct: float = 0.045, # 修复：确保属性在初始化时被赋值
        soft_limit_ratio: float = 0.6,
        upper_limit : float = 0.7,
        unit_pct_scale:float = 0.7
    ):
        self.executor = executor
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.max_layers = max_layers
        self.risk_per_unit = risk_per_unit
        self.upper_limit = upper_limit
        self.unit_pct_scale = unit_pct_scale
        
        # 核心风控参数
        self.max_daily_loss_pct = max_daily_loss_pct # 修复：必须明确赋值
        self.soft_limit_ratio = soft_limit_ratio
        
        self.curr_layers = 0 
        self.day_start_equity = None
        self.last_trade_date = None
        self.is_halted_today = False 
        
        self.logger = logging.getLogger("TurtleBrain")

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 1. 计算 TR (True Range)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 2. 严格的 Wilder's ATR 计算逻辑
        atr_period = self.atr_period
        df['atr'] = 0.0
        
        if len(df) >= atr_period:
            # A. 计算初始种子：前 n 个 TR 的简单平均值 (SMA)
            sma_seed = df['tr'].iloc[:atr_period].mean()
            df.loc[df.index[atr_period-1], 'atr'] = sma_seed
            
            # B. 执行递归计算：Current ATR = (Prior ATR * (n-1) + Current TR) / n
            # 由于 Pandas 的 ewm 无法在向量化计算中途识别手动修改的种子点
            # 我们使用循环来确保每一行都基于上一行的正确值
            tr_values = df['tr'].values
            atr_values = df['atr'].values
            for i in range(atr_period, len(df)):
                atr_values[i] = (atr_values[i-1] * (atr_period - 1) + tr_values[i]) / atr_period
            df['atr'] = atr_values

        # 3. 计算唐奇安通道
        df['entry_high'] = df['high'].shift(1).rolling(window=self.entry_period).max()
        df['entry_low'] = df['low'].shift(1).rolling(window=self.entry_period).min()
        df['exit_high'] = df['high'].shift(1).rolling(window=self.exit_period).max()
        df['exit_low'] = df['low'].shift(1).rolling(window=self.exit_period).min()
        
        return df

    def _update_daily_equity(self, current_time: datetime, account_balance: float):
        current_date = current_time.date()
        if self.last_trade_date != current_date:
            self.day_start_equity = account_balance
            self.last_trade_date = current_date
            self.is_halted_today = False

    def decide(self, df: pd.DataFrame, current_time: datetime, account_balance: float, 
               curr_dir: PositionDir, curr_pos_size: float, last_entry_price: float) -> TradingAction:
        
        if len(df) < self.entry_period:
            return TradingAction(ActionType.HOLD)

        self._update_daily_equity(current_time, account_balance)
        if curr_dir == PositionDir.FLAT:
            self.curr_layers = 0

        if self.is_halted_today:
            return TradingAction(ActionType.HOLD)

        # 1. 风险预算审计
        daily_loss_abs = max(0, self.day_start_equity - account_balance)
        max_loss_allowed_abs = self.day_start_equity * self.max_daily_loss_pct
        remaining_budget = max_loss_allowed_abs - daily_loss_abs

        if daily_loss_abs > max_loss_allowed_abs:
            self.is_halted_today = True
            self.logger.warning(f"🚨 [MELTDOWN] Daily loss limit breached {(daily_loss_abs/self.day_start_equity)*100:.4f}% ! time:{df.iloc[-1].name} Closing all positions.")
            self.executor.user_close()
            return TradingAction(ActionType.CLOSE) # 强制返回全平信号
        
        # 2. 指标计算
        df = self._calculate_indicators(df)
        self._check_gaps(df, current_time) # 调用封装的检测函数
        last_row = df.iloc[-1]
        current_price = last_row['close']
        atr = last_row['atr']
        
        # 3. 仓位与止损的数学锚定 (修复虚假安全)
        # 预估加仓后的总名义价值占比
        target_layers = (self.curr_layers + 1) if self.curr_layers < self.max_layers else self.max_layers
        
        # 根据风险公式计算单笔 Unit 大小
        unit_size = (account_balance * self.risk_per_unit) / atr if atr > 0 else 0
        # unit_size = unit_size*0.2
        unit_pct = (unit_size * current_price) / account_balance
        # 3. 强制约束占比（如 50% 上限）
        if unit_pct > self.upper_limit:
            unit_pct = self.upper_limit
        # === 关键：同步更新实际下单的股数 ===
        unit_pct = unit_pct* self.unit_pct_scale
        unit_size = (unit_pct * account_balance) / current_price
        # unit_pct = unit_pct*0.2
        
        # 目标总仓位名义价值
        total_target_pct = unit_pct * target_layers

        # === 核心修正：止损逻辑数学脱节 ===
        # 公式：止损比例 <= 剩余预算 / 总名义价值
        # 增加 0.8 系数应对滑点，确保即使触发止损，今日亏损也不超 4.8%
        max_sl_ratio = (remaining_budget / (total_target_pct * account_balance if total_target_pct > 0 else 1)) * 0.8
        
        turtle_sl = (2.0 * atr) / current_price
        # 取两者最小值，确保安全第一
        final_sl_ratio = min(turtle_sl, max_sl_ratio)

        # 4. 决策逻辑 (使用 user_order 独立下单)
        action = TradingAction(ActionType.HOLD)
        # self.logger.info(f"entry_high {last_row['entry_high']}, entry_low: {last_row['entry_low']}, exit_low: {last_row['exit_low']}, exit_high: {last_row['exit_high']}")
        if curr_dir == PositionDir.FLAT:
            if current_price > last_row['entry_high']:
                action = TradingAction(ActionType.OPEN, PositionDir.LONG, 1, unit_pct)
            elif current_price < last_row['entry_low']:
                action = TradingAction(ActionType.OPEN, PositionDir.SHORT, 1, unit_pct)
        elif self.curr_layers < self.max_layers:
            threshold = 0.5 * atr
            if curr_dir == PositionDir.LONG and current_price > last_entry_price + threshold:
                action = TradingAction(ActionType.PYRAMID, PositionDir.LONG, target_layers, unit_pct)
            elif curr_dir == PositionDir.SHORT and current_price < last_entry_price - threshold:
                action = TradingAction(ActionType.PYRAMID, PositionDir.SHORT, target_layers, unit_pct)

        # 出场
        if (curr_dir == PositionDir.LONG and current_price < last_row['exit_low']) or \
           (curr_dir == PositionDir.SHORT and current_price > last_row['exit_high']):
            action = TradingAction(ActionType.CLOSE)

        # 5. 执行：由 executor.user_order 处理
        if action.action != ActionType.HOLD:
            if action.action == ActionType.CLOSE:
                self.curr_layers = 0
                self.executor.user_close()
            else:
                self.curr_layers = action.target_layers
                is_buy = action.target_dir == PositionDir.LONG
                self.logger.info(f"🐢 Order: {action.action} | is_buy: {is_buy} | Layer: {self.curr_layers} | Size_Pct: {unit_pct:.2%} | SL: {final_sl_ratio:.2%}")
                # 改造点：使用 user_order 替代 target_percent
                self.executor.user_order(unit_size, is_buy, stop_loss=final_sl_ratio)
        
        return action
    
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