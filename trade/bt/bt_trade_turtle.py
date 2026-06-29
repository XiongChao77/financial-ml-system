import backtrader as bt
import pandas as pd
import logging
from trade.bt.bt_executor import BtExecutor
from trade.strategy.strategy_turtle import TurtleBrain
from trade.strategy.strategy_ml import PositionDir

class TurtleStrategy(BtExecutor):
    params = dict(
        # 海龟系统参数
        entry_period=20,
        exit_period=10,
        atr_period=20,
        max_layers=1,
        risk_per_unit=0.01,
        max_daily_loss_pct = 0.045,
        upper_limit = 0.7,
        unit_pct_scale = 0.7,

    )

    def __init__(self):
        super().__init__()
        # === 核心：初始化全能大脑 ===
        # 将 self (即 BtExecutor) 作为执行器传入
        self.brain = TurtleBrain(
            executor=self, 
            entry_period=self.params.entry_period,
            exit_period=self.params.exit_period,
            atr_period=self.params.atr_period,
            max_layers=self.params.max_layers,
            risk_per_unit=self.params.risk_per_unit,
            max_daily_loss_pct = self.params.max_daily_loss_pct,
            upper_limit = self.params.upper_limit,
            unit_pct_scale = self.params.unit_pct_scale,
        )
        
        # 仅用于审计的变量
        self.max_margin_level = 0.0
        self.atr = bt.ind.ATR(period=self.params.atr_period)

    def next(self):
        """
        每根 K 线触发一次：数据转换 -> 驱动决策
        """
        # 1. 风险审计
        self._audit_margin()
        current_atr = self.atr[0]

        # 2. 将 Backtrader 序列转换为 Pandas DataFrame
        lookback = max(self.params.entry_period, self.params.atr_period * 4) + 10
        if len(self.data) < lookback:
            return
            
        dt_list = [self.data.datetime.datetime(-i) for i in range(lookback-1, -1, -1)]

        df = pd.DataFrame({
            'open': self.data.open.get(size=lookback),
            'high': self.data.high.get(size=lookback),
            'low': self.data.low.get(size=lookback),
            'close': self.data.close.get(size=lookback),
            'volume': self.data.volume.get(size=lookback)
        }, index=dt_list) # 关键：设置时间索引

        # 3. 获取实时环境参数
        current_time = self.data.datetime.datetime(0)
        account_equity = self.broker.getvalue() # 获取当前账户总净值
        current_price = self.data.close[0]

        # --- 新增：计算传递给 BrainBase 的持仓状态 ---
        # A. 确定持仓方向 (PositionDir)
        if self.position.size > 0:
            curr_dir = PositionDir.POSITIVE 
        elif self.position.size < 0:
            curr_dir = PositionDir.NEGATIVE
        else:
            curr_dir = PositionDir.FLAT

        # B. 计算仓位名义价值占比 (curr_pos_size)
        # 公式：(abs(持仓数量) * 当前价格) / 总净值
        pos_value = abs(self.position.size) * current_price
        curr_pos_size = pos_value / account_equity if account_equity > 0 else 0

        # Detect current position state from Backtrader internals
        if not self.position:
            last_entry_price = 0.0
        else:
            if self.live_trades:
                # Get the price of the most recent order added
                last_trade = self.live_trades[-1]
                last_entry_price = last_trade['main'].created.price
            else:
                last_entry_price = current_price # Fallback

        # 4. 驱动决策：传入重构后的 5 个参数
        self.brain.decide(
            df=df, 
            current_time=current_time, 
            account_equity=account_equity,
            curr_dir=curr_dir,
            curr_pos_size=curr_pos_size,
            last_entry_price = last_entry_price,
            # atr = current_atr,
        )


    # ----------------------------------------------------------------
    # 辅助审计逻辑
    # ----------------------------------------------------------------

    def _audit_margin(self):
        equity = self.broker.getvalue()
        pos_value = abs(self.position.size * self.data.close[0])
        leverage = self.broker.getcommissioninfo(self.data).p.leverage
        
        if equity > 0:
            margin_level = (pos_value / leverage) / equity
            self.max_margin_level = max(self.max_margin_level, margin_level)
            if margin_level > 0.8:
                self.logger.warning(f"⚠️ 风险：保证金占用率 {margin_level:.2%}")

    def stop(self):
        self.logger.info(f"🚩 回测结束 | 最大保证金占用: {self.max_margin_level:.2%}")