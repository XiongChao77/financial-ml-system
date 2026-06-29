import backtrader as bt
import pandas as pd
from trade.bt.bt_executor import BtExecutor
from trade.strategy.strategy_rules import RulesBrain
from trade.strategy.strategy_ml import PositionDir

class RulesStrategy(BtExecutor):
    def __init__(self):
        super().__init__()

        # 3. 初始化全能大脑
        self.brain = RulesBrain(
            executor=self,
        )
        # 1. 定义海龟策略参数 (需与 RulesBrain 同步)
        self.entry_period = self.brain.entry_period
        self.exit_period = self.brain.exit_period
        self.atr_period = self.brain.atr_period

        # 2. 在 init 中预创建 Backtrader 指标 (会被向量化加速计算)
        # 注意：使用 (-1) 偏移量来模拟 shift(1)，防止未来函数
        self.entry_high = bt.indicators.Highest(self.data.high(-1), period=self.entry_period)
        self.entry_low = bt.indicators.Lowest(self.data.low(-1), period=self.entry_period)
        self.exit_high = bt.indicators.Highest(self.data.high(-1), period=self.exit_period)
        self.exit_low = bt.indicators.Lowest(self.data.low(-1), period=self.exit_period)
        
        # 标准海龟使用 Wilder's ATR，Backtrader 默认为平滑移动平均
        self.atr = bt.indicators.ATR(self.data, period=self.atr_period)

        self.max_margin_level = 0.0

    def next(self):
        # 1. 风险审计
        self._audit_margin()

        # 2. 动态确定回溯长度
        # 至少需要 2 行来支持跳空检测；如果需要 Brain 重新计算指标，则需要 entry_period + 1
        lookback = self.brain.entry_period*10
        
        # 确保当前已有足够的数据点，否则跳过
        if len(self.data) < lookback:
            return

        # 获取历史切片数据
        # Backtrader 的 get(size=N) 返回的是从当前点往前的 N 个值的列表
        times = [self.data.datetime.datetime(-i) for i in range(lookback)][::-1]
        
        df = pd.DataFrame({
            'open': self.data.open.get(size=lookback),
            'high': self.data.high.get(size=lookback),
            'low': self.data.low.get(size=lookback),
            'close': self.data.close.get(size=lookback),
            
            # 同样获取预计算指标的历史值
            'atr': self.atr.get(size=lookback),
            'entry_high': self.entry_high.get(size=lookback),
            'entry_low': self.entry_low.get(size=lookback),
            'exit_high': self.exit_high.get(size=lookback),
            'exit_low': self.exit_low.get(size=lookback)
        }, index=times)

        # 3. 获取实时持仓状态
        if self.position.size > 0:
            curr_dir = PositionDir.POSITIVE 
        elif self.position.size < 0:
            curr_dir = PositionDir.NEGATIVE
        else:
            curr_dir = PositionDir.FLAT

        # 4. 驱动决策
        # 注意：现在传入的 df 长度为 lookback，满足了 iloc[-1] 和 iloc[-2] 的需求
        current_time = times[-1]
        self.brain.decide(
            df=df, 
            current_time=current_time, 
            account_equity=self.broker.getvalue(),
            curr_dir=curr_dir,
            curr_pos_qty=self.position.size,
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