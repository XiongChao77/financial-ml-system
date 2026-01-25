import backtrader as bt
from trade.bt.bt_executor import BtExecutor
from trade.strategy.strategy_ma import MaCrossoverBrain, MaMarketState
from trade.strategy.strategy_ml import PositionDir
from trade.strategy.strategy_ml import ActionType

class MaCrossoverStrategy(BtExecutor):
    params = dict(
        fast_period=50,
        slow_period=200,
        trade_risk=0.95,
        stop_loss=0.05,  # 5% 固定止损
    )

    def __init__(self):
        super().__init__()
        # 声明指标：Backtrader 会自动处理这些指标的预热
        self.fast_ma = bt.ind.SMA(period=self.params.fast_period)
        self.slow_ma = bt.ind.SMA(period=self.params.slow_period)
        
        self.brain = MaCrossoverBrain(trade_risk=self.params.trade_risk)

    def next(self):
        # 预热期保护
        if len(self) < self.params.slow_period:
            return

        # 同步当前持仓状态
        current_dir = PositionDir.FLAT
        if self.position.size > 0:
            current_dir = PositionDir.LONG
        elif self.position.size < 0:
            current_dir = PositionDir.SHORT

        # 构造状态并交给大脑决策
        state = MaMarketState(
            price=self.data.close[0],
            signal=None, # 均线策略不依赖外部信号列
            pred_prob=0.0,
            position_dir=current_dir,
            layers=1,
            fast_ma=self.fast_ma[0],
            slow_ma=self.slow_ma[0]
        )

        action = self.brain.decide(state)
        self.execute_action(action)

    def execute_action(self, action):
        if action.action == ActionType.HOLD:
            return
        
        # 使用你设计的 user_order_target_percent，透传止损比例
        if action.action in (ActionType.OPEN, ActionType.REVERSE):
            self.user_order_target_percent(
                target_pct=action.target_pct, 
                stop_loss=self.params.stop_loss
            )