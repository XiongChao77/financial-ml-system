import backtrader as bt
from trade.bt.bt_executor import BtExecutor
from trade.strategy.strategy_triple_screen import TripleScreenBrain, TripleMarketState
from trade.strategy.strategy_ftmo import PositionDir
from trade.strategy.strategy_ftmo import ActionType

class TripleScreenStrategy(BtExecutor):
    params = dict(
        macd_p1=12, macd_p2=26, macd_p3=9, # 大周期指标
        rsi_period=14, rsi_low=30, rsi_high=70, # 中周期指标
        stop_loss=0.03
    )

    def __init__(self):
        super().__init__()
        # Data0 是 1h (中周期), Data1 是 12h (大周期)
        self.d_long = self.datas[1] 
        self.d_mid = self.datas[0]

        # 第一层：大周期趋势 (MACD 柱状图斜率)
        self.macd = bt.ind.MACDHisto(self.d_long, period_me1=self.params.macd_p1, 
                                     period_me2=self.params.macd_p2, period_signal=self.params.macd_p3)
        
        # 第二层：中周期波段 (RSI)
        self.rsi = bt.ind.RSI(self.d_mid, period=self.params.rsi_period)
        
        self.brain = TripleScreenBrain()

    def next(self):
        # 同步大周期趋势方向 (Screen 1)
        tide_dir = PositionDir.FLAT
        if self.macd[0] > self.macd[-1]: # 柱状图上升
            tide_dir = PositionDir.LONG
        elif self.macd[0] < self.macd[-1]: # 柱状图下降
            tide_dir = PositionDir.SHORT

        # 同步中周期回调状态 (Screen 2)
        is_wave_pullback = False
        if tide_dir == PositionDir.LONG and self.rsi[0] < self.params.rsi_low:
            is_wave_pullback = True
        elif tide_dir == PositionDir.SHORT and self.rsi[0] > self.params.rsi_high:
            is_wave_pullback = True

        state = TripleMarketState(
            price=self.d_mid.close[0],
            signal=None, pred_prob=0.0,
            position_dir=PositionDir.LONG if self.position.size > 0 else (PositionDir.SHORT if self.position.size < 0 else PositionDir.FLAT),
            layers=1,
            tide_dir=tide_dir,
            is_wave_pullback=is_wave_pullback,
            prev_high=self.d_mid.high[-1],
            prev_low=self.d_mid.low[-1]
        )

        action = self.brain.decide(state)
        # 执行逻辑透传 stop_loss
        if action.action != ActionType.HOLD:
            self.user_order_target_percent(action.target_pct, stop_loss=self.params.stop_loss)