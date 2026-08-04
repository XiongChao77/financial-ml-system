import backtrader as bt
import logging
import numpy as np
from trade.venue.bt.bt_venue import BtVenue
from trade.strategy.strategy_ml import MlSignalStrategy,Observation,TradeIntent,ActionType,PositionDir,Signal
# --- Strategy ---
class MlBtVenue(BtVenue):
    params = dict(
        predict_num = None,
        min_hold_bars=1,
        init_equity = 0,
        risk_per_trade_pct=0.98,  # share of total equity added per layer. 0-1
        max_layers=1,  # maximum number of layers
        allow_short=True,
        allow_long=True,
        prob_thresh=None,  # confidence threshold
        stop_loss_mult = 1,
        atr_sl_long_mult = 3,
        atr_sl_short_mult = 3,
        atr_tp_mult = 5,
        max_daily_loss_pct = 0.99,
        decide_version = 0,
    )

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.bar_executed = None
        self.dir = 0  # current position direction: 1(long), -1(short), 0(flat)
        self.layers = 0  # current number of layers
        self.params.risk_per_trade_pct = self.params.risk_per_trade_pct
        self.strategy = MlSignalStrategy(
            self,
            init_equity = self.params.init_equity,
            risk_per_trade_pct=self.params.risk_per_trade_pct,
            min_hold_bars=self.params.min_hold_bars,
            exist_hold_bars = 0,
            allow_long=self.params.allow_long,
            allow_short=self.params.allow_short,
            prob_thresh=self.params.prob_thresh,
            atr_sl_long_mult = self.params.atr_sl_long_mult,
            atr_sl_short_mult = self.params.atr_sl_short_mult,
            atr_tp_mult = self.params.atr_tp_mult,
            max_daily_loss_pct = self.params.max_daily_loss_pct,
            leverage = self.leverage,
            decide_version = self.params.decide_version,
        )

        self.logger.warning(f"stop_loss is {self.params.stop_loss_mult}")

    def notify_order(self, order):
        # The generic part (fill log / order log) is handled by BtVenue
        super().notify_order(order)
        if order.status == order.Completed:
            self.bar_executed = len(self)
        elif order.status in [order.Margin, order.Rejected]:
            self.logger.warning(f"Order Canceled/Margin/Rejected: {order.getstatusname()}")

    def next(self):
        # Margin usage + input signal collection + label alignment audit (generic venue side)
        self.collect_bar_metrics()

        state = self.observe()
        self.dir = state.position.dir
        self.layers = state.position.layers
        self.strategy.process(state)
