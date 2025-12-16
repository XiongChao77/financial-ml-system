import backtrader as bt
import logging
from datetime import timezone
import numpy as np
from trade.bt.bt_executor import BtExecutor
from trade.strategy.strategy_ftmo import FtmoBrain,MarketState,TradingAction,ActionType,PositionDir,Signal
# --- Strategy ---
class FtmoStrategy(BtExecutor):
    params = dict(
        holdbar=1,
        trade_risk=0.98,  # 每次加仓 10% 总资金. 0-1
        max_layers=1,  # 最大加仓层数
        allow_short=True,
        allow_long=True,
        thresh=None,  # 置信度阈值
        stop_loss = 1
    )

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.bar_executed = None
        self.dir = 0  # 当前持仓方向: 1(多), -1(空), 0(无)
        self.layers = 0  # 当前加仓层数
        self.trade_logs = []
        self.params.trade_risk = self.params.position_ratio * self.params.trade_risk
        self.brain = FtmoBrain(
            self,
            trade_risk=self.params.trade_risk,
            max_layers=self.params.max_layers,
            holdbar=self.params.holdbar,
            allow_long=self.params.allow_long,
            allow_short=self.params.allow_short,
            thresh=self.params.thresh,
        )
        self.logger.warning(f"stop_loss is {self.params.stop_loss}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.exectype == bt.Order.Market:
                order_type_name = "🟢 开仓/平仓 (市价单)"
            elif order.exectype == bt.Order.Limit:
                order_type_name = "💰 止盈成交 (限价单)"
                self.debug_limitation(order)
            elif order.exectype == bt.Order.Stop:
                order_type_name = "🛑 止损成交 (止损单)"
                self.debug_limitation(order)
            elif order.exectype == bt.Order.StopTrail:
                order_type_name = "📉 移动止损成交 (追踪单)"
                self.debug_limitation(order)
            if order.isbuy():
                self.logger.debug(
                    f"BUY EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm {order.executed.comm}"
                )
            elif order.issell():
                self.logger.debug(
                    f"SELL EXECUTED, Price: {order.executed.price}, Cost: {order.executed.value}, Comm {order.executed.comm}"
                )
            self.bar_executed = len(self)

            # 记录交易日志 (修复 UTC 时间戳问题)
            dt = self.data.datetime.datetime()
            dt_utc = dt.replace(tzinfo=timezone.utc)
            record = {
                "dt": int(dt_utc.timestamp()),
                "price": order.executed.price,
                "size": order.executed.size,
                "is_buy": order.isbuy(),
            }
            self.trade_logs.append(record)

        elif order.status in [ order.Margin, order.Rejected]:
            self.logger.warning(f"Order Canceled/Margin/Rejected: {order.getstatusname()}")
        elif order.status == order.Canceled:
            pass

    def debug_limitation(self,order):
        # 1. 忽略未完成的订单状态 (Submitted/Accepted 等)
        if order.status not in [order.Completed]:
            return

        # 2. 判断订单类型
        order_type_name = "未知"
        if order.exectype == bt.Order.Market:
            order_type_name = "🟢 开仓/平仓 (市价单)"
        elif order.exectype == bt.Order.Limit:
            order_type_name = "💰 止盈成交 (限价单)"
        elif order.exectype == bt.Order.Stop:
            order_type_name = "🛑 止损成交 (止损单)"
        elif order.exectype == bt.Order.StopTrail:
            order_type_name = "📉 移动止损成交 (追踪单)"

        # 3. 判断方向
        dir_str = "买入" if order.isbuy() else "卖出"
        
        # 4. 计算滑点 (对于止损单特别重要)
        # order.created.price 是你设定的触发价
        # order.executed.price 是实际撮合的成交价
        slippage = 0.0
        if order.exectype in [bt.Order.Stop, bt.Order.StopTrail]:
            slippage = order.executed.price - order.created.price
            
        # 5. 打印输出
        self.logger.debug(f'>>> debug_limitation {order_type_name} | {dir_str} | 数量: {order.executed.size} | '
                 f'成交价: {order.executed.price:.2f} | '
                 f'设定价: {order.created.price:.2f} | '
                 f'滑点: {slippage:.4f}')

    def stop(self):
        value = self.broker.getvalue()
        self.logger.record(f"Start Value: {self.broker.startingcash:2f} | End Value: {value:.2f}")
        # UI
        self.cerebro.trade_logs = self.trade_logs

    def next(self):
        # 获取预测结果
        pred = self.data.pred[0]
        pred_prob = self.data.pred_prob[0]

        # 1. 数据有效性检查
        if np.isnan(pred) or np.isnan(pred_prob):
            return

        if not self.position:   #sync with stopprice
            self.dir = PositionDir.FLAT
            self.layers = 0

        state = MarketState(
            price=self.data.close[0],
            signal=Signal(pred),
            pred_prob=float(pred_prob),
            position_dir=self.dir,
            layers=self.layers,
        )

        self.brain.decide(state)

        # # 强制最后平仓
        # if len(self.data) - 1 == len(self) - 1 and self.position:
        #     self.close()