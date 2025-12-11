import backtrader as bt
import logging
from datetime import timezone
import numpy as np
from trade_simulation.strategy.base_strategy import BaseStrategy
# --- Strategy ---
class SimpleStrategy(BaseStrategy):
    params = dict(
        holdbar=1,
        trade_risk=0.6,  # 每次加仓 10% 总资金
        max_layers=1,  # 最大加仓层数
        allow_short=True,
        allow_long=True,
        thresh=None,  # 置信度阈值
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.bar_executed = None
        self.held_bars = 0
        self.dir = 0  # 当前持仓方向: 1(多), -1(空), 0(无)
        self.layers = 0  # 当前加仓层数
        self.trade_logs = []

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                self.logger.debug(
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}"
                )
            elif order.issell():
                self.logger.debug(
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}"
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

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.warning(f"Order Canceled/Margin/Rejected: {order.getstatusname()}")

    def stop(self):
        value = self.broker.getvalue()
        self.logger.record(f"End Value: {value:.2f}")
        # UI
        self.cerebro.trade_logs = self.trade_logs

    def next(self):
        # 获取预测结果
        pred = self.data.pred[0]
        conf = self.data.conf[0]

        # 1. 数据有效性检查
        if np.isnan(pred) or np.isnan(conf):
            return

        # 2. 置信度过滤
        if self.params.thresh is not None and conf < self.params.thresh:
            # 如果置信度不够，我们可以选择“保持不动”或者“视为震荡”
            # 这里简单处理：视为震荡信号(target_dir=0)，如果不持仓则不开，如果持仓则可能平仓
            pred = 1  # 强制视为震荡/观望

        pred = int(pred)

        # 3. 映射信号到方向
        # 假设: 0=空(Short), 1=震荡(Neutral), 2=多(Long)
        target_dir = 0
        if pred == 2 and self.params.allow_long:
            target_dir = 1
        elif pred == 0 and self.params.allow_short:
            target_dir = -1
        else:
            target_dir = 0  # 震荡或不允许的方向

        # 更新持仓时间计数
        if self.position:
            self.held_bars += 1
        else:
            self.held_bars = 0
            self.dir = 0
            self.layers = 0

        # === 4. 核心交易逻辑优化 ===

        # 情况 A: 当前无持仓
        if not self.position:
            if target_dir != 0:
                # 只有明确的多/空信号才开仓
                self.dir = target_dir
                self.layers = 1
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.order_target_percent(target=target_pct)
            return

        # 情况 B: 当前有持仓 (self.dir != 0)

        # B-1: 信号变为震荡 (Label 1) -> 立即平仓
        if target_dir == 0:
            if self.held_bars >= self.params.holdbar:
                self.close()
                self.dir = 0
                self.layers = 0
            return

        # B-2: 信号反转 (多转空 或 空转多) -> 反手 (Reverse)
        if target_dir != self.dir:
            if self.held_bars >= self.params.holdbar:
                # 记录新方向
                self.dir = target_dir
                self.layers = 1  # 重置层数为1
                # 直接计算反向的目标仓位 (例如从 +0.05 变成 -0.05)
                # Backtrader 会自动平掉旧仓位并开新仓位
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.order_target_percent(target=target_pct)
            return

        # B-3: 信号同向 (同向预测) -> 加仓 (Pyramiding)
        if target_dir == self.dir:
            if self.layers < self.params.max_layers:
                self.layers += 1
                target_pct = self.params.trade_risk * self.layers * self.dir
                self.order_target_percent(target=target_pct)
                self.logger.debug(
                    f"Pyramiding: Layer {self.layers}, Target {target_pct:.2%}"
                )
            return

        # 强制最后平仓
        if len(self.data) - 1 == len(self) - 1 and self.position:
            self.close()
