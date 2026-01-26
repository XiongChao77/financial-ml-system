import backtrader as bt
import logging
from datetime import timezone
import numpy as np
from trade.bt.bt_executor import BtExecutor
from trade.strategy.strategy_ml import FtmoBrain,MarketState,TradingAction,ActionType,PositionDir,Signal
from data_process import common
# --- Strategy ---
class FtmoStrategy(BtExecutor):
    params = dict(
        holdbar=1,
        trade_risk=0.98,  # 每次加仓 10% 总资金. 0-1
        max_layers=1,  # 最大加仓层数
        allow_short=True,
        allow_long=True,
        thresh=None,  # 置信度阈值
        stop_loss = 1,
        stop_loss_long = 0.05,
        stop_loss_short = 0.05,
        atr_sl_mult_long = 3,
        atr_sl_mult_short = 3,
        max_daily_loss_pct = 0.99,
    )

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.bar_executed = None
        self.dir = 0  # 当前持仓方向: 1(多), -1(空), 0(无)
        self.layers = 0  # 当前加仓层数
        self.trade_logs = []
        #  新增：用于校验的数据容器
        self.all_preds = []
        self.all_labels = []
        self.audit_results = {}
        self.audit_results['long_total'] = 0
        self.audit_results['long_correct'] = 0
        self.audit_results['short_total'] = 0
        self.audit_results['short_correct'] = 0
        self.params.trade_risk = self.params.trade_risk
        self.brain = FtmoBrain(
            self,
            trade_risk=self.params.trade_risk,
            max_layers=self.params.max_layers,
            holdbar=self.params.holdbar,
            allow_long=self.params.allow_long,
            allow_short=self.params.allow_short,
            thresh=self.params.thresh,
            stop_loss_long = self.params.stop_loss_long,
            stop_loss_short = self.params.stop_loss_short,
            atr_sl_mult_long = self.params.atr_sl_mult_long,
            atr_sl_mult_short = self.params.atr_sl_mult_short,
            max_daily_loss_pct = self.params.max_daily_loss_pct,
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
        self.brain.stop()
        value = self.broker.getvalue()
        self.logger.info(f"Start Value: {self.broker.startingcash:2f} | End Value: {value:.2f}")
        #  新增：在回测结束时打印输入准确性校验报告
        if self.all_preds:
            from sklearn.metrics import classification_report, f1_score
            y_true = np.array(self.all_labels)
            y_pred = np.array(self.all_preds)
            
            self.logger.info("\n" + "🔍" + "="*25 + " Strategy Input Integrity Check " + "="*25)
            self.logger.info("\n" + classification_report(y_true, y_pred, digits=4))
            
            input_f1 = f1_score(y_true, y_pred, average='macro')
            self.logger.info(f"📊 Final Input Macro-F1: {input_f1:.4f}")
            self.logger.info("="*75 + "\n")
        self._print_audit_report()
        # UI
        self.cerebro.trade_logs = self.trade_logs

    def next(self):
        # 获取预测结果
        pred = self.data.pred[0]
        pred_prob = self.data.pred_prob[0]
        label = self.data.label[0]

        self._audit_label_integrity(lookback=common.PREDICT_NUM)
        #  新增：收集非空数据用于校验
        if not np.isnan(pred) and not np.isnan(label):
            self.all_preds.append(int(pred))
            self.all_labels.append(int(label))

        # 1. 数据有效性检查
        current_signal = Signal.INVALID if np.isnan(pred) else Signal(int(pred))
        current_prob = 0.0 if np.isnan(pred_prob) else float(pred_prob)

        self.dir = PositionDir.FLAT
        if not self.position:   #sync with stopprice
            self.dir = PositionDir.FLAT
            self.layers = 0
        elif self.position.size > 0:
            self.dir = PositionDir.LONG
            self.layers = 1
        elif self.position.size < 0:
            self.dir = PositionDir.SHORT
            self.layers = 1

        state = MarketState(
            price=self.data.close[0],
            signal=current_signal,
            pred_prob=float(current_prob),
            position_dir=self.dir,
            layers=self.layers,
            current_time= self.data.datetime.datetime(0),
            account_balance=self.broker.getvalue(),
            atr=self.data.atr[0] if hasattr(self.data, 'atr') else 0.0,
            slow_atr = self.data.slow_atr[0] if hasattr(self.data, 'slow_atr') else 0.0,
            vol_regime = self.data.vol_regime[0] if hasattr(self.data, 'vol_regime') else None,
        )

        self.brain.decide(state)

        # # 强制最后平仓
        # if len(self.data) - 1 == len(self) - 1 and self.position:
        #     self.close()

    def _audit_label_integrity(self, lookback=common.CANDLESTICK_NUM):
        """
        封装的校验函数：对比 [当前价格] 与 [lookback 根 K 线前的价格及标签]
        """
        if len(self) <= lookback:
            return

        past_label = self.data.label[-lookback]
        past_price = self.data.close[-lookback]
        current_price = self.data.close[0]

        # 校验做多标签 (Label 2)
        if past_label == common.Signal.LONG:
            self.audit_results['long_total'] += 1
            if current_price > past_price:
                self.audit_results['long_correct'] += 1
        
        # 校验做空标签 (Label 0)
        elif past_label == common.Signal.SHORT:
            self.audit_results['short_total'] += 1
            if current_price < past_price:
                self.audit_results['short_correct'] += 1

    def _print_audit_report(self):
        """打印审计总结"""
        self.logger.info("\n" + "🔍" * 5 + " 数据标签对齐审计 (Integrity Audit) " + "🔍" * 5)
        
        for side in ['long', 'short']:
            correct = self.audit_results[f'{side}_correct']
            total = self.audit_results[f'{side}_total']
            acc = (correct / total * 100) if total > 0 else 0
            icon = "📈" if side == 'long' else "📉"
            self.logger.info(f"{icon} {side.upper()} Label 一致性: {acc:.2f}% ({correct}/{total})")
        
        # 深度诊断建议
        total_acc = (self.audit_results['long_correct'] + self.audit_results['short_correct']) / \
                    (max(1, self.audit_results['long_total'] + self.audit_results['short_total']))
        
        if total_acc < 0.99:
            self.logger.error("🚨 警告：标签一致性低于 99%！数据处理阶段可能存在 index shift。")
        else:
            self.logger.info("✅ 标签对齐校验通过。")
        self.logger.info("=" * 55 + "\n")           