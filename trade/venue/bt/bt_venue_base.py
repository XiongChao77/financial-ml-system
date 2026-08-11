from enum import IntEnum
from datetime import timezone
import backtrader as bt
import numpy as np
import logging,sys,os
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", "..", ".."))
# Import project modules
from data_process import common
from trade.core.venue_base import VenueBase
from trade.core.protocol import (
    PositionDir, Signal, Observation, MarketView, PositionView, AccountView,
)


def valid_number(value) -> bool:
    return value is not None and isinstance(value, (int, float, np.number)) and np.isfinite(value)


class TradeRole:
    OPEN = "open"
    STOP_LOSS = "sl"
    TAKE_PROFIT = "tp"
    CLOSE = "close"

class BtVenue(VenueBase,bt.Strategy):
    params = dict(
        strategy_config = None,  # opaque strategy-owned config; subclasses know its concrete type
        initial_equity = None,   # runtime account baseline, not a strategy parameter
        predict_num = None,    # look-ahead horizon of the label alignment audit; None skips the audit
        margin_warn_pct = 0.8,  # margin usage warning threshold
    )

    def __init__(self):
        self.logger = logging.getLogger("trade")
        # === key: keeps every "live" trade group ===
        # shape: [{'id': id, 'stop': stop_ord, 'limit': limit_ord, 'size': size}, ...]
        self.live_trades = []
        self.closed_pnl = []

        # === generic venue level metrics (shared by every strategy) ===
        self.trade_logs = []          # per-order fill log, used by the frontend/report
        self.max_margin_level = 0.0   # peak margin usage
        self.all_preds = []           # predictions the strategy actually saw (input signal quality check)
        self.all_labels = []
        self.audit_results = {
            'long_total': 0, 'long_correct': 0,
            'short_total': 0, 'short_correct': 0,
        }
        self.leverage = self.broker.getcommissioninfo(self.data).p.leverage

    # ================================================================
    # Generic metric collection: subclasses just call collect_bar_metrics() once in next()
    # ================================================================
    def collect_bar_metrics(self):
        """Per-bar generic audit: margin usage + input signal quality + label alignment"""
        self._audit_margin()
        self._collect_prediction()
        self._audit_label_integrity(self.p.predict_num)

    def line_value(self, name, idx=0, default=None):
        """Safely read an optional line off data (not every feed carries atr_pct/label etc.)"""
        line = getattr(self.data, name, None)
        if line is None:
            return default
        try:
            val = line[idx]
        except IndexError:
            return default
        return default if val is None else val

    def current_position_dir(self) -> PositionDir:
        if not self.position:
            return PositionDir.FLAT
        return PositionDir.POSITIVE if self.position.size > 0 else PositionDir.NEGATIVE

    # ================================================================
    # Inbound: assemble the feed's market signals + this venue's account/position into an Observation
    # ================================================================
    def observe(self, layers=None) -> Observation:
        """
        Subclasses call self.observe() in next() to get the strategy layer input.
        When layers is omitted it degrades to "in position means 1 layer"; strategies needing the real count (martingale) pass it explicitly.
        """
        pred = self.line_value("pred")
        pred_prob = self.line_value("pred_prob")
        atr_pct = self.line_value("atr_pct")
        threshold_long = self.line_value("threshold_long")
        threshold_short = self.line_value("threshold_short")
        bars_to_close = self.line_value("bars_to_close")
        position_dir = self.current_position_dir()
        if layers is None:
            layers = 0 if position_dir == PositionDir.FLAT else 1

        market = MarketView(
            price=self.data.close[0],
            open=self.data.open[0],
            high=self.data.high[0],
            low=self.data.low[0],
            close=self.data.close[0],
            signal=Signal.INVALID if pred is None or np.isnan(pred) else Signal(int(pred)),
            pred_prob=0.0 if pred_prob is None or np.isnan(pred_prob) else float(pred_prob),
            atr_pct=0.0 if atr_pct is None or np.isnan(atr_pct) else float(atr_pct),
            threshold_long=(
                None
                if threshold_long is None or np.isnan(threshold_long)
                else float(threshold_long)
            ),
            threshold_short=(
                None
                if threshold_short is None or np.isnan(threshold_short)
                else float(threshold_short)
            ),
            slow_atr=self.line_value("slow_atr"),
            vol_regime=self.line_value("vol_regime"),
            bars_to_close=(
                float("inf")
                if bars_to_close is None or np.isnan(bars_to_close)
                else float(bars_to_close)
            ),
        )
        return Observation(
            market=market,
            position=PositionView(dir=position_dir, layers=layers),
            account=AccountView(equity=self.broker.getvalue()),
            current_time=self.data.datetime.datetime(0),
        )

    def _audit_margin(self):
        equity = self.broker.getvalue()
        pos_value = abs(self.position.size * self.data.close[0])
        if equity <= 0:
            return
        margin_level = (pos_value / self.leverage) / equity
        self.max_margin_level = max(self.max_margin_level, margin_level)
        if margin_level > self.p.margin_warn_pct:
            self.logger.warning(f"⚠️ 风险：保证金占用率 {margin_level:.2%}")

    def _collect_prediction(self):
        """Collect the (pred, label) pairs the strategy really saw, to recompute the input F1 at the end"""
        pred = self.line_value("pred")
        label = self.line_value("label")
        if pred is None or label is None:
            return
        if np.isnan(pred) or np.isnan(label):
            return
        self.all_preds.append(int(pred))
        self.all_labels.append(int(label))

    def _audit_label_integrity(self, lookback):
        """Compare [current price] with [price and label lookback klines ago] to detect index shift"""
        if not lookback or len(self) <= lookback:
            return
        past_label = self.line_value("label", -lookback)
        if past_label is None or np.isnan(past_label):
            return

        past_price = self.data.close[-lookback]
        current_price = self.data.close[0]

        if past_label == common.Signal.POSITIVE:
            self.audit_results['long_total'] += 1
            if current_price > past_price:
                self.audit_results['long_correct'] += 1
        elif past_label == common.Signal.NEGATIVE:
            self.audit_results['short_total'] += 1
            if current_price < past_price:
                self.audit_results['short_correct'] += 1

    def record_order_log(self, order, **extra):
        """Append one fill to trade_logs (uniform fields; sl/tp info of bracket orders comes along)"""
        dt = self.data.datetime.datetime()
        dt_utc = dt.replace(tzinfo=timezone.utc)
        record = {
            "order_ref": order.ref,
            "dt": int(dt_utc.timestamp()),
            "date_utc": str(dt_utc.date()),
            "price": order.executed.price,
            "size": order.executed.size,
            "is_buy": order.isbuy(),
            "execution_type": order.getordername(),
            "role": order.info.get("role", None),
            "sl_pct": order.info.get("sl_pct", None),
            "tp_pct": order.info.get("tp_pct", None),
            "sl_price": order.info.get("sl_price", None),
            "tp_price": order.info.get("tp_price", None),
        }
        record.update(self.order_execution_diagnostics(order))
        record.update(extra)
        self.trade_logs.append(record)
        return record

    def order_execution_diagnostics(self, order) -> dict:
        """Diagnose gap-through stops and same-bar TP/SL ambiguity for bracket exits."""
        role = order.info.get("role", None)
        is_long = order.info.get("is_long", None)
        entry_ref_price = order.info.get("entry_ref_price", None)
        sl_price = order.info.get("sl_price", None)
        tp_price = order.info.get("tp_price", None)
        exec_price = order.executed.price

        bar_open = float(self.data.open[0])
        bar_high = float(self.data.high[0])
        bar_low = float(self.data.low[0])
        bar_close = float(self.data.close[0])
        prev_close = None
        if len(self) > 1:
            try:
                prev_close = float(self.data.close[-1])
            except IndexError:
                prev_close = None

        diagnostics = {
            "bar_index": len(self),
            "bar_open": bar_open,
            "bar_high": bar_high,
            "bar_low": bar_low,
            "bar_close": bar_close,
            "prev_close": prev_close,
            "entry_ref_price": entry_ref_price,
            "is_long": is_long,
            "parent_ref": order.info.get("parent_ref", None),
            "same_bar_tp_sl_hit": False,
            "same_bar_outcome": None,
            "gap_through_stop_pct": None,
            "actual_stop_loss_pct": None,
            "planned_stop_loss_pct": order.info.get("sl_pct", None),
        }
        if prev_close and prev_close > 0:
            diagnostics["open_gap_pct"] = (bar_open - prev_close) / prev_close
        else:
            diagnostics["open_gap_pct"] = None

        if is_long is None or not valid_number(entry_ref_price) or entry_ref_price <= 0:
            return diagnostics

        hit_sl = False
        hit_tp = False
        if valid_number(sl_price) and valid_number(tp_price):
            if is_long:
                hit_sl = bar_low <= sl_price
                hit_tp = bar_high >= tp_price
            else:
                hit_sl = bar_high >= sl_price
                hit_tp = bar_low <= tp_price

        if role in (TradeRole.STOP_LOSS, TradeRole.TAKE_PROFIT):
            diagnostics["same_bar_tp_sl_hit"] = bool(hit_sl and hit_tp)
            if hit_sl and hit_tp:
                diagnostics["same_bar_outcome"] = role

        if role == TradeRole.STOP_LOSS and valid_number(sl_price) and valid_number(exec_price):
            if is_long:
                diagnostics["actual_stop_loss_pct"] = (entry_ref_price - exec_price) / entry_ref_price
                diagnostics["gap_through_stop_pct"] = max(0.0, (sl_price - exec_price) / entry_ref_price)
            else:
                diagnostics["actual_stop_loss_pct"] = (exec_price - entry_ref_price) / entry_ref_price
                diagnostics["gap_through_stop_pct"] = max(0.0, (exec_price - sl_price) / entry_ref_price)
            diagnostics["gap_stop_at_open"] = (
                (bar_open < sl_price) if is_long else (bar_open > sl_price)
            )
        else:
            diagnostics["gap_stop_at_open"] = None

        return diagnostics

    # ================================================================
    # Generic wrap-up: subclasses need not override stop() (if they do, call super().stop())
    # ================================================================
    def stop(self):
        # Let the strategy settle and print its own report first, then the venue side metrics
        strategy = getattr(self, 'strategy', None)
        if strategy is not None and hasattr(strategy, 'finalize'):
            strategy.finalize()
        self.print_signal_quality_report()
        self.print_label_audit_report()
        self.logger.info(
            f"Start Value: {self.broker.startingcash:.2f} | End Value: {self.broker.getvalue():.2f}"
        )
        self.logger.info(f"🚩 回测结束 | 最大保证金占用: {self.max_margin_level:.2%}")

    def print_signal_quality_report(self):
        """Input signal macro-F1 recomputed on the strategy side (matches the training metric, catches bad data)"""
        input_f1 = self.executor_metrics().get('input_f1')
        if input_f1 is None:
            return
        self.logger.info("\n" + "🔍" + "=" * 25 + " Strategy Input Integrity Check " + "=" * 25)
        self.logger.info(f"📊 Final Input Macro-F1 count by strategy: {input_f1:.4f}")
        self.logger.info("=" * 75 + "\n")

    # ================================================================
    # Single metric exit: the layer above (backtest_runner.generate_backtest_report) only knows this interface
    # ================================================================
    def executor_metrics(self) -> dict:
        """Generic venue level metrics (present for every strategy)"""
        metrics = {
            'strategy': type(self).__name__,
            'max_margin_level': float(self.max_margin_level),
        }
        metrics.update(self.order_diagnostics_summary())
        if self.all_preds:
            from sklearn.metrics import f1_score
            metrics['input_f1'] = float(
                f1_score(np.array(self.all_labels), np.array(self.all_preds), average='macro')
            )
        for side in ('long', 'short'):
            total = self.audit_results[f'{side}_total']
            if total:
                metrics[f'label_{side}_consistency'] = self.audit_results[f'{side}_correct'] / total
                metrics[f'label_{side}_count'] = total
        return metrics

    def strategy_metrics(self) -> dict:
        """
        Merge [generic venue metrics] + [strategy specific statistics] into {'summary':..., 'detail':...}.

        summary holds scalars only, so it can go straight into the report and into jsonl;
        detail holds list/dict records (e.g. the martingale death log) and goes to report_additional.
        A strategy only has to implement report() in its own class, nothing above changes.
        """
        summary = self.executor_metrics()
        detail = {}

        strategy = getattr(self, 'strategy', None)
        payload = {}
        if strategy is not None and hasattr(strategy, 'report'):
            try:
                payload = strategy.report() or {}
            except Exception as e:   # a statistics failure must not affect the backtest result
                self.logger.warning(f"⚠️ strategy.report() 失败，跳过策略专属统计: {e}")
                payload = {}

        strategy_summary, strategy_detail = self._split_metrics(payload)
        summary.update(strategy_summary)
        detail.update(self.order_diagnostics_detail())
        detail.update(strategy_detail)
        return {'summary': summary, 'detail': detail}

    def order_diagnostics_summary(self) -> dict:
        exits = [
            item for item in self.trade_logs
            if item.get("role") in (TradeRole.STOP_LOSS, TradeRole.TAKE_PROFIT)
        ]
        stop_exits = [item for item in exits if item.get("role") == TradeRole.STOP_LOSS]
        same_bar = [item for item in exits if item.get("same_bar_tp_sl_hit")]
        gap_stops = [
            item for item in stop_exits
            if (item.get("gap_through_stop_pct") or 0.0) > 0.0
        ]
        stop_slippages = [
            float(item.get("gap_through_stop_pct") or 0.0)
            for item in stop_exits
        ]
        return {
            "exit_order_count": len(exits),
            "stop_loss_order_count": len(stop_exits),
            "take_profit_order_count": sum(1 for item in exits if item.get("role") == TradeRole.TAKE_PROFIT),
            "gap_through_stop_count": len(gap_stops),
            "gap_through_stop_at_open_count": sum(1 for item in gap_stops if item.get("gap_stop_at_open")),
            "max_gap_through_stop_pct": max(stop_slippages) if stop_slippages else 0.0,
            "max_gap_through_stop_date": (
                max(gap_stops, key=lambda item: item.get("gap_through_stop_pct") or 0.0).get("date_utc")
                if gap_stops else None
            ),
            "same_bar_tp_sl_hit_count": len(same_bar),
            "same_bar_tp_sl_as_stop_count": sum(1 for item in same_bar if item.get("role") == TradeRole.STOP_LOSS),
            "same_bar_tp_sl_as_tp_count": sum(1 for item in same_bar if item.get("role") == TradeRole.TAKE_PROFIT),
        }

    def order_diagnostics_detail(self) -> dict:
        gap_stops = [
            item for item in self.trade_logs
            if item.get("role") == TradeRole.STOP_LOSS
            and (item.get("gap_through_stop_pct") or 0.0) > 0.0
        ]
        same_bar = [
            item for item in self.trade_logs
            if item.get("same_bar_tp_sl_hit")
        ]
        gap_stops = sorted(
            gap_stops,
            key=lambda item: item.get("gap_through_stop_pct") or 0.0,
            reverse=True,
        )
        gap_by_day = {}
        for item in gap_stops:
            date_key = item.get("date_utc")
            if date_key is None:
                continue
            bucket = gap_by_day.setdefault(
                date_key,
                {
                    "date": date_key,
                    "count": 0,
                    "at_open_count": 0,
                    "max_gap_through_stop_pct": 0.0,
                    "sum_gap_through_stop_pct": 0.0,
                },
            )
            gap_pct = float(item.get("gap_through_stop_pct") or 0.0)
            bucket["count"] += 1
            bucket["at_open_count"] += 1 if item.get("gap_stop_at_open") else 0
            bucket["sum_gap_through_stop_pct"] += gap_pct
            bucket["max_gap_through_stop_pct"] = max(bucket["max_gap_through_stop_pct"], gap_pct)
        gap_by_day = sorted(
            gap_by_day.values(),
            key=lambda item: (item["max_gap_through_stop_pct"], item["count"]),
            reverse=True,
        )
        return {
            "gap_through_stop_orders": gap_stops[:50],
            "gap_through_stop_by_day": gap_by_day,
            "same_bar_tp_sl_orders": same_bar[:50],
        }

    @staticmethod
    def _split_metrics(payload: dict):
        """Split by "scalar vs detail" automatically; a strategy may also return summary/detail explicitly"""
        if 'summary' in payload or 'detail' in payload:
            return dict(payload.get('summary') or {}), dict(payload.get('detail') or {})

        summary, detail = {}, {}
        for k, v in payload.items():
            if isinstance(v, np.generic):
                v = v.item()
            if v is None or isinstance(v, (int, float, str, bool)):
                summary[k] = v
            else:
                detail[k] = v
        return summary, detail

    def print_label_audit_report(self):
        """Label alignment audit summary"""
        if not (self.audit_results['long_total'] or self.audit_results['short_total']):
            return
        self.logger.info("\n" + "🔍" * 5 + " 数据标签对齐审计 (Integrity Audit) " + "🔍" * 5)
        for side in ['long', 'short']:
            correct = self.audit_results[f'{side}_correct']
            total = self.audit_results[f'{side}_total']
            acc = (correct / total * 100) if total > 0 else 0
            icon = "📈" if side == 'long' else "📉"
            self.logger.info(f"{icon} {side.upper()} Label 一致性: {acc:.2f}% ({correct}/{total})")

        total_acc = (self.audit_results['long_correct'] + self.audit_results['short_correct']) / \
                    (max(1, self.audit_results['long_total'] + self.audit_results['short_total']))
        if total_acc < 0.99:
            self.logger.error("🚨 警告：标签一致性低于 99%！数据处理阶段可能存在 index shift。")
        else:
            self.logger.info("✅ 标签对齐校验通过。")
        self.logger.info("=" * 55 + "\n")

    def submit_order(self, size, is_buy, stop_loss_pct=None, take_profit_pct=None):
        self._open_bracket(abs(size), is_buy=is_buy, stop_loss_pct=stop_loss_pct, take_profit_pct= take_profit_pct)

    def close_position(self, size=None, **kwargs):
        self.logger.debug(f"close_position ammount :{size}")
        current_size = self.position.size
        if size is None or size >= current_size:
            close_order = self.close(**kwargs)
            close_order.addinfo(role=TradeRole.CLOSE, close_type="full")
            self._cancel_all_live_orders() # helper: cancel every pending order
            self.live_trades.clear()
        else:
            raise RuntimeError("reduce not support")
    # ----------------------------------------------------------------
    # Helper logic
    # ----------------------------------------------------------------
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            # --- generic fill log + subclass hook ---
            # Let the subclass update its own state (avg price / layers) first, so the extra fields are post-fill values
            self.on_order_filled(order)
            self.record_order_log(order, **self.order_log_extra(order))

            # --- core: figure out what the order really meant ---

            # 1. Is this an increase in exposure (open or pyramid)?
            # i.e. buying while long or flat, or selling while short or flat
            is_entry = (order.isbuy() and self.position.size >= 0) or \
                       (not order.isbuy() and self.position.size <= 0)

            if is_entry:
                type_str = "🚀 开仓/加仓 (ENTRY)"
            else:
                # 2. If exposure is reduced, derive the intent from the order type and the pnl
                if order.exectype == bt.Order.Stop:
                    type_str = "🛡️ 硬核止损 (STOP LOSS)"
                elif order.exectype == bt.Order.Limit:
                    type_str = "🎯 自动止盈 (TAKE PROFIT)"
                else:
                    # For a market close, tell stop loss from take profit by the pnl
                    # Note: this compares the execution price with the entry average price
                    pnl = (order.executed.price - self.position.price) * self.position.size
                    type_str = "🛑 信号止损 (SIGNAL SL)" if pnl < 0 else "🛑 信号止盈 (SIGNAL TP)"

            direction = "🟢 买入" if order.isbuy() else "🔴 卖出"
            self.logger.debug(
                f"✅ 【订单成交】 {direction} | 意图: {type_str} | "
                f"价格: {order.executed.price:.4f} | 数量: {order.executed.size:.2f}"
            )

        # 3. Order failures
        elif order.status == order.Margin:
            order_price = float(order.created.price)
            order_size = abs(float(order.created.size))
            notional = order_price * order_size
            leverage = max(float(self.leverage), 1e-12)
            required_margin = notional / leverage
            self.logger.error(
                f"❌ 【订单失败】 保证金不足！"
                f"价格: {order_price:.4f} | 数量: {order.created.size:.2f} | "
                f"订单金额: {notional:.2f} | 需要保证金: {required_margin:.2f} | "
                f"cash: {self.broker.getcash():.2f} | balance/value: {self.broker.getvalue():.2f} | "
                f"leverage: {self.leverage:.2f}"
            )
        elif order.status == order.Rejected:
            self.logger.error(f"❌ 【订单失败】 订单被拒绝！")
        elif order.status == order.Canceled:
            self.logger.debug(f"⚠️ 【订单取消】 订单已撤单。")

    # --- subclass extension points: no need to override notify_order ---
    def on_order_filled(self, order):
        """Forward entry fills to strategies that maintain their own position ledger."""
        is_entry = order.info.get("is_entry", False) or (
            order.info.get("role") == TradeRole.OPEN
        )
        strategy = getattr(self, "strategy", None)
        on_fill = getattr(strategy, "on_fill", None)
        if is_entry and callable(on_fill):
            on_fill(
                price=order.executed.price,
                size=order.executed.size,
                is_buy=order.isbuy(),
            )

    def order_log_extra(self, order) -> dict:
        """Subclasses may append custom fields to trade_logs (e.g. the martingale layer/avg_price)"""
        return {}

    def notify_trade(self, trade):
        """
        Trade notification: fires only when a trade closes (whether by take profit or stop loss),
        which is the only place with the real pnl figures.
        """
        # if not trade.isclosed:
        #     return

        # trade.pnl: gross profit (commission excluded)
        # trade.pnlcomm: net profit (commission included)
        # Record the net profit (commission included)
        self.closed_pnl.append(trade.pnlcomm)
        # Print the net pnl including commission
        direction = ( "🟢 多" if trade.size > 0  else "🔴 空" )
        self.logger.debug(f"💸 交易结算 {direction} | price {trade.price} | 毛利: {trade.pnl:.2f} | 手续费: {trade.commission:.2f} | 净利: {trade.pnlcomm:.2f}")

        # Writing the pnl back into trade_logs above is awkward,
        # because trade_logs is per order while this callback is per round trip.
        # Keeping a separate closed_trades list is usually the better option.

    def _open_bracket(self, size, is_buy, stop_loss_pct, take_profit_pct):
        """Place a bracket order and keep the returned orders"""
        price = self.data.close[0]
        args = {}

        if is_buy:
            stop_price = price * (1.0 - stop_loss_pct)
            limit_price = price * (1.0 + take_profit_pct)

            self.logger.debug(
                f"_open_bracket price:{price}, size:{size}, "
                f"stop_price:{stop_price}, limit_price:{limit_price}, "
                f"stop_loss_pct:{stop_loss_pct}"
            )

            orders = self.buy_bracket(
                size=size,
                price=price,
                stopprice=stop_price,
                limitprice=limit_price,
                exectype=bt.Order.Market,
                **args
            )

            main_order = orders[0]
            stop_order = orders[1]
            limit_order = orders[2] if len(orders) > 2 else None

            main_order.addinfo(
                role=TradeRole.OPEN,
                is_long=True,
                entry_ref_price=price,
                sl_price=stop_price,
                tp_price=limit_price,
                sl_pct=stop_loss_pct,
                tp_pct=take_profit_pct,
            )

            stop_order.addinfo(
                role=TradeRole.STOP_LOSS,
                parent_ref=main_order.ref,
                is_long=True,
                entry_ref_price=price,
                sl_price=stop_price,
                tp_price=limit_price,
                sl_pct=stop_loss_pct,
                tp_pct=take_profit_pct,
            )

            if limit_order is not None:
                limit_order.addinfo(
                    role=TradeRole.TAKE_PROFIT,
                    parent_ref=main_order.ref,
                    is_long=True,
                    entry_ref_price=price,
                    sl_price=stop_price,
                    tp_price=limit_price,
                    sl_pct=stop_loss_pct,
                    tp_pct=take_profit_pct,
                )

            self.live_trades.append({
                "main": main_order,
                "stop": stop_order,
                "limit": limit_order,
                "size": size,
            })

        else:
            stop_price = price * (1.0 + stop_loss_pct)
            limit_price = price * (1.0 - take_profit_pct)

            self.logger.debug(
                f"_open_bracket price:{price}, size:{size}, "
                f"stop_price:{stop_price}, limit_price:{limit_price}, "
                f"stop_loss_pct:{stop_loss_pct}"
            )

            orders = self.sell_bracket(
                size=size,
                price=price,
                stopprice=stop_price,
                limitprice=limit_price,
                exectype=bt.Order.Market,
                **args
            )

            main_order = orders[0]
            stop_order = orders[1]
            limit_order = orders[2] if len(orders) > 2 else None

            main_order.addinfo(
                role=TradeRole.OPEN,
                is_long=False,
                entry_ref_price=price,
                sl_price=stop_price,
                tp_price=limit_price,
                sl_pct=stop_loss_pct,
                tp_pct=take_profit_pct,
            )

            stop_order.addinfo(
                role=TradeRole.STOP_LOSS,
                parent_ref=main_order.ref,
                is_long=False,
                entry_ref_price=price,
                sl_price=stop_price,
                tp_price=limit_price,
                sl_pct=stop_loss_pct,
                tp_pct=take_profit_pct,
            )

            if limit_order is not None:
                limit_order.addinfo(
                    role=TradeRole.TAKE_PROFIT,
                    parent_ref=main_order.ref,
                    is_long=False,
                    entry_ref_price=price,
                    sl_price=stop_price,
                    tp_price=limit_price,
                    sl_pct=stop_loss_pct,
                    tp_pct=take_profit_pct,
                )

            self.live_trades.append({
                "main": main_order,
                "stop": stop_order,
                "limit": limit_order,
                "size": size,
            })

    def _cancel_all_live_orders(self):
        """Clean up every pending order before reversing"""
        for trade in self.live_trades:
            if trade['stop']: self.cancel(trade['stop'])
            if trade['limit']: self.cancel(trade['limit'])
