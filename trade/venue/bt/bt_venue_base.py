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
        margin_warn_pct = None,  # copied from BrokerConfig by the runner
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
            position=PositionView(dir=position_dir, layers=layers, size=self.position.size, price= self.position.price ),
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
            self.logger.warning(f"⚠️ Risk: margin utilization {margin_level:.2%}")

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
        record.update(extra)
        self.trade_logs.append(record)
        return record

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
        self.logger.info(f"🚩 Backtest complete | Peak margin utilization: {self.max_margin_level:.2%}")

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
                self.logger.warning(f"⚠️ strategy.report() failed; skipping strategy metrics: {e}")
                payload = {}

        strategy_summary, strategy_detail = self._split_metrics(payload)
        summary.update(strategy_summary)
        detail.update(strategy_detail)
        return {'summary': summary, 'detail': detail}

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
        self.logger.info("\n" + "🔍" * 5 + " Label Alignment Audit " + "🔍" * 5)
        for side in ['long', 'short']:
            correct = self.audit_results[f'{side}_correct']
            total = self.audit_results[f'{side}_total']
            acc = (correct / total * 100) if total > 0 else 0
            icon = "📈" if side == 'long' else "📉"
            self.logger.info(f"{icon} {side.upper()} label consistency: {acc:.2f}% ({correct}/{total})")

        total_acc = (self.audit_results['long_correct'] + self.audit_results['short_correct']) / \
                    (max(1, self.audit_results['long_total'] + self.audit_results['short_total']))
        if total_acc < 0.99:
            self.logger.error("🚨 Warning: label consistency is below 99%; the data pipeline may contain an index shift.")
        else:
            self.logger.info("✅ Label alignment check passed.")
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

            if order.info['role'] == 'open':
                type_str = "🚀 ENTRY/ADD"
            else:
                # 2. If exposure is reduced, derive the intent from the order type and the pnl
                if order.exectype == bt.Order.Stop:
                    type_str = "🛡️ STOP LOSS"
                elif order.exectype == bt.Order.Limit:
                    type_str = "🎯 TAKE PROFIT"
                else:
                    # For a market close, tell stop loss from take profit by the pnl
                    # Note: this compares the execution price with the entry average price
                    pnl = (order.executed.price - self.position.price) * self.position.size
                    type_str = "🛑 SIGNAL SL" if pnl < 0 else "🛑 SIGNAL TP"

            direction = "🟢 BUY" if order.isbuy() else "🔴 SELL"
            self.logger.debug(
                f"✅ ORDER FILLED {direction} | Intent: {type_str} | "
                f"Price: {order.executed.price:.4f} | Quantity: {order.executed.size:.2f}"
                f"SL: {order.info['sl_price']}, TP: {order.info['tp_price']}"
            )

        # 3. Order failures
        elif order.status == order.Margin:
            order_price = float(order.created.price)
            order_size = abs(float(order.created.size))
            notional = order_price * order_size
            leverage = max(float(self.leverage), 1e-12)
            required_margin = notional / leverage
            self.logger.error(
                f"❌ ORDER FAILED: insufficient margin | "
                f"Price: {order_price:.4f} | Quantity: {order.created.size:.2f} | "
                f"Notional: {notional:.2f} | Required margin: {required_margin:.2f} | "
                f"cash: {self.broker.getcash():.2f} | balance/value: {self.broker.getvalue():.2f} | "
                f"leverage: {self.leverage:.2f}"
            )
            self.env.runstop()
        elif order.status == order.Rejected:
            self.logger.error("❌ ORDER FAILED: order rejected")
        elif order.status == order.Canceled:
            self.logger.debug(f"⚠️ ORDER CANCELED, Role {order.info['role']}")

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
        Log trade-open and trade-close transitions with their corresponding times.
        """
        direction = "🟢 LONG" if trade.long else "🔴 SHORT"

        if trade.justopened:
            trade_status = "OPEN"
            event_time = bt.num2date(trade.dtopen)
        elif trade.isclosed:
            trade_status = "CLOSE"
            event_time = bt.num2date(trade.dtclose)
            self.closed_pnl.append(trade.pnlcomm)
        else:
            return

        self.logger.debug(
            f"💸 Trade {trade_status} {direction} | "
            f"time: {event_time:%Y-%m-%d %H:%M:%S} | "
            f"price: {trade.price} | Gross PnL: {trade.pnl:.2f} | "
            f"Commission: {trade.commission:.2f} | Net PnL: {trade.pnlcomm:.2f}"
        )

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
