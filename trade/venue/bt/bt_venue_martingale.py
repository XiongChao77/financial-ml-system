"""
backtrader execution layer of the restartable martingale.

Same layer as bt_venue_ml.py: BtVenue is the venue (assembles the Observation + places orders), the strategy only decides.
The difference is that the martingale uses no bracket orders (no hard stop, and the take profit follows the average price),
so plain market orders are used here and take profit / safety orders / wipe-out are all decided per bar by the strategy.

Account model: the broker holds a single pool of cash; the reserve account is a "virtual isolation" ledger inside the strategy.
Position sizing may only use the trade account equity, so the reserve can never be eaten by a blow-up.
"""

from trade.venue.bt.bt_venue import BtVenue, TradeRole
from trade.core.protocol import PositionDir
from trade.strategy.strategy_martingale import RestartableMartingaleStrategy, AccountPhase


class MartingaleBtVenue(BtVenue):
    params = dict(
        init_equity=10000.0,
        # capital isolation
        reserve_pct=0.7,
        restart_capital_pct=0.3,
        min_restart_capital_pct=0.05,
        restart_cost_pct=0.0,
        pause_days=7,               # pause trading for a week after the trade account is wiped out
        # profit sweep
        sweep_trigger_pct=0.10,
        compound_pct=0.0,
        sweep_min_interval_days=0,
        # martingale grid
        base_order_pct=0.02,
        max_safety_orders=8,
        price_deviation_pct=0.01,
        step_mult=1.2,
        volume_mult=1.6,
        tp_pct=0.01,
        atr_grid_mult=None,
        atr_tp_mult=None,
        max_hold_bars=None,
        margin_usage_cap_pct=0.9,
        # death rule
        death_equity_pct=0.2,
        cycle_stop_pct=None,
        # entry
        entry_mode="signal",        # signal / long / short / reversion
        prob_thresh=None,
        allow_long=True,
        allow_short=True,
        # A martingale is heavy by nature, so relax the margin warning threshold (generic venue parameter)
        margin_warn_pct=0.95,
    )

    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.dir = PositionDir.FLAT
        self.stopped_early = False

        self.strategy = RestartableMartingaleStrategy(
            self,
            init_equity=self.params.init_equity,
            reserve_pct=self.params.reserve_pct,
            restart_capital_pct=self.params.restart_capital_pct,
            min_restart_capital_pct=self.params.min_restart_capital_pct,
            restart_cost_pct=self.params.restart_cost_pct,
            pause_days=self.params.pause_days,
            sweep_trigger_pct=self.params.sweep_trigger_pct,
            compound_pct=self.params.compound_pct,
            sweep_min_interval_days=self.params.sweep_min_interval_days,
            base_order_pct=self.params.base_order_pct,
            max_safety_orders=self.params.max_safety_orders,
            price_deviation_pct=self.params.price_deviation_pct,
            step_mult=self.params.step_mult,
            volume_mult=self.params.volume_mult,
            tp_pct=self.params.tp_pct,
            atr_grid_mult=self.params.atr_grid_mult,
            atr_tp_mult=self.params.atr_tp_mult,
            max_hold_bars=self.params.max_hold_bars,
            margin_usage_cap_pct=self.params.margin_usage_cap_pct,
            leverage=self.leverage,
            death_equity_pct=self.params.death_equity_pct,
            cycle_stop_pct=self.params.cycle_stop_pct,
            entry_mode=self.params.entry_mode,
            prob_thresh=self.params.prob_thresh,
            allow_long=self.params.allow_long,
            allow_short=self.params.allow_short,
        )

    # ------------------------------------------------------------------
    # Executor interface (overrides BtVenue's bracket version, the martingale only uses market orders)
    # ------------------------------------------------------------------
    def submit_order(self, size, is_buy, stop_loss_pct=None, take_profit_pct=None):
        size = abs(size)
        if size <= 0:
            return
        order = self.buy(size=size) if is_buy else self.sell(size=size)
        if order is not None:
            order.addinfo(role=TradeRole.OPEN, is_entry=True, is_long=is_buy)

    def close_position(self, size=None, **kwargs):
        if not self.position:
            return
        order = self.close(**kwargs)
        if order is not None:
            order.addinfo(role=TradeRole.CLOSE, is_entry=False, close_type="full")

    def withdraw_cash(self, amount):
        """Restart friction cost: really taken out of the account cash"""
        if amount > 0:
            self.broker.add_cash(-amount)

    def request_halt(self):
        """Both accounts wiped out -> end the backtest"""
        if not self.stopped_early:
            self.stopped_early = True
            self.logger.error("🛑 [EXIT] 交易账户与储备账户均已清零，提前结束回测。")
            self.env.runstop()

    # ------------------------------------------------------------------
    # Orders / fills
    # ------------------------------------------------------------------
    def on_order_filled(self, order):
        """Fill hook: only opens/safety orders update the martingale average price and layer count, closes do not"""
        if order.info.get("is_entry", False):
            self.strategy.on_fill(
                price=order.executed.price,
                size=order.executed.size,
                is_buy=order.isbuy(),
            )

    def order_log_extra(self, order):
        """Martingale specific log fields: current layer and average entry price"""
        return {"layer": self.strategy.cycle_layers, "avg_price": self.strategy.cycle_avg_price}

    def notify_order(self, order):
        super().notify_order(order)
        if order.status in [order.Margin, order.Rejected]:
            self.logger.warning(
                f"❌ 订单失败: {order.getstatusname()} | size={order.created.size:.4f} "
                f"| trade_equity={self.strategy.trade_equity:.2f}"
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def next(self):
        # Margin usage + input signal F1 + label alignment audit (generic venue side)
        self.collect_bar_metrics()

        if self.strategy.phase == AccountPhase.DEAD and not self.position:
            self.request_halt()
            return

        # The martingale layer count is the real number of safety orders it maintains, so pass it explicitly
        state = self.observe(layers=self.strategy.cycle_layers)
        self.dir = state.position.dir
        self.strategy.process(state)

    def stop(self):
        self.martingale_report = self.strategy.report()
        # Strategy settlement + input F1 / label alignment / margin usage / start-end equity are all printed by BtVenue
        super().stop()
