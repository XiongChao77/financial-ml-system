"""
backtrader execution layer of the restartable martingale.

Same layer as bt_venue_ml.py: BtVenue is the venue (assembles the Observation + places orders), the strategy only decides.
The difference is that the martingale uses no bracket orders (no hard stop, and the take profit follows the average price),
so plain market orders are used here and take profit / safety orders / wipe-out are all decided per bar by the strategy.

Account model: the broker holds a single pool of cash; the reserve account is a "virtual isolation" ledger inside the strategy.
Position sizing may only use the trade account equity, so the reserve can never be eaten by a blow-up.
"""

import backtrader as bt

from trade.venue.bt.bt_venue_base import BtVenue, TradeRole
from trade.core.protocol import OrderType, PositionDir
from trade.strategy.strategy_martingale import RestartableMartingaleStrategy, AccountPhase


class MartingaleBtVenue(BtVenue):
    def __init__(self):
        super().__init__()
        self.dataclose = self.datas[0].close
        self.dir = PositionDir.FLAT
        self.stopped_early = False

        strategy_config = self.p.strategy_config
        self.strategy = RestartableMartingaleStrategy(
            self,
            init_equity=self.p.initial_equity,
            config=strategy_config,
            leverage=self.leverage,
        )

    # ------------------------------------------------------------------
    # Executor interface (overrides BtVenue's bracket version, the martingale only uses market orders)
    # ------------------------------------------------------------------
    def submit_order(
        self,
        size,
        is_buy,
        stop_loss_pct=None,
        take_profit_pct=None,
        *,
        order_type=OrderType.MARKET,
        price=None,
    ):
        order_type, price = self.normalize_order_request(order_type, price)
        size = abs(size)
        if size <= 0:
            return
        order_args = {"size": size}
        if order_type == OrderType.LIMIT:
            order_args.update(price=price, exectype=bt.Order.Limit)
        order = self.buy(**order_args) if is_buy else self.sell(**order_args)
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
            self.logger.error("🛑 [EXIT] Trading and reserve accounts are depleted; stopping the backtest early.")
            self.env.runstop()


    def order_log_extra(self, order):
        """Martingale specific log fields: current layer and average entry price"""
        return {"layer": self.strategy.cycle_layers, "avg_price": self.strategy.cycle_avg_price}

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
