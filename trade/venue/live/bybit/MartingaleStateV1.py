import time, threading, math, logging, sys, os, signal
from dataclasses import dataclass, field
from enum import Enum
import argparse
from typing import Dict
from bybit_engine import BybitEngine

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", "..", "..", ".."))
from data_process.common import setup_session_logger
from collections import OrderedDict


class OrderLabel:
    ENTRY = "Entry"  # base order / initial position
    SO = "SO"  # safety order
    TP = "TP"  # take profit order
    SL = "SL"  # stop loss order
    MARKET = "MARKET"  # market order / choppy mode handling
    CANCEL_ALL = "CANCEL_ALL"


class MarketState(Enum):
    OSCILLATION = "OSCILLATION"  # choppy
    TREND_UP = "TREND_UP"  # one-way up
    TREND_DOWN = "TREND_DOWN"  # one-way down


@dataclass
class SymbolConfig:
    """Martingale parameters for one symbol"""

    symbol: str
    budget_pct: float  # total budget (percentage)
    base_gap: float  # base grid spacing (0.005 = 0.5%)
    trend_bias: float  # trend bias (0.002 = 0.2%). How far the base order leans to one side, must be below base_gap
    max_layers: int  # maximum number of safety orders
    volume_mult: float  # size multiplier (martingale)
    step_mult: float  # spacing multiplier
    profit_target: float  # fixed take profit percentage (0.005 = 0.5%)
    # this profit margin is kept no matter which layer we are on
    # The parameters below are overwritten dynamically by the exchange
    qty_step: float  # quantity precision (exchange limit)
    tick_size: float  # price precision (exchange limit)
    min_qty: float  # minimum order size
    #  fee parameters (filled in from the API)
    maker_fee: float = 0.0002  # default 0.02%
    taker_fee: float = 0.00055  # default 0.055%
    # Core matrix: the factors of every layer
    # shape: {layer_int: {"p_factor": f, "q_factor": f, "avg_p_factor": f, ...}}
    matrix: list = field(default_factory=list)

    # A few convenience helpers can live on the class
    def get_max_so_gap(self, layer: int) -> float:
        """Percentage distance of layer N relative to the average price"""
        return self.base_gap * (self.step_mult ** (max(0, layer - 1)))

    @property
    def estimated_round_trip_fee(self):
        # Fees of the main operations (pure maker mode)
        return self.maker_fee * 2

    def build_matrix(self, is_long: bool):
        self.matrix = []
        cum_q_factor = 0.0
        cum_notional_factor = 0.0
        cum_gap = 0.0

        #  Two-sided fee cost factor
        # cost = opening fee + closing fee
        fee_cost_pct = self.maker_fee + self.maker_fee

        for i in range(1, self.max_layers + 1):
            if i > 1:
                cum_gap += self.base_gap * (self.step_mult ** (i - 2))

            p_factor = (1 - cum_gap) if is_long else (1 + cum_gap)
            q_factor = self.volume_mult ** (i - 1)

            cum_q_factor += q_factor
            cum_notional_factor += q_factor * p_factor
            avg_p_factor = cum_notional_factor / cum_q_factor

            #  "Theoretical net profit factor" of this layer
            # Logic: (avg price * (1 + profit target)) - (avg price * (1 + fee cost))
            net_profit_factor = self.profit_target - fee_cost_pct

            self.matrix.append(
                {
                    "layer": i,
                    "p_factor": p_factor,
                    "q_factor": q_factor,
                    "avg_p_factor": avg_p_factor,
                    "cum_q_factor": cum_q_factor,
                    "net_profit_factor": net_profit_factor,  # stored in the matrix, later read straight from the table
                }
            )


# ================= core state management =================
class SymbolState:
    def __init__(self, symbol: str, conf):
        self.symbol = symbol
        self.conf: SymbolConfig = conf
        self.lock = threading.RLock()
        self.last_order_ts = int(time.time() * 1000)
        # print(f"{sys._getframe().f_lineno} {time.time()} last_order_ts {self.last_order_ts}")

        #  Core: signed position (positive = long, negative = short, 0 = flat)
        self.signed_pos_qty = 0.0
        self.avg_entry_price = 0.0
        self.base_price = 0
        self.base_qty = 0

        # State flags
        self.layer_count = 0  # which safety order layer we are on
        self.trend_score = 0.0  # Z-score trend score (positive = up, negative = down)
        self.order_submit_sl = False
        # running status
        self.last_result = 0  # -1:SL, 1:TP
        self.loss_count = 0
        self.last_result_updte = time.time()
        self.market_state = MarketState.OSCILLATION
        # 0: no trend, 1: up, -1: down
        self.trend_direction = 0
        # Task debounce lock
        self.is_processing = False
        self.last_processing_time = 0

        # Statistics
        self.total_profit = 0.0
        self.total_fees = 0.0
        self.tp_count = 0  #  number of take profits so far
        self.sl_count = 0  #  number of stop losses so far
        self.tp_layer_dist = {}  # e.g. {1: 15, 2: 5, 3: 1}


class MartingaleBot:
    def __init__(self, engine: BybitEngine, configs: list, is_long_account):
        self.version = "V1"  # martingale grid
        self.logger, _ = setup_session_logger(sub_folder=self.__class__.__name__ + self.version, console_level=logging.INFO, file_level=logging.INFO)
        self.engine = engine
        self.configs: list[SymbolConfig] = configs
        self.markets: Dict[str, SymbolState] = {}
        self.total_equity = 0.0
        self.init_total_equity = -1
        self.max_loss_ratio = 0.2
        self.stop_signal = False
        self.is_long_account = is_long_account
        self.leverage = 10
        self.setup()

    def setup(self):
        self.logger.info("🚀 Starting V10 strategy...")
        self.update_wallet_balance()

        for cfg in self.configs:
            # 1. Sync the exchange precision
            info = self.engine.get_symbol_info(cfg.symbol)  # assumes you added this helper to the engine
            if info:
                cfg.tick_size, cfg.qty_step, cfg.min_qty = info["tick_size"], info["qty_step"], info["min_qty"]
            # 2.  Sync the real fee rates
            maker, taker = self.engine.get_real_fee_rate(cfg.symbol)
            cfg.maker_fee = maker
            cfg.taker_fee = taker
            cfg.build_matrix(is_long=self.is_long_account)
            self.logger.info(
                f"💎 [{cfg.symbol}] info update: Maker {maker:.4%} | Taker {taker:.4%} | tick_size {cfg.tick_size} | qty_step {cfg.qty_step} | min_qty {cfg.min_qty} "
            )
            self.check_profit_viability(cfg)
            # 2. Initialize the environment
            # self.engine.http.switch_position_mode(category="linear", symbol=cfg.symbol, mode=0)
            self.engine.set_leverage(cfg.symbol, self.leverage)
            self.markets[cfg.symbol] = SymbolState(cfg.symbol, cfg)

        self.initial_risk_report()

        self.engine.start_stream(self.on_ws_order_notify)
        for symbol in self.markets.keys():
            self.engine.market_close_all(symbol)

    def check_profit_viability(self, cfg: SymbolConfig):
        """
        Start-up self check: make sure the take profit target covers the real maker fee
        """
        # The martingale grid flow is: maker buy -> maker sell
        # cost = opening fee + closing fee
        cost = cfg.maker_fee * 2

        # Net profit margin
        net_margin = cfg.profit_target - cost

        # Safety threshold: at least 0.2% net profit, or twice the fee
        if net_margin < 0.001:
            self.logger.warning("=" * 60)
            self.logger.warning(f"⚠️ [{cfg.symbol}] Profit margin is too narrow | Estimated net margin: {net_margin:.2%} (may be negligible after slippage)")
            # self.logger.warning(f"   - configured take profit: {cfg.profit_target:.2%}")
            # self.logger.warning(f"   - real maker fee: {cfg.maker_fee:.4%} (both sides: {cost:.4%})")
            # self.logger.warning(f"   - expected net profit: {net_margin:.2%} (slippage may leave close to nothing)")
            # self.logger.warning("   -> suggestion: raise profit_target or upgrade the VIP tier")
            self.logger.warning("=" * 60)
        else:
            self.logger.info(f"✅ [{cfg.symbol}] Profit model is healthy | Net margin: {net_margin:.2%}")

    def initial_risk_report(self):
        """
        🛡️ Extended risk report - adds fee drag and net profit analysis
        """
        for m in self.markets.values():
            with m.lock:
                p_ref = self.get_last_price(m.symbol)
                if p_ref <= 0:
                    continue

                side = "Long" if self.is_long_account else "Short"
                self.logger.info("=" * 115)
                self.logger.info(f"📊 [{m.symbol} {side}] Deep-drawdown and net-return analysis | Maker fee: {m.conf.maker_fee:.4%}")
                self.logger.info("-" * 115)

                # Add an "estimated net profit (U)" column
                header = (
                    f"{'Layer':<6} | {'Fill price':<10} | {'Live average':<12} | {'Rebound %':<10} | {'Total position (U)':<18} | {'Estimated net PnL (U)'}"
                )
                print(header)
                print("-" * 115)

                m.base_qty = 5 / p_ref
                for row in m.conf.matrix:
                    layer = row["layer"]
                    fill_p = p_ref * row["p_factor"]
                    avg_p = p_ref * row["avg_p_factor"]

                    # Take profit level
                    tp_p = avg_p * (1 + m.conf.profit_target if self.is_long_account else 1 - m.conf.profit_target)
                    rebound_needed = abs((tp_p - fill_p) / fill_p) * 100

                    # Statistics
                    q_total = m.base_qty * row["cum_q_factor"]
                    total_notional = q_total * avg_p

                    #  "Net profit" when leaving from this layer
                    # Formula: total notional * net profit factor
                    net_profit_u = total_notional * row["net_profit_factor"]

                    # Colour warning (in case the fees leave almost no net profit)
                    profit_status = "⚠️ LOW" if net_profit_u < 0.1 else "OK"

                    print(
                        f"L{layer:<3} | {fill_p:>10.4f} | {avg_p:>10.4f} | {rebound_needed:>7.2f}% | {total_notional:>12.2f} | {net_profit_u:>10.4f} {profit_status}"
                    )

                # --- final risk summary ---
                final_layer = m.conf.matrix[-1]
                avg_price = p_ref * final_layer["avg_p_factor"]
                tp_price = avg_price * (1 + m.conf.profit_target if self.is_long_account else 1 - m.conf.profit_target)
                sl_price = avg_price * (1 - m.conf.profit_target if self.is_long_account else 1 + m.conf.profit_target)

                # Stop loss drag
                loss_at_sl = abs((m.base_qty * final_layer["cum_q_factor"]) * (sl_price - avg_price))
                loss_pct = (loss_at_sl / self.total_equity) * 100

                self.logger.info(f"🔍 [{m.symbol} {side}] Key findings:")
                self.logger.info(f"   - Average-price anchor: {avg_price:.5f} ({((avg_price-p_ref)/p_ref*100):.3f}% from reference)")
                self.logger.info(f"   - 🎯 Take-profit target: {tp_price:.5f} ({((tp_price-p_ref)/p_ref*100):.3f}% from reference)")
                self.logger.info(f"   - 💀 Mirrored stop loss: {sl_price:.5f} ({((sl_price-p_ref)/p_ref*100):.3f}% from reference)")
                self.logger.info(f"   - 💥 Full-position risk: estimated loss {loss_at_sl:.2f} USDT ({loss_pct:.2f}% of equity)")

                # Required bounce
                max_rebound = abs((tp_price - (p_ref * final_layer["p_factor"])) / (p_ref * final_layer["p_factor"])) * 100
                self.logger.info(
                    f"   - ⚡ Break-even rebound: after reaching L{final_layer['layer']}, price must rebound {max_rebound:.2f}% for a profitable exit"
                )

        self.logger.info("=" * 105)

    def update_wallet_balance(self):
        res = self.engine.http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if res.get("retCode") == 0:
            self.total_equity = float(res["result"]["list"][0]["coin"][0]["equity"])
            if self.init_total_equity == -1:
                self.init_total_equity = self.total_equity

    def on_ws_order_notify(self, message):
        orders = message.get("data", [])
        if not orders:
            return
        # 1. Preprocess: sort by timestamp ascending (Bybit may push out of order)
        # so that L1 is handled before L2 when both are in data
        orders.sort(key=lambda x: int(x.get("execTime", 0)))
        for order in message.get("data", []):
            symbol = order["symbol"]
            if symbol in self.markets:
                if order["orderStatus"] == "Filled":
                    if order["orderLinkId"] == "":
                        continue  # cancle/market order, ignore
                    self.handle_fill(symbol, order)  # call directly when test
                    # threading.Thread(target=self.handle_fill, args=(symbol,order), daemon=True).start()
                elif order["orderStatus"] in ["Cancelled", "Rejected"]:
                    self.logger.debug(f"ℹ️ Order {order['orderLinkId']} status changed: {order['orderStatus']}")
            else:
                self.logger.warning(f"unecpected symbol {symbol}, close all")
                self.engine.market_close_all(symbol)

    # take profit; Bybit cancels the TP order with the higher price automatically
    def place_tp_order(self, m: SymbolState):
        tp_raw_price = m.avg_entry_price * (1 + (m.conf.profit_target * (1 if (m.signed_pos_qty > 0) else -1)))
        tp_price = round(tp_raw_price / m.conf.tick_size) * m.conf.tick_size
        side = "Sell" if (m.signed_pos_qty > 0) else "Buy"
        tp_qty = abs(m.signed_pos_qty)
        order_id = self.generate_order_link_id(m.last_order_ts, m.layer_count, label=OrderLabel.TP)
        self.engine.place_order(m.symbol, side, tp_qty, tp_price, order_id, is_reduce=True)
        self.logger.info(f"place tp order {side} order , qty {tp_qty}, price {tp_price} order_id {order_id}")

    def _prepare_so_order_params(self, m: SymbolState, layer: int):
        """
        🧪 Generic SO order parameter generator (does not place the order)
        Formula:
        Price_{layer} = BasePrice /times P_Factor_{layer}$
        Qty_{layer} = BaseQty /times Q_Factor_{layer}$
        """
        # 1. Read the factors from the table
        row = m.conf.matrix[layer - 1]

        # 2. Physical price and quantity
        raw_p = m.base_price * row["p_factor"]
        price = round(raw_p / m.conf.tick_size) * m.conf.tick_size
        self.logger.debug(f"_prepare_so_order_params layer {layer} price {price}")
        raw_q = m.base_qty * row["q_factor"]
        qty = round(raw_q / m.conf.qty_step) * m.conf.qty_step

        # 3. Build the parameter dict (Bybit V5 batch order format)
        side = "Buy" if self.is_long_account else "Sell"
        label = OrderLabel.ENTRY if layer == 1 else OrderLabel.SO
        # print(f"{sys._getframe().f_lineno} {time.time()} last_order_ts {m.last_order_ts}, layer {layer}")
        order_id = self.generate_order_link_id(m.last_order_ts, layer, label=label)

        return {"symbol": m.symbol, "side": side, "orderType": "Limit", "qty": str(qty), "price": str(price), "orderLinkId": order_id, "reduceOnly": False}

    def _prepare_sl_order_params(self, m: SymbolState):
        """
        🛡️ Mirrored stop loss parameter generator (maker mode)
        Logic: from the full-load average price factor, derive a stop symmetric to the take profit
        """
        # 1. Last layer of the matrix (fully loaded state)
        final_layer = m.conf.matrix[-1]

        # 2. Derive the theoretical full-load average price
        # $AvgP_{final} = BasePrice \times AvgPFactor_{final}$
        theo_avg_p = m.base_price * final_layer["avg_p_factor"]

        # 3. Mirrored symmetric distance
        # distance = full-load average price * take profit target
        dist = theo_avg_p * m.conf.profit_target

        # 4. Stop loss price (below the average for longs, above it for shorts)
        if self.is_long_account:
            sl_raw_p = theo_avg_p - dist
            side = "Sell"
        else:
            sl_raw_p = theo_avg_p + dist
            side = "Buy"

        sl_price = round(sl_raw_p / m.conf.tick_size) * m.conf.tick_size

        # 5. Slightly oversized close quantity (1.001x + reduceOnly to guarantee a full close)
        full_qty = m.base_qty * final_layer["cum_q_factor"] * 1.001
        sl_qty = round(full_qty / m.conf.qty_step) * m.conf.qty_step
        #  Extra logging: monitor the core SL parameters
        # print(f"[{sys._getframe().f_lineno}] 🛡️ {m.symbol} SL Calc | Side: {side} | TheoAvgP: {theo_avg_p:.6f} | Dist: {dist:.6f} | SL_Price: {sl_price} | SL_Qty: {sl_qty} | TS: {m.last_order_ts}")
        return {
            "symbol": m.symbol,
            "side": side,
            "orderType": "Limit",
            "qty": str(sl_qty),
            "price": str(sl_price),
            "orderLinkId": self.generate_order_link_id(m.last_order_ts, m.conf.max_layers, OrderLabel.SL),
            "reduceOnly": True,  #  must be on, so it can only reduce the position
        }

    # symmetric stop loss
    def place_sl_order(self, m: SymbolState):
        """🛠️ Place the stop loss order"""
        params = self._prepare_sl_order_params(m)

        # limit price
        # self.engine.place_order(
        #     symbol=params['symbol'],
        #     side=params['side'],
        #     qty=float(params['qty']),
        #     price=float(params['price']),
        #     link_id=params['orderLinkId'],
        #     is_reduce=True # maps to reduceOnly=True internally
        # )
        #  Note: a conditional market order needs no price before it triggers, only triggerPrice
        self.engine.place_order(
            symbol=params["symbol"],
            side=params["side"],
            qty=float(params["qty"]),
            price=None,  #  a market order carries no limit price
            triggerPrice=params["price"],  #  the trigger price
            link_id=params["orderLinkId"],
            is_reduce=True,
            order_type="Market",  #  Market stated explicitly
            triggerDirection=2 if self.is_long_account else 1,
        )
        self.logger.info(f"🛡️ [{m.symbol}] Mirrored stop order is active | Price: {params['price']} | Quantity: {params['qty']}")

    def handle_fill(self, symbol, order):
        m = self.markets[symbol]
        price = float(order.get("avgPrice", order["price"]))
        qty = float(order["cumExecQty"])
        signed_delta = qty if order["side"] == "Buy" else -qty

        with m.lock:
            # Increase or decrease of exposure
            valid, ts, layer_count, label = self.parse_order_link_id(order["orderLinkId"])
            if valid == False or ts != m.last_order_ts:
                # self.engine.cancel_single_order(order['symbol'], order['orderLinkId'])
                if valid == False:
                    self.logger.error(f"cancle invalid order {order['orderLinkId']}, current version {self.version} ts {m.last_order_ts}")
                elif ts < m.last_order_ts:
                    self.logger.error(f"cancle expired order {order['orderLinkId']}, current version {self.version} ts {m.last_order_ts}")
                elif ts > m.last_order_ts:
                    self.logger.warning(f"cancle newer order {order['orderLinkId']}, current version {self.version} ts {m.last_order_ts}")
                return
            if layer_count != m.layer_count:  # allowed, kept for robustness
                self.logger.warning(f"Cross-layer fill detected | layer {m.layer_count} -> {layer_count}")
                m.layer_count = layer_count
            is_inc = (m.signed_pos_qty * signed_delta) > 0 or m.signed_pos_qty == 0
            if is_inc:
                if m.order_submit_sl == False:
                    # --- 3. Place the "final limit stop loss" (goal: close as maker) ---
                    # Computed symmetrically from the full-load average price factor of the last matrix layer
                    m.order_submit_sl = True
                    self.place_sl_order(m)
                old_abs = abs(m.signed_pos_qty)
                m.avg_entry_price = ((m.avg_entry_price * old_abs) + (price * qty)) / (old_abs + qty)
                m.signed_pos_qty += signed_delta
                if m.layer_count < m.conf.max_layers:
                    m.layer_count += 1
                    self.place_tp_order(m)  # no need to cancle the previous TP
                    # self.logger.info(f"handle_fill {side} Buy order , qty {tp_qty}, price {tp_price} order_id {order_id}")
            else:
                m.signed_pos_qty += signed_delta
                if label == OrderLabel.TP:
                    m.tp_count += 1  #  record the take profit
                    m.tp_layer_dist[layer_count] = m.tp_layer_dist.get(layer_count, 0) + 1
                    m.last_result = OrderLabel.TP
                    m.last_result_updte = time.time()
                    m.loss_count = 0
                    self.logger.info("Take profit triggered")
                elif label == OrderLabel.SL:
                    m.sl_count += 1  #  record the stop loss
                    if m.last_result == OrderLabel.SL:
                        m.loss_count += 1
                    else:
                        m.loss_count = 1
                    m.last_result = OrderLabel.SL
                    m.last_result_updte = time.time()
                    self.logger.info("Stop loss triggered")
                if abs(m.signed_pos_qty) >= m.conf.min_qty:
                    self.logger.error("unexpected uncompleted decrease in position, please check !")
                    # self.engine.cancel_all_http(symbol)
                    # self.engine.market_close_all(symbol)
                else:
                    if label != OrderLabel.TP and label != OrderLabel.SL:  # take profit / stop loss
                        self.logger.error("unexpected order id {order['orderLinkId']} layer_count {layer_count} label {label}")
                    else:
                        self.logger.info(" position is 0, new order ")
                self.engine.market_close_all(symbol)
                self.deploy_full_martingale_grid(symbol)  # new order
            self.logger.info(f"📊 {symbol} position updated: {m.signed_pos_qty:.2f} @ {m.avg_entry_price:.4f}")

    def generate_order_link_id(self, last_order_ts=0, layer_count=0, label=""):
        """Build a unique ID: V9:SYMBOL:INDEX:SIDE:TIMESTAMP"""
        # Millisecond precision to avoid collisions
        link_id = f"{self.version}:{last_order_ts}:{layer_count}:{label}"
        self.logger.debug(f"generate new order {link_id}")
        return link_id

    def parse_order_link_id(self, link_id):
        """
        Parse the ID
        Returns: (valid, timestamp, layer_count, label)
        """
        try:
            parts = link_id.split(":")
            if len(parts) < 4 or parts[0] != self.version:
                return False, None, None, None

            ts = int(parts[1])
            layer_count = int(parts[2])
            label = str(parts[3])
            return True, ts, layer_count, label
        except Exception:
            return False, None, None, None

    def update_micro_market_status_volume(self, symbol):
        """
        Trend circuit breaker (V9 flavour): Z-score probability model + volume profile (VPVR)
        Aims to pinpoint the 20% of time with extreme moves
        """
        import math

        m = self.markets[symbol]

        # 1. Fetch the kline data (100 x 1m bars)
        res = self.engine.http.get_kline(category=self.engine.category, symbol=symbol, interval=30, limit=100)  # 1, 3, 5, 15, 30
        if res.get("retCode") != 0 or not res["result"]["list"]:
            return

        # Extract close prices and volumes
        k_list = res["result"]["list"]
        prices = [float(k[4]) for k in k_list]
        volumes = [float(k[5]) for k in k_list]
        prices.reverse()  # into chronological order
        volumes.reverse()

        # 2. Z-score (statistical deviation)
        n = len(prices)
        ma = sum(prices) / n
        variance = sum((p - ma) ** 2 for p in prices) / n
        std_dev = math.sqrt(variance)
        curr = prices[-1]

        # Z-score (sign kept, not absolute)
        z_val = (curr - ma) / std_dev if std_dev > 0 else 0
        z_score_abs = abs(z_val)

        # 3. Volume concentration (simplified VPVR)
        bin_count = 10
        min_p, max_p = min(prices), max(prices)
        # Avoid dividing by zero
        price_range = max_p - min_p
        interval = price_range / bin_count if price_range > 0 else 0.0001

        profile = [0.0] * bin_count
        for p, v in zip(prices, volumes):
            # Put the price into its volume bucket
            idx = min(int((p - min_p) / interval), bin_count - 1)
            profile[idx] += v

        # Share of the largest volume bucket
        total_vol = sum(profile)
        concentration = max(profile) / total_vol if total_vol > 0 else 0

        # 4. Decide state and direction
        with m.lock:
            # Logic: price deviates too much (Z > 1.28) or the volume is extremely spread out (concentration < 0.10)
            # 1.28 corresponds to roughly the 20% two-sided tail of a normal distribution
            # if z_score_abs > 1.5 or concentration < 0.10:
            if concentration < 0.10:
                if z_val > 0:
                    m.market_state = MarketState.TREND_UP
                    m.trend_direction = 1
                else:
                    m.market_state = MarketState.TREND_DOWN
                    m.trend_direction = -1

                self.logger.warning(f"🚨 [{symbol}] Trend trigger: Z={z_val:.2f}, POC={concentration:.2f}")
            else:
                m.market_state = MarketState.OSCILLATION
                m.trend_direction = 0

    def deploy_full_martingale_grid(self, symbol, p_ref=0, q1=0):
        """
        ⚡ Deploy the whole grid: batch the opening/safety orders + a deterministic limit stop loss (maker)
        """
        left_equity_ratio = self.total_equity / self.init_total_equity
        if left_equity_ratio < (1 - self.max_loss_ratio):
            self.stop_signal = True
            self.logger.warning(f" Quity ratio {left_equity_ratio} is less than {self.max_loss_ratio}, Emergency Stop ")
        m = self.markets[symbol]
        with m.lock:
            if m.loss_count == 1 and time.time() - m.last_result_updte > 60:  # no trading within 1 min
                self.logger.info(f"trade forbiden loss_count {m.loss_count}")
                return
            elif m.loss_count > 1 and time.time() - m.last_result_updte > 60 * 2 * m.loss_count:  # abnormal market, trading disabled
                self.logger.info(f"trade forbiden loss_count {m.loss_count}")
                return
            elif (m.market_state == MarketState.TREND_DOWN and self.is_long_account) or (m.market_state == MarketState.TREND_UP and not self.is_long_account):
                self.logger.info(f"trade forbiden symbol {symbol} market_state {m.market_state.value}")
                return
            m.order_submit_sl = False
            #  Core: lock the reference anchor here
            res = self.engine.http.get_tickers(category=self.engine.category, symbol=symbol)

            if res.get("retCode") == 0:
                m.base_price = float(res["result"]["list"][0]["lastPrice"])
                m.base_qty = self.total_equity * m.conf.budget_pct / m.base_price
            else:
                self.logger.warning("deploy update price fail! ")
                return

            # 1. Build the order requests of every layer (L1-Lmax)
            order_requests = []
            m.last_order_ts = int(time.time() * 1000)
            # print(f"{sys._getframe().f_lineno} {time.time()} last_order_ts {m.last_order_ts}")
            for i in range(1, m.conf.max_layers + 1):
                req = self._prepare_so_order_params(m, i)
                order_requests.append(req)

            # --- 2. Batch order placement (L1-Lmax) ---
            if order_requests:
                res = self.engine.http.place_batch_order(category="linear", request=order_requests)
                if res.get("retCode") == 0:
                    self.logger.info(f"✅ [{symbol}] L1-L{m.conf.max_layers} batch order placement succeeded")
                else:
                    self.logger.error(f"❌ Batch order placement failed: {res.get('retMsg')}")
                    return
            m.layer_count = 1

    def get_all_open_orders(self, category="linear"):
        """
        🚀 Fetch every active order of the account in one call
        """
        #  Key point: without a symbol parameter Bybit returns every order in this category
        res = self.engine.http.get_open_orders(category="linear", settleCoin="USDT")

        if res.get("retCode") == 0:
            return res.get("result", {}).get("list", [])
        else:
            self.logger.error(f"❌ Failed to fetch orders in batch: {res.get('retMsg')}")
            return []

    def reconcile_all_markets(self):
        # 1. Fetch every open order at once
        all_orders = self.get_all_open_orders()

        # 2. Group the orders by symbol
        # Uses a dict[str, list]: {'MNTUSDT': [...], 'BTCUSDT': [...]}
        orders_by_symbol = {}
        for order in all_orders:
            s = order["symbol"]
            if s not in orders_by_symbol:
                orders_by_symbol[s] = []
            orders_by_symbol[s].append(order)
        return orders_by_symbol

    def get_open_orders(self, symbol):
        res = self.engine.http.get_open_orders(category="linear", symbol=symbol)
        return res.get("result", {}).get("list", [])

    def get_last_price(self, symbol):
        res = self.engine.http.get_tickers(category="linear", symbol=symbol)
        return float(res["result"]["list"][0]["lastPrice"])

    def recover_layer_from_history(self, symbol):
        """
        📜 Recover the layer count precisely from the execution history
        """
        result = False, 0, 0, 0
        try:
            # 1. Fetch the last 20 fills (Bybit V5); order frequency differs per symbol, so read them one by one
            res = self.engine.http.get_executions(category="linear", symbol=symbol, limit=100)

            if res.get("retCode") != 0:
                return result

            exec_list = res.get("result", {}).get("list", [])
            if not exec_list:
                return result

            # 2. Find the last valid fill
            order_list: dict[int, list] = {}
            for order in exec_list:
                link_id = order.get("orderLinkId", "")
                if link_id == "":
                    continue
                valid, ts, layer_count, label = self.parse_order_link_id(link_id)
                if valid:
                    if ts not in order_list:
                        order_list[ts] = [(order, layer_count, label)]
                    else:
                        order_list[ts].append((order, layer_count, label))
            newest_orders_index = max(order_list.keys())
            newest_orders = order_list[newest_orders_index]
            if not newest_orders:
                return result
            so_orders = {}
            for order, layer_count, label in newest_orders:
                self.logger.info(f" label {label}| {OrderLabel.ENTRY}| {order.get('orderLinkId', '')} |layer_count{layer_count} ")
                if label in [OrderLabel.ENTRY, OrderLabel.SO]:  # opening / pyramiding order
                    so_orders[layer_count] = order
                    self.logger.info(f"add to so_orders layer_count {layer_count} | {order.get('orderLinkId', '')}")
            sorted_so_order_list = OrderedDict(sorted(so_orders.items()))
            self.logger.info(f"num of sorted_so_order_list {len(sorted_so_order_list)}")
            last_layer_count = 0
            total_position = 0
            find_all_order = True
            for layer_count, order in sorted_so_order_list.items():
                if layer_count != last_layer_count + 1:
                    self.logger.warning(f"layer count {last_layer_count} miss!!!")
                    find_all_order = False
                    total_position += float(order["cumExecQty"])
                    last_layer_count = layer_count
                    break
            if find_all_order != True:
                return result
            result = True, newest_orders_index, last_layer_count, total_position
            return result
        except Exception as e:
            self.logger.error(f"❌ Failed to restore layer count: {e}")

        return result  # safe default

    def sync_local_pos(self):
        """Sync the local position and recover the layer count"""
        res = self.engine.http.get_positions(category="linear", settleCoin="USDT")
        if res.get("retCode") != 0:
            self.logger.warning("sync get_positions fail ")
            return
        pos_list = res.get("result", {}).get("list", [])
        active_symbols = []
        all_market_orders: Dict[str:list] = self.reconcile_all_markets()

        if not pos_list:
            for symbol in self.markets.keys():
                market_orders = all_market_orders.get(symbol, [])
                if len(market_orders) == 0:
                    self.logger.info(" no position1,no order, start new order")
                    self.deploy_full_martingale_grid(symbol)
            return

        for p in pos_list:
            qty = float(p["size"])
            symbol = p["symbol"]
            # test stage
            market_orders = all_market_orders.get(symbol, [])
            if qty != 0 or len(market_orders) != 0:
                continue
            else:
                self.logger.info(" no position,no order, start new order")
                self.deploy_full_martingale_grid(symbol)
                continue
            if qty > 0:
                active_symbols.append(symbol)
                m = self.markets[symbol]
                with m.lock:
                    self.logger.debug(f" symbol {symbol} signed_pos_qty {m.signed_pos_qty} qty {qty} side {p['side']}")
                    self.logger.debug(f" symbol {symbol} avg_entry_price {m.avg_entry_price} avgPrice {p['avgPrice']}")
                    m.signed_pos_qty = qty * (1 if p["side"] == "Buy" else -1)
                    m.avg_entry_price = float(p["avgPrice"])
                    #  Derive it from the synced average price
                    valid, ts, last_layer_count, total_position = self.recover_layer_from_history(symbol)
                    sync_result = False
                    if valid == False:
                        self.logger.error(f"{symbol} position sync fail ! position {qty}")
                    else:
                        if qty != total_position:
                            self.logger.error(f"{symbol} position sync fail ! qty {qty}, total_position {total_position}")
                        elif ts != m.last_order_ts:
                            self.logger.error(f"{symbol} position sync fail ! ts {ts}, last_order_ts {m.last_order_ts}")
                        elif last_layer_count != m.layer_count:
                            self.logger.warning(f"{symbol} sync layer, update layer_count {m.layer_count} -> {last_layer_count}")
                            sync_result = True
                        else:
                            sync_result = True
                    if sync_result == False:
                        # self.engine.cancel_all_http(symbol)
                        self.engine.market_close_all(symbol)
                        self.update_wallet_balance()
                        self.deploy_full_martingale_grid(symbol)
                    else:  # check exist oder, TP/SO/SL
                        market_orders = all_market_orders.get(symbol, [])
                        # There should be two orders: the take profit + the stop loss / safety order
                        find_tp = False
                        find_so = False
                        find_sl = False
                        target_tp_order_id = self.generate_order_link_id(m.last_order_ts, m.layer_count, label=OrderLabel.TP)  # take profit order
                        target_sl_order_id = self.generate_order_link_id(m.last_order_ts, m.layer_count, label=OrderLabel.SL)  # stop loss order
                        miss_so_layers = {}
                        for i in range(last_layer_count, m.conf.max_layers + 1):
                            m.layer_count = i
                            target_so_order_id = self.generate_order_link_id(m.last_order_ts, m.layer_count, label=OrderLabel.SO)  # safety order
                            miss_so_layers[target_so_order_id] = i
                        m.layer_count = last_layer_count  # recover

                        for order in market_orders:
                            link_id = order.get("orderLinkId", "")
                            if link_id == target_tp_order_id:
                                find_tp = True
                            elif link_id in miss_so_layers:
                                miss_so_layers.pop(link_id)
                            elif link_id == target_sl_order_id:
                                find_sl = True
                        if find_tp == False:
                            self.logger.warning(f"{symbol} take-profit order is missing; placing a replacement")
                            self.place_tp_order(m)
                        so_order_requests = []
                        for layer in miss_so_layers.values():
                            so_order_requests.append(self._prepare_so_order_params(m, layer))
                        if so_order_requests:
                            res = self.engine.http.place_batch_order(category="linear", request=so_order_requests)
                            if res.get("retCode") == 0:
                                self.logger.info(f"✅ [{symbol}] L1-L{m.conf.max_layers} batch order replacement succeeded")
                            else:
                                self.logger.error(f"❌ Batch order replacement failed: {res.get('retMsg')}")
                                return
                        if find_sl == False:
                            self.logger.warning(f"{symbol} stop-loss order is missing; placing a replacement")
                            self.place_sl_order(m)
            else:
                self.logger.info(" no position,check market_orders first")
                # No position, first check whether any order exists
                market_orders = all_market_orders.get(symbol, [])
                if len(market_orders) == 0:
                    self.logger.info(f"sync market_orders len {len(market_orders)}")
                    # self.engine.cancel_all_http(symbol)
                    self.update_wallet_balance()
                    self.deploy_full_martingale_grid(symbol)
        # for symbol in self.markets.keys():
        #     if symbol not in active_symbols:
        #         self.logger.warning(f" {symbol} not found , cancel all ")
        #         m = self.markets[symbol]
        #         with m.lock:
        #             m.signed_pos_qty, m.avg_entry_price, m.layer_count = 0.0, 0.0, 0
        #             self.engine.cancel_all_http(symbol)
        #             self.update_wallet_balance()
        #             self.deploy_full_martingale_grid(symbol)

    def print_runtime_report(self):
        """📊 Live status report"""
        self.update_wallet_balance()  # refresh the latest equity

        self.logger.info(f"{'Symbol':<10} | {'Layer':<6} | {'Take-profit count (distribution)':<38} | {'Stops':<7} | {'Win rate':<8}")
        self.logger.info("-" * 110)

        for symbol, m in self.markets.items():
            with m.lock:
                # Turn the dict {1: 10, 2: 5} into the string "L1:10, L2:5"
                dist_items = sorted(m.tp_layer_dist.items())
                dist_str = ", ".join([f"L{k}:{v}" for k, v in dist_items]) if dist_items else "None"

                total_trades = m.tp_count + m.sl_count
                win_rate = (m.tp_count / total_trades * 100) if total_trades > 0 else 0

                self.logger.info(f"{symbol:<10} | L{m.layer_count:<3} | {dist_str:<35} | {m.sl_count:>5} | {win_rate:>7.1f}%")

    def run(self):
        last_report_time = 0
        while not self.stop_signal:
            server_time = time.time()

            # 1. Update the market state (trend detection)
            for symbol in self.markets.keys():
                self.update_micro_market_status_volume(symbol)

            # 2. Sync the position and refill the orders
            self.sync_local_pos()

            # 3. Print a status report every 60 seconds
            if server_time - last_report_time >= 60:
                self.print_runtime_report()
                last_report_time = server_time

            time.sleep(30)  # keep a 30 second scan interval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Martingale Bot V1")
    group = parser.add_mutually_exclusive_group(required=True)
    #  simplified to -l and -s
    group.add_argument("-l", "--long", action="store_true", help="Run Long Account")
    group.add_argument("-s", "--short", action="store_true", help="Run Short Account")

    args = parser.parse_args()
    is_long_account = args.long
    # Path to your key file
    demoTrading = False
    keypath = "Testnet" if demoTrading == True else "Maringale"
    side_path = "Long" if is_long_account == True else "Short"
    BASE = os.path.dirname(os.path.abspath(__file__))
    API_K = os.path.join(BASE, "keys", keypath, side_path, "hmac_api_key")
    API_S = os.path.join(BASE, "keys", keypath, side_path, "hmac_secret")
    RSA_K = os.path.join(BASE, "keys", keypath, side_path, "api_key")
    RSA_P = os.path.join(BASE, "keys", "bybit_rsa.pem")

    # ================= configuration =================
    # Tune gap and trend_bias per symbol volatility
    CONFIGS = [
        # SymbolConfig(symbol, budget, base gap, trend bias, max layers, size mult, step mult, profit target)
        SymbolConfig("MNTUSDT", 0.05, 0.008, 0.002, 6, 1.4, 1.1, 0.0008, 0, 0, 0),
        SymbolConfig("DOGEUSDT", 0.05, 0.0008, 0.0003, 6, 1.4, 1.2, 0.0008, 0, 0, 0),
        SymbolConfig("RAVEUSDT", 0.05, 0.002, 0.0003, 6, 1.7, 1.2, 0.002, 0, 0, 0),  # fee rate 0.0004
        # SymbolConfig("ADAUSDT", 0.05,  0.01,   0.001,      5,      1.3,    1.1,        0.008,   0,0,0),
    ]

    # Splitting long/short into two accounts decouples the logic a lot and keeps the statistics clean
    # Initialize the engine
    engine = BybitEngine(API_K, API_S, RSA_K, RSA_P, testnet=demoTrading)
    # Start the bot
    bot = MartingaleBot(engine, CONFIGS, is_long_account=is_long_account)

    # Signal handling
    def signal_handler(sig, frame):
        print("\n👋 Stop signal received...")
        bot.stop_signal = True
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    bot.run()
