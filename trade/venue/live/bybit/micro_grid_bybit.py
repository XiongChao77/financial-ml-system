import os, sys, logging, time, uuid, json, threading, math, signal
from enum import Enum
from typing import Dict, Set, List, Tuple

# bybit_engine.py is expected in the same directory
from bybit_engine import BybitEngine 
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", "..", "..", ".."))
from data_process import common

# -----------------------------------------------------------------------------
# Basic enums and dataclasses
# -----------------------------------------------------------------------------
class MarketState(Enum):
    OSCILLATION = "OSCILLATION"  # choppy
    TREND_UP = "TREND_UP"        # one-way up
    TREND_DOWN = "TREND_DOWN"    # one-way down
    
class NodeStatus(Enum):
    WAITING = "WAITING"   # order resting
    FILLED = "FILLED"     # filled (used for the history)
    UNKNOWN = "UNKNOWN"

    @property
    def short(self) -> str:
        return self.value[0]

    @classmethod
    def from_short(cls, short: str) -> "NodeStatus":
        if not short: return cls.UNKNOWN
        for s in cls:
            if s.value.startswith(short.upper()): return s
        return cls.UNKNOWN

class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"
    
    @property
    def short(self) -> str:
        return self.value[0]  # 'B' or 'S'

    @classmethod
    def from_short(cls, short: str) -> "OrderSide":
        if short == 'B': return cls.BUY
        if short == 'S': return cls.SELL
        return cls.BUY # Default

class GridNode:
    """
    Grid node (V9): id here is the relative index of the grid
    e.g. 0 (centre), -1 (one step below), 5 (five steps above)
    """
    def __init__(self, index: int, qty: float, price: float, side: OrderSide, 
                 order_id: str = None, status: NodeStatus = NodeStatus.WAITING):
        self.index = index  # grid index (may be negative)
        self.qty = qty
        self.price = price
        self.side = side
        self.order_id = order_id
        self.status = status
        self.timestamp = int(time.time())

    def __repr__(self):
        return f"<Idx:{self.index} {self.side.value} @ {self.price:.4f}>"

class SymbolConfig:
    def __init__(self, symbol, budget_pct, max_layers, base_offset, qty_step):
        self.symbol = symbol
        self.budget_pct = budget_pct    # budget share
        self.max_layers = max_layers    # layers per side (e.g. 3)
        self.base_offset = base_offset  # grid spacing (e.g. 0.005)
        self.qty_step = qty_step        # quantity precision

class SymbolState:
    def __init__(self, config: SymbolConfig):
        self.config = config
        
        # 🔒 mutex: protects the dynamic data
        self.lock = threading.RLock()
        
        # --- dynamic grid core ---
        self.initial_price = 0.0     # initial reference price (the price at index=0)
        self.center_index = 0        # logical centre index of the current window
        
        # Currently active nodes {index: GridNode}
        # The key is the index (-1, 0, 1...)
        self.active_nodes: Dict[int, GridNode] = {} 
        
        # Cached exchange parameters
        self.tick_size = 0.0001
        self.min_qty = 1.0
        self.market_state = MarketState.OSCILLATION
        # 0: no trend, 1: up, -1: down
        self.trend_direction = 0 
        # Cached position size, used for decisions
        self.current_pos_size = 0.0

        self.total_profit_usdt = 0.0  # cumulative net profit (fees deducted)
        self.total_fee_spent = 0.0    # cumulative fees paid
        self.profit_count = 0         # number of profitable fills
        
        # Core: holding cost computation
        self.current_pos_size = 0.0   # current position size
        self.avg_entry_price = 0.0    # dynamic weighted average cost
        self.initial_entry_done = False
# -----------------------------------------------------------------------------
# Main bot class (initialization)
# -----------------------------------------------------------------------------
class UnifiedGridBot:
    def __init__(self, engine: BybitEngine, symbol_configs: dict, clean= False):
        self.version = 'V9' # dynamic sliding window
        self.logger, _ = common.setup_session_logger(
            sub_folder=self.__class__.__name__, 
            console_level=logging.INFO, 
            file_level=logging.INFO
        )
        self.engine = engine
        self.markets: Dict[str, SymbolState] = {}
        self.stop_signal = False
        self.start_time = time.time()
        self.starting_balance = 0.0  # starting capital
        self.is_stopped = False      # global circuit breaker switch
        self.clean = clean

        self.current_balance = 0.0
        self.fee_rate = 0.0002 # taker estimate
        for symbol, cfg in symbol_configs.items():
            conf_obj = SymbolConfig(
                symbol=symbol,
                budget_pct=cfg['budget_pct'],
                max_layers=cfg['max_layers'],
                base_offset=cfg['base_offset'],
                qty_step=cfg['qty_step'],
            )
            self.markets[symbol] = SymbolState(conf_obj)
            # One-way position by default, set the leverage here if needed
            self.engine.set_leverage(symbol, 5) 
            self.logger.info(f"Load Config: {symbol} | Gap:{cfg['base_offset']:.2%} | Layers:±{cfg['max_layers']}")

    def startup(self):
        """Start-up sequence"""
        self.logger.info(f"🚀 Starting {self.version} dynamic sliding-window grid...")
        
        # 1. Sync the basic information
        self.update_instrument_info()
        self.get_wallet_balance()
        
        # 2. Start the WS listener
        self.engine.start_stream(self.on_order_update)
        
        if self.clean >1 :
            self.emergency_wipe_all_account_positions(use_market = (self.clean ==2))

        self.get_wallet_balance()
        for symbol in self.markets:
            if self.clean ==1 :
                self.engine.cancel_all_http(symbol) 
                time.sleep(1)
            self.start_dynamic_grid(symbol)
            time.sleep(1)
            
        # 4. Start the background daemon threads
        threading.Thread(target=self.run_loop, daemon=True).start()
        
        self.logger.info("✅ System ready; waiting for market data...")
        # Block the main thread
        while not self.stop_signal:
            time.sleep(1)

    # -------------------------------------------------------------------------
    # Helpers: price computation and ID handling
    # -------------------------------------------------------------------------
    def check_profit_viability(self, symbol):
        """
        Grid profit check: does base_offset cover the fees?
        """
        m = self.markets[symbol]
        
        # 1. Latest price
        res = self.engine.http.get_tickers(category=self.engine.category, symbol=symbol)
        if res.get('retCode') != 0 or not res['result']['list']:
            self.logger.error(f"❌ [{symbol}] Health check failed: unable to fetch the latest price")
            return
            
        last_price = float(res['result']['list'][0]['lastPrice'])
        
        # 2. Physical gap and two-sided fee
        # fee_rate is best defined in __init__; Bybit V5 taker is about 0.0006, maker about 0.0002
        gap_value = last_price * m.config.base_offset
        
        # Assuming we mostly act as maker, use 0.0002. Both sides means * 2
        # Change 0.0002 to 0.0006 if you often end up as taker
        total_fee_rate = self.fee_rate * 2 
        fee_value = last_price * total_fee_rate
        
        # 3. Net profit percentage
        net_profit_pct = ((gap_value - fee_value) / last_price) * 100
        
        # 4. How many ticks the gap covers
        ticks = gap_value / m.tick_size if m.tick_size > 0 else 0
        
        self.logger.info(f"📊 [{symbol}] Grid health report:")
        self.logger.info(f"   - Configured spacing: {m.config.base_offset:.2%}")
        self.logger.info(f"   - Price spacing: {gap_value:.5f} (about {ticks:.1f} ticks)")
        self.logger.info(f"   - Estimated net profit per trade: {net_profit_pct:.4f}%")
        
        # 5. Risk warning
        if net_profit_pct <= 0:
            self.logger.error(f"🚨 Warning: [{symbol}] spacing is too narrow to cover fees")
        elif net_profit_pct < 0.05:
            self.logger.warning(f"⚠️ Warning: [{symbol}] profit is extremely thin ({net_profit_pct:.4f}%); increase base_offset")
        elif ticks < 3:
            self.logger.warning(f"⚠️ Warning: [{symbol}] spacing is only {ticks:.1f} ticks and is vulnerable to spread and slippage")
        else:
            self.logger.info(f"✅ [{symbol}] Profit model is healthy")

    def get_price_by_index(self, symbol, index):
        """Core formula: target price from the index"""
        m = self.markets[symbol]
        if m.initial_price <= 0: return 0.0
        
        # price = reference price * (1 + index * spacing)
        raw_price = m.initial_price * (1 + index * m.config.base_offset)
        
        # Align to the precision
        return round(raw_price / m.tick_size) * m.tick_size

    def adjust_qty(self, raw_qty, qty_step, min_qty):
        """Quantity precision alignment"""
        if raw_qty < min_qty: raw_qty = min_qty
        
        # Integer and fractional steps are handled separately
        if qty_step == 0: qty_step = 1 # guard
        
        qty = round(raw_qty / qty_step) * qty_step
        
        # Format as a string, dropping the trailing zeros
        if qty_step >= 1:
            return str(int(qty))
        else:
            precision = int(math.ceil(-math.log10(qty_step)))
            return f"{qty:.{precision}f}"

    def generate_order_link_id(self, symbol, index, side: OrderSide):
        """Build a unique ID: V9:SYMBOL:INDEX:SIDE:TIMESTAMP"""
        short_sym = symbol.replace("USDT", "")
        ts = int(time.time() * 1000) # millisecond precision to avoid collisions
        # Note: the index may be negative, the f-string handles that (e.g. -1)
        return f"{self.version}:{short_sym}:{index}:{side.short}:{ts}"

    def parse_order_link_id(self, link_id):
        """
        Parse the ID
        Returns: (valid, symbol, index, side, timestamp)
        """
        try:
            parts = link_id.split(':')
            if len(parts) != 5 or parts[0] != self.version:
                return False, None, 0, None, 0
            
            sym = parts[1]
            index = int(parts[2]) # parses "-1" as well
            side = OrderSide.from_short(parts[3])
            ts = int(parts[4])
            
            return True, sym, index, side, ts
        except Exception:
            return False, None, 0, None, 0

    def get_wallet_balance(self):
        """Refresh the balance: total equity"""
        res = self.engine.http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if res.get('retCode') == 0:
            try:
                coin_data = res['result']['list'][0]['coin'][0]
                # totalEquity = wallet balance + unrealized pnl
                # Sum it manually when the Bybit API response has no totalEquity
                equity = float(coin_data.get('equity', coin_data.get('walletBalance', 0)))
                self.current_balance = equity
                
                # First run: record the starting baseline
                if self.starting_balance == 0:
                    self.starting_balance = equity
                    self.logger.info(f"🏦 Recorded starting balance: {self.starting_balance:.2f} USDT")
            except Exception as e:
                self.logger.error(f"❌ Balance parsing error: {e}")
            
    def check_account_safe(self):
        """
        Account level safety check: circuit breaker on the balance change
        """
        if self.is_stopped or self.starting_balance <= 0:
            return True

        # Total loss share
        loss_amount = self.starting_balance - self.current_balance
        loss_ratio = loss_amount / self.starting_balance if self.starting_balance > 0 else 0

        # More than 10% (0.10) lost
        if loss_ratio >= 0.10:
            self.logger.critical(f"🚨🚨 Account risk triggered | Initial: {self.starting_balance:.2f}, current: {self.current_balance:.2f}")
            self.logger.critical(f"📉 Total loss ratio: {loss_ratio:.2%}; initiating global liquidation")
            self.global_emergency_halt()
            return False
        
        return True

    def global_emergency_halt(self):
        """
        Account wide emergency stop: cancel every order, close every position and exit the program
        """
        self.is_stopped = True
        self.stop_signal = True
        
        #  Improvement: instead of looping over self.markets, sweep everything
        # so legacy positions are cleared as well
        self.emergency_wipe_all_account_positions(use_market=True)
            
        self.logger.critical("🛑 System shut down safely; all positions are flat")
        os._exit(0)

    def update_instrument_info(self):
        """Sync the exchange precision"""
        res = self.engine.http.get_instruments_info(category=self.engine.category)
        if res.get('retCode') == 0:
            info_map = {item['symbol']: item for item in res['result']['list']}
            for s in self.markets:
                if s in info_map:
                    self.markets[s].tick_size = float(info_map[s]['priceFilter']['tickSize'])
                    self.markets[s].min_qty = float(info_map[s]['lotSizeFilter']['minOrderQty'])
                    step = info_map[s]['lotSizeFilter']['qtyStep']
                    self.markets[s].config.qty_step = float(step)
                    self.check_profit_viability(s)

    def on_place_result(self, response):
        """Handle the WebSocket order acknowledgement"""
        if response.get('retCode') != 0:
            self.logger.warning(f"⚠️ Order failure response: {response.get('retMsg')}")
        else:
            # self.logger.debug(f"order confirmed: {response.get('result', {}).get('orderId')}")
            pass

    def perform_rebase(self, symbol):
        """
        Re-anchor: reset initial_price and zero out center_index
        """
        m = self.markets[symbol]
        
        # 1. Cancel every order of this symbol to avoid ID collisions
        self.engine.cancel_all_http(symbol)
        time.sleep(1) # wait for the exchange to process it

        # 2. Use the current market price as the new anchor
        res = self.engine.http.get_tickers(category=self.engine.category, symbol=symbol)
        if res.get('retCode') == 0:
            new_price = float(res['result']['list'][0]['lastPrice'])
            
            with m.lock:
                old_price = m.initial_price
                # Core reset
                m.initial_price = new_price 
                m.center_index = 0
                self.logger.info(f"✨ [{symbol}] Rebase complete: {old_price:.4f} -> {new_price:.4f} (index reset to zero)")
            
            # 3. Re-run the grid alignment
            self.reconcile_dynamic_grid(symbol)

    def smart_close_all_maker(self, symbol, max_retries=10, use_market=False):
        """
        Fully automatic close: maker chasing (limit) or a straight market close
        :param use_market: True uses a market order, False rests a limit order at the mark price
        """
        mode_str = "Market" if use_market else "Maker (price chasing)"
        self.logger.info(f"🧹 Starting the {mode_str} close sequence for {symbol}...")

        for attempt in range(max_retries):
            # 1. Cancel every HTTP order of this symbol so old and new orders do not fight
            self.engine.cancel_all_http(symbol)
            time.sleep(0.5)

            # 2. Fetch the live position data
            pos_res = self.engine.http.get_positions(category=self.engine.category, symbol=symbol)
            if pos_res.get('retCode') != 0:
                self.logger.error(f"❌ Failed to fetch positions: {pos_res.get('retMsg')}")
                continue

            pos_list = pos_res.get('result', {}).get('list', [])
            # Keep the entries with a size above 0
            active_pos = [p for p in pos_list if float(p.get('size', 0)) > 0]

            if not active_pos:
                self.logger.info(f"✅ [{symbol}] Position is flat; close succeeded")
                return True

            # 3. Walk the positions and send the closing orders
            for pos in active_pos:
                p_idx = int(pos['positionIdx'])
                size = pos['size']
                
                #  Core: a market order needs no price, a limit order uses the mark price
                price = "" if use_market else pos['markPrice']
                order_type = "Market" if use_market else "Limit"
                
                # Direction (works for one-way and hedge mode)
                if p_idx == 1: side = "Sell"    # close long
                elif p_idx == 2: side = "Buy"   # close short
                else: side = "Sell" if pos['side'] == "Buy" else "Buy" # one-way

                # 4. Call the order engine
                # Note: make sure your engine.place_order accepts the order_type parameter
                self.engine.place_order(
                    symbol=symbol, 
                    side=side, 
                    qty=size, 
                    price=price,
                    order_type=order_type, #  this keyword must match the engine class definition
                    link_id=f"CLOSE_{p_idx}_{int(time.time())}",
                    is_reduce=True, 
                    pos_idx=p_idx,
                    callback=self.on_place_result
                )
                self.logger.info(f"🔄 [{symbol}] {mode_str} attempt {attempt+1}: {side} {size}")

            # A market order usually fills at once, a maker order needs time to be watched
            time.sleep(1.0 if use_market else 5.0) 

        self.logger.error(f"❌ [{symbol}] Position remains open after {max_retries} attempts; check for an extreme move preventing fills")
        return False

    # -------------------------------------------------------------------------
    # Core strategy logic (inside the UnifiedGridBot class)
    # -------------------------------------------------------------------------
    def start_dynamic_grid(self, symbol):
        """
        Initialize the dynamic grid:
        1. Lock the current price as index=0
        2. Trigger one alignment immediately to place the initial orders
        """
        m = self.markets[symbol]
        res = self.engine.http.get_tickers(category=self.engine.category, symbol=symbol)
        
        if res.get('retCode') == 0:
            current_price = float(res['result']['list'][0]['lastPrice'])
            with m.lock:
                m.initial_price = current_price # anchor the reference price
                m.center_index = 0              # the initial centre is 0
            
            self.logger.info(f"🚀 [{symbol}] Dynamic grid started | Reference price (index=0): {current_price:.4f}")
            # Place the first orders right away
            self.reconcile_dynamic_grid(symbol)

    def on_order_update(self, message):
        """WS callback entry point"""
        data = message.get('data', [])
        for order in data:
            if order['orderStatus'] == "Filled":
                self.handle_filled_order(
                    order['symbol'], 
                    order['orderLinkId'], 
                    float(order.get('avgPrice', order['price'])), #  prefer the fill average price avgPrice
                    float(order['qty']),
                )

    def handle_filled_order(self, symbol, order_id, fill_price, qty):
        """
        ⚡ Fill handling: shift the centre + trigger an alignment
        """
        m = self.markets.get(symbol)
        if not m: return

        # 1. Parse the ID to get the grid index
        valid, _, index, side, _ = self.parse_order_link_id(order_id)
        if not valid: return

        self.logger.info(f"⚡ [{symbol}] Index {index} ({side.value}) filled -> shifting grid")

        # 2.  Core: move the grid centre (centre follows price)
        # Whether a buy or a sell filled, the centre moves to that level
        # e.g. the -1 buy fills, the centre becomes -1; -1 is the new axis and the old 0 becomes a sell above it.
        with m.lock:
            if not m.initial_entry_done:
                m.initial_entry_done = True
            # 1. Fee of this fill (assumed maker 0.0002)
            fee = fill_price * qty * self.fee_rate
            m.total_fee_spent += fee
            
            # 2. Update the holding cost or book the profit
            if side == OrderSide.BUY:
                # Case: buy (increase) -> dilute the cost
                new_total_qty = m.current_pos_size + qty
                m.avg_entry_price = ((m.avg_entry_price * m.current_pos_size) + (fill_price * qty)) / new_total_qty
                m.current_pos_size = new_total_qty
                self.logger.debug(f"📥 [{symbol}] Buy added; new average price: {m.avg_entry_price:.4f}")
                
            else:
                # Case: sell (reduce / take profit) -> book the profit
                # profit = (sell price - average cost) * qty - fee of this trade
                # Note: the fee paid on the matching buy should also be deducted (simplified to the current fee here)
                # local profit = price gap - fees
                grid_gap_price = m.initial_price * m.config.base_offset
                trade_profit = grid_gap_price * qty - fee # how much "this cell" earned
                m.total_profit_usdt += trade_profit
                m.profit_count += 1
                
                # Update the position size (does not affect the average cost)
                m.current_pos_size = max(0, m.current_pos_size - qty)
                #  Core fix: reset the average price once the position is flat
                if m.current_pos_size <= 0:
                    m.avg_entry_price = 0.0
                    self.logger.info(f"✨ [{symbol}] Position fully closed; average cost reset")
                self.logger.info(f"💰 [{symbol}] Position closed in profit | PnL: {trade_profit:.4f} USDT")

            # 3. Shift the centre dynamically (original logic)
            m.center_index = index
        
        # 3. Trigger the dynamic alignment (better run async so the WS is not blocked)
        threading.Thread(target=self.reconcile_dynamic_grid, args=(symbol,), daemon=True).start()

    def emergency_wipe_all_account_positions(self, use_market=True):
        """
        Account wide sweep: ignore the local configuration and force close every linear position
        """
        self.logger.critical("🚨 Starting account-wide liquidation for linear instruments...")

        # 1. Fetch every position of the account (without a symbol the API returns all of them)
        pos_res = self.engine.http.get_positions(
            category=self.engine.category, 
            settleCoin="USDT"
        )
        
        if pos_res.get('retCode') != 0:
            self.logger.error(f"❌ Unable to fetch account-wide positions: {pos_res.get('retMsg')}")
            return False

        # 2. Keep the positions with size > 0
        all_pos_list = pos_res.get('result', {}).get('list', [])
        active_symbols = set([p['symbol'] for p in all_pos_list if float(p.get('size', 0)) > 0])

        if not active_symbols:
            self.logger.info("✅ No active positions in the account")
            return True

        self.logger.warning(f"🔍 Instruments requiring action: {list(active_symbols)}")

        # 3. Cancel and close for every symbol found
        for sym in active_symbols:
            # Cancel every order of this symbol (so it cannot interfere with the close)
            self.engine.cancel_all_http(sym) 
            
            # Reuse the existing closing logic
            # It does not matter if the symbol is missing from self.markets, self.engine is enough
            self.smart_close_all_maker(sym, max_retries=5, use_market=use_market)

        self.logger.critical("💀 Account-wide liquidation complete")
        return True

    def reconcile_dynamic_grid(self, symbol):
        """
        🧠 Strategy brain: set difference algorithm
        [wanted orders] - [existing orders] = [orders to place]
        [existing orders] - [wanted orders] = [orders to cancel]
        """
        m = self.markets[symbol]
        cfg = m.config
        REBASE_THRESHOLD = cfg.max_layers
        # --- 0. Check whether a re-anchor is due ---
        if abs(m.center_index) >= REBASE_THRESHOLD:
            self.logger.warning(f"🔄 [{symbol}] Rebase triggered: center index {m.center_index} drifted too far; resetting coordinates...")
            self.perform_rebase(symbol)
            return # the re-anchor places the orders again, so this alignment stops here
        
        # --- 1. Read the real position (decides whether to shut the grid down) ---
        # Note: this has to be synchronous so the data is fresh
        pos_res = self.engine.http.get_positions(category=self.engine.category, symbol=symbol)
        
        real_pos_size = 0.0
        if pos_res.get('retCode') == 0:
            pos_list = pos_res.get('result', {}).get('list', [])
            # For simplicity take the sum of the absolute values (one-way / hedge compatible)
            real_pos_size = sum(abs(float(p.get('size', 0))) for p in pos_list)
        else:
            self.logger.error(f"❌ [{symbol}] Failed to fetch position: {pos_res.get('retMsg')}")
            return # on a query failure skip the alignment, so we do not act on a false flat
        with m.lock:
            m.current_pos_size = real_pos_size
            center = m.center_index
            layers = cfg.max_layers
            
            # --- A. Target topology ---
            # Requirement: sells above the centre, buys below it
            target_indices = set()
            
            # 🎯 The centre order is only placed in the initial stage (before the first entry)
            if not m.initial_entry_done:
                target_indices.add(center)
                m.initial_entry_done = True
            # Buys below: [Center-1, Center-2, ... Center-N]
            for i in range(1, layers + 1):
                target_indices.add(center - i)
                
            # Sells above: [Center+1, Center+2, ... Center+N]
            for i in range(1, layers + 1):
                target_indices.add(center + i)


            # =========================================================
            # 🛡️ Trend circuit breaker and direction filter (user logic interceptor)
            # =========================================================
            if m.market_state != MarketState.OSCILLATION:
                
                # Case 1: a trend starts while flat -> shut down immediately (clear every target)
                if m.current_pos_size == 0:
                    self.logger.warning(f"🛑 [{symbol}] Trending while flat; clearing the grid to reduce risk")
                    target_indices.clear() # this makes the diff below cancel every order
                
                else:
                    # Case 2: in position -> keep only the closing direction, drop the opening one
                    # Assuming a one-way (long only) position: Buy opens, Sell closes
                    
                    if m.market_state == MarketState.TREND_DOWN:
                        # Downtrend: no Buy (catching a falling knife), keep Sell (cut the loss on a bounce)
                        self.logger.warning(f"🛡️ [{symbol}] Downtrend circuit breaker: removing all buy orders")
                        target_indices = {idx for idx in target_indices if idx > center}
                        
                    elif m.market_state == MarketState.TREND_UP:
                        # Uptrend: (in a long grid)
                        # Ambiguity: while rising, Buy chases (opens) and Sell takes profit (closes).
                        # The user asked to "drop the opening direction (Buy), keep the closing one (Sell)"
                        # so we follow that instruction: remove Buy
                        self.logger.warning(f"🛡️ [{symbol}] Uptrend circuit breaker: pausing momentum buys")
                        target_indices = {idx for idx in target_indices if idx > center}
            # =========================================================

            # --- B. Current orders (reality) ---
            # In production read memory first (fast) and correct it over HTTP periodically. Here HTTP is read directly for accuracy.
            current_map = {} # {index: order_link_id}
            
            # Every active order of this symbol
            open_orders = self.fetch_symbol_open_orders(symbol)
            for o in open_orders:
                valid, _, idx, _, _ = self.parse_order_link_id(o['orderLinkId'])
                if valid:
                    current_map[idx] = o['orderLinkId']
            
            current_indices = set(current_map.keys())

            # --- C. Set difference ---
            # 1. Out of range / wrong orders -> cancel
            to_cancel = current_indices - target_indices
            
            # 2. Missing / new orders -> place
            to_place = target_indices - current_indices
            
            if not to_cancel and not to_place:
                return # perfect state, nothing to do

            self.logger.info(f"🧮 Alignment: center={center} | Cancel={list(to_cancel)} | Place={list(to_place)}")

            # --- D. Cancel ---
            for idx in to_cancel:
                self.engine.http.cancel_order(
                    category=self.engine.category, 
                    symbol=symbol, 
                    orderLinkId=current_map[idx]
                )
            
            # --- E. Place ---
            # Pre-compute the capital per cell
            self.get_wallet_balance()
            if self.current_balance <= 0: return
            budget_per_node = (self.current_balance * cfg.budget_pct) / (cfg.max_layers * 2)
            
            for idx in to_place:
                self.place_grid_order(symbol, idx, budget_per_node, center)
                time.sleep(0.1)

    def place_grid_order(self, symbol, index, budget, center_index):
        """Place one concrete order"""
        m = self.markets[symbol]
        price = self.get_price_by_index(symbol, index)
        if price <= 0: return

        #  Fix A: enforce the minimum notional value
        # Bybit linear contracts usually require > 5 USDT per order, we use 5.5 to be safe
        safe_budget = max(budget, 5.5) 
        
        # Direction
        # ---  trend driven direction decision ---
        if index < center_index:
            side = OrderSide.BUY
        elif index > center_index:
            side = OrderSide.SELL
        else:
            # 🎯 Core logic: when the index sits exactly at the centre
            # take the direction from the trend: m.trend_direction 1 (up) -> buy, -1 (down) -> sell
            # When choppy (0) we lean short (closing longs) or adjust by the position, to lower the risk
            if m.trend_direction == 1:
                side = OrderSide.BUY
            elif m.trend_direction == -1:
                side = OrderSide.SELL
            else:
                # While choppy: rest a sell when long, otherwise a buy
                side = OrderSide.BUY if m.current_pos_size < 0 else OrderSide.SELL
        
        # Quantity
        qty = safe_budget / price
        qty_str = self.adjust_qty(qty, m.config.qty_step, m.min_qty)
        
        # 4. Build the ID
        link_id = self.generate_order_link_id(symbol, index, side)
        
        # 5. Send it (one-way mode: pos_idx=0)
        # Note: is_reduce=False here, because this is a neutral grid where a sell may open a short or close a long
        # Bybit's one-way mode handles the netting
        self.engine.place_order(
            symbol, side.value, qty_str, price, link_id, 
            pos_idx=0, is_reduce=False, callback=self.on_place_result
        )

    def fetch_symbol_open_orders(self, symbol):
        """Helper: open orders of one symbol"""
        orders = []
        cursor = ""
        while True:
            res = self.engine.http.get_open_orders(
                category=self.engine.category, 
                symbol=symbol, 
                limit=50, 
                cursor=cursor
            )
            if res.get('retCode') != 0: break
            data = res.get('result', {})
            orders.extend(data.get('list', []))
            cursor = data.get('nextPageCursor', "")
            if not cursor: break
        return orders

    def update_micro_market_status_volume(self, symbol):
        """
        Trend circuit breaker (V9 flavour): Z-score probability model + volume profile (VPVR)
        Aims to pinpoint the 20% of time with extreme moves
        """
        import math
        m = self.markets[symbol]
        
        # 1. Fetch the kline data (100 x 1m bars)
        res = self.engine.http.get_kline(category=self.engine.category, symbol=symbol, interval=30, limit=100)   #1, 3, 5, 15, 30
        if res.get('retCode') != 0 or not res['result']['list']: 
            return

        # Extract close prices and volumes
        k_list = res['result']['list']
        prices = [float(k[4]) for k in k_list]
        volumes = [float(k[5]) for k in k_list]
        prices.reverse() # into chronological order
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
        price_range = (max_p - min_p)
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

    def update_micro_market_status(self, symbol):
        """
        Trend detection engine (V9)
        Logic: use the Z-score to tell whether the price strayed too far from the mean
        """
        m = self.markets[symbol]
        
        # 1. Fetch the klines (1 minute bars, the last 60)
        # interval=1 can be tuned to your grid frequency, 1 or 5 for high frequency
        res = self.engine.http.get_kline(category=self.engine.category, symbol=symbol, interval=1, limit=60)
        
        if res.get('retCode') != 0 or not res['result']['list']: 
            return

        # Bybit returns the data reversed (newest first), so flip it
        k_data = res['result']['list']
        prices = [float(k[4]) for k in k_data] # Close price
        # volumes = [float(k[5]) for k in k_data] # volume (optional VPVR logic)
        
        # 2. Statistics
        n = len(prices)
        avg_price = sum(prices) / n
        variance = sum((p - avg_price) ** 2 for p in prices) / n
        std_dev = math.sqrt(variance)
        
        current_price = prices[0] # latest close
        
        # 3. Z-score (how many standard deviations the price is away)
        if std_dev == 0: z_score = 0
        else: z_score = (current_price - avg_price) / std_dev
        
        # 4. Threshold (2.0 means a very large deviation)
        THRESHOLD = 2.0 
        
        with m.lock:
            if z_score > THRESHOLD:
                m.market_state = MarketState.TREND_UP
                m.trend_direction = 1
                self.logger.warning(f"📈 [{symbol}] One-way uptrend detected (Z:{z_score:.2f})")
            elif z_score < -THRESHOLD:
                m.market_state = MarketState.TREND_DOWN
                m.trend_direction = -1
                self.logger.warning(f"📉 [{symbol}] One-way downtrend detected (Z:{z_score:.2f})")
            else:
                m.market_state = MarketState.OSCILLATION
                m.trend_direction = 0
                # self.logger.info(f"[{symbol}] choppy market (Z:{z_score:.2f})")

    def get_uptime(self) -> str:
        """
        Compute and format the uptime
        """
        uptime_sec = time.time() - self.start_time
        days, remainder = divmod(int(uptime_sec), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        else:
            return f"{minutes}m {seconds}s"
        
    def report_status(self):
        """
        Upgraded live briefing: adds tick efficiency analysis
        """
        self.get_wallet_balance()
        uptime_str = self.get_uptime() 

        # Widen the header to fit the new columns
        print("\n" + "═"*110)
        print(f"📊 {self.version} Live Summary | Uptime: {uptime_str} | Balance: {self.current_balance:.2f} USDT")
        print("─" * 110)
        
        # Adds GAP(T) -> gap in ticks, FEE(T) -> fees in ticks
        header = f"{'SYMBOL':<10} | {'CTR':<5} | {'GAP(T)':<8} | {'FEE(T)':<8} | {'NET(T)':<8} | {'PROFIT(U)':<12} | {'COUNT':<6}"
        print(header)
        print("─" * 110)

        total_p = 0.0
        total_count = 0
        for s, m in self.markets.items():
            with m.lock:
                total_p += m.total_profit_usdt
                total_count += m.profit_count
                
                # --- dynamic tick efficiency ---
                # Use the reference price for the current tick coverage
                ref_price = m.initial_price if m.initial_price > 0 else 1.0
                
                # 1. Gap in ticks: (price * spacing) / tickSize
                gap_ticks = (ref_price * m.config.base_offset) / m.tick_size if m.tick_size > 0 else 0
                
                # 2. Fee in ticks: (price * two-sided fee) / tickSize
                # Both sides assumed maker (0.0002 * 2)
                fee_ticks = (ref_price * (self.fee_rate * 2)) / m.tick_size if m.tick_size > 0 else 0
                
                # 3. Net profit in ticks
                net_ticks = gap_ticks - fee_ticks
                
                # Colouring logic (for reference only)
                status_icon = "✅" if net_ticks > 3 else "⚠️"
                
                print(f"{s:<10} | {m.center_index:<5} | {gap_ticks:>8.1f} | {fee_ticks:>8.1f} | {net_ticks:>8.1f} {status_icon} | {m.total_profit_usdt:>12.4f} | {m.profit_count:<6}")
        
        print("─" * 110)
        print(f"📈 Total realized profit: {total_p:.4f} USDT | Count: {total_count}")
        print("═"*110 + "\n")
    # -------------------------------------------------------------------------
    # Background loops
    # -------------------------------------------------------------------------
    def run_loop(self):
        """Background daemon: print the status + fallback alignment"""
        last_report_time = 0
        while not self.stop_signal:
            # 1. Refresh the balance and run the safety check
            self.get_wallet_balance()
            if not self.check_account_safe():
                break # circuit breaker fired, leave the loop

            # 2. Original reporting and state update logic
            if time.time() - last_report_time > 30:
                self.report_status()
                for s, m in self.markets.items():
                    self.update_micro_market_status_volume(s)
                last_report_time = time.time()
            time.sleep(10) # print every 10 seconds
            
            self.logger.info("-" * 40)
            self.logger.info(f"💰 Balance: {self.current_balance:.2f}")
            
            for s, m in self.markets.items():
                # --- 0. Refresh the trend state first ---
                # Better done periodically in the main loop, or here every time (slower)
                with m.lock:
                    ctr = m.center_index
                    # Estimate the current order range
                    low = ctr - m.config.max_layers
                    high = ctr + m.config.max_layers
                    self.logger.info(f"   {s}: Center={ctr} | Range=[{low}, {high}] | BasePrice={m.initial_price:.4f}")
            self.logger.info("-" * 40)
        self.logger.info("👋 run_loop exited")
        # os._exit makes sure the main process goes down with it
        os._exit(0)
# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Path configuration
    BASE = os.path.dirname(os.path.abspath(__file__))
    API_K = os.path.join(BASE, "keys", "hmac_api_key")
    API_S = os.path.join(BASE, "keys", "hmac_secret")
    # Configure WS when RSA is required, otherwise leave it empty or keep the old logic
    RSA_K = os.path.join(BASE, "keys", "api_key")     
    RSA_P = os.path.join(BASE, "keys", "bybit_rsa.pem")

    CONFIGS = {
        "DOGEUSDT": {"budget_pct": 0.5, "max_layers": 10, "base_offset": 0.0008, "qty_step": 0},
        "PIPPINUSDT": {"budget_pct": 0.3, "max_layers": 10, "base_offset": 0.003, "qty_step": 0}, 
        "RAVEUSDT": {"budget_pct": 0.8, "max_layers": 10, "base_offset": 0.004, "qty_step": 0},     #fee rate 0.0004
        "MNTUSDT": {"budget_pct": 0.5, "max_layers": 10, "base_offset": 0.001, "qty_step": 0}, 
        "BEATUSDT": {"budget_pct": 0.3, "max_layers": 10, "base_offset": 0.004, "qty_step": 0}, #fee rate 0.0004
        "ADAUSDT": {"budget_pct": 0.3, "max_layers": 10, "base_offset": 0.0012, "qty_step": 0}, 

        # "ARCUSDT":  {"budget_pct": 0.5, "max_layers": 4, "base_offset": 0.002, "qty_step": 0}, 
        # "SUIUSDT": {"budget_pct": 0.3, "max_layers": 4, "base_offset": 0.002, "qty_step": 0}, 
        # "LUNA2USDT": {"budget_pct": 0.3, "max_layers": 4, "base_offset": 0.002, "qty_step": 0},
    }
    # Initialize the engine
    # Note: the engine code must support place_order(..., pos_idx=0)
    engine = BybitEngine(API_K, API_S, RSA_K, RSA_P)
    
    # Start the bot
    bot = UnifiedGridBot(engine, CONFIGS, clean =1) #1:close all exist order

    # Signal handling (Ctrl+C to exit)
    def signal_handler(sig, frame):
        print("\n👋 Stopping...")
        bot.stop_signal = True
        engine.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    # GO!
    bot.startup()
