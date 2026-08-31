import logging
import sys,os
import itertools
import math
import re
import time
from datetime import datetime, timezone

# Path setup
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", "..", "..", ".."))

from trade.venue.live.bybit.bybit_engine import BybitEngine
from trade.core.protocol import ActionType, OrderType, PositionDir, PositionView
from trade.core.venue_base import VenueBase

class BybitVenue(VenueBase):
    def __init__(
        self,
        key_path,
        symbol: str,
        magic: str | None = None,
        *,
        logger: logging.Logger | None = None,
    ):
        self.engine = BybitEngine(key_path)
        self.symbol = symbol
        self.magic = str(magic or "financial-ml-system")
        self._order_id_sequence = itertools.count()
        self.logger = logger or logging.getLogger("BybitVenue")
        self.logger.info(f"BybitVenue key_path:{key_path} symbol {symbol}")
        
        # Initialize precision info
        self.qty_step = 0.0
        self.tick_size = 0.0
        self.min_qty = 0.0
        self._init_symbol_info()

    def _new_order_link_id(self, action: str) -> str:
        safe_magic = re.sub(r"[^A-Za-z0-9_-]", "_", self.magic)
        nonce = f"{time.time_ns():x}{next(self._order_id_sequence):x}"[-14:]
        suffix = f"_{action}_{nonce}"
        return f"{safe_magic[:36 - len(suffix)]}{suffix}"

    def _init_symbol_info(self):
        """Sync exchange precision settings to prevent Invalid Volume errors."""
        try:
            res = self.engine.http.get_instruments_info(category="linear", symbol=self.symbol)
            if res['retCode'] == 0:
                info = res['result']['list'][0]
                self.qty_step = float(info['lotSizeFilter']['qtyStep'])
                self.min_qty = float(info['lotSizeFilter']['minOrderQty'])
                self.tick_size = float(info['priceFilter']['tickSize'])
                self.logger.info(f"✅ Precision synced: QtyStep={self.qty_step}, Tick={self.tick_size}")
        except Exception as e:
            self.logger.error(f"Failed to sync precision info: {e}")

    def get_account_equity(self):
        """Get USDT account equity."""
        res = self.engine.http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if res.get('retCode') == 0:
            return float(res['result']['list'][0]['coin'][0]['equity'])
        return 0.0

    def get_current_state(self) -> PositionView:
        """Return the current position direction, size, and average price."""
        try:
            res = self.engine.http.get_positions(category="linear", symbol=self.symbol)
            if res['retCode'] != 0: 
                return PositionView()
            
            pos_list = res['result']['list']
            if not pos_list: 
                return PositionView()

            pos = pos_list[0]
            size = float(pos['size'])
            avg_price = float(pos['avgPrice']) if size > 0 else 0.0
            side = pos['side'] # 'Buy' or 'Sell'

            if size == 0:
                return PositionView()
            
            direction = PositionDir.POSITIVE  if side == 'Buy' else PositionDir.NEGATIVE
            return PositionView(dir=direction, size=size, price=avg_price)

        except Exception as e:
            self.logger.error(f"Failed to get position state: {e}")
            return PositionView()

    def _latest_price(self) -> float:
        response = self.engine.http.get_tickers(
            category="linear",
            symbol=self.symbol,
        )
        price = float(response["result"]["list"][0]["lastPrice"])
        if not math.isfinite(price) or price <= 0:
            raise RuntimeError("Bybit returned an invalid ticker price")
        return price

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
        """
        Place an order.

        size: base coin quantity
        is_buy: True = Buy/Long, False = Sell/Short
        stop_loss_pct: stop-loss ratio, e.g. 0.05 means 5%
        take_profit_pct: take-profit ratio, e.g. 0.10 means 10%
        """
        order_type, price = self.normalize_order_request(order_type, price)

        # 1. Align quantity to exchange precision
        qty = round(float(size) / self.qty_step) * self.qty_step
        qty = max(self.min_qty, qty)
        qty_str = str(qty)

        reference_price = self._latest_price() if price is None else price
        if order_type == OrderType.LIMIT:
            reference_price = round(reference_price / self.tick_size) * self.tick_size

        # 3. Compute stop-loss / take-profit concrete prices.
        # Bybit expects concrete price, while the strategy layer provides ratios.
        sl_price = 0.0
        tp_price = 0.0

        if stop_loss_pct:
            if is_buy:
                raw_sl = reference_price * (1 - stop_loss_pct)
            else:
                raw_sl = reference_price * (1 + stop_loss_pct)

            sl_price = round(raw_sl / self.tick_size) * self.tick_size

        if take_profit_pct:
            if is_buy:
                raw_tp = reference_price * (1 + take_profit_pct)
            else:
                raw_tp = reference_price * (1 - take_profit_pct)

            tp_price = round(raw_tp / self.tick_size) * self.tick_size

        side = "Buy" if is_buy else "Sell"

        self.logger.info(
            f"🐢 Placing {order_type.value} order: {side} {qty_str} | "
            f"SL: {sl_price} | TP: {tp_price}"
        )

        try:
            # 4. Place order via engine.
            # Market orders do not need a limit price.
            order_params = {
                "category": "linear",
                "symbol": self.symbol,
                "side": side,
                "orderType": "Market" if order_type == OrderType.MARKET else "Limit",
                "qty": qty_str,
                "positionIdx": 0,  # one-way position mode
                "reduceOnly": False,
                "orderLinkId": self._new_order_link_id("open"),
            }
            if order_type == OrderType.LIMIT:
                order_params["price"] = str(reference_price)
                order_params["timeInForce"] = "GTC"

            if sl_price > 0:
                order_params["stopLoss"] = str(sl_price)

            if tp_price > 0:
                order_params["takeProfit"] = str(tp_price)

            # Optional but often recommended:
            # Use MarkPrice to reduce wick-trigger noise.
            # If you prefer LastPrice, remove these two lines.
            if sl_price > 0:
                order_params["slTriggerBy"] = "MarkPrice"

            if tp_price > 0:
                order_params["tpTriggerBy"] = "MarkPrice"

            res = self.engine.http.place_order(**order_params)

            if res["retCode"] == 0:
                self.logger.info(
                    f"✅ Order placed successfully: ID {res['result']['orderId']}"
                )
            else:
                self.logger.error(f"❌ Order failed: {res['retMsg']}")

        except Exception as e:
            self.logger.error(f"Order exception: {e}")

    def close_position(self):
        """Close all open positions for this symbol."""
        try:
            # Fetch positions
            res = self.engine.http.get_positions(category="linear", symbol=self.symbol)
            for pos in res['result']['list']:
                size = float(pos['size'])
                if size > 0:
                    side = "Sell" if pos['side'] == "Buy" else "Buy"
                    self.logger.info(f"Closing position: {pos['side']} {size}")
                    
                    self.engine.http.place_order(
                        category="linear",
                        symbol=self.symbol,
                        side=side,
                        orderType="Market",
                        qty=str(size),
                        positionIdx=0,
                        reduceOnly=True,
                        orderLinkId=self._new_order_link_id("close"),
                    )
        except Exception as e:
            self.logger.error(f"Close position exception: {e}")

    def get_last_position_open_time(self):
        try:
            res = self.engine.http.get_executions(
                category="linear",
                symbol=self.symbol,
                limit=50
            )
            
            if res.get("retCode") != 0:
                return None
            
            executions = res["result"]["list"]
            if not executions:
                return None
            
            # Current position direction
            position = self.get_current_state()
            if position.dir == PositionDir.FLAT:
                return None
            
            target_side = "Buy" if position.dir == PositionDir.POSITIVE else "Sell"
            
            # Find the most recent fill in the opening direction
            for exe in executions:
                if exe["execType"] == "Trade" and exe["side"] == target_side:
                    ts = int(exe["execTime"])
                    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            
            return None

        except Exception as e:
            self.logger.error(f"Failed to get position open time: {e}")
            return None
