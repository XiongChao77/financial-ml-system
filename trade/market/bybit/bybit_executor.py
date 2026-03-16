import logging
import sys,os
from datetime import datetime, timezone

# Path setup
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..', '..'))

from trade.market.bybit.bybit_engine import BybitEngine
from trade.strategy.strategy_ml import PositionDir, ActionType
from trade.strategy.base_executor import BaseExecutor

class BybitExecutor(BaseExecutor):
    def __init__(self, key_path, symbol: str):
        self.engine = BybitEngine(key_path)
        self.symbol = symbol
        self.logger = logging.getLogger("BybitExecutor")
        self.logger.info(f"BybitExecutor key_path:{key_path} symbol {symbol}")
        
        # Initialize precision info
        self.qty_step = 0.0
        self.tick_size = 0.0
        self.min_qty = 0.0
        self._init_symbol_info()

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

    def get_current_state(self):
        """
        Returns: (PositionDir, layers, avg_price)
        Compatible with TurtleBrain interface expectations.
        """
        try:
            res = self.engine.http.get_positions(category="linear", symbol=self.symbol)
            if res['retCode'] != 0: 
                return PositionDir.FLAT, 0, 0.0
            
            pos_list = res['result']['list']
            if not pos_list: 
                return PositionDir.FLAT, 0, 0.0

            pos = pos_list[0]
            size = float(pos['size'])
            avg_price = float(pos['avgPrice']) if size > 0 else 0.0
            side = pos['side'] # 'Buy' or 'Sell'

            if size == 0:
                return PositionDir.FLAT, 0, 0.0
            
            # Simplified layer estimate (turtle systems often track this externally).
            # If strict layer logic is required, track it outside or infer via size/unit_size.
            direction = PositionDir.POSITIVE  if side == 'Buy' else PositionDir.NEGATIVE
            return direction, 1, avg_price

        except Exception as e:
            self.logger.error(f"Failed to get position state: {e}")
            return PositionDir.FLAT, 0, 0.0

    def get_server_time(self):
        return datetime.now(timezone.utc)
    
    def user_order(self, size, is_buy, stop_loss=None):
        """
        Place an order.
        size: base coin quantity
        is_buy: direction
        stop_loss: stop-loss ratio (e.g. 0.05 means 5%)
        """
        # 1. Align quantity to exchange precision
        qty = round(float(size) / self.qty_step) * self.qty_step
        qty = max(self.min_qty, qty)
        qty_str = str(qty)

        # 2. Get current price for computing SL price.
        # Note: this uses a market order, so entry_price ~= current ticker price.
        tickers = self.engine.http.get_tickers(category="linear", symbol=self.symbol)
        curr_price = float(tickers['result']['list'][0]['lastPrice'])
        
        # 3. Compute stop-loss price (Bybit expects a concrete price; Brain provides a ratio)
        sl_price = 0.0
        if stop_loss:
            if is_buy:
                raw_sl = curr_price * (1 - stop_loss)
            else:
                raw_sl = curr_price * (1 + stop_loss)
            sl_price = round(raw_sl / self.tick_size) * self.tick_size

        side = "Buy" if is_buy else "Sell"
        self.logger.info(f"🐢 Placing order: {side} {qty_str} @ market | SL: {sl_price}")

        try:
            # 4. Place order via engine; include stopLoss when present.
            # Market orders do not need a limit price.
            order_params = {
                "category": "linear",
                "symbol": self.symbol,
                "side": side,
                "orderType": "Market",
                "qty": qty_str,
                "positionIdx": 0, # one-way position mode
                "reduceOnly": False
            }
            
            if sl_price > 0:
                order_params["stopLoss"] = str(sl_price)

            # HTTP placement is typically more reliable than ws_trade.place_order for this use case.
            # This turtle strategy is not high-frequency, so HTTP is fine.
            res = self.engine.http.place_order(**order_params)
            
            if res['retCode'] == 0:
                self.logger.info(f"✅ Order placed successfully: ID {res['result']['orderId']}")
            else:
                self.logger.error(f"❌ Order failed: {res['retMsg']}")
                
        except Exception as e:
            self.logger.error(f"Order exception: {e}")

    def user_close(self):
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
                        reduceOnly=True
                    )
        except Exception as e:
            self.logger.error(f"Close position exception: {e}")