import time, json, logging, threading, contextlib, os, sys, websocket,uuid
from pybit.unified_trading import HTTP, WebSocket, WebSocketTrading
from websocket import WebSocketConnectionClosedException 

class BybitEngine:
    """
    Bybit V5 generic quant base (V8.1)
    Combines HTTP (configuration/queries) + WS_Trade (trading) + environment self-healing
    """
    def __init__(self, key_path, testnet=False):
        self.api_key = self._load_key(os.path.join(key_path, "hmac_api_key"))
        self.hmac_secret = self._load_key(os.path.join(key_path, "hmac_secret"))
        self.rsa_key = self._load_key(os.path.join(key_path, "api_key"))
        self.rsa_pem_path = self._load_key(os.path.join(key_path, "bybit_rsa.pem"))
        self.testnet = testnet
        self.category = "linear"
        
        # 1. HTTP client (configuration hub)
        self.http = HTTP(testnet=testnet, api_key=self.api_key, api_secret=self.hmac_secret,timeout=10,)
        
        # 2. WS trading client
        self.ws_trade = WebSocketTrading(testnet=testnet, api_key=self.api_key, api_secret=self.hmac_secret)
        
        # 3. WS market data client
        self.ws_stream = WebSocket(
            testnet=testnet, channel_type="private", 
            rsa_authentication=True, api_key=self.rsa_key, api_secret=self.rsa_pem_path
        )

    def _load_key(self, path):
        try: return open(path, 'r').read().strip()
        except Exception as e:
            raise RuntimeError(f" load key fail: {path}, e")

    # --- added: environment configuration ---

    def setup_hedge_env(self, symbol, leverage="10"):
        """
        One-shot trading environment setup: hedge mode + leverage
        """
        print(f"🛠️ Configuring hedge mode for {symbol}...")
        
        # 1. Try to switch to hedge mode (mode=3)
        # Note: this fails if there is an open position or a pending order
        try:
            res = self.http.switch_position_mode(
                category=self.category,
                symbol=symbol,
                mode=3 
            )
            if res.get('retCode') == 0:
                print("   ✅ Mode switched successfully: hedge mode enabled")
            elif res.get('retCode') == 110025:
                print("   ℹ️ Mode confirmed: hedge mode is already enabled")
        except Exception as e:
            print(f"   ⚠️ Position-mode switch error: {e}")

        # 2. Set the leverage
        try:
            self.http.set_leverage(
                category=self.category,
                symbol=symbol,
                buyLeverage=leverage,
                sellLeverage=leverage
            )
            print(f"   ✅ Leverage set successfully: {leverage}x")
        except Exception as e:
            # Bybit raises when the leverage did not change, which can usually be ignored
            pass

    def set_leverage(self, symbol, leverage):
        """
        Smart leverage setup: check first, only change when it differs
        leverage: target leverage, e.g. "5"
        """
        try:
            # 1. Read the current position / risk limit information
            # Note: this endpoint returns the symbol's default leverage even without a position
            pos_res = self.http.get_positions(category=self.category, symbol=symbol)
            
            if pos_res.get('retCode') == 0 and pos_res['result']['list']:
                # Leverage of the first record (usually POS 0 or POS 1)
                current_lev = pos_res['result']['list'][0].get('leverage', "0")
                
                # 2. Compare: skip when it already matches
                if str(current_lev) == str(leverage):
                    # self.logger.debug(f"[{symbol}] leverage is already {leverage}x, skipping")
                    return True

            # 3. Only apply when it differs
            res = self.http.set_leverage(
                category=self.category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            
            if res.get('retCode') == 0:
                print(f"✅ [{symbol}] Leverage updated to {leverage}x")
            elif res.get('retCode') == 110043:
                # Fallback: the query may lag, so still catch the not-modified error here
                pass 
            else:
                print(f"⚠️ [{symbol}] Failed to set leverage: {res.get('retMsg')}")
                
        except Exception as e:
            print(f"❌ [{symbol}] Error while checking or setting leverage: {e}")

    # --- core trading logic ---

    def place_order(self, symbol, side, qty, price, link_id, 
                    order_type="Limit", pos_idx=0, is_reduce=False, callback=None):
        """
        Send an order over WebSocket: works around the missing pybit positional argument
        """
        # Provide a default no-op when the caller passes no callback, to avoid an error
        if callback is None:
            callback = lambda response: None 

        order_params = {
            "category": self.category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "orderLinkId": link_id,
            "positionIdx": pos_idx,
            "reduceOnly": is_reduce,
        }

        if order_type == "Limit":
            order_params["price"] = str(price)
            order_params["timeInForce"] = "PostOnly"
        else:
            order_params["timeInForce"] = "GTC"

        try:
            #  key fix: pass callback as the first positional argument
            self.ws_trade.place_order(callback, **order_params)
            
        except Exception as e:
            self.logger.error(f"❌ WebSocket order error: {e}")

    def start_stream(self, order_callback):
        self.ws_stream.order_stream(callback=order_callback)

    def cancel_all_http(self, symbol):
        """
        Cancel every pending order over HTTP: hardened version
        """
        try:
            #  fix: state the category explicitly and add settleCoin as a hint
            # For USDT perpetuals settleCoin must be "USDT"
            return self.http.cancel_all_orders(
                category=self.category, # make sure it is "linear"
                symbol=symbol,
                settleCoin="USDT"       # adding this usually clears error 110074
            )
        except Exception as e:
            print(f"❌ [{symbol}] HTTP cancel-all request failed: {e}")
            return {"retCode": -1, "retMsg": str(e)}
    
    def stop(self):
        try:
            if hasattr(self, 'ws_stream'): self.ws_stream.exit()
            if hasattr(self, 'ws_trade'): self.ws_trade.exit()
        except WebSocketConnectionClosedException: pass
        print("engine safety close")

# --- silence thread exceptions (unchanged) ---
def silent_thread_exception_handler(args):
    if args.exc_type == websocket._exceptions.WebSocketConnectionClosedException:
        return
    sys.__excepthook__(args.exc_type, args.exc_value, args.exc_traceback)

threading.excepthook = silent_thread_exception_handler
