import MetaTrader5 as mt5
import logging,time
from datetime import datetime, timezone
from trade.core.venue_base import VenueBase
from trade.core.protocol import OrderType, PositionDir

MT5_SYMBOL_FTMO_MAP = {"DOGEUSDT": "DOGEUSD", "ETHUSDT": "ETHUSD", "BTCUSDT": "BTCUSD"}

class MT5Venue(VenueBase):
    def __init__(
        self,
        path,
        symbol,
        magic,
        logger,
        login=None,
        password=None,
        server=None,
    ):
        self.symbol = MT5_SYMBOL_FTMO_MAP[symbol]
        self.magic = magic
        self.logger = logger
        self.path = path
        self.login = int(login) if login not in (None, "") else None
        self.password = password
        self.server = server

        self._connect_and_login()
        
        # make sure the symbol exist
        if not mt5.symbol_select(self.symbol, True):
            mt5.shutdown()
            raise RuntimeError(f"{symbol} not support | {self.magic}")

    def _connect_and_login(self):
        mt5.shutdown()
        initialize_args = {
            "path": self.path,
            "timeout": 60_000,
        }
        if self.login is not None:
            initialize_args["login"] = self.login
            if self.password is not None:
                initialize_args["password"] = self.password
            if self.server is not None:
                initialize_args["server"] = self.server

        initialized = mt5.initialize(**initialize_args)
        if initialized and self._is_target_account(mt5.account_info()):
            self.logger.info(
                "MT5 connection ready | login=%s magic=%s",
                self.login,
                self.magic,
            )
            return

        first_error = mt5.last_error()
        if self.login is None:
            self.logger.error(
                "MT5 initialization or account check failed | error=%s",
                first_error,
            )
            mt5.shutdown()
            raise RuntimeError("MT5 initialization failed")

        if not initialized:
            self.logger.warning(
                "MT5 credential initialization failed | error=%s; retrying login",
                first_error,
            )
            mt5.shutdown()
            if not mt5.initialize(path=self.path, timeout=60_000):
                error = mt5.last_error()
                self.logger.error("MT5 initialization retry failed | error=%s", error)
                mt5.shutdown()
                raise RuntimeError("MT5 initialization failed")
        else:
            account = mt5.account_info()
            self.logger.warning(
                "MT5 initialized with unexpected account | current=%s target=%s; "
                "retrying login",
                getattr(account, "login", None),
                self.login,
            )

        login_args = {"login": self.login}
        if self.password is not None:
            login_args["password"] = self.password
        if self.server is not None:
            login_args["server"] = self.server
        if not mt5.login(**login_args):
            error = mt5.last_error()
            self.logger.error(
                "MT5 login failed | login=%s server=%s error=%s",
                self.login,
                self.server,
                error,
            )
            mt5.shutdown()
            raise RuntimeError("MT5 login failed")

        account = mt5.account_info()
        if not self._is_target_account(account):
            self.logger.error(
                "MT5 account verification failed | current=%s target=%s",
                getattr(account, "login", None),
                self.login,
            )
            mt5.shutdown()
            raise RuntimeError("MT5 account verification failed")

        self.logger.info(
            "MT5 login succeeded | login=%s server=%s magic=%s",
            self.login,
            self.server,
            self.magic,
        )

    def _is_target_account(self, account):
        if account is None:
            return False
        if self.login is None:
            return True
        return int(account.login) == self.login

    def _ensure_target_account(self):
        account = mt5.account_info()
        if self._is_target_account(account):
            return account

        self.logger.warning(
            "MT5 account changed or disconnected | current=%s target=%s; reconnecting",
            getattr(account, "login", None),
            self.login,
        )
        self._connect_and_login()
        account = mt5.account_info()
        if not self._is_target_account(account):
            raise RuntimeError("MT5 target account is not active")
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"MT5 symbol is unavailable after login: {self.symbol}")
        return account

    def shutdown(self):
        mt5.shutdown()

    def get_account_equity(self):
        """Used by the daily risk audit"""
        return self._ensure_target_account().equity

    def get_current_state(self):
        """
        Current position state (direction, layers, average entry price)
        Note: price_open must be returned for TurtleStrategy's pyramiding check
        """
        self._ensure_target_account()
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        if not positions:
            return PositionDir.FLAT, 0, 0.0

        pos = positions[0] 
        direction = PositionDir.POSITIVE  if pos.type == 0 else PositionDir.NEGATIVE
        
        # Fix: return pos.price_open (average entry price) instead of pos.volume
        # so that last_price in ftmo_turtle.py gets the correct value
        return direction, 1, pos.price_open

    def get_server_time(self):
        self._ensure_target_account()
        tick = mt5.symbol_info_tick(self.symbol)
        server_time = datetime.fromtimestamp(tick.time)
        return server_time

    def submit_order(
        self,
        size,
        is_buy,
        stop_loss_pct=None,
        take_profit_pct=None,
        interval_ms=500,
        *,
        order_type=OrderType.MARKET,
        price=None,
    ):
        """
        size: Notional value in currency units
        is_buy: Boolean, True for BUY, False for SELL
        stop_loss_pct: Percentage value, e.g. 0.02 for 2%
        take_profit_pct: Percentage value, e.g. 0.04 for 4%
        interval_ms: Delay between split orders in milliseconds
        """
        order_type, price = self.normalize_order_request(order_type, price)
        self._ensure_target_account()
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol not found: {self.symbol}")
            return

        # 2. Calculate total lots
        total_raw_lots = float(size / symbol_info.trade_contract_size)
        total_lots = round(total_raw_lots / symbol_info.volume_step) * symbol_info.volume_step
        total_lots = round(total_lots, 2)

        if total_lots < symbol_info.volume_min:
            self.logger.warning(f"Total lots {total_lots} below minimum {symbol_info.volume_min}")
            return

        self.logger.info(
            f"Start execution: total_lots={total_lots} | max_per_order={symbol_info.volume_max}"
        )

        remaining_lots = total_lots
        order_count = 0
        benchmark_price = None
        responses = []

        # 3. Execute split orders
        while remaining_lots > 0:
            current_batch_lots = min(remaining_lots, symbol_info.volume_max)

            current_batch_lots = round(
                round(current_batch_lots / symbol_info.volume_step) * symbol_info.volume_step,
                2
            )

            if current_batch_lots < symbol_info.volume_min:
                break

            if order_type == OrderType.MARKET:
                tick = mt5.symbol_info_tick(self.symbol)
                if tick is None:
                    self.logger.error("Tick fetch failed, aborting")
                    break
                order_price = tick.ask if is_buy else tick.bid
            else:
                order_price = price
            order_price = round(order_price, symbol_info.digits)
            if benchmark_price is None:
                benchmark_price = order_price

            # Stop Loss
            if stop_loss_pct is not None:
                sl_price = (
                    order_price * (1.0 - stop_loss_pct)
                    if is_buy else order_price * (1.0 + stop_loss_pct)
                )
                sl_price = round(sl_price, symbol_info.digits)
            else:
                sl_price = 0.0

            # Take Profit
            if take_profit_pct is not None:
                tp_price = (
                    order_price * (1.0 + take_profit_pct)
                    if is_buy else order_price * (1.0 - take_profit_pct)
                )
                tp_price = round(tp_price, symbol_info.digits)
            else:
                tp_price = 0.0

            request = {
                "action": (
                    mt5.TRADE_ACTION_DEAL
                    if order_type == OrderType.MARKET
                    else mt5.TRADE_ACTION_PENDING
                ),
                "symbol": self.symbol,
                "volume": current_batch_lots,
                "type": (
                    mt5.ORDER_TYPE_BUY
                    if order_type == OrderType.MARKET and is_buy
                    else mt5.ORDER_TYPE_SELL
                    if order_type == OrderType.MARKET
                    else mt5.ORDER_TYPE_BUY_LIMIT
                    if is_buy
                    else mt5.ORDER_TYPE_SELL_LIMIT
                ),
                "price": order_price,
                "sl": sl_price,
                "tp": tp_price,
                "magic": self.magic,
                "comment": f"Split_Order_{order_count}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": (
                    mt5.ORDER_FILLING_IOC
                    if order_type == OrderType.MARKET
                    else mt5.ORDER_FILLING_RETURN
                ),
            }

            self._ensure_target_account()
            res = mt5.order_send(request)
            responses.append(res)

            success_codes = {
                mt5.TRADE_RETCODE_DONE,
                getattr(mt5, "TRADE_RETCODE_PLACED", mt5.TRADE_RETCODE_DONE),
            }
            if res is None or res.retcode not in success_codes:
                err_msg = res.comment if res else "Order failed"
                self.logger.error(f"Batch {order_count} failed: {err_msg}")
                break

            self.logger.info(
                f"Batch {order_count} executed | "
                f"lots={current_batch_lots} | req_price={order_price} | "
                f"SL={sl_price} | TP={tp_price}"
            )

            remaining_lots -= current_batch_lots
            remaining_lots = round(max(0.0, remaining_lots), 2)
            order_count += 1

            if remaining_lots > 0:
                time.sleep(interval_ms / 1000.0)

        if order_type == OrderType.LIMIT:
            return responses

        # 4. Wait briefly to ensure position is updated
        time.sleep(0.2)

        # 5. Get current positions and compute weighted average price
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)

        if not positions:
            self.logger.error("No positions found after execution")
            return

        total_volume = 0.0
        weighted_price_sum = 0.0

        for p in positions:
            total_volume += p.volume
            weighted_price_sum += p.volume * p.price_open

        if total_volume == 0:
            self.logger.error("Total position volume is zero")
            return

        avg_price = weighted_price_sum / total_volume

        # 6. Calculate slippage
        slippage = (avg_price - benchmark_price) if is_buy else (benchmark_price - avg_price)
        slippage_pct = slippage / benchmark_price

        self.logger.info(
            f"Execution finished: batches={order_count} | "
            f"benchmark_price={benchmark_price} | avg_price={avg_price:.6f} | "
            f"slippage={slippage_pct * 100:.4f}%"
        )
        return responses
        
    def close_position(self, **kwargs):
        """Close every position of the current magic number"""
        self._ensure_target_account()
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        for pos in positions:
            self._ensure_target_account()
            tick = mt5.symbol_info_tick(self.symbol)
            mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "position": pos.ticket,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                "price": tick.bid if pos.type == 0 else tick.ask,
                "magic": self.magic,
                # "type_filling": mt5.ORDER_FILLING_IOC,
            })
        self.logger.info(f"order close {self.magic}")

    def get_last_position_open_time(self):
        try:
            self._ensure_target_account()
            positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
            
            # No position
            if not positions:
                return None
            
            pos = positions[0]
            
            # MT5 returns a second level timestamp (int)
            open_time = pos.time
            
            if open_time is None or open_time == 0:
                return None
            
            return datetime.fromtimestamp(open_time, tz=timezone.utc)

        except Exception as e:
            self.logger.error(f"Failed to get last position open time: {e}")
            return None
