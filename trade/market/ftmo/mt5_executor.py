import MetaTrader5 as mt5
import logging,time
from datetime import datetime, timezone
from trade.strategy.base_executor import BaseExecutor
from trade.strategy.strategy_ml import PositionDir

MT5_SYMBOL_FTMO_MAP = {"DOGEUSDT": "DOGEUSD", "ETHUSDT": "ETHUSD", "BTCUSDT": "BTCUSD"}

class MT5Executor(BaseExecutor):
    def __init__(self, path, symbol, magic, logger):
        self.symbol = MT5_SYMBOL_FTMO_MAP[symbol]
        self.magic = magic
        self.logger = logger
        self.path = path
        
        if not mt5.initialize(path=path):
            self.logger.error(f"❌ 初始化失败! 错误码: {mt5.last_error()}")
            raise RuntimeError("MT5 初始化失败")
        else:
            self.logger.info(f"init success | {self.magic}")
        
        # make sure the symbol exist
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"{symbol} not support | {self.magic}")

    def get_account_equity(self):
        """用于每日风控审计"""
        return mt5.account_info().equity

    def get_current_state(self):
        """
        返回当前持仓状态 (方向, 层数, 持仓均价) 
        注意：为了适配 TurtleBrain 的加仓判断，必须返回 price_open
        """
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        if not positions:
            return PositionDir.FLAT, 0, 0.0

        pos = positions[0] 
        direction = PositionDir.POSITIVE  if pos.type == 0 else PositionDir.NEGATIVE
        
        # 修正：返回 pos.price_open (开仓均价) 而不是 pos.volume
        # 这样 ftmo_turtle.py 里的 last_price 才能拿到正确的值
        return direction, 1, pos.price_open

    def get_server_time(self):
        tick = mt5.symbol_info_tick(self.symbol)
        server_time = datetime.fromtimestamp(tick.time)
        return server_time

    def user_order(self, size, is_buy, stop_loss=None, interval_ms=500):
        """
        size: Notional value in currency units
        is_buy: Boolean, True for BUY, False for SELL
        stop_loss: Percentage value (e.g., 0.02 for 2%)
        interval_ms: Delay between split orders in milliseconds
        """
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.logger.error(f"Symbol not found: {self.symbol}")
            return

        # 1. Get benchmark price (first tick before execution)
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            self.logger.error("Failed to get initial tick")
            return

        benchmark_price = tick.ask if is_buy else tick.bid
        benchmark_price = round(benchmark_price, symbol_info.digits)

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

        # 3. Execute split orders
        while remaining_lots > 0:
            current_batch_lots = min(remaining_lots, symbol_info.volume_max)

            current_batch_lots = round(
                round(current_batch_lots / symbol_info.volume_step) * symbol_info.volume_step,
                2
            )

            if current_batch_lots < symbol_info.volume_min:
                break

            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                self.logger.error("Tick fetch failed, aborting")
                break

            price = tick.ask if is_buy else tick.bid
            price = round(price, symbol_info.digits)

            if stop_loss is not None:
                sl_price = (
                    price * (1.0 - stop_loss)
                    if is_buy else price * (1.0 + stop_loss)
                )
                sl_price = round(sl_price, symbol_info.digits)
            else:
                sl_price = 0.0

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": current_batch_lots,
                "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl_price,
                "magic": self.magic,
                "comment": f"Split_Order_{order_count}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            res = mt5.order_send(request)

            if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                err_msg = res.comment if res else "Order failed"
                self.logger.error(f"Batch {order_count} failed: {err_msg}")
                break

            self.logger.info(
                f"Batch {order_count} executed | lots={current_batch_lots} | req_price={price}"
            )

            remaining_lots -= current_batch_lots
            remaining_lots = round(max(0.0, remaining_lots), 2)
            order_count += 1

            if remaining_lots > 0:
                time.sleep(interval_ms / 1000.0)

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
            f"slippage={slippage_pct*100:.4f}%"
        )
        
    def user_close(self, **kwargs):
        """全平当前 Magic 订单"""
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        for pos in positions:
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
            positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
            
            # 没有持仓
            if not positions:
                return None
            
            pos = positions[0]
            
            # MT5 返回的是秒级时间戳（int）
            open_time = pos.time
            
            if open_time is None or open_time == 0:
                return None
            
            return datetime.fromtimestamp(open_time, tz=timezone.utc)

        except Exception as e:
            self.logger.error(f"Failed to get last position open time: {e}")
            return None