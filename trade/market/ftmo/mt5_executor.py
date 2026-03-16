import MetaTrader5 as mt5
import logging
from datetime import datetime
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

    def user_order(self, size, is_buy, stop_loss=None):
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            self.logger.error(f"❌ can't find symbol: {self.symbol} | {self.magic}")
            return

        # 1. 计算原始手数
        raw_lots = float(size / symbol_info.trade_contract_size)
        
        # 2. 强制对齐步长 (解决 6.14 这种无效数值)
        # 例如：step 为 0.1，则 6.14 会变成 6.1
        lots = round(raw_lots / symbol_info.volume_step) * symbol_info.volume_step
        
        # 3. 限制在 [最小值, 最大值] 范围内 (解决 1K 这种越权数值)
        lots = max(symbol_info.volume_min, min(symbol_info.volume_max, lots))
        
        # 4. 获取价格并对齐精度
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None: return
        
        price = tick.ask if is_buy else tick.bid
        sl_price = price * (1.0 - stop_loss) if is_buy else price * (1.0 + stop_loss)
        
        # 价格也要 round 到品种的小数位数
        price = round(price, symbol_info.digits)
        sl_price = round(sl_price, symbol_info.digits)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": round(lots, 2), # 最终确保传给服务器的是干净的浮点数
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl_price,
            "magic": self.magic,
            "comment": "Turtle_Live",
            # "type_filling": mt5.ORDER_FILLING_IOC, # 如果还报错，尝试换成 ORDER_FILLING_FOK/ORDER_FILLING_IOC/ORDER_FILLING_RETURN
        }
        
        res = mt5.order_send(request)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            err_msg = res.comment if res else "Unknown Error"
            self.logger.error(f"order fail: {err_msg} | volume: {lots} | {self.magic}")
            if res is None:
                code, msg = mt5.last_error()
                self.logger.error(f"order_send return None | last_error: {code} - {msg}")
        else:
            self.logger.info(f"order success Lots {lots} | {self.magic}")

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