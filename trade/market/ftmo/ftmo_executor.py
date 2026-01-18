import MetaTrader5 as mt5
import logging
from trade.strategy.base_executor import BaseExecutor
from trade.strategy.strategy_ftmo import PositionDir

class MT5Executor(BaseExecutor):
    def __init__(self, symbol, magic):
        self.symbol = symbol
        self.magic = magic
        self.logger = logging.getLogger("MT5Executor")
        if not mt5.initialize():
            raise RuntimeError("MT5 初始化失败")

    def get_account_equity(self):
        """实时获取净值，用于 Brain 内部的 FTMO 每日风控"""
        return mt5.account_info().equity

    def get_current_state(self):
        """
        精简状态同步：只需返回是否有持仓
        因为 max_layers=1，所以层数只有 0 或 1
        """
        positions = mt5.positions_get(symbol=self.symbol, magic=self.magic)
        if not positions:
            return PositionDir.FLAT, 0, 0.0 # 方向, 层数, 入场价

        pos = positions[0] # 只取第一笔
        direction = PositionDir.LONG if pos.type == 0 else PositionDir.SHORT
        return direction, 1, pos.price_open

    def user_order(self, size, is_buy, stop_loss=None):
        """执行下单：自动处理 Units -> Lots 转换"""
        symbol_info = mt5.symbol_info(self.symbol)
        # 核心转换：Lots = 币数 / 合约大小 (DOGE 通常是 1 或 10000)
        lots = float(size / symbol_info.trade_contract_size)
        
        # 价格与止损
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.ask if is_buy else tick.bid
        sl_price = price * (1.0 - stop_loss) if is_buy else price * (1.0 + stop_loss)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl_price,
            "magic": self.magic,
            "comment": "Turtle_L1",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"下单失败: {res.comment}")

    def user_close(self, **kwargs):
        """全平当前 Magic 下的所有单子"""
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
                "type_filling": mt5.ORDER_FILLING_IOC,
            })