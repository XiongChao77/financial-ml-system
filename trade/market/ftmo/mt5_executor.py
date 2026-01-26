import MetaTrader5 as mt5
import logging
from datetime import datetime
from trade.strategy.base_executor import BaseExecutor
from trade.strategy.strategy_ml import PositionDir

class MT5Executor(BaseExecutor):
    def __init__(self, path, symbol, magic, logger):
        self.symbol = symbol
        self.magic = magic
        self.logger = logger
        
        if not mt5.initialize(path=path):
            self.logger.error(f"❌ 初始化失败! 错误码: {mt5.last_error()}")
            raise RuntimeError("MT5 初始化失败")
        
        # 确保品种已在市场报价中
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"{symbol} not support")

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
        direction = PositionDir.LONG if pos.type == 0 else PositionDir.SHORT
        
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
            self.logger.error(f"❌ 找不到品种信息: {self.symbol}")
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
            "type_filling": mt5.ORDER_FILLING_IOC, # 如果还报错，尝试换成 ORDER_FILLING_FOK
        }
        
        res = mt5.order_send(request)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            err_msg = res.comment if res else "Unknown Error"
            self.logger.error(f"❌ 下单失败: {err_msg} | 尝试手数: {lots}")
        else:
            self.logger.info(f"✅ 下单成功: {lots} Lots")

    def user_order_target_percent(self, target_pct):
        """实现按百分比调仓逻辑"""
        equity = self.get_account_equity()
        current_dir, _, current_vol = self.get_current_state()
        
        if target_pct == 0:
            self.user_close()
            return

        # 若方向反转，先平后开 (满足单层仓位逻辑)
        target_is_buy = target_pct > 0
        if current_dir != PositionDir.FLAT:
            if (target_is_buy and current_dir == PositionDir.SHORT) or \
               (not target_is_buy and current_dir == PositionDir.LONG):
                self.user_close()

        tick = mt5.symbol_info_tick(self.symbol)
        target_value = abs(target_pct) * equity
        size = target_value / tick.bid # 将金额转换为币数
        
        return self.user_order(size, target_is_buy)

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
                "type_filling": mt5.ORDER_FILLING_IOC,
            })

    def close_all(self):
        """别名方法适配 test_execution"""
        self.user_close()