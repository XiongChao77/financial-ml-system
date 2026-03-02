import logging
import sys,os
from datetime import datetime, timezone

# 路径适配
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..', '..'))

from trade.market.bybit.bybit_engine import BybitEngine
from trade.strategy.strategy_ml import PositionDir, ActionType
from trade.strategy.base_executor import BaseExecutor

class BybitExecutor(BaseExecutor):
    def __init__(self, key_path,rsa_pem_path, symbol: str):
        self.engine = BybitEngine(key_path,rsa_pem_path)
        self.symbol = symbol
        self.logger = logging.getLogger("BybitExecutor")
        
        # 初始化精度信息
        self.qty_step = 0.0
        self.tick_size = 0.0
        self.min_qty = 0.0
        self._init_symbol_info()

    def _init_symbol_info(self):
        """同步交易所精度配置，防止 Invalid Volume"""
        try:
            res = self.engine.http.get_instruments_info(category="linear", symbol=self.symbol)
            if res['retCode'] == 0:
                info = res['result']['list'][0]
                self.qty_step = float(info['lotSizeFilter']['qtyStep'])
                self.min_qty = float(info['lotSizeFilter']['minOrderQty'])
                self.tick_size = float(info['priceFilter']['tickSize'])
                self.logger.info(f"✅ 精度同步: QtyStep={self.qty_step}, Tick={self.tick_size}")
        except Exception as e:
            self.logger.error(f"精度同步失败: {e}")

    def get_account_equity(self):
        """获取 USDT 账户净值"""
        res = self.engine.http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if res.get('retCode') == 0:
            return float(res['result']['list'][0]['coin'][0]['equity'])
        return 0.0

    def get_current_state(self):
        """
        返回: (PositionDir, layers, avg_price)
        适配 TurtleBrain 的接口需求
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
            
            # 简单的层数估算（海龟逻辑通常需要自己记录，这里简化为 1 层代表有持仓）
            # 如果需要严格的层数逻辑，需要在外部记录或通过 size/unit_size 推算
            direction = PositionDir.POSITIVE  if side == 'Buy' else PositionDir.NEGATIVE
            return direction, 1, avg_price

        except Exception as e:
            self.logger.error(f"获取持仓状态失败: {e}")
            return PositionDir.FLAT, 0, 0.0

    def get_server_time(self):
        return datetime.now(timezone.utc)
    
    def user_order(self, size, is_buy, stop_loss=None):
        """
        执行下单逻辑
        size: 币的数量 (Base Coin)
        is_buy: 方向
        stop_loss: 止损比例 (如 0.05 代表 5%)
        """
        # 1. 精度对齐
        qty = round(float(size) / self.qty_step) * self.qty_step
        qty = max(self.min_qty, qty)
        qty_str = str(qty)

        # 2. 获取当前价格用于计算 SL 价格
        # 注意：这里使用市价单，所以 entry_price 近似为当前 ticker 价格
        tickers = self.engine.http.get_tickers(category="linear", symbol=self.symbol)
        curr_price = float(tickers['result']['list'][0]['lastPrice'])
        
        # 3. 计算止损价格 (Bybit 需要具体价格，BrainBase 给的是比例)
        sl_price = 0.0
        if stop_loss:
            if is_buy:
                raw_sl = curr_price * (1 - stop_loss)
            else:
                raw_sl = curr_price * (1 + stop_loss)
            sl_price = round(raw_sl / self.tick_size) * self.tick_size

        side = "Buy" if is_buy else "Sell"
        self.logger.info(f"🐢 执行下单: {side} {qty_str} @ 市价 | SL: {sl_price}")

        try:
            # 4. 调用 engine 下单，带上 stopLoss 参数
            # Market 单不需要传 price
            order_params = {
                "category": "linear",
                "symbol": self.symbol,
                "side": side,
                "orderType": "Market",
                "qty": qty_str,
                "positionIdx": 0, # 单向持仓模式
                "reduceOnly": False
            }
            
            if sl_price > 0:
                order_params["stopLoss"] = str(sl_price)

            # 使用 HTTP 接口下单更稳妥，或者用 engine.ws_trade.place_order
            # 这里为了简单直接用 HTTP，因为海龟不是高频策略
            res = self.engine.http.place_order(**order_params)
            
            if res['retCode'] == 0:
                self.logger.info(f"✅ 下单成功: ID {res['result']['orderId']}")
            else:
                self.logger.error(f"❌ 下单失败: {res['retMsg']}")
                
        except Exception as e:
            self.logger.error(f"下单异常: {e}")

    def user_close(self):
        """全平当前持仓"""
        try:
            # 获取持仓
            res = self.engine.http.get_positions(category="linear", symbol=self.symbol)
            for pos in res['result']['list']:
                size = float(pos['size'])
                if size > 0:
                    side = "Sell" if pos['side'] == "Buy" else "Buy"
                    self.logger.info(f"正在平仓: {pos['side']} {size}")
                    
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
            self.logger.error(f"平仓异常: {e}")