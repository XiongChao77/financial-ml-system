import MetaTrader5 as mt5
import logging, sys, os, time
import math

current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..'))

from trade.strategy.base_executor import BaseExecutor
from trade.strategy.strategy_ftmo import PositionDir, ActionType

class MT5Executor(BaseExecutor):
    """
    MT5 执行器：实现与 BtExecutor 一致的订单逻辑 (反手/加减仓/SLTP)
    适配加密货币 (自动获取 Contract Size 和 Digits)
    """
    def __init__(self, symbol, magic_number, sl_scale=1.0, tp_ratio=0.90):
        """
        :param sl_scale: 止损缩放系数. 最终止损距离 = stop_threshold * sl_scale
        :param tp_ratio: 止盈相对止损的盈亏比. TP距离 = stop_threshold * tp_ratio (或者基于价格的比例)
                         注意：BtExecutor里写的是 limit_price = price * (1.0 + self.params.take_profit)
                         如果是固定比例止盈，这里保持一致。
        """
        self.symbol = symbol
        self.magic = magic_number
        self.sl_scale = sl_scale   
        self.tp_ratio = tp_ratio   
        
        self.logger = logging.getLogger("ftmo_live")
        
        if not mt5.initialize():
            self.logger.critical("MT5 Initialize Failed!")
            raise RuntimeError("MT5 Init Failed")
            
        self.logger.info(f"Connected to MT5. Account: {mt5.account_info().login}")
        
        # === 关键：动态获取品种规格 (适配 Crypto) ===
        info = mt5.symbol_info(self.symbol)
        if not info:
            raise ValueError(f"Symbol {self.symbol} not found in MT5")
            
        self.digits = info.digits             # 价格精度 (BTC=2, DOGE=5)
        self.min_vol = info.volume_min        # 最小手数
        self.vol_step = info.volume_step      # 手数步长
        self.contract_size = info.trade_contract_size # 合约大小 (1手=多少币)
        
        # 市场上下文缓存 (由 LiveBot 每轮注入)
        self._ctx_stop_threshold = 0.05 # 默认 5% (防止未注入时报错)

    def update_context(self, stop_threshold_pct):
        """
        [关键] 每轮循环前调用，注入策略计算出的动态波动率阈值
        """
        self._ctx_stop_threshold = stop_threshold_pct
        self.logger.debug(f"Context Updated: StopThreshold={stop_threshold_pct:.2%}")

    def get_current_state(self):
        """获取当前持仓状态 (方向, 层数, 手数)"""
        positions = mt5.positions_get(symbol=self.symbol)
        my_pos = [p for p in positions if p.magic == self.magic]
        
        if not my_pos:
            return PositionDir.FLAT, 0, 0.0
        
        # 聚合持仓
        total_vol = sum(p.volume for p in my_pos)
        # 判断方向 (简单逻辑：取第一个订单的方向)
        direction = PositionDir.LONG if my_pos[0].type == mt5.ORDER_TYPE_BUY else PositionDir.SHORT
        layers = 1 if total_vol > 0 else 0

        # self.logger.debug(f"State Check: {direction.name} | Vol: {total_vol} | PosCount: {len(my_pos)}")
        return direction, layers, total_vol

    def user_order_target_percent(self, target_pct: float):
        """
        [核心] 全能下单函数：将目标仓位比例转换为具体的买卖操作
        """
        # 1. 获取实时价格 (Bid/Ask)
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            self.logger.error(f"Failed to get tick for {self.symbol}")
            return
            
        # 使用中间价计算市值 (公允价)
        mid_price = (tick.bid + tick.ask) / 2.0
        
        # 2. 获取账户余额
        account = mt5.account_info()
        if not account: return
        equity = account.equity
        
        # 3. 获取当前持仓 (转换为带符号的 volume)
        curr_dir, _, curr_vol = self.get_current_state()
        current_signed_vol = curr_vol if curr_dir == PositionDir.LONG else -curr_vol
        
        # 4. 计算目标手数
        # 公式：TargetValue = Balance * Pct
        #       Lots = TargetValue / (Price * ContractSize)
        target_value = equity * target_pct # target_pct 带符号 (+/-)
        
        # 防御除零
        c_size = self.contract_size if self.contract_size > 0 else 1.0
        raw_target_lots = target_value / (mid_price * c_size)
        target_lots_signed = self._round_volume(raw_target_lots)
        
        # 只有当目标发生变化时才打印，或者设为 DEBUG
        self.logger.debug(
            f"Calc Lots: Equity={equity:.2f} * Pct={target_pct:.2f} = Val={target_value:.2f} | "
            f"Price={mid_price:.2f} * Size={c_size} | "
            f"RawLots={raw_target_lots:.4f} -> Target={target_lots_signed} (Curr={current_signed_vol})"
        )

        # 如果目标与当前差距太小(小于半个步长)，忽略
        if math.isclose(target_lots_signed, current_signed_vol, abs_tol=self.min_vol/2):
            return

        # 5. 判断操作类型 (反手 / 加仓 / 减仓)
        
        # === A: 反手 (Reverse) ===
        # 符号相反，且都不为0 (或者从有仓位变到反向仓位)
        is_reversing = (current_signed_vol > 0 and target_lots_signed < 0) or \
                       (current_signed_vol < 0 and target_lots_signed > 0)
                       
        if is_reversing:
            self.logger.info(f"🔄 反手: {current_signed_vol} -> {target_lots_signed}")
            self.close_all() # 全平
            time.sleep(1)    # 等待成交
            
            # 开新仓
            new_vol = abs(target_lots_signed)
            is_buy = target_lots_signed > 0
            self._open_position(new_vol, is_buy)
            return

        # === B: 开仓 / 加仓 (Increase) ===
        if abs(target_lots_signed) > abs(current_signed_vol):
            gap_vol = abs(target_lots_signed) - abs(current_signed_vol)
            gap_vol = self._round_volume(gap_vol)
            
            if gap_vol >= self.min_vol:
                is_buy = target_lots_signed > 0
                action_name = "开仓" if current_signed_vol == 0 else "加仓"
                self.logger.info(f"➕ {action_name}: 目标 {target_lots_signed}, 需加 {gap_vol}")
                self._open_position(gap_vol, is_buy)
        
        # === C: 减仓 (Reduce) ===
        elif abs(target_lots_signed) < abs(current_signed_vol):
            gap_vol = abs(current_signed_vol) - abs(target_lots_signed)
            gap_vol = self._round_volume(gap_vol)
            
            if gap_vol >= self.min_vol:
                self.logger.info(f"➖ 减仓: 目标 {target_lots_signed}, 需减 {gap_vol}")
                # 减仓不需要 SL/TP，直接反向开单平仓
                self._reduce_position(gap_vol, is_buy_close=(current_signed_vol > 0))

    def user_close(self, size=None, **kwargs):
        """Brain 调用的强制平仓接口"""
        if size is None:
            self.close_all()
        else:
            # 部分平仓
            curr_dir, _, curr_vol = self.get_current_state()
            if curr_dir == PositionDir.FLAT: return
            
            reduce_vol = self._round_volume(size)
            if reduce_vol > curr_vol: reduce_vol = curr_vol
            
            self._reduce_position(reduce_vol, is_buy_close=(curr_dir == PositionDir.LONG))

    # ----------------------------------------------------------------
    # 内部具体执行逻辑
    # ----------------------------------------------------------------

    def _open_position(self, volume, is_buy):
        """执行开仓，并自动计算 SL/TP"""
        # 获取最新报价 (用于计算 SL/TP 的基准)
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick: return
        
        # 买单用 Ask 开仓，卖单用 Bid 开仓
        price = tick.ask if is_buy else tick.bid
        
        # === 计算 SL / TP 价格 ===
        # 根据 update_context 注入的 stop_threshold (百分比) 计算
        # BtExecutor逻辑: stop_loss_dist = stop_threshold * sl_scale
        sl_pct_dist = self._ctx_stop_threshold * self.sl_scale
        tp_pct_dist = self.tp_ratio # 假设止盈是固定比例 (如 0.5 即 50%)
        
        sl_price = 0.0
        tp_price = 0.0
        
        if is_buy:
            # 多单：SL 在下方，TP 在上方
            sl_price = price * (1.0 - sl_pct_dist)
            tp_price = price * (1.0 + tp_pct_dist)
            order_type = mt5.ORDER_TYPE_BUY
        else:
            # 空单：SL 在上方，TP 在下方
            sl_price = price * (1.0 + sl_pct_dist)
            tp_price = price * (1.0 - tp_pct_dist)
            order_type = mt5.ORDER_TYPE_SELL
            
        # === 精度修正 (Crypto 必须) ===
        sl_price = round(sl_price, self.digits)
        tp_price = round(tp_price, self.digits)
        
        # 发送订单
        self._send_order(order_type, volume, price, sl_price, tp_price, comment="Open/Add")

    def _reduce_position(self, volume, is_buy_close):
        """
        减仓逻辑 (FIFO):
        如果是 Hedging 账户，建议遍历持仓按时间平仓。
        如果是 Netting 账户，直接反向开单即可。
        这里为了通用性，采用『遍历平仓』的方式 (模拟 FIFO)。
        """
        positions = mt5.positions_get(symbol=self.symbol)
        my_pos = [p for p in positions if p.magic == self.magic]
        
        # 按开仓时间排序 (旧的在前)
        my_pos.sort(key=lambda x: x.time_msc)
        
        remaining = volume
        
        for pos in my_pos:
            if remaining <= 0: break
            
            # 只有方向匹配才平 (多单只能平多单)
            pos_is_buy = (pos.type == mt5.ORDER_TYPE_BUY)
            if pos_is_buy != is_buy_close:
                continue
                
            close_vol = min(pos.volume, remaining)
            close_vol = self._round_volume(close_vol)
            
            if close_vol < self.min_vol: continue
            
            # 执行平仓 (对冲模式下 Close By Ticket)
            self._close_by_ticket(pos.ticket, close_vol, is_buy_close)
            
            remaining -= close_vol
            remaining = self._round_volume(remaining)

    def _close_by_ticket(self, ticket, volume, is_buy_close):
        """按 TicketID 平仓"""
        tick = mt5.symbol_info_tick(self.symbol)
        # 平多(Sell), 平空(Buy)
        order_type = mt5.ORDER_TYPE_SELL if is_buy_close else mt5.ORDER_TYPE_BUY
        price = tick.bid if is_buy_close else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket, # 指定 Ticket
            "price": price,
            "magic": self.magic,
            "comment": "Reduce FIFO",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(request)

    def _send_order(self, order_type, volume, price, sl, tp, comment=""):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 50, # 滑点
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        self.logger.debug(f"Sending Order: {request}")
        res = mt5.order_send(request)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.error(f"Order Fail: {res.comment} (Code: {res.retcode})")
        else:
            self.logger.info(f"Order OK: #{res.order} | {order_type} {volume} lots @ {price} | SL: {sl} | TP: {tp}")

    def close_all(self):
        """平掉所有持仓"""
        positions = mt5.positions_get(symbol=self.symbol)
        my_pos = [p for p in positions if p.magic == self.magic]
        for pos in my_pos:
            is_buy = (pos.type == mt5.ORDER_TYPE_BUY)
            self._close_by_ticket(pos.ticket, pos.volume, is_buy)

    def _round_volume(self, vol):
        if vol < self.min_vol: return 0.0
        
        steps = round(vol / self.vol_step)
        rounded = steps * self.vol_step
        
        # 动态计算需要保留的小数位
        # 例如 step=0.01 -> decimals=2; step=0.001 -> decimals=3
        decimals = 0
        if self.vol_step < 1:
            decimals = int(math.ceil(-math.log10(self.vol_step)))
        
        return round(rounded, decimals)