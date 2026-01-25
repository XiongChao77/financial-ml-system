from enum import IntEnum
import backtrader as bt
import logging,sys,os
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..'))
# 引入自定义模块
from data_process import common
from trade.strategy.base_executor import BaseExecutor

class BtExecutor(BaseExecutor,bt.Strategy):
    params = dict(
        stop_loss = 0.05,  # 5% 止损
        take_profit = 0.50, # 10% 止盈 (可选)
        position_ratio = 1,
    )

    def __init__(self):
        self.logger = logging.getLogger("trade")
        # === 关键：用于记录每一笔"存活"的交易组 ===
        # 结构: [{'id': id, 'stop': stop_ord, 'limit': limit_ord, 'size': size}, ...]
        self.live_trades = []
        self.closed_pnl = []

    def user_order_target_percent(self, target_pct:float, stop_loss: float = None, take_profit = None):
        """
        全能下单函数 (带订单追踪版):
        1. 自动计算股数
        2. 反手：全平 + 清空记录 -> 开新仓
        3. 加仓：开带止损的新单 -> 记录到 live_trades
        4. 减仓：FIFO (先进先出) 逻辑 -> 取消旧单止损 -> 平仓
        """
        # 1. 获取基础数据
        total_value = self.broker.get_value()
        current_price = self.data.close[0]
        current_size = self.position.size 
        
        # 2. 计算目标股数
        target_value = total_value * target_pct
        target_size = target_value / current_price
        
        if target_size == current_size:
            return

        # 3. 计算差额
        gap_size = target_size - current_size

        # 4. 判断操作类型
        
        # === A: 反手 (Reverse) ===
        is_reversing = (current_size > 0 and target_size < 0) or \
                       (current_size < 0 and target_size > 0)
        
        if is_reversing:
            self.logger.debug(f'反手从 {current_size} 到 {target_size}')
            
            # 1. 彻底清空旧状态
            self.close() # 全平仓位
            self._cancel_all_live_orders() # 辅助函数：取消所有挂单
            self.live_trades.clear()       # 清空记录队列
            
            # 2. 开新仓 (反手后 gap_size 实际上就是 target_size，因为旧的已经视为0了)
            # 但为了逻辑统一，这里直接按 target_size 开仓
            action_size = abs(target_size)
            if target_size > 0: 
                self._open_bracket(action_size, is_buy=True, stop_loss=stop_loss, take_profit=take_profit)
            else:
                self._open_bracket(action_size, is_buy=False, stop_loss=stop_loss, take_profit=take_profit)
            return

        # === B: 同向加仓 (Increase) ===
        if abs(target_size) > abs(current_size):
            action_size = abs(gap_size)
            is_buy = target_size > 0
            
            self.logger.debug(f'【加仓】方向:{"多" if is_buy else "空"} 数量:{action_size}, current_size:{current_size}')
            self._open_bracket(action_size, is_buy=is_buy, stop_loss=stop_loss)

        # === C: 同向减仓 (Reduce - FIFO模式) ===
        elif abs(target_size) < abs(current_size):
            reduce_amount = abs(gap_size)
            self.logger.debug(f'【减仓】需要减持: {reduce_amount}')
            
            # 调用专门的减仓处理函数
            self._reduce_position_fifo(reduce_amount, is_buy_close=(current_size > 0))

    def user_order(self, size, is_buy, stop_loss=None, take_profit=None):
        self._open_bracket(abs(size), is_buy=is_buy, stop_loss=stop_loss, take_profit= take_profit)

    def user_close(self, size=None, **kwargs):
        self.logger.debug(f"user_close ammount :{size}")
        current_size = self.position.size
        if size is None or size >= current_size:
            self.close(**kwargs) # 全平仓位
            self._cancel_all_live_orders() # 辅助函数：取消所有挂单
            self.live_trades.clear() 
        else:
            gap_size = current_size - size
            self._reduce_position_fifo(gap_size, is_buy_close=(current_size > 0))
    # ----------------------------------------------------------------
    # 辅助逻辑封装
    # ----------------------------------------------------------------
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            # --- 核心：判定订单的真实意图 ---
            
            # 1. 判定是否为“增仓”（开仓或加仓）
            # 如果买入时原本持有多头或空仓，或者卖出时原本持有空头或空仓
            is_entry = (order.isbuy() and self.position.size >= 0) or \
                       (not order.isbuy() and self.position.size <= 0)

            if is_entry:
                type_str = "🚀 开仓/加仓 (ENTRY)"
            else:
                # 2. 如果是“减仓”，则根据类型和盈亏判定意图
                if order.exectype == bt.Order.Stop:
                    type_str = "🛡️ 硬核止损 (STOP LOSS)"
                elif order.exectype == bt.Order.Limit:
                    type_str = "🎯 自动止盈 (TAKE PROFIT)"
                else:
                    # 如果是市价平仓，根据盈亏判定是止损还是止盈离场
                    # 注意：这里需要对比执行价与入场均价
                    pnl = (order.executed.price - self.position.price) * self.position.size
                    type_str = "🛑 信号止损 (SIGNAL SL)" if pnl < 0 else "🛑 信号止盈 (SIGNAL TP)"

            direction = "🟢 买入" if order.isbuy() else "🔴 卖出"
            self.logger.debug(
                f"✅ 【订单成交】 {direction} | 意图: {type_str} | "
                f"价格: {order.executed.price:.4f} | 数量: {order.executed.size:.2f}"
            )

        # 3. 订单失败判定
        elif order.status == order.Margin:
            self.logger.error(f"❌ 【订单失败】 保证金不足！价格: {order.created.price:.4f} | 数量: {order.created.size:.2f}")
        elif order.status == order.Rejected:
            self.logger.error(f"❌ 【订单失败】 订单被拒绝！")
        elif order.status == order.Canceled:
            self.logger.debug(f"⚠️ 【订单取消】 订单已撤单。")

    def notify_trade(self, trade):
        """
        交易通知：只有当一笔交易平仓（不管是止盈还是止损）时，才会触发此函数
        这里才能拿到真正的盈亏数据。
        """
        # if not trade.isclosed:
        #     return

        # trade.pnl: 毛利 (不含手续费)
        # trade.pnlcomm: 净利 (含手续费)
        # 记录净利润 (含手续费)
        self.closed_pnl.append(trade.pnlcomm)
        # 打印包含手续费的净盈亏
        direction = ( "🟢 多" if trade.size > 0  else "🔴 空" )
        self.logger.debug(f"💸 交易结算 {direction} | price {trade.price} | 毛利: {trade.pnl:.2f} | 手续费: {trade.commission:.2f} | 净利: {trade.pnlcomm:.2f}")

        # 如果你想把盈亏回写到上面的 trade_logs 里，比较麻烦，
        # 因为 trade_logs 是按单(order)记的，而这里是按回合(trade)记的。
        # 通常建议单独存一个 closed_trades 列表。

    def _open_bracket(self, size, is_buy, stop_loss=None, take_profit=None):
        """执行 Bracket 下单并记录返回值"""
        price = self.data.close[0]
        if stop_loss is None:
            actual_stop_loss = 0.99
        else:
            actual_stop_loss = stop_loss

        args = {}
        if is_buy:
            stop_price = price * (1.0 - actual_stop_loss)
            if take_profit == None: 
                limit_price = None
                args['limitexec'] = None
            else:    limit_price = price * (1.0 + take_profit)
            self.logger.debug(f"_open_bracket price:{price},size:{size} , stop_price{stop_price}, limit_price {limit_price}, actual_stop_loss:{actual_stop_loss}")
            if limit_price != None:
                self.logger.error(f"_open_bracket limit_price:{limit_price}")
            # returns: [Main, Stop, Limit]
            orders = self.buy_bracket(
                size=size, 
                price=price, 
                stopprice=stop_price, 
                limitprice=limit_price,
                exectype=bt.Order.Market,
                **args
            )
            # 记录到队列
            self.live_trades.append({
                'main': orders[0],  # 主单
                'stop': orders[1],  # 止损单
                'limit': orders[2], # 止盈单
                'size': size
            })
            
        else: # Sell
            stop_price = price * (1.0 + actual_stop_loss)
            if take_profit == None: 
                limit_price = None
                args['limitexec'] = None
            else:    limit_price = price * (1.0 - take_profit)
            self.logger.debug(f"_open_bracket price:{price}, size:{size}, stop_price{stop_price}, limit_price {limit_price}, actual_stop_loss:{actual_stop_loss}")
            if limit_price != None:
                self.logger.error(f"_open_bracket limit_price:{limit_price}")
            orders = self.sell_bracket(
                size=size, 
                price=price, 
                stopprice=stop_price, 
                limitprice=limit_price,
                exectype=bt.Order.Market,
                **args
            )
            self.live_trades.append({
                'main': orders[0],
                'stop': orders[1],
                'limit': orders[2],
                'size': size
            })

    def _reduce_position_fifo(self, amount_needed, is_buy_close):
        """
        FIFO 减仓逻辑：
        从 live_trades 的头部（旧订单）开始处理
        1. 取消旧的 Stop/Limit
        2. 发送平仓单
        3. 维护 live_trades 列表
        """
        remaining_to_close = amount_needed

        # 使用切片拷贝遍历，因为我们可能会在循环中修改列表
        for trade_record in self.live_trades[:]:
            if remaining_to_close <= 0:
                break

            current_record_size = trade_record['size']
            
            # 1. 无论全平还是半平，首先取消关联的止损/止盈单
            # 防止平仓后，止损单被意外触发导致反向开仓
            if trade_record['stop']: 
                self.cancel(trade_record['stop'])
            if trade_record['limit']: 
                self.cancel(trade_record['limit'])

            # 2. 判断是全平这个块，还是平一部分
            if current_record_size <= remaining_to_close:
                # === 情况 A: 当前块不够扣，全平 ===
                close_size = current_record_size
                
                # 发送平仓单 (如果是多头持仓，则卖出平仓)
                if is_buy_close:
                    self.sell(size=close_size, exectype=bt.Order.Market)
                else:
                    self.buy(size=close_size, exectype=bt.Order.Market)
                
                # 从记录中移除该块
                self.live_trades.remove(trade_record)
                remaining_to_close -= close_size
                self.logger.debug(f'  >> 平掉旧单 ID {trade_record["main"].ref}, 数量 {close_size}')

            else:
                # === 情况 B: 当前块够扣，平一部分 ===
                close_size = remaining_to_close
                
                if is_buy_close:
                    self.sell(size=close_size, exectype=bt.Order.Market)
                else:
                    self.buy(size=close_size, exectype=bt.Order.Market)
                
                # 更新记录中的剩余股数
                trade_record['size'] -= close_size
                remaining_to_close = 0
                
                self.logger.debug(f'  >> 部分平掉旧单 ID {trade_record["main"].ref}, 数量 {close_size}, 剩余 {trade_record["size"]}')
                
                # 【重要提示】
                # 此时该剩余部分处于"裸奔"状态（因为上面取消了旧止损）。
                # 如果你想非常严谨，应该在这里为 trade_record['size'] 的剩余部分
                # 重新发生一个新的 StopOrder，并更新到 trade_record['stop'] 中。
                self._reissue_protective_orders(trade_record)

    def _cancel_all_live_orders(self):
        """反手前清理所有挂单"""
        for trade in self.live_trades:
            if trade['stop']: self.cancel(trade['stop'])
            if trade['limit']: self.cancel(trade['limit'])

    def _reissue_protective_orders(self, trade_record):
        """
        为部分平仓后剩余的仓位，重新根据原价格挂出止损/止盈单
        """
        # 1. 获取剩余的股数和方向
        remaining_size = trade_record['size']
        if remaining_size == 0:
            return

        # 这里的 size 是带符号的（正数代表持多，负数代表持空）
        # 但下单函数通常接受正数 size，通过 buy/sell 方法区分方向
        # 这里的 trade_record['size'] 已经在 FIFO 逻辑里更新过了
        
        # 判断是持多单还是持空单
        is_long_position = remaining_size > 0
        abs_size = abs(remaining_size)

        # 2. 从旧订单对象中提取原定的价格参数
        # 注意：虽然旧订单被 cancel 了，但 Python 对象还在，参数依然可读
        old_stop_ord = trade_record['stop']
        old_limit_ord = trade_record['limit']

        # 提取价格 (如果原单存在)
        stop_price = old_stop_ord.params.price if old_stop_ord else None
        limit_price = old_limit_ord.params.price if old_limit_ord else None

        self.logger.debug(f'  >> 为剩余 {remaining_size} 股重发保护单 (Stop: {stop_price}, Limit: {limit_price})')

        # 3. 根据持仓方向重新下单
        new_stop_ord = None
        new_limit_ord = None

        if is_long_position:
            # === 持有多单，需要 Sell 来止损/止盈 ===
            if stop_price:
                new_stop_ord = self.sell(
                    size=abs_size, 
                    price=stop_price, 
                    exectype=bt.Order.Stop,
                    transmit=True 
                )
            if limit_price:
                new_limit_ord = self.sell(
                    size=abs_size, 
                    price=limit_price, 
                    exectype=bt.Order.Limit,
                    transmit=True
                )
        else:
            # === 持有空单，需要 Buy 来止损/止盈 ===
            if stop_price:
                new_stop_ord = self.buy(
                    size=abs_size, 
                    price=stop_price, 
                    exectype=bt.Order.Stop,
                    transmit=True
                )
            if limit_price:
                new_limit_ord = self.buy(
                    size=abs_size, 
                    price=limit_price, 
                    exectype=bt.Order.Limit,
                    transmit=True
                )

        # 4. 更新 trade_record，替换为新的订单对象
        # 这样下一次循环如果要继续减仓，取消的就是这些新单子
        trade_record['stop'] = new_stop_ord
        trade_record['limit'] = new_limit_ord