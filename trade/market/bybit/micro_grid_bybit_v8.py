import os, sys, logging, time, uuid, json, threading, math, signal
from enum import Enum
# 假设上面的类保存在 bybit_engine.py 中
from bybit_engine import BybitEngine 
from typing import Dict
# -----------------------------------------------------------------------------
# 配置区域
# -----------------------------------------------------------------------------
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..' , '..'))
from data_process import common

class NodeStatus(Enum):
    """网格节点生命周期状态枚举"""
    WAITING_ENTRY = "WAITING_ENTRY"  # 初始状态：挂出，等待成交
    HOLDING = "HOLDING"              # 持仓状态：一半已成交，止盈单已挂出
    UNKNOWN = "UNKNOWN"
    # 如果未来增加逻辑，可以扩展如下状态：
    # CANCELLED = "CANCELLED"        # 异常状态：订单被手动撤销或失效
    # CLOSED = "CLOSED"              # 结束状态：卖单成交，利润落袋
    @property
    def short(self) -> str:
        """状态短码（用于 orderLinkId 等）"""
        return self.value[0]

    @classmethod
    def from_short(cls, short: str) -> "NodeStatus":
        """从短码还原状态"""
        if not short:
            return cls.UNKNOWN
        short = short.upper()
        for s in cls:
            if s.value.startswith(short):
                return s
        return cls.UNKNOWN

class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"
    UNKNOWN = "UNKNOWN"
    @property
    def short(self) -> str:
        """状态短码（用于 orderLinkId 等）"""
        return self.value[0]

    @classmethod
    def from_short(cls, short: str) -> "NodeStatus":
        """从短码还原状态"""
        if not short:
            return cls.UNKNOWN
        short = short.upper()
        for s in cls:
            if s.value.startswith(short):
                return s
        return cls.UNKNOWN
class MarketState(Enum):
    """市场行情趋势枚举"""
    OSCILLATION = "OSCILLATION"      # 震荡行情：允许网格正常补单
    TREND = "TREND"                  # 趋势/单边行情：触发熔断，暂停新开买单

class OrderOpStatus(Enum):
    """
    操作中间态：专门解决“10秒超时”和“异步确认”问题
    """
    IDLE = 0            # 空闲：没有正在处理的订单
    SUBMITTING = 1      # 提交中：已发出 place_order 请求，尚未收到确认
    ACTIVE = 2          # 已激活：交易所已确认挂单成功
    CANCELING = 3       # 撤单中：已发出 cancel_order 请求，尚未确认撤回
    FAILED = 4          # 失败：Post-Only 拒绝、余额不足或网络丢包

class GridNode:
    def __init__(self, node_id: int, qty: float, price: float, side = OrderSide.BUY,
                 entry_id: str =None, exit_id =None, timestamp= timestamp, status: NodeStatus = NodeStatus.WAITING_ENTRY):
        self.id = node_id   #双向 ID,通过奇数(buy)/偶数(sell)代表方向
        self.qty = qty
        self.price = price
        self.entry_id = entry_id
        self.status:NodeStatus = status
        self.exit_id = exit_id  # 初始为空，持仓挂卖单时赋值
        self.timestamp = timestamp
        # --- 新增中间态管理变量 ---
        self.side = side
        self.op_status:OrderOpStatus= OrderOpStatus.IDLE
        self.last_error = ""     # 记录失败原因（如 PostOnly 拒绝
    @property
    def timestamp(self) -> int:
        """从复合 node_id 中提取时间戳 (假设你的 node_id 是 TS * 1000 + N)"""
        # 这里的解析逻辑根据你 generate_node_id 的算法来定
        parts = self.id.split(':')
        return parts[-1]

    @property
    def initial_side(self) -> OrderSide:
        """从复合 node_id 中提取方向"""
        return OrderSide.BUY if self.id % 2 != 0 else OrderSide.OrderSide.SELL

    @property
    def pair_id(self) -> OrderSide:
        return self.id*2 if self.initial_side == OrderSide.BUY else self.id//2

    @property
    def sequence_num(self) -> int:
        """从复合 node_id 中提取它是第几个网格"""
        return (self.id % 1000) // 4

    def __repr__(self):
        """定义打印格式，方便调试"""
        return f"<Node #{self.id} {self.status} Qty:{self.qty} P:{self.price}>"

class SymbolConfig:
    def __init__(self, symbol, budget_pct, max_layers, base_offset, qty_step):
        self.symbol = symbol
        self.budget_pct = budget_pct    # 预算比例
        self.max_layers = max_layers    # 最大网格层数
        self.base_offset = base_offset  # 网格间距
        self.qty_step = qty_step        # 数量步长精度

class SymbolState:
    def __init__(self, config: SymbolConfig):
        self.config = config  # 组合静态配置
        self.stop_loss = 0.005
        
        # 🔒 互斥锁：保护以下所有动态数据
        self.lock = threading.RLock()
        
        # --- 核心动态数据 (读写需加锁) ---
        self.grid_nodes:Dict[int,GridNode] = {}     # 活跃节点列表
        self.node_counter = 0    # 订单 ID 计数器
        self.market_state = MarketState.OSCILLATION # 趋势熔断状态
        self.last_check_time = 0
        self.start_price_gap = 0.0005
        self.last_rebalance_time = time.time()
        # --- 交易所精度缓存 (初始化后很少改动，但属于状态) ---
        self.tick_size = 0.0001
        self.qty_step_actual = 1.0
        self.min_qty = 1.0

class UnifiedGridBot:
    def __init__(self, engine:BybitEngine, symbol_configs):
        self.version = 'V8'
        self.logger, _ = common.setup_session_logger(sub_folder=f'{self.__class__}', console_level=logging.INFO, file_level=logging.INFO)
        self.engine:BybitEngine = engine
        self.markets :Dict[str,SymbolState]= {}
        self.trade_stats = {s: 0 for s in symbol_configs.keys()}
        self.trade_history = {s: [] for s in symbol_configs.keys()} # V4: 用于权重计算
        
        self.start_balance = 0
        self.current_balance = 0
        self.total_profit = 0.0
        self.start_time = time.time()
        self.stop_new_pairs = False
        self.fee_rate = 0.0002
        self.initial_price_gap = 0.0002

        for symbol, cfg in symbol_configs.items():
            # 1. 创建静态配置实例
            config_obj = SymbolConfig(
                symbol=symbol,
                budget_pct=cfg['budget_pct'],
                max_layers=cfg['max_layers'],
                base_offset=cfg['base_offset'],
                qty_step=cfg['qty_step'],
            )
            # 2. 创建并存储动态状态实例
            self.markets[symbol] = SymbolState(config_obj)
            config = self.markets[symbol].config
            self.engine.set_leverage(symbol, leverage = 5)
            self.logger.info(f"load conf: symbol {symbol} | budget_pct {config.budget_pct} | max_layers {config.max_layers} base_offset {config.base_offset}")  

    def startup(self):
        """V8 启动流程: 包含 V5 的所有体检项目"""
        self.logger.info(f"🚀 {self.__class__} 网格启动 | ID标记恢复 | 趋势熔断 | 利润保底")
        
        # 1. 同步精度 & 利润体检 (V5)
        self.update_instrument_info()
        # 2. 获取初始余额
        self.get_wallet_balance()
        self.start_balance = self.current_balance
        # 3. 状态恢复 (V5+V7: 不死鸟机制)
        self.reconcile_global_state()
        self.update_micro_market_status() # 3. 初始趋势分析

        # 4. 注册 WS 监听 (V7)
        self.engine.start_stream(self.on_order_update)
        self.init_order()
        # 5. 启动后台线程 (V4+V5: 巡逻、趋势、交互)
        threading.Thread(target=self.run_loop, daemon=True).start()
        threading.Thread(target=self.listen_keyboard, daemon=True).start()
        
        # 6. 初始启动单
        self.check_initial_entry()
        
        # 阻塞主线程
        while True: time.sleep(1)

    # -------------------------------------------------------------------------
    # 核心交易逻辑
    # -------------------------------------------------------------------------
    def on_order_update(self, message):
        """WS 推送入口 (极速)"""
        data = message.get('data', [])
        for order in data:
            if order['orderStatus'] == "Filled":
                self.handle_filled_order(order['symbol'], order['orderLinkId'], float(order['avgPrice']), float(order['qty']))

    def handle_filled_order(self, symbol, order_id, fill_price, qty):
        m = self.markets.get(symbol)
        if not m: return

        # 解析 ID
        valid, _, node_id, side, status, ts = self.parse_order_link_id(order_id)
        if valid==False:
            self.logger.warning(f"handle_filled_order invalid order_id {order_id}")
            return

        # 查找内存节点
        node:GridNode = m.grid_nodes.get(node_id)
        reserve_side = OrderSide.BUY if side== OrderSide.SELL else OrderSide.SELL
        if not node: # 容错：如果是刚恢复的单子
            node = GridNode(node_id= node_id, qty=qty, price=fill_price, side =reserve_side, entry_id= order_id, timestamp=ts,status= status)
            m.grid_nodes[node_id] = node # 存入
        
        self.logger.info(f"⚡ [{symbol}] {node.side.value}成交 #{node_id} @ {fill_price}")
        # 1. 挂平仓单 (止盈)
        self.place_order(symbol, reserve_side, node, base_price=fill_price, status =NodeStatus.WAITING_ENTRY)
        # 2. 补新单 (循序)
        if len(m.grid_nodes) < m.config.max_layers and not self.stop_new_pairs:
            self.place_order(symbol, node.side, None, base_price=fill_price, status= NodeStatus.WAITING_ENTRY)
        if side == OrderSide.BUY:
            reserve_price = fill_price* (1+ m.config.base_offset)
        else:
            reserve_price = fill_price* (1 - m.config.base_offset)
        #平仓单
        self.place_order(symbol, node.side, None, base_price=reserve_price, status= NodeStatus.WAITING_ENTRY) #反向单

    def place_order(self, symbol, side:OrderSide, node=None, base_price=0, status = NodeStatus.WAITING_ENTRY, stop_loss=False) -> bool:
        """统一发单入口"""
        m = self.markets[symbol]
        if m.market_state == "TREND" and status == NodeStatus.WAITING_ENTRY:
            self.logger.debug(f"place_order skip market_state {m.market_state}")
            return False# 趋势中不接刀
        cfg = m.config
        
        if m.market_state == MarketState.TREND and status == NodeStatus.HOLDING: return False# V5: 熔断

        # 价格计算
        if status == NodeStatus.WAITING_ENTRY:
            offset = self.initial_price_gap
            self.get_wallet_balance()
            budget = self.current_balance * cfg.budget_pct/ cfg.max_layers
            qty = budget / price
            m.node_counter += 1
            node_id = m.node_counter
        else:
            offset = cfg.base_offset
            qty = node.qty
            node_id = node.id

        if stop_loss == True:
            offset = 0.001
        if side == OrderSide.BUY:
            price = base_price * (1 - offset)
            # 精度对齐
            price = int(price / m.tick_size) * m.tick_size
        else:
            price = base_price * (1 + offset)
            # 精度对齐
            price = round(price / m.tick_size) * m.tick_size

        # 步长对齐 (qty_step)
        qty_str = self.adjust_qty(qty, m.config.qty_step, m.min_qty)
        # ID 生成
        link_id = self.generate_order_link_id(symbol, node_id, side, status)
        # 执行
        self.engine.place_order(symbol, side, qty_str, f"{price:.5f}", link_id, self.on_place_result)
        
        # 乐观更新内存 (WAITING_ENTRY only)
        if status == NodeStatus.WAITING_ENTRY:
            node = GridNode(node_id= node_id, qty=qty, price=price, side=side, entry_id= link_id, status= NodeStatus.WAITING_ENTRY)
            node.op_status = OrderOpStatus.SUBMITTING
            m.grid_nodes[node_id] = node # 存入
            m.node_counter += 1
            self.logger.info(f"📤 [{symbol}] 发送{side.value} #{node_id} Price:{price:.5f}")
        else:
            node.exit_id = link_id
            node.op_status = OrderOpStatus.SUBMITTING
            self.logger.info(f"📤 [{symbol}] 发送{side.value} #{node_id} Price:{price:.5f}")
        return True

    def on_place_result(self, msg):
        if msg.get('retCode') != 0:
            self.logger.warning(f"⚠️ 下单回执报错: {msg.get('retMsg')}")

    def cancel_grid_order(self, symbol, order_link_id):
        """通过自定义 ID 取消订单"""
        res = self.engine.http.cancel_order(
            category=self.engine.category,
            symbol=symbol,
            orderLinkId=order_link_id
        )
        
        if res.get('retCode') == 0:
            self.logger.info(f"✅ [{symbol}] 订单 {order_link_id} 撤单成功")
            return True
        else:
            self.logger.error(f"❌ [{symbol}] 撤单失败: {res.get('retMsg')}")
            return False

    def init_order(self):
        all_tickers = self.get_all_tickers()
        for symbol in self.markets:
            m = self.markets[symbol]
            curr_price = all_tickers.get(symbol)
            with m.lock:
                if True != self.place_order(symbol, OrderSide.BUY, None, base_price=curr_price, status= NodeStatus.WAITING_ENTRY):
                    self.logger.debug(f"init_order BUY place_order fail")

                if True != self.place_order(symbol, OrderSide.SELL, None, base_price=curr_price, status= NodeStatus.WAITING_ENTRY):
                    self.logger.debug(f"init_order SELL place_order fail")
                
                
    # -------------------------------------------------------------------------
    # 后台管理 (V4/V5 的精髓)
    # -------------------------------------------------------------------------
    def run_loop(self):
        last_report_tick = time.time()
        last_order_tick = last_report_tick
        while True:
            now = time.time()
            if now - last_order_tick > 60:
                #检查止损
                all_tickers = self.get_all_tickers()
                for symbol in self.markets.keys():
                    m = self.markets[symbol]
                    curr_price = all_tickers.get(symbol)
                    if not curr_price: continue
                    pos_res = self.engine.http.get_positions(category=self.engine.category, symbol=symbol)
                    if pos_res.get('retCode') != 0: continue
                    avg_price = float(pos_res.get('avgPrice', 0))
                    mark_price = float(pos_res.get('markPrice'))
                    size = float(pos_res.get('size', 0))
                    price_change_pct = (mark_price - avg_price) / avg_price
                    if size != 0 and abs(price_change_pct) > (self.stop_loss):
                        self.smart_close_all_maker(symbol)

            if now - last_report_tick > 30:
                # 1. 趋势分析 (V5)
                self.update_micro_market_status()
                
                # 2. 状态对账 (V7.2 新增: 防WS丢包)
                for symbol in self.markets:
                    self.reconcile_state(symbol)
                
                # 3. 动态权重调整 (V4 回归)
                if now - self.markets[list(self.markets.keys())[0]].last_rebalance_time > 1800:
                    self.rebalance_weights()
                
                # 4. 战报打印 (V5)
                self.report_status()

    def fetch_all_open_orders(self):
        """分页获取全场所有活跃挂单"""
        all_orders = []
        cursor = ""
        
        while True:
            # 发起请求，传入当前的 cursor
            res = self.engine.http.get_open_orders(
                category=self.engine.category, 
                limit=50, 
                cursor=cursor
            )
            
            if res.get('retCode') != 0:
                break
            
            data = res.get('result', {})
            all_orders.extend(data.get('list', []))
            
            # 🌟 检查是否还有下一页
            cursor = data.get('nextPageCursor')
            if not cursor: # 如果没有 cursor 了，说明抓完了
                break
                
        return all_orders

 
    def reconcile_state(self, symbol):
        """HTTP 巡逻：全量对比内存与链上 (V8 高性能版)"""
        m = self.markets[symbol]
        
        # 1. 批量获取当前挂单 (Open Orders)
        res_open = self.engine.http.get_open_orders(category=self.engine.category, symbol=symbol, limit=50)
        if res_open.get('retCode') != 0: return
        on_chain_ids = {o['orderLinkId'] for o in res_open['result']['list']}
        
        # 2. 🌟 批量获取历史成交订单 (Order History)
        # 一次拿 50 条，覆盖过去几分钟的成交绰绰有余
        res_hist = self.engine.http.get_order_history(category=self.engine.category, symbol=symbol, limit=50)
        history_map = {}
        if res_hist.get('retCode') == 0:
            # 将历史订单组织成 map，方便 $O(1)$ 查询
            history_map = {o['orderLinkId']: o for o in res_hist['result']['list'] if o['orderStatus'] == "Filled"}
        # 3. 使用递归锁保护内存状态
        with m.lock:
            for nid in list(m.grid_nodes.keys()):
                node = m.grid_nodes[nid]
                
                # 状态 A: 内存说在等买，但链上挂单里没了
                if node.status == NodeStatus.WAITING_ENTRY and node.entry_id not in on_chain_ids:
                    
                    # 🌟 核心检查：去批量抓取的历史映射表中查找
                    hist_order = history_map.get(node.entry_id)
                    
                    if hist_order:
                        # 走到这里，才代表“确认成交”
                        self.logger.warning(f"🔍 [{symbol}] 巡逻确认漏单成交 #{node.id} (ID:{node.entry_id})")
                        fill_price = float(hist_order['avgPrice'])
                        
                        # 手动补偿执行成交逻辑
                        self.handle_filled_order(symbol, node.entry_id, fill_price, node.qty)
                    else:
                        # 🌟 备选逻辑：如果历史里也没有，说明订单被“手动撤销”或“系统废弃”了
                        self.logger.error(f"🚨 [{symbol}] 订单 #{node.id} 彻底失踪（非成交，非挂单），需清理内存")
                        # 这种情况通常需要从内存删除该节点，否则网格会卡死在这层
                        # m.grid_nodes.pop(nid, None)

    def rebalance_weights(self):
        """V4 回归：根据活跃度调整资金比例"""
        now = time.time()
        scores = {}
        total_score = 0
        for s in self.markets:
            # 统计过去1小时成交数
            recent_trades = [t for t in self.trade_history[s] if now - t < 3600]
            self.trade_history[s] = recent_trades
            score = len(recent_trades) + 1 # 保底1分
            scores[s] = score
            total_score += score
        
        self.logger.info(f"⚖️ [权重重组] 总分: {total_score}")
        for s in self.markets:
            # 简单分配：占比 * 总Budget (假设总投入 80%)
            new_pct = (scores[s] / total_score) * 0.8 
            new_pct = max(0.05, min(new_pct, 0.4)) # 限制 5% - 40%
            self.markets[s].config.budget_pct= new_pct
            m = self.markets[s]
            m.last_rebalance_time = now

    def listen_keyboard(self):
        """V4/V5 回归：控制台指令"""
        while True:
            cmd = input()
            if cmd == 'x': # 核按钮
                if input("🔥 确认全平? (y/n): ") == 'y':
                    self.stop_new_pairs = True
                    for s in self.markets:
                        self.smart_close_all_maker(s)
                    self.logger.info("🚨 全场已清空")
            elif cmd == 's':
                self.stop_new_pairs = not self.stop_new_pairs
                self.logger.info(f"🛑 停止开新单: {self.stop_new_pairs}")

    def market_close_all(self, symbol):
        """
        极致市价平仓：撤单并结清所有头寸
        """
        # 1. 先撤掉所有挂单，防止平仓过程中成交了反向挂单
        self.engine.cancel_all_http(symbol)
        
        # 2. 获取当前实际持仓
        pos_res = self.engine.http.get_positions(category=self.engine.category, symbol=symbol)
        if pos_res.get('retCode') != 0: return

        for pos in pos_res['result']['list']:
            size = float(pos.get('size', 0))
            if size == 0: continue
            
            # 3. 确定平仓方向：多头仓位(Positive)用Sell，空头仓位(Negative)用Buy
            side = "Sell" if size > 0 else "Buy"
            
            # 4. 发起市价单
            res = self.engine.http.place_order(
                category=self.engine.category,
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(abs(size)),
                reduceOnly=True # 🌟 核心：确保只减仓不反向开仓
            )
            
            if res.get('retCode') == 0:
                self.logger.warning(f"💥 [{symbol}] 市价平仓指令下达成功：{side} {abs(size)}")
            else:
                self.logger.error(f"❌ [{symbol}] 市价平仓失败：{res.get('retMsg')}")

    def smart_close_all_maker(self, symbol, max_retries=10, use_market=False):
        """
        全自动平仓逻辑：支持 Maker 追价 (Limit) 或 Market 直接平仓
        :param use_market: True 使用市价单直接成交，False 使用限价单挂在标记价追价
        """
        mode_str = "Market (市价)" if use_market else "Maker (追价)"
        self.logger.info(f"🧹 开始执行 {symbol} 的 {mode_str} 平仓序列...")

        for attempt in range(max_retries):
            # 1. 撤销该币种的所有 HTTP 挂单，防止新老单子打架
            self.engine.cancel_all_http(symbol)
            time.sleep(0.5)

            # 2. 获取实时持仓数据
            pos_res = self.engine.http.get_positions(category=self.engine.category, symbol=symbol)
            if pos_res.get('retCode') != 0:
                self.logger.error(f"❌ 获取持仓失败: {pos_res.get('retMsg')}")
                continue

            pos_list = pos_res.get('result', {}).get('list', [])
            # 过滤出持仓量大于 0 的房间
            active_pos = [p for p in pos_list if float(p.get('size', 0)) > 0]

            if not active_pos:
                self.logger.info(f"✅ [{symbol}] 持仓已清零，平仓成功！")
                return True

            # 3. 遍历持仓并发送平仓指令
            for pos in active_pos:
                p_idx = int(pos['positionIdx'])
                size = pos['size']
                
                # 🌟 核心逻辑：市价单不需要价格，限价单使用标记价
                price = "" if use_market else pos['markPrice']
                order_type = "Market" if use_market else "Limit"
                
                # 判定方向 (单向或双向持仓兼容)
                if p_idx == 1: side = "Sell"    # 平多
                elif p_idx == 2: side = "Buy"   # 平空
                else: side = "Sell" if pos['side'] == "Buy" else "Buy" # 单向

                # 4. 调用下单引擎
                # 注意：确保你的 engine.place_order 能够接收 order_type 参数
                self.engine.place_order(
                    symbol=symbol, side=side, qty=size, price=price,
                    order_type=order_type, # 注入下单类型
                    link_id=f"CLOSE_{p_idx}_{int(time.time())}",
                    is_reduce=True, pos_idx=p_idx,
                    callback=self.on_place_result
                )
                self.logger.info(f"🔄 [{symbol}] {mode_str} 尝试第 {attempt+1} 次: {side} {size}")

            # 市价单通常一次见效，Maker 需要留时间观察
            time.sleep(1.0 if use_market else 2.0) 

        self.logger.error(f"❌ [{symbol}] 尝试 {max_retries} 次后仍未平仓，请检查是否有极速暴跌导致无法成交！")
        return False

    # -------------------------------------------------------------------------
    # 辅助函数 (简化版)
    # -------------------------------------------------------------------------
    def update_instrument_info(self):
        res = self.engine.http.get_instruments_info(category=self.engine.category)
        if res.get('retCode') == 0:
            info_map = {item['symbol']: item for item in res['result']['list']}
            for s in self.markets:
                if s in info_map:
                    # 获取价格精度
                    self.markets[s].tick_size = float(info_map[s]['priceFilter']['tickSize'])
                    # 🌟 获取数量步长 和 最小下单量
                    self.markets[s].config.qty_step = float(info_map[s]['lotSizeFilter']['qtyStep'])
                    self.markets[s].min_qty = float(info_map[s]['lotSizeFilter']['minOrderQty'])
                    self.check_profit_viability(s)

    def check_profit_viability(self, symbol):
        m = self.markets[symbol]
        res = self.engine.http.get_tickers(category=self.engine.category, symbol=symbol)
        if res.get('retCode') != 0: 
            self.logger(f'check_profit_viability fail symbol {symbol}')
            return
        last_price = float(res['result']['list'][0]['lastPrice'])
        
        gap = last_price * m.base_offset
        fee = last_price * (self.fee_rate * 2) # 双边费率
        net_pct = ((gap - fee) / last_price) * 100
        ticks = gap / m.tick_size
        
        self.logger.info(f"📊 [{symbol}] 体检: Offset={m.base_offset*100:.2f}% | 净利预测={net_pct:.3f}% | Ticks={ticks:.1f}")
        if ticks < 4 or net_pct < 0.05:
            self.logger.warning(f"   ❌ {symbol} 利润过薄！请增加 base_offset！ ticks:{ticks} net_pct:{net_pct}")

    def get_wallet_balance(self):
        res = self.engine.http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if res.get('retCode') == 0:
            self.current_balance = float(res['result']['list'][0]['coin'][0]['walletBalance'])

    # -----------------------------------------------------------------------------
    # ID 编解码与状态恢复 (核心功能)
    # -----------------------------------------------------------------------------
    def generate_order_link_id(self, symbol, node_id, side = OrderSide.BUY,  status: NodeStatus = NodeStatus.WAITING_ENTRY, timestamp=None):
        short_sym = symbol.replace("USDT", "")
        ts = timestamp if timestamp else int(time.time())
        st = status.short
        sd = side.short
        # 🌟 极致精简：[版本]:[币种]:[ID]:[时间戳]
        return f"{self.version}:{short_sym}:{node_id}:{sd}:{st}:{ts}"

    def parse_order_link_id(self, order_link_id):
        """
        解析返回: (valid, short_sym, node_id, side, status, timestamp)
        """
        try:
            parts = order_link_id.split(':')
            # 校验长度为 5，且版本号匹配
            if len(parts) != 5 or parts[0] != self.version: 
                return False, None, None, None, None, None
            status = NodeStatus.from_short(parts[4])
            side = OrderSide.from_short(parts[3])
            return (True, parts[1], int(parts[2]), side, status, int(parts[5]))   #short_sym,node_id,side,status,ts
        except Exception as e:
            print(f"parse_order_link_id failed: {order_link_id}, err={e}")
            return False, None, None, None, None, None

    def is_order_sell(self, node_id):
        if (node_id % 2 == 0):  return True
        return False

    def reconcile_global_state(self):
        """
        全场状态对账与自愈中心 (V8 确定性版)
        """
        self.logger.info("📡 [全场巡检] 启动：1.同步内存 2.风险对齐")

        # --- 步骤 0: 获取全场快照 ---
        all_open_orders = self.fetch_all_open_orders()
        pos_res = self.engine.http.get_positions(category=self.engine.category)
        if pos_res.get('retCode') != 0: return

        # 整理仓位字典 {symbol: actual_size}
        positions = {p['symbol']: float(p['size']) for p in pos_res['result']['list'] if float(p['size']) != 0}
        avg_prices = {p['symbol']: float(p['avgPrice']) for p in pos_res['result']['list'] if float(p['size']) != 0}

        # 整理挂单字典 {symbol: { 'waiting': [], 'holding': [] }}
        remote_logic_map:Dict[str,Dict[int,GridNode]] = {}
        # remote_logic_map :Dict[int,GridNode]= {}
        for o in all_open_orders:
            symbol = o['symbol']
            if symbol not in remote_logic_map:
                remote_logic_map[symbol] = {}
            valid, _, nid, status, ts = self.parse_order_link_id(o['orderLinkId'])
            if not valid: continue
            remote_node = GridNode(nid, float(o['qty']), float(o['price']), timestamp= ts, status= status)
            remote_logic_map[symbol][nid] = remote_node
        # # --- 遍历所有配置的币种执行逻辑 ---
        # for symbol, m in self.markets.items():
        #     with m.lock:
        #         m.grid_nodes = remote_logic_map[symbol]
                
        #         # ---------------------------------------------------------
        #         # 步骤 2: 处理订单和 position 不一致的问题 (Risk Alignment)
        #         # ---------------------------------------------------------
        #         holiding_buy_orders = [n.status== NodeStatus.WAITING_ENTRY and n.initial_side ==OrderSide.BUY for n in m.grid_nodes.values()]
        #         holiding_sell_orders = [n.status== NodeStatus.WAITING_ENTRY and n.initial_side ==OrderSide.SELL for n in m.grid_nodes.values()]
        #         actual_size = positions.get(symbol, 0)
        #         # 统计当前已经在挂的止盈单总量 (基于 ID 识别)
        #         if actual_size > 0: #多头
        #             current_sell_tp_qty = sum(info['qty'] for info in holiding_sell_orders)

        #         else:
        #             current_buy_tp_qty = sum(info['qty'] for info in holiding_buy_orders)

        #         # 判定 A: 漏挂止盈 (持仓 > 止盈单) -> 补平
        #         if actual_size > current_tp_qty:
        #             gap = actual_size - current_tp_qty
        #             self.logger.warning(f"🚨 [{symbol}] 仓位裸奔！缺口: {gap}，执行补挂 Maker 止盈...")
        #             # 补单逻辑使用 PostOnly ，并标记为 HOLDING 状态
        #             self.place_order(symbol, reserve_side, node, base_price=fill_price, status =NodeStatus.HOLDING)

        #         # 判定 B: 冗余止盈 (止盈单 > 持仓) -> 撤单
        #         elif current_tp_qty > actual_size:
        #             redundant = current_tp_qty - actual_size
        #             self.logger.warning(f"🚨 [{symbol}] 止盈冗余！多出: {redundant}，执行撤单清理...")
        #             self._cleanup_redundant_tp(symbol, current_remote['holding'], redundant)

        self.logger.info("✅ [全场巡检] 同步与一致性处理完成。")

    def sync_state_from_exchange(self):
        self.logger.info("🔗 正在恢复状态...")
        for s in self.markets.keys():
            m = self.markets[s]
            m.grid_nodes = [] 
            res = self.engine.http.get_open_orders(category=self.engine.category, symbol=s, limit=50)
            if res.get('retCode') != 0: continue
            
            recovered_nodes:Dict[int,GridNode] = {}
            active_orders = res['result']['list']
            
            for order in active_orders:
                oid = order['orderLinkId']
                valid, sym_short, node_id, ts = self.parse_order_link_id(oid)
                
                # 过滤非本策略订单
                if not valid or sym_short != s.replace("USDT", ""): continue
                
                m.node_counter = max(m.node_counter, node_id)
                
                if node_id not in recovered_nodes:
                    recovered_nodes[node_id] = GridNode(node_id= node_id, qty=float(order['qty'], price=0, entry_id= None, status= NodeStatus.UNKNOWN))
                node = recovered_nodes[node_id]
                price = float(order['price'])

                if not self.is_order_sell(node_id):
                    node.status = NodeStatus.WAITING_ENTRY
                    node.entry_id = oid
                    node.price = price
                else:
                    node.status = NodeStatus.HOLDING
                    node.exit_id = oid
                    node.price = price
            m.grid_nodes = recovered_nodes
            self.logger.info(f"   ✅ {s} 状态同步完成，当前层数: {len(m.grid_nodes)}")

    def adjust_qty(self, raw_qty, qty_step, min_qty):
        """
        校准下单数量：
        1. 确保大于 min_qty
        2. 确保是 qty_step 的整数倍
        3. 处理精度问题 (避免 100.0000001 这种情况)
        """
        # 1. 保底最小数量
        if raw_qty < min_qty:
            raw_qty = min_qty
            
        # 2. 步长对齐 (向下取整或四舍五入均可，这里用四舍五入)
        # 逻辑：先除以步长，取整，再乘以步长
        # 例子：qty=505, step=10 -> 50.5 -> 51 -> 510
        qty = round(raw_qty / qty_step) * qty_step
        
        # 3. 精度格式化 (关键！消除浮点数尾数)
        # 根据 qty_step 计算需要保留的小数位数
        if qty_step >= 1:
            precision = 0
        else:
            # e.g. 0.001 -> 3
            precision = int(math.ceil(-math.log10(qty_step)))
            
        # 再次 round 确保 Python 浮点数不飘移
        qty = round(qty, precision)
        
        # 返回字符串格式，避免传给 API 时变成科学计数法
        if precision == 0:
            return str(int(qty))
        else:
            return f"{qty:.{precision}f}"
        
    def update_micro_market_status(self):
        """趋势熔断逻辑: 动态分位数版 (旨在仅熔断约 20% 的极端波动时间)"""
        import math
        now = time.time()
        for symbol in self.markets:
            m = self.markets[symbol]
            #
            res = self.engine.http.get_kline( category=self.engine.category, symbol=symbol, interval=1, limit=100)
            if res.get('retCode') != 0 or not res['result']['list']: continue

            prices = [float(k[4]) for k in res['result']['list']]
            volumes = [float(k[5]) for k in res['result']['list']]
            prices.reverse() 

            # 1. 计算动态波动率 (标准差)
            n = len(prices)
            ma = sum(prices) / n
            # 计算方差与标准差
            variance = sum((p - ma) ** 2 for p in prices) / n
            std_dev = math.sqrt(variance)
            
            curr = prices[-1]
            # 计算当前价格偏离了多少个标准差 (Z-Score)
            # 💡 逻辑：只有当偏离度超过 1.28 sigma 时，才认为属于那 20% 的极端行情
            z_score = abs(curr - ma) / std_dev if std_dev > 0 else 0
            
            # 2. 筹码集中度 (简化版 VPVR)
            bin_count = 10
            min_p, max_p = min(prices), max(prices)
            interval = (max_p - min_p) / bin_count if max_p != min_p else 0.0001
            profile = [0] * bin_count
            for p, v in zip(prices, volumes):
                idx = min(int((p - min_p) / interval), bin_count - 1)
                profile[idx] += v
            concentration = max(profile) / sum(profile) if sum(profile) > 0 else 0
            
            # 3. 动态判定
            # z_score > 1.28 替代了死板的 0.003
            # concentration 阈值从 0.15 调低到 0.10，增加对轻微趋势的容忍度
            if z_score > 1.28 or concentration < 0.10:
                m.market_state = MarketState.TREND
            else:
                m.market_state = MarketState.OSCILLATION
            
            m.last_check_time = now
            self.logger.info(f"🔍 [{symbol}] 状态: {m.market_state} (check_time:{now}, POC:{concentration:.2f})")

    def check_initial_entry(self):
        # 启动时如果空仓则开单
        time.sleep(2)
        for s in self.markets:
            if len(self.markets[s].grid_nodes) == 0:
                 res = self.engine.http.get_tickers(category=self.engine.category, symbol=s)
                 if res.get('retCode')==0:
                     price = float(res['result']['list'][0]['lastPrice'])
                     self.place_order(s, OrderSide.BUY, None, base_price=price, is_initial=True)

    def report_status(self):
        self.get_wallet_balance()
        self.logger.info("-" * 40)
        self.logger.info(f"💰 余额: {self.current_balance:.2f} | 总利: {self.total_profit:.4f}")
        for s, m in self.markets.items():
            self.logger.info(f"   {s}: {len(m.grid_nodes)}层 | 状态: {m.market_state} | 权重: {m.config.budget_pct:.2%}")
        self.logger.info("-" * 40)

    def exit(self):
        self.engine.stop()
        time.sleep(0.5)
        sys.exit(0)
# -------------------------------------------------------------------------
# 启动入口
# -------------------------------------------------------------------------
if __name__ == "__main__":
    BASE = os.path.dirname(os.path.abspath(__file__))
    # 准备好 4 个密钥
    API_K = os.path.join(BASE, "keys", "hmac_api_key")
    API_S = os.path.join(BASE, "keys", "hmac_secret")
    RSA_K = os.path.join(BASE, "keys", "api_key")     
    RSA_P = os.path.join(BASE, "keys", "bybit_rsa.pem")

    CONFIGS = {
        "DOGEUSDT": {"budget_pct": 0.2, "max_layers": 5, "base_offset": 0.005, "qty_step": 0},
        "ARCUSDT":  {"budget_pct": 0.1, "max_layers": 5, "base_offset": 0.006, "qty_step": 0}, 
        "ASTRUSDT": {"budget_pct": 0.1, "max_layers": 5, "base_offset": 0.006, "qty_step": -2}, 
        "APEXUSDT": {"budget_pct": 0.1, "max_layers": 5, "base_offset": 0.006, "qty_step": 0}, 
        "1000RATSUSDT": {"budget_pct": 0.1, "max_layers": 5, "base_offset": 0.008, "qty_step": -2},
    }

    engine = BybitEngine(API_K, API_S, RSA_K, RSA_P)
    bot = UnifiedGridBot(engine, CONFIGS)

    def signal_handler(sig, frame):
        """信号处理器：当按下 Ctrl+C 时触发"""
        bot.logger.warning("\n\n👋 检测到 Ctrl+C，正在执行安全退出程序...")
        bot.exit()
        bot.logger.warning("✅ 机器人已离线。")
        sys.exit(0)
    # 注册 SIGINT 信号 (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    bot.startup()