import os, sys, logging, time, uuid, json, threading, math, signal
from enum import Enum
from typing import Dict, Set, List, Tuple

# 假设 bybit_engine.py 就在同级目录下
from bybit_engine import BybitEngine 
current_work_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_work_dir, "..", '..' , '..'))
from data_process import common

# -----------------------------------------------------------------------------
# 基础枚举与数据类
# -----------------------------------------------------------------------------
class MarketState(Enum):
    OSCILLATION = "OSCILLATION"  # 震荡
    TREND_UP = "TREND_UP"        # 单边上涨
    TREND_DOWN = "TREND_DOWN"    # 单边下跌
    
class NodeStatus(Enum):
    WAITING = "WAITING"   # 挂单中
    FILLED = "FILLED"     # 已成交 (用于历史记录)
    UNKNOWN = "UNKNOWN"

    @property
    def short(self) -> str:
        return self.value[0]

    @classmethod
    def from_short(cls, short: str) -> "NodeStatus":
        if not short: return cls.UNKNOWN
        for s in cls:
            if s.value.startswith(short.upper()): return s
        return cls.UNKNOWN

class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"
    
    @property
    def short(self) -> str:
        return self.value[0]  # 'B' or 'S'

    @classmethod
    def from_short(cls, short: str) -> "OrderSide":
        if short == 'B': return cls.BUY
        if short == 'S': return cls.SELL
        return cls.BUY # Default

class GridNode:
    """
    网格节点 (V9): 这里的 id 是网格的相对索引 (Index)
    例如: 0 (中心), -1 (下方一格), 5 (上方五格)
    """
    def __init__(self, index: int, qty: float, price: float, side: OrderSide, 
                 order_id: str = None, status: NodeStatus = NodeStatus.WAITING):
        self.index = index  # 网格索引 (可为负数)
        self.qty = qty
        self.price = price
        self.side = side
        self.order_id = order_id
        self.status = status
        self.timestamp = int(time.time())

    def __repr__(self):
        return f"<Idx:{self.index} {self.side.value} @ {self.price:.4f}>"

class SymbolConfig:
    def __init__(self, symbol, budget_pct, max_layers, base_offset, qty_step):
        self.symbol = symbol
        self.budget_pct = budget_pct    # 预算比例
        self.max_layers = max_layers    # 单边层数 (如 3)
        self.base_offset = base_offset  # 网格间距 (如 0.005)
        self.qty_step = qty_step        # 数量精度

class SymbolState:
    def __init__(self, config: SymbolConfig):
        self.config = config
        
        # 🔒 互斥锁：保护动态数据
        self.lock = threading.RLock()
        
        # --- 动态网格核心 ---
        self.initial_price = 0.0     # 初始基准价格 (Index=0 的价格)
        self.center_index = 0        # 当前窗口的逻辑中心索引
        
        # 记录当前活跃的节点 {index: GridNode}
        # 这里的 key 是索引 (-1, 0, 1...)
        self.active_nodes: Dict[int, GridNode] = {} 
        
        # 交易所参数缓存
        self.tick_size = 0.0001
        self.min_qty = 1.0
        self.market_state = MarketState.OSCILLATION
        # 0: 无趋势, 1: 上涨, -1: 下跌
        self.trend_direction = 0 
        # 缓存当前持仓数量，用于决策
        self.current_pos_size = 0.0

        self.total_profit_usdt = 0.0  # 累计净利润 (扣除手续费后)
        self.total_fee_spent = 0.0    # 累计消耗的手续费
        self.profit_count = 0         # 获利成交次数
        
        # 核心：持有成本计算
        self.current_pos_size = 0.0   # 当前持仓数量
        self.avg_entry_price = 0.0    # 动态加权平均成本价
        self.initial_entry_done = False
# -----------------------------------------------------------------------------
# 机器人主类 (初始化部分)
# -----------------------------------------------------------------------------
class UnifiedGridBot:
    def __init__(self, engine: BybitEngine, symbol_configs: dict, clean= False):
        self.version = 'V9' # 动态滑窗版
        self.logger, _ = common.setup_session_logger(
            sub_folder=self.__class__.__name__, 
            console_level=logging.INFO, 
            file_level=logging.INFO
        )
        self.engine = engine
        self.markets: Dict[str, SymbolState] = {}
        self.stop_signal = False
        self.start_time = time.time()
        self.starting_balance = 0.0  # 初始启动资金
        self.is_stopped = False      # 全局熔断开关
        self.clean = clean

        self.current_balance = 0.0
        self.fee_rate = 0.0002 # Taker 预估
        for symbol, cfg in symbol_configs.items():
            conf_obj = SymbolConfig(
                symbol=symbol,
                budget_pct=cfg['budget_pct'],
                max_layers=cfg['max_layers'],
                base_offset=cfg['base_offset'],
                qty_step=cfg['qty_step'],
            )
            self.markets[symbol] = SymbolState(conf_obj)
            # 默认单向持仓，如需杠杆请在此处设置
            self.engine.set_leverage(symbol, 5) 
            self.logger.info(f"Load Config: {symbol} | Gap:{cfg['base_offset']:.2%} | Layers:±{cfg['max_layers']}")

    def startup(self):
        """启动流程"""
        self.logger.info(f"🚀 {self.version} 动态滑窗网格启动...")
        
        # 1. 基础信息同步
        self.update_instrument_info()
        self.get_wallet_balance()
        
        # 2. 启动 WS 监听
        self.engine.start_stream(self.on_order_update)
        
        if self.clean >1 :
            self.emergency_wipe_all_account_positions(use_market = (self.clean ==2))

        self.get_wallet_balance()
        for symbol in self.markets:
            if self.clean ==1 :
                self.engine.cancel_all_http(symbol) 
                time.sleep(1)
            self.start_dynamic_grid(symbol)
            time.sleep(1)
            
        # 4. 启动后台守护线程
        threading.Thread(target=self.run_loop, daemon=True).start()
        
        self.logger.info("✅ 系统就绪，等待行情驱动...")
        # 阻塞主线程
        while not self.stop_signal:
            time.sleep(1)

    # -------------------------------------------------------------------------
    # 辅助工具：价格计算与 ID 处理
    # -------------------------------------------------------------------------
    def check_profit_viability(self, symbol):
        """
        网格利润体检：计算 base_offset 是否能覆盖手续费
        """
        m = self.markets[symbol]
        
        # 1. 获取最新价格
        res = self.engine.http.get_tickers(category=self.engine.category, symbol=symbol)
        if res.get('retCode') != 0 or not res['result']['list']:
            self.logger.error(f"❌ [{symbol}] 体检失败: 无法获取最新价")
            return
            
        last_price = float(res['result']['list'][0]['lastPrice'])
        
        # 2. 计算物理间距 (Gap) 和 双边手续费 (Fee)
        # 这里的 fee_rate 建议在 __init__ 中定义，Bybit V5 Taker 默认约 0.0006, Maker 约 0.0002
        gap_value = last_price * m.config.base_offset
        
        # 假设我们尽量做 Maker，使用 0.0002 费率。双边即 * 2
        # 如果你经常触发 Taker，请将 0.0002 改为 0.0006
        total_fee_rate = self.fee_rate * 2 
        fee_value = last_price * total_fee_rate
        
        # 3. 计算净利润百分比
        net_profit_pct = ((gap_value - fee_value) / last_price) * 100
        
        # 4. 计算间距覆盖了多少个价格最小变动单位 (Ticks)
        ticks = gap_value / m.tick_size if m.tick_size > 0 else 0
        
        self.logger.info(f"📊 [{symbol}] 网格体检报告:")
        self.logger.info(f"   - 设定间距: {m.config.base_offset:.2%}")
        self.logger.info(f"   - 价格间距: {gap_value:.5f} (约 {ticks:.1f} Ticks)")
        self.logger.info(f"   - 预估单次净利: {net_profit_pct:.4f}%")
        
        # 5. 风险预警
        if net_profit_pct <= 0:
            self.logger.error(f"🚨 警告: [{symbol}] 间距太小，利润无法覆盖手续费！")
        elif net_profit_pct < 0.05:
            self.logger.warning(f"⚠️ 警告: [{symbol}] 利润极其微薄 ({net_profit_pct:.4f}%)，建议调大 base_offset。")
        elif ticks < 3:
            self.logger.warning(f"⚠️ 警告: [{symbol}] 间距仅 {ticks:.1f} Ticks，极易因点差导致无法成交或滑点损失。")
        else:
            self.logger.info(f"✅ [{symbol}] 利润模型健康。")

    def get_price_by_index(self, symbol, index):
        """核心公式：根据 Index 计算目标价格"""
        m = self.markets[symbol]
        if m.initial_price <= 0: return 0.0
        
        # 价格 = 基准价 * (1 + 索引 * 间距)
        raw_price = m.initial_price * (1 + index * m.config.base_offset)
        
        # 精度对齐
        return round(raw_price / m.tick_size) * m.tick_size

    def adjust_qty(self, raw_qty, qty_step, min_qty):
        """数量精度校准"""
        if raw_qty < min_qty: raw_qty = min_qty
        
        # 整数步长与小数步长分别处理
        if qty_step == 0: qty_step = 1 # 防呆
        
        qty = round(raw_qty / qty_step) * qty_step
        
        # 格式化为字符串，去掉多余的 0
        if qty_step >= 1:
            return str(int(qty))
        else:
            precision = int(math.ceil(-math.log10(qty_step)))
            return f"{qty:.{precision}f}"

    def generate_order_link_id(self, symbol, index, side: OrderSide):
        """生成唯一 ID: V9:SYMBOL:INDEX:SIDE:TIMESTAMP"""
        short_sym = symbol.replace("USDT", "")
        ts = int(time.time() * 1000) # 毫秒级防碰撞
        # 注意：Index 可能是负数，f-string 会自动处理 (e.g. -1)
        return f"{self.version}:{short_sym}:{index}:{side.short}:{ts}"

    def parse_order_link_id(self, link_id):
        """
        解析 ID
        返回: (valid, symbol, index, side, timestamp)
        """
        try:
            parts = link_id.split(':')
            if len(parts) != 5 or parts[0] != self.version:
                return False, None, 0, None, 0
            
            sym = parts[1]
            index = int(parts[2]) # 支持解析 "-1"
            side = OrderSide.from_short(parts[3])
            ts = int(parts[4])
            
            return True, sym, index, side, ts
        except Exception:
            return False, None, 0, None, 0

    def get_wallet_balance(self):
        """刷新余额：获取总权益 (Equity)"""
        res = self.engine.http.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if res.get('retCode') == 0:
            try:
                coin_data = res['result']['list'][0]['coin'][0]
                # totalEquity = 钱包余额 + 未实现盈亏
                # 如果 Bybit API 返回中没有 totalEquity，则手动相加
                equity = float(coin_data.get('equity', coin_data.get('walletBalance', 0)))
                self.current_balance = equity
                
                # 如果是第一次运行，记录启动基准
                if self.starting_balance == 0:
                    self.starting_balance = equity
                    self.logger.info(f"🏦 记录启动基准资金: {self.starting_balance:.2f} USDT")
            except Exception as e:
                self.logger.error(f"❌ 解析余额异常: {e}")
            
    def check_account_safe(self):
        """
        账户级安全检查：基于余额变动的熔断机制
        """
        if self.is_stopped or self.starting_balance <= 0:
            return True

        # 计算总亏损比例
        loss_amount = self.starting_balance - self.current_balance
        loss_ratio = loss_amount / self.starting_balance if self.starting_balance > 0 else 0

        # 如果亏损超过 10% (0.10)
        if loss_ratio >= 0.10:
            self.logger.critical(f"🚨🚨 账户风险触发！初始: {self.starting_balance:.2f}, 当前: {self.current_balance:.2f}")
            self.logger.critical(f"📉 总亏损比例: {loss_ratio:.2%}, 立即执行全局清仓！")
            self.global_emergency_halt()
            return False
        
        return True

    def global_emergency_halt(self):
        """
        全账户紧急停机：清空所有订单、所有仓位并退出程序
        """
        self.is_stopped = True
        self.stop_signal = True
        
        # 🌟 改进点：不再循环 self.markets，而是执行地毯式平仓
        # 这样即使是历史遗留仓位也能一扫而光
        self.emergency_wipe_all_account_positions(use_market=True)
            
        self.logger.critical("🛑 系统已安全关闭，账户已排空。")
        os._exit(0)

    def update_instrument_info(self):
        """同步交易所精度"""
        res = self.engine.http.get_instruments_info(category=self.engine.category)
        if res.get('retCode') == 0:
            info_map = {item['symbol']: item for item in res['result']['list']}
            for s in self.markets:
                if s in info_map:
                    self.markets[s].tick_size = float(info_map[s]['priceFilter']['tickSize'])
                    self.markets[s].min_qty = float(info_map[s]['lotSizeFilter']['minOrderQty'])
                    step = info_map[s]['lotSizeFilter']['qtyStep']
                    self.markets[s].config.qty_step = float(step)
                    self.check_profit_viability(s)

    def on_place_result(self, response):
        """处理 WebSocket 下单回执"""
        if response.get('retCode') != 0:
            self.logger.warning(f"⚠️ 下单失败回执: {response.get('retMsg')}")
        else:
            # self.logger.debug(f"✅ 下单确认: {response.get('result', {}).get('orderId')}")
            pass

    def perform_rebase(self, symbol):
        """
        执行重基准操作：重置 initial_price 并归零 center_index
        """
        m = self.markets[symbol]
        
        # 1. 撤销该币种所有挂单，防止 ID 冲突
        self.engine.cancel_all_http(symbol)
        time.sleep(1) # 等待交易所处理

        # 2. 获取当前市场价格作为新锚点
        res = self.engine.http.get_tickers(category=self.engine.category, symbol=symbol)
        if res.get('retCode') == 0:
            new_price = float(res['result']['list'][0]['lastPrice'])
            
            with m.lock:
                old_price = m.initial_price
                # 核心重置
                m.initial_price = new_price 
                m.center_index = 0
                self.logger.info(f"✨ [{symbol}] 重基准完成: {old_price:.4f} -> {new_price:.4f} (Index归零)")
            
            # 3. 重新激活网格对齐
            self.reconcile_dynamic_grid(symbol)

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
                    symbol=symbol, 
                    side=side, 
                    qty=size, 
                    price=price,
                    order_type=order_type, # 🌟 这里的参数名必须与引擎类定义一致
                    link_id=f"CLOSE_{p_idx}_{int(time.time())}",
                    is_reduce=True, 
                    pos_idx=p_idx,
                    callback=self.on_place_result
                )
                self.logger.info(f"🔄 [{symbol}] {mode_str} 尝试第 {attempt+1} 次: {side} {size}")

            # 市价单通常一次见效，Maker 需要留时间观察
            time.sleep(1.0 if use_market else 5.0) 

        self.logger.error(f"❌ [{symbol}] 尝试 {max_retries} 次后仍未平仓，请检查是否有极速暴跌导致无法成交！")
        return False

    # -------------------------------------------------------------------------
    # 核心策略逻辑 (接上文 UnifiedGridBot 类内部)
    # -------------------------------------------------------------------------
    def start_dynamic_grid(self, symbol):
        """
        初始化动态网格：
        1. 锁定当前价格为 Index=0
        2. 立即触发一次对齐，挂出初始订单
        """
        m = self.markets[symbol]
        res = self.engine.http.get_tickers(category=self.engine.category, symbol=symbol)
        
        if res.get('retCode') == 0:
            current_price = float(res['result']['list'][0]['lastPrice'])
            with m.lock:
                m.initial_price = current_price # 锚定基准价
                m.center_index = 0              # 初始中心为 0
            
            self.logger.info(f"🚀 [{symbol}] 动态网格启动 | 基准价(Idx=0): {current_price:.4f}")
            # 立即执行首次挂单
            self.reconcile_dynamic_grid(symbol)

    def on_order_update(self, message):
        """WS 回调入口"""
        data = message.get('data', [])
        for order in data:
            if order['orderStatus'] == "Filled":
                self.handle_filled_order(
                    order['symbol'], 
                    order['orderLinkId'], 
                    float(order.get('avgPrice', order['price'])), # 🌟 优先使用成交均价 avgPrice
                    float(order['qty']),
                )

    def handle_filled_order(self, symbol, order_id, fill_price, qty):
        """
        ⚡ 成交处理：中心平移 + 触发对齐
        """
        m = self.markets.get(symbol)
        if not m: return

        # 1. 解析 ID 拿到网格索引
        valid, _, index, side, _ = self.parse_order_link_id(order_id)
        if not valid: return

        self.logger.info(f"⚡ [{symbol}] 索引 {index} ({side.value}) 成交 -> 触发平移")

        # 2. 🌟 核心：更新网格中心 (Center Follows Price)
        # 无论买单还是卖单成交，中心都移动到该成交位置
        # 例如：-1 买单成交，中心变为 -1。此时 -1 变成新的中轴，原本的 0 变成上方的卖单。
        with m.lock:
            if not m.initial_entry_done:
                m.initial_entry_done = True
            # 1. 计算这一单的手续费 (假设为 Maker 0.0002)
            fee = fill_price * qty * self.fee_rate
            m.total_fee_spent += fee
            
            # 2. 更新持仓成本或计算利润
            if side == OrderSide.BUY:
                # 场景：买入（增仓）-> 摊薄成本
                new_total_qty = m.current_pos_size + qty
                m.avg_entry_price = ((m.avg_entry_price * m.current_pos_size) + (fill_price * qty)) / new_total_qty
                m.current_pos_size = new_total_qty
                self.logger.debug(f"📥 [{symbol}] 买入补仓，新均价: {m.avg_entry_price:.4f}")
                
            else:
                # 场景：卖出（减仓/获利）-> 计算结转利润
                # 利润 = (卖出价 - 成本均价) * 数量 - 该笔交易手续费
                # 注意：这里我们还要扣除该笔对应的买入时的手续费(简单处理则直接扣除当前fee)
                # 局部利润 = 价格间距 - 手续费
                grid_gap_price = m.initial_price * m.config.base_offset
                trade_profit = grid_gap_price * qty - fee # 这反映了“这一格”赚了多少
                m.total_profit_usdt += trade_profit
                m.profit_count += 1
                
                # 更新持仓数量 (不影响成本均价)
                m.current_pos_size = max(0, m.current_pos_size - qty)
                # 🌟 核心修复：如果仓位清零，重置均价
                if m.current_pos_size <= 0:
                    m.avg_entry_price = 0.0
                    self.logger.info(f"✨ [{symbol}] 仓位已结清，成本均价重置")
                self.logger.info(f"💰 [{symbol}] 获利平仓! 盈亏: {trade_profit:.4f} USDT")

            # 3. 动态平移中心 (原有逻辑)
            m.center_index = index
        
        # 3. 触发动态对齐 (建议异步执行，防止阻塞 WS)
        threading.Thread(target=self.reconcile_dynamic_grid, args=(symbol,), daemon=True).start()

    def emergency_wipe_all_account_positions(self, use_market=True):
        """
        全账户地毯式清仓：无视本地配置，强平账户内所有 Linear 仓位
        """
        self.logger.critical("🚨 开始执行全账户地毯式清仓（Linear 类别）...")

        # 1. 获取账户中所有的持仓（不传 symbol 参数，API 会返回所有有变动的仓位）
        pos_res = self.engine.http.get_positions(
            category=self.engine.category, 
            settleCoin="USDT"
        )
        
        if pos_res.get('retCode') != 0:
            self.logger.error(f"❌ 无法获取全账户持仓: {pos_res.get('retMsg')}")
            return False

        # 2. 提取所有 size > 0 的仓位
        all_pos_list = pos_res.get('result', {}).get('list', [])
        active_symbols = set([p['symbol'] for p in all_pos_list if float(p.get('size', 0)) > 0])

        if not active_symbols:
            self.logger.info("✅ 账户内无任何活跃持仓。")
            return True

        self.logger.warning(f"🔍 发现待处理币种: {list(active_symbols)}")

        # 3. 针对每一个发现的币种执行撤单和平仓
        for sym in active_symbols:
            # 撤销该币种所有挂单（防止干扰平仓）
            self.engine.cancel_all_http(sym) 
            
            # 直接复用现有的平仓逻辑
            # 即使该币种不在 self.markets 中也没关系，只要有 self.engine 即可执行
            self.smart_close_all_maker(sym, max_retries=5, use_market=use_market)

        self.logger.critical("💀 全账户地毯式清仓执行完毕。")
        return True

    def reconcile_dynamic_grid(self, symbol):
        """
        🧠 策略大脑：集合差分算法
        计算 [应有订单] - [现有订单] = [需补订单]
        计算 [现有订单] - [应有订单] = [需撤订单]
        """
        m = self.markets[symbol]
        cfg = m.config
        REBASE_THRESHOLD = cfg.max_layers
        # --- 0. 检查是否触发重基准 ---
        if abs(m.center_index) >= REBASE_THRESHOLD:
            self.logger.warning(f"🔄 [{symbol}] 触发重基准：当前索引 {m.center_index} 偏离过远，正在重置坐标系...")
            self.perform_rebase(symbol)
            return # 重基准会重新触发挂单，本次对齐退出
        
        # --- 1. 获取当前真实持仓 (用于判定是否关网格) ---
        # 注意：这里需要同步获取，确保数据新鲜
        pos_res = self.engine.http.get_positions(category=self.engine.category, symbol=symbol)
        
        real_pos_size = 0.0
        if pos_res.get('retCode') == 0:
            pos_list = pos_res.get('result', {}).get('list', [])
            # 简单起见，取绝对值总和 (单向/双向兼容)
            real_pos_size = sum(abs(float(p.get('size', 0))) for p in pos_list)
        else:
            self.logger.error(f"❌ [{symbol}] 获取持仓失败: {pos_res.get('retMsg')}")
            return # 如果查询失败，不执行后续对齐，防止误判空仓
        with m.lock:
            m.current_pos_size = real_pos_size
            center = m.center_index
            layers = cfg.max_layers
            
            # --- A. 计算目标拓扑 (Target Topology) ---
            # 你的需求：以 Center 为中心，上方挂 Sell，下方挂 Buy
            target_indices = set()
            
            # 🎯 只有在初始阶段（未完成首次进场）时，才挂中心单
            if not m.initial_entry_done:
                target_indices.add(center)
                m.initial_entry_done = True
            # 下方买单: [Center-1, Center-2, ... Center-N]
            for i in range(1, layers + 1):
                target_indices.add(center - i)
                
            # 上方卖单: [Center+1, Center+2, ... Center+N]
            for i in range(1, layers + 1):
                target_indices.add(center + i)


            # =========================================================
            # 🛡️ 趋势熔断与方向过滤 (User Logic Interceptor)
            # =========================================================
            if m.market_state != MarketState.OSCILLATION:
                
                # 场景 1: 发生趋势且当前无持仓 -> 立即关停 (清空所有目标)
                if m.current_pos_size == 0:
                    self.logger.warning(f"🛑 [{symbol}] 趋势中且空仓，清空网格避险！")
                    target_indices.clear() # 这会导致下面的 Diff 逻辑撤销所有挂单
                
                else:
                    # 场景 2: 有持仓 -> 只保留平仓方向，取消开仓方向
                    # 假设单向持仓 (Long Only): Buy是开仓, Sell是平仓
                    
                    if m.market_state == MarketState.TREND_DOWN:
                        # 下跌趋势：禁止 Buy (接飞刀)，只保留 Sell (反弹减亏)
                        self.logger.warning(f"🛡️ [{symbol}] 下跌熔断：移除所有买单")
                        target_indices = {idx for idx in target_indices if idx > center}
                        
                    elif m.market_state == MarketState.TREND_UP:
                        # 上涨趋势：(在做多网格中)
                        # 逻辑歧义点：上涨时 Buy 是追高(开仓)，Sell 是止盈(平仓)。
                        # 用户要求 "取消开仓方向(Buy)，保留平仓(Sell)"
                        # 这里我们遵循用户指令：移除 Buy
                        self.logger.warning(f"🛡️ [{symbol}] 上涨熔断：暂停追涨(Buy)")
                        target_indices = {idx for idx in target_indices if idx > center}
            # =========================================================

            # --- B. 获取当前挂单 (Current Reality) ---
            # 生产环境建议先读内存(快)，配合定期 HTTP 矫正。这里直接读 HTTP 保证准确性。
            current_map = {} # {index: order_link_id}
            
            # 获取该币种所有活动委托
            open_orders = self.fetch_symbol_open_orders(symbol)
            for o in open_orders:
                valid, _, idx, _, _ = self.parse_order_link_id(o['orderLinkId'])
                if valid:
                    current_map[idx] = o['orderLinkId']
            
            current_indices = set(current_map.keys())

            # --- C. 差分计算 (Set Difference) ---
            # 1. 越界/错误的单子 -> 撤销
            to_cancel = current_indices - target_indices
            
            # 2. 缺失/新增的单子 -> 补挂
            to_place = target_indices - current_indices
            
            if not to_cancel and not to_place:
                return # 完美状态，无需操作

            self.logger.info(f"🧮 对齐: Center={center} | 撤:{list(to_cancel)} | 补:{list(to_place)}")

            # --- D. 执行撤单 ---
            for idx in to_cancel:
                self.engine.http.cancel_order(
                    category=self.engine.category, 
                    symbol=symbol, 
                    orderLinkId=current_map[idx]
                )
            
            # --- E. 执行补单 ---
            # 预计算每格资金
            self.get_wallet_balance()
            if self.current_balance <= 0: return
            budget_per_node = (self.current_balance * cfg.budget_pct) / (cfg.max_layers * 2)
            
            for idx in to_place:
                self.place_grid_order(symbol, idx, budget_per_node, center)
                time.sleep(0.1)

    def place_grid_order(self, symbol, index, budget, center_index):
        """执行具体的下单动作"""
        m = self.markets[symbol]
        price = self.get_price_by_index(symbol, index)
        if price <= 0: return

        # 🌟 修复 A: 强制最小下单价值 (Notional Value)
        # Bybit 线性合约通常要求单笔 > 5 USDT，我们设 5.5 以防万一
        safe_budget = max(budget, 5.5) 
        
        # 确定方向
        # --- 🌟 趋势驱动的下单方向判定 ---
        if index < center_index:
            side = OrderSide.BUY
        elif index > center_index:
            side = OrderSide.SELL
        else:
            # 🎯 核心逻辑：当 index 恰好在中心位置时
            # 根据趋势判定方向：m.trend_direction 为 1 (涨) 则买入，为 -1 (跌) 则卖出
            # 如果是震荡 (0)，为了降低风险，我们倾向于做空（平多）或根据持仓调整
            if m.trend_direction == 1:
                side = OrderSide.BUY
            elif m.trend_direction == -1:
                side = OrderSide.SELL
            else:
                # 震荡状态下，如果有多仓则挂卖单，无仓位则挂买单
                side = OrderSide.BUY if m.current_pos_size < 0 else OrderSide.SELL
        
        # 计算数量
        qty = safe_budget / price
        qty_str = self.adjust_qty(qty, m.config.qty_step, m.min_qty)
        
        # 4. 生成 ID
        link_id = self.generate_order_link_id(symbol, index, side)
        
        # 5. 发送 (单向模式：pos_idx=0)
        # 注意：这里 is_reduce=False，因为这是一个中性网格，卖单可能是开空，也可能是平多
        # Bybit 单向模式会自动处理对冲
        self.engine.place_order(
            symbol, side.value, qty_str, price, link_id, 
            pos_idx=0, is_reduce=False, callback=self.on_place_result
        )

    def fetch_symbol_open_orders(self, symbol):
        """辅助：获取单个币种挂单"""
        orders = []
        cursor = ""
        while True:
            res = self.engine.http.get_open_orders(
                category=self.engine.category, 
                symbol=symbol, 
                limit=50, 
                cursor=cursor
            )
            if res.get('retCode') != 0: break
            data = res.get('result', {})
            orders.extend(data.get('list', []))
            cursor = data.get('nextPageCursor', "")
            if not cursor: break
        return orders

    def update_micro_market_status_volume(self, symbol):
        """
        趋势熔断逻辑 (V9 适配版)：Z-Score 概率模型 + 筹码分布 (VPVR)
        旨在精准识别 20% 的极端波动时间
        """
        import math
        m = self.markets[symbol]
        
        # 1. 获取 K 线数据 (100根 1m 线)
        res = self.engine.http.get_kline(category=self.engine.category, symbol=symbol, interval=30, limit=100)   #1, 3, 5, 15, 30
        if res.get('retCode') != 0 or not res['result']['list']: 
            return

        # 提取收盘价与成交量
        k_list = res['result']['list']
        prices = [float(k[4]) for k in k_list]
        volumes = [float(k[5]) for k in k_list]
        prices.reverse() # 转为正序
        volumes.reverse()

        # 2. 计算 Z-Score (统计学偏离度)
        n = len(prices)
        ma = sum(prices) / n
        variance = sum((p - ma) ** 2 for p in prices) / n
        std_dev = math.sqrt(variance)
        curr = prices[-1]
        
        # 计算 Z-Score (不取绝对值，保留方向)
        z_val = (curr - ma) / std_dev if std_dev > 0 else 0
        z_score_abs = abs(z_val)
        
        # 3. 计算筹码集中度 (VPVR 简化版)
        bin_count = 10
        min_p, max_p = min(prices), max(prices)
        # 防止除零
        price_range = (max_p - min_p)
        interval = price_range / bin_count if price_range > 0 else 0.0001
        
        profile = [0.0] * bin_count
        for p, v in zip(prices, volumes):
            # 将价格归入对应的成交量桶
            idx = min(int((p - min_p) / interval), bin_count - 1)
            profile[idx] += v
        
        # 计算最大成交量桶占比
        total_vol = sum(profile)
        concentration = max(profile) / total_vol if total_vol > 0 else 0
        
        # 4. 判定状态与方向
        with m.lock:
            # 逻辑：价格偏离过大 (Z > 1.28) 或者 筹码极度分散 (Concentration < 0.10)
            # 1.28 对应正态分布双侧约 20% 的尾部区域
            # if z_score_abs > 1.5 or concentration < 0.10:
            if concentration < 0.10:
                if z_val > 0:
                    m.market_state = MarketState.TREND_UP
                    m.trend_direction = 1
                else:
                    m.market_state = MarketState.TREND_DOWN
                    m.trend_direction = -1
                
                self.logger.warning(f"🚨 [{symbol}] 趋势触发: Z={z_val:.2f}, POC={concentration:.2f}")
            else:
                m.market_state = MarketState.OSCILLATION
                m.trend_direction = 0

    def update_micro_market_status(self, symbol):
        """
        趋势判定引擎 (V9版)
        逻辑：利用 Z-Score 判定价格是否偏离均线过远
        """
        m = self.markets[symbol]
        
        # 1. 获取 K 线 (1分钟线，取最近 60 根)
        # 这里的 interval=1 可以根据你的网格频率调整，高频用 1 或 5
        res = self.engine.http.get_kline(category=self.engine.category, symbol=symbol, interval=1, limit=60)
        
        if res.get('retCode') != 0 or not res['result']['list']: 
            return

        # Bybit 返回的数据是反序的 (最新在最前)，需要反转
        k_data = res['result']['list']
        prices = [float(k[4]) for k in k_data] # Close price
        # volumes = [float(k[5]) for k in k_data] # Volume (可选 VPVR 逻辑)
        
        # 2. 计算统计指标
        n = len(prices)
        avg_price = sum(prices) / n
        variance = sum((p - avg_price) ** 2 for p in prices) / n
        std_dev = math.sqrt(variance)
        
        current_price = prices[0] # 最新的收盘价
        
        # 3. 计算 Z-Score (当前价格偏离了多少个标准差)
        if std_dev == 0: z_score = 0
        else: z_score = (current_price - avg_price) / std_dev
        
        # 4. 设定阈值 (例如 2.0 表示偏离非常大)
        THRESHOLD = 2.0 
        
        with m.lock:
            if z_score > THRESHOLD:
                m.market_state = MarketState.TREND_UP
                m.trend_direction = 1
                self.logger.warning(f"📈 [{symbol}] 识别到单边上涨趋势 (Z:{z_score:.2f})")
            elif z_score < -THRESHOLD:
                m.market_state = MarketState.TREND_DOWN
                m.trend_direction = -1
                self.logger.warning(f"📉 [{symbol}] 识别到单边下跌趋势 (Z:{z_score:.2f})")
            else:
                m.market_state = MarketState.OSCILLATION
                m.trend_direction = 0
                # self.logger.info(f"〰️ [{symbol}] 震荡行情 (Z:{z_score:.2f})")

    def get_uptime(self) -> str:
        """
        计算并格式化系统运行时间
        """
        uptime_sec = time.time() - self.start_time
        days, remainder = divmod(int(uptime_sec), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        else:
            return f"{minutes}m {seconds}s"
        
    def report_status(self):
        """
        实战简报升级版：增加 Tick 效能分析
        """
        self.get_wallet_balance()
        uptime_str = self.get_uptime() 

        # 调整表头宽度以容纳新列
        print("\n" + "═"*110)
        print(f"📊 {self.version} 实战简报 | 运行: {uptime_str} | 余额: {self.current_balance:.2f} USDT")
        print("─" * 110)
        
        # 增加 GAP(T) -> 物理间距Tick数，FEE(T) -> 手续费消耗Tick数
        header = f"{'SYMBOL':<10} | {'CTR':<5} | {'GAP(T)':<8} | {'FEE(T)':<8} | {'NET(T)':<8} | {'PROFIT(U)':<12} | {'COUNT':<6}"
        print(header)
        print("─" * 110)

        total_p = 0.0
        total_count = 0
        for s, m in self.markets.items():
            with m.lock:
                total_p += m.total_profit_usdt
                total_count += m.profit_count
                
                # --- 动态 Tick 效能计算 ---
                # 使用基准价计算当前的 Tick 覆盖情况
                ref_price = m.initial_price if m.initial_price > 0 else 1.0
                
                # 1. 物理间距 Tick 数: (价格 * 间距比例) / TickSize
                gap_ticks = (ref_price * m.config.base_offset) / m.tick_size if m.tick_size > 0 else 0
                
                # 2. 手续费 Tick 数: (价格 * 双边费率) / TickSize
                # 我们假设双边均为 Maker (0.0002 * 2)
                fee_ticks = (ref_price * (self.fee_rate * 2)) / m.tick_size if m.tick_size > 0 else 0
                
                # 3. 净利润 Tick 数
                net_ticks = gap_ticks - fee_ticks
                
                # 状态着色提醒逻辑 (仅用于参考)
                status_icon = "✅" if net_ticks > 3 else "⚠️"
                
                print(f"{s:<10} | {m.center_index:<5} | {gap_ticks:>8.1f} | {fee_ticks:>8.1f} | {net_ticks:>8.1f} {status_icon} | {m.total_profit_usdt:>12.4f} | {m.profit_count:<6}")
        
        print("─" * 110)
        print(f"📈 累计总实现利润: {total_p:.4f} USDT | COUNT {total_count}")
        print("═"*110 + "\n")
    # -------------------------------------------------------------------------
    # 后台循环
    # -------------------------------------------------------------------------
    def run_loop(self):
        """后台守护进程：打印状态 + 兜底对齐"""
        last_report_time = 0
        while not self.stop_signal:
            # 1. 刷新余额并进行安全检查
            self.get_wallet_balance()
            if not self.check_account_safe():
                break # 触发熔断，跳出循环

            # 2. 原有的报表与状态更新逻辑
            if time.time() - last_report_time > 30:
                self.report_status()
                for s, m in self.markets.items():
                    self.update_micro_market_status_volume(s)
                last_report_time = time.time()
            time.sleep(10) # 每10秒打印一次
            
            self.logger.info("-" * 40)
            self.logger.info(f"💰 余额: {self.current_balance:.2f}")
            
            for s, m in self.markets.items():
                # --- 0. 先更新一下趋势状态 ---
                # 建议在主循环定期做，或者在这里每次做（会增加耗时）
                with m.lock:
                    ctr = m.center_index
                    # 估算当前挂单范围
                    low = ctr - m.config.max_layers
                    high = ctr + m.config.max_layers
                    self.logger.info(f"   {s}: Center={ctr} | Range=[{low}, {high}] | BasePrice={m.initial_price:.4f}")
            self.logger.info("-" * 40)
        self.logger.info("👋 run_loop 已退出")
        # 此时通过 os._exit 确保主进程也一起带走
        os._exit(0)
# -----------------------------------------------------------------------------
# 程序入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 配置路径
    BASE = os.path.dirname(os.path.abspath(__file__))
    API_K = os.path.join(BASE, "keys", "hmac_api_key")
    API_S = os.path.join(BASE, "keys", "hmac_secret")
    # WS 如果需要 RSA 则配置，不需要则留空或沿用旧逻辑
    RSA_K = os.path.join(BASE, "keys", "api_key")     
    RSA_P = os.path.join(BASE, "keys", "bybit_rsa.pem")

    CONFIGS = {
        "DOGEUSDT": {"budget_pct": 0.5, "max_layers": 10, "base_offset": 0.0008, "qty_step": 0},
        "PIPPINUSDT": {"budget_pct": 0.3, "max_layers": 10, "base_offset": 0.003, "qty_step": 0}, 
        "RAVEUSDT": {"budget_pct": 0.8, "max_layers": 10, "base_offset": 0.004, "qty_step": 0},     #fee rate 0.0004
        "MNTUSDT": {"budget_pct": 0.5, "max_layers": 10, "base_offset": 0.001, "qty_step": 0}, 
        "BEATUSDT": {"budget_pct": 0.3, "max_layers": 10, "base_offset": 0.004, "qty_step": 0}, #fee rate 0.0004
        "ADAUSDT": {"budget_pct": 0.3, "max_layers": 10, "base_offset": 0.0012, "qty_step": 0}, 

        # "ARCUSDT":  {"budget_pct": 0.5, "max_layers": 4, "base_offset": 0.002, "qty_step": 0}, 
        # "SUIUSDT": {"budget_pct": 0.3, "max_layers": 4, "base_offset": 0.002, "qty_step": 0}, 
        # "LUNA2USDT": {"budget_pct": 0.3, "max_layers": 4, "base_offset": 0.002, "qty_step": 0},
    }
    # 初始化引擎
    # 注意：engine 代码需支持 place_order(..., pos_idx=0)
    engine = BybitEngine(API_K, API_S, RSA_K, RSA_P)
    
    # 启动机器人
    bot = UnifiedGridBot(engine, CONFIGS, clean =1) #1:close all exist order

    # 信号处理 (Ctrl+C 退出)
    def signal_handler(sig, frame):
        print("\n👋 正在停止...")
        bot.stop_signal = True
        engine.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    # GO!
    bot.startup()